"""Reproducible orchestration for the open-geometry generation pipeline."""

from __future__ import annotations

import json
import inspect
from collections import Counter
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from .appearance import _select_reference, checkerboard_preview, render_appearance, scale_preview
from .contracts import (
    AppearanceSpec,
    AssetRequest,
    GeometrySpec,
    ReferenceAsset,
    ReferenceRole,
    ShapeDescriptor,
    ValidationResult,
    is_overlay_part,
    to_jsonable,
)
from .geometry import CompiledGeometry, compile_geometry
from .packaging import build_resourcepack
from .reference_geometry import (
    reference_fallback_geometry,
    reference_partition_geometry,
    reference_silhouette_partition,
)
from .uv_layout import render_entity_preview, render_front_preview, render_isometric_preview
from .validation import SemanticCritic, StructuralCritic, aggregate_results, validate_geometry, validate_render_alpha


def _geometry_can_be_rendered(spec: GeometrySpec, compiled: CompiledGeometry) -> bool:
    """Allow visual feedback for a compilable candidate with soft metric errors.

    Size, occupancy and model-authored metric constraints are useful quality
    signals, but they should not hide a raster from the blind reviewer. The
    renderer still requires a non-empty union, every required physical part,
    and every required connection; these are the structural conditions that
    make a candidate meaningful to revise.
    """
    if compiled.mask.getbbox() is None:
        return False
    part_points: dict[str, set[tuple[int, int]]] = {}
    part_by_id = {part.id: part for part in spec.parts}
    for part in spec.parts:
        points = {
            (x, y)
            for y in range(compiled.height)
            for x in range(compiled.width)
            if compiled.part_masks[part.id].getpixel((x, y)) > 0
        }
        part_points[part.id] = points
        if part.required and not part.paint_only and not points:
            return False
    neighbours = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    for connection in spec.connections:
        if not connection.required:
            continue
        first = part_points.get(connection.a, set())
        second = part_points.get(connection.b, set())
        if part_by_id[connection.a].paint_only or part_by_id[connection.b].paint_only:
            continue
        if not first or not second:
            return False
        if not first.intersection(second) and not any(
            (x + dx, y + dy) in second
            for x, y in first
            for dx, dy in neighbours
        ):
            return False
    return True


def _reference_shape_is_locked(descriptor: ShapeDescriptor) -> bool:
    """Read a model-authored strategy for a no-contour-change variant.

    Two statements can answer this question, and they disagree more often than
    they should. The art direction is a reference-free brief with one boolean
    summarising several sentences; the descriptor is written *after* seeing the
    source raster and carries a structured mode plus the concrete strategy.

    The structured mode therefore outranks the boolean, because the boolean is
    a lossy summary that regularly contradicts its own prose. A live crystal
    bow wrote requires_silhouette_change=true next to "the outer bow contour
    stays continuous ... without opening the bow body contour" and
    shape_edit_mode=local_silhouette_edit with no named edit, so the anchor
    frame skipped the contour lock and shipped the *next* frame's silhouette.

    The brief still decides when the descriptor does not: a bare
    model_decides run, or one whose strategy names no contour at all, keeps
    its original veto -- which is what protects the broken-limb case, where
    the strategy names the edit the brief asked for.
    """
    direction = descriptor.art_direction
    mode = str(descriptor.shape_edit_mode or "").strip().lower()
    strategy = str(descriptor.reference_strategy).lower()
    preserve_cues = (
        "same silhouette", "same contour", "preserve", "unchanged", "keep the same",
        "保持", "不改变", "保留",
    )
    edit_cues = (
        "break", "broken", "chip", "chipped", "damage", "damaged", "fracture",
        "notch", "tear", "new silhouette", "new shape", "redesign", "change contour",
        "改变轮廓", "断裂", "缺口", "新轮廓", "重设计",
    )
    named_edit = any(cue in strategy for cue in edit_cues)
    # `shape_edit_mode` is the structured form of the question the prose below
    # is scraped for, and it is answered after the source raster is visible.
    if mode in {"appearance_only", "preserve_silhouette"}:
        return True
    if mode == "new_silhouette":
        return False
    # "local_silhouette_edit" is the smallest model-authored alpha edit *at a
    # named host*. With no host named there is no edit to make, so the source
    # contour is the answer after all. The label on its own is not evidence:
    # one crystal-bow run wrote appearance_only and the next wrote
    # local_silhouette_edit for the same recolour, with a strategy that only
    # said to use the source for silhouette and proportions.
    if mode == "local_silhouette_edit":
        return not named_edit
    # Nothing structured was said, so the reference-free brief decides. This is
    # the broken-limb guard: it asked for a frayed contour and wrote "preserve
    # the intact limb arc", and only the brief keeps that from locking.
    if direction is not None and direction.requires_silhouette_change:
        return False
    if named_edit:
        return False
    if any(cue in strategy for cue in preserve_cues):
        return True
    # A model often says "use the shape reference ... apply the material"
    # without repeating the word preserve, so a material/palette-only strategy
    # is itself a same-silhouette cue. It used to require an ownership map as
    # corroboration, but the model supplies that map rarely, so the cue almost
    # never fired; the map is no longer load-bearing now that the geometry
    # stage can partition the source from the part masks directly.
    material_cues = (
        "material", "palette", "recolor", "recolour", "hue", "finish",
        "材质", "材质", "调色", "改色",
    )
    return any(cue in strategy for cue in material_cues)


def _candidate_shape_matches_reference(
    compiled: CompiledGeometry,
    references: list[ReferenceAsset],
    threshold: float = 0.9,
) -> bool:
    """Compare a candidate alpha mask with an exact same-size shape reference."""
    candidate = {
        (x, y)
        for y in range(compiled.height)
        for x in range(compiled.width)
        if compiled.mask.getpixel((x, y)) > 0
    }
    for reference in references:
        if ReferenceRole.SHAPE not in reference.roles:
            continue
        try:
            with Image.open(reference.path) as loaded:
                source = loaded.convert("RGBA")
                if source.size != (compiled.width, compiled.height):
                    continue
                expected = {
                    (x, y)
                    for y in range(compiled.height)
                    for x in range(compiled.width)
                    if source.getpixel((x, y))[3] >= 8
                }
        except (OSError, ValueError):
            continue
        union = candidate | expected
        return len(candidate & expected) / float(max(len(union), 1)) >= threshold
    return True


def _target_overlay_descriptor(descriptor: ShapeDescriptor) -> bool:
    """Whether a descriptor has source supports plus a local overlay part.

    A target map is helpful but optional after descriptor correction.  Source
    partition evidence still lets the pipeline restore a support the geometry
    model omitted while leaving the new motif to the model-authored geometry.
    """
    if not isinstance(descriptor.reference_part_map, dict):
        return False
    text = " ".join(
        [descriptor.target, descriptor.semantic, descriptor.reference_strategy]
        + list(descriptor.visual_identity)
        + [part.meaning + " " + part.style_role for part in descriptor.parts]
    ).lower()
    return any(token in text for token in (
        "embedded", "inlaid", "inlay", "inserted", "mounted", "attached",
        "motif", "emblem", "boss", "socket", "gem", "eye", "镶嵌", "嵌入",
    ))


def _source_supports_drift(
    candidate: CompiledGeometry,
    baseline: CompiledGeometry,
    descriptor: ShapeDescriptor,
) -> bool:
    """Detect a model draft that redesigned unchanged support parts.

    The comparison is driven by the descriptor's own part roles and maps. It
    does not know what a sword, tool or creature is; it only protects source
    supports when a model explicitly declared a local target overlay.
    """
    support_ids = [part.id for part in descriptor.parts if not is_overlay_part(part)]
    if not support_ids:
        return False
    for part_id in support_ids:
        candidate_points = {
            (x, y)
            for y in range(candidate.height)
            for x in range(candidate.width)
            if candidate.part_masks[part_id].getpixel((x, y)) > 0
        }
        baseline_points = {
            (x, y)
            for y in range(baseline.height)
            for x in range(baseline.width)
            if baseline.part_masks[part_id].getpixel((x, y)) > 0
        }
        if not baseline_points:
            continue
        iou = len(candidate_points & baseline_points) / float(max(len(candidate_points | baseline_points), 1))
        if iou < 0.72:
            return True
    return False


def _overlay_part_ids(descriptor: ShapeDescriptor) -> set[str]:
    return {part.id for part in descriptor.parts if is_overlay_part(part)}


class GeometryRepairer(Protocol):
    """Optional repair loop for model-produced geometry that failed hard checks."""

    def repair_geometry(
        self,
        request: AssetRequest,
        descriptor: ShapeDescriptor,
        geometry: GeometrySpec,
        validation: ValidationResult,
        compiled: CompiledGeometry | None,
        references: list[ReferenceAsset] | None = None,
    ) -> GeometrySpec:
        """Return a replacement geometry spec with the same declared parts."""


class _ReferenceFallbackRepairer:
    """Conservative final fallback used only by an explicit auto request."""

    def __init__(self, references: list[ReferenceAsset]) -> None:
        self.references = references

    def repair_geometry(self, request, descriptor, geometry, validation, compiled):
        fallback = reference_partition_geometry(
            descriptor,
            self.references,
            request.width,
            request.height,
            getattr(request, "name", None),
        ) or reference_fallback_geometry(
            descriptor,
            self.references,
            request.width,
            request.height,
            getattr(request, "name", None),
        )
        if fallback is None:
            raise ValueError("reference fallback requires model-authored partition evidence or one required part")
        return fallback


class _ReferenceAwareRepairer:
    """Keep a model repair when it is usable, otherwise recover the source map."""

    def __init__(self, primary: GeometryRepairer, references: list[ReferenceAsset]) -> None:
        self.primary = primary
        self.references = references

    def repair_geometry(self, request, descriptor, geometry, validation, compiled, references=None):
        candidate = None
        primary_error: Exception | None = None
        try:
            repair_fn = self.primary.repair_geometry
            args = (request, descriptor, geometry, validation, compiled)
            if "references" in inspect.signature(repair_fn).parameters:
                candidate = repair_fn(*args, references=self.references)
            else:
                candidate = repair_fn(*args)
            candidate_compiled = compile_geometry(candidate)
            if (
                _geometry_can_be_rendered(candidate, candidate_compiled)
                and (
                    not _reference_shape_is_locked(descriptor)
                    or _candidate_shape_matches_reference(candidate_compiled, self.references)
                )
            ):
                return candidate
        except Exception as exc:  # noqa: BLE001 - fallback remains auditable
            primary_error = exc
        member_name = getattr(request, "name", None)
        fallback = reference_partition_geometry(
            descriptor, self.references, request.width, request.height, member_name
        )
        if fallback is None and _reference_shape_is_locked(descriptor):
            # Same recovery the render path uses, so a locked request whose
            # repair kept returning a loose contour still lands on the source.
            fallback = reference_silhouette_partition(
                descriptor,
                geometry,
                self.references,
                request.width,
                request.height,
                member_name,
            )
        fallback = fallback or reference_fallback_geometry(
            descriptor, self.references, request.width, request.height, member_name
        )
        if fallback is not None:
            return fallback
        if candidate is not None:
            return candidate
        if primary_error is not None:
            raise primary_error
        raise ValueError("model repair returned no geometry")


def _rgb_luma(pixel: tuple[int, int, int, int]) -> float:
    return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]


def _texture_pattern_metrics(
    source: Image.Image,
    rendered: Image.Image,
    bbox: tuple[int, int, int, int],
) -> dict[str, float | int | None]:
    """Compare local value structure without requiring identical colours."""
    left, top, right, bottom = bbox
    pairs: list[tuple[float, float]] = []
    for y in range(top, bottom):
        for x in range(left, right):
            source_pixel = source.getpixel((x, y))
            rendered_pixel = rendered.getpixel((x, y))
            if source_pixel[3] >= 8 and rendered_pixel[3] >= 8:
                pairs.append((_rgb_luma(source_pixel), _rgb_luma(rendered_pixel)))
    if len(pairs) < 2:
        return {
            "opaque_overlap_pixels": len(pairs),
            "luma_correlation": None,
            "high_low_iou": None,
            "edge_agreement": None,
        }
    source_mean = sum(pair[0] for pair in pairs) / len(pairs)
    rendered_mean = sum(pair[1] for pair in pairs) / len(pairs)
    covariance = sum((first - source_mean) * (second - rendered_mean) for first, second in pairs)
    source_variance = sum((first - source_mean) ** 2 for first, _second in pairs)
    rendered_variance = sum((second - rendered_mean) ** 2 for _first, second in pairs)
    denominator = sqrt(source_variance * rendered_variance)
    correlation = covariance / denominator if denominator > 1e-9 else 1.0
    source_sorted = sorted(pair[0] for pair in pairs)
    rendered_sorted = sorted(pair[1] for pair in pairs)
    source_threshold = source_sorted[len(source_sorted) // 2]
    rendered_threshold = rendered_sorted[len(rendered_sorted) // 2]
    source_high = {
        index for index, pair in enumerate(pairs) if pair[0] >= source_threshold
    }
    rendered_high = {
        index for index, pair in enumerate(pairs) if pair[1] >= rendered_threshold
    }
    union = source_high | rendered_high
    intersection = source_high & rendered_high
    high_low_iou = len(intersection) / float(max(len(union), 1))
    edge_total = edge_same = 0
    # Compare local transitions in row order; this catches a missing patch
    # even when the overall face has a similar average brightness.
    for index in range(len(pairs) - 1):
        source_delta = pairs[index + 1][0] - pairs[index][0]
        rendered_delta = pairs[index + 1][1] - pairs[index][1]
        if abs(source_delta) < 10.0 and abs(rendered_delta) < 10.0:
            continue
        edge_total += 1
        if (source_delta >= 0.0) == (rendered_delta >= 0.0):
            edge_same += 1
    return {
        "opaque_overlap_pixels": len(pairs),
        "luma_correlation": round(max(-1.0, min(1.0, correlation)), 4),
        "high_low_iou": round(high_low_iou, 4),
        "edge_agreement": round(edge_same / float(max(edge_total, 1)), 4),
    }


def _reference_for_comparison(
    references: list[ReferenceAsset],
    size: tuple[int, int],
) -> tuple[Image.Image, str, str] | None:
    """Load the same evidence image used by the texture audit.

    A block face is often stored as a compact 16x16 tile while the generated
    atlas repeats it horizontally.  Expanding that tile here keeps the visual
    comparison and the numeric comparison on the same pixel coordinates.  A
    style/material reference is preferred when several references share a
    canvas, but a shape-only reference remains a usable fallback for custom
    callers that supplied no paint evidence.
    """
    # Shape/alpha comparison must use the selected structural reference.  A
    # secondary material or motif swatch can share the canvas size (an eye
    # sprite beside a sword, for example); preferring it here makes the audit
    # report compare the generated sword against the eye and produces a false
    # shape-drift diagnosis.  Material/palette references remain available to
    # the appearance planner and are used only as a fallback when no shape
    # raster matches the generated canvas.
    shape_roles = {"shape"}
    preferred_roles = {"pixel_style", "material", "palette"}
    ordered = sorted(
        enumerate(references),
        key=lambda item: (
            not any(getattr(role, "value", role) in shape_roles for role in item[1].roles),
            # Once a same-size shape exists, preserve the caller/router order
            # among shape references; only use material/palette preference
            # when no structural raster is available at all.
            0 if any(getattr(role, "value", role) in shape_roles for role in item[1].roles)
            else not any(getattr(role, "value", role) in preferred_roles for role in item[1].roles),
            item[0],
        ),
    )
    width, height = size
    for _index, reference in ordered:
        try:
            with Image.open(reference.path) as loaded:
                image = loaded.convert("RGBA")
        except (OSError, ValueError):
            continue
        if image.size == size:
            return image, reference.path, "exact"
        if image.height == height and image.width < width and width % image.width == 0:
            expanded = Image.new("RGBA", size, (0, 0, 0, 0))
            for left in range(0, width, image.width):
                expanded.alpha_composite(image, (left, 0))
            return expanded, reference.path, "face_tile"
    return None


def _alpha_bbox(pixels: list[tuple[int, int, int, int]], width: int, height: int) -> list[int] | None:
    points = [
        (index % width, index // width)
        for index, pixel in enumerate(pixels)
        if pixel[3] >= 8
    ]
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]


def _reference_comparison_data(
    source: Image.Image,
    generated: Image.Image,
    reference_path: str,
    reference_mode: str,
) -> dict[str, object]:
    """Return compact shape, colour and texture diagnostics for two RGBA images."""
    source_pixels = list(source.convert("RGBA").get_flattened_data())
    generated_pixels = list(generated.convert("RGBA").get_flattened_data())
    width, height = source.size
    source_opaque = [pixel[3] >= 8 for pixel in source_pixels]
    generated_opaque = [pixel[3] >= 8 for pixel in generated_pixels]
    intersection = sum(a and b for a, b in zip(source_opaque, generated_opaque))
    union = sum(a or b for a, b in zip(source_opaque, generated_opaque))
    source_count = sum(source_opaque)
    generated_count = sum(generated_opaque)
    alpha_extra = sum(b and not a for a, b in zip(source_opaque, generated_opaque))
    alpha_missing = sum(a and not b for a, b in zip(source_opaque, generated_opaque))
    changed: list[tuple[int, int]] = []
    color_changed = 0
    rgb_abs_total = 0
    luma_abs_total = 0.0
    source_luma_total = 0.0
    generated_luma_total = 0.0
    diff_pixels: list[tuple[int, int, int, int]] = []
    for index, (source_pixel, generated_pixel) in enumerate(zip(source_pixels, generated_pixels)):
        source_is_opaque = source_pixel[3] >= 8
        generated_is_opaque = generated_pixel[3] >= 8
        if source_is_opaque != generated_is_opaque:
            changed.append((index % width, index // width))
            diff_pixels.append((225, 50, 190, 255))
            continue
        if source_is_opaque and generated_is_opaque:
            source_luma = _rgb_luma(source_pixel)
            generated_luma = _rgb_luma(generated_pixel)
            source_luma_total += source_luma
            generated_luma_total += generated_luma
            rgb_delta = sum(abs(source_pixel[channel] - generated_pixel[channel]) for channel in range(3))
            rgb_abs_total += rgb_delta
            luma_abs_total += abs(source_luma - generated_luma)
            if source_pixel[:3] != generated_pixel[:3]:
                color_changed += 1
                changed.append((index % width, index // width))
                diff_pixels.append((255, 145, 0, 255))
            else:
                diff_pixels.append((70, 70, 70, 255))
        else:
            diff_pixels.append((24, 24, 24, 255))
    if changed:
        xs = [point[0] for point in changed]
        ys = [point[1] for point in changed]
        changed_bbox: list[int] | None = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    else:
        changed_bbox = None
    overlap_count = intersection
    texture = _texture_pattern_metrics(source, generated, (0, 0, width, height))
    shape_iou = intersection / float(max(union, 1))
    texture_correlation = texture.get("luma_correlation")
    texture_edges = texture.get("edge_agreement")
    if shape_iou >= 0.9:
        shape_signal = "aligned"
    elif shape_iou >= 0.6:
        shape_signal = "partially_changed"
    else:
        shape_signal = "diverged"
    if color_changed == 0:
        color_signal = "unchanged"
    elif color_changed / float(max(overlap_count, 1)) >= 0.5:
        color_signal = "substantially_changed"
    else:
        color_signal = "locally_changed"
    if overlap_count < 8 or texture_correlation is None:
        texture_signal = "insufficient_overlap"
    elif float(texture_correlation) >= 0.8 and float(texture_edges or 0.0) >= 0.7:
        texture_signal = "preserved"
    else:
        texture_signal = "diverged"
    return {
        "reference": reference_path,
        "reference_mode": reference_mode,
        "source_size": [width, height],
        "generated_size": [generated.width, generated.height],
        "changed_pixels": len(changed),
        "changed_bbox": changed_bbox,
        "shape": {
            "source_opaque_pixels": source_count,
            "generated_opaque_pixels": generated_count,
            "intersection_pixels": intersection,
            "union_pixels": union,
            "alpha_extra_pixels": alpha_extra,
            "alpha_missing_pixels": alpha_missing,
            "iou": round(intersection / float(max(union, 1)), 4),
            "precision": round(intersection / float(max(generated_count, 1)), 4),
            "recall": round(intersection / float(max(source_count, 1)), 4),
            "source_bbox": _alpha_bbox(source_pixels, width, height),
            "generated_bbox": _alpha_bbox(generated_pixels, width, height),
        },
        "color": {
            "overlap_pixels": overlap_count,
            "changed_pixels": color_changed,
            "change_ratio": round(color_changed / float(max(overlap_count, 1)), 4),
            "source_unique_colors": len(Counter(pixel[:3] for pixel in source_pixels if pixel[3] >= 8)),
            "generated_unique_colors": len(Counter(pixel[:3] for pixel in generated_pixels if pixel[3] >= 8)),
            "mean_abs_rgb_delta": round(rgb_abs_total / float(max(overlap_count * 3, 1)), 4),
            "mean_abs_luma_delta": round(luma_abs_total / float(max(overlap_count, 1)), 4),
            "mean_source_luma": round(source_luma_total / float(max(overlap_count, 1)), 4),
            "mean_generated_luma": round(generated_luma_total / float(max(overlap_count, 1)), 4),
        },
        "texture": texture,
        "assessment": {
            "shape": shape_signal,
            "color": color_signal,
            "texture": texture_signal,
            "meaning": "descriptive comparison signals; intentional new silhouettes or palettes still require human/model judgement",
        },
        "_diff_pixels": diff_pixels,
    }


def is_material_only_descriptor(descriptor: ShapeDescriptor) -> bool:
    """Recognize a material/swatch request without an object noun."""
    if len(descriptor.parts) != 1:
        return False
    text = " ".join([
        str(descriptor.target),
        str(descriptor.semantic),
        *[str(item) for item in descriptor.visual_identity],
    ]).lower()
    material_cues = {
        "material", "material sample", "swatch", "patch", "leather", "hide", "cloth", "fabric",
        "texture sample", "皮革", "材料", "材质", "样本", "皮片", "布料",
    }
    return any(cue in text for cue in material_cues)


@dataclass(frozen=True)
class GenerationPlan:
    request: AssetRequest
    descriptor: ShapeDescriptor
    geometry: GeometrySpec
    appearance: AppearanceSpec
    references: list[ReferenceAsset] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.geometry.width != self.request.width or self.geometry.height != self.request.height:
            raise ValueError("geometry dimensions must match asset request")
        descriptor_parts = {part.id for part in self.descriptor.parts}
        geometry_parts = {part.id for part in self.geometry.parts}
        # Paint-only parts are intentionally rendered inside an existing host
        # through AppearanceSpec; requiring an alpha/UV geometry part for them
        # contradicts that contract. Physical required parts still need one.
        missing = {
            part.id for part in self.descriptor.parts
            if part.required and not part.paint_only
        } - geometry_parts
        if missing:
            raise ValueError("geometry does not implement descriptor parts: %s" % ", ".join(sorted(missing)))
        if not descriptor_parts.issuperset(geometry_parts):
            raise ValueError("geometry includes parts absent from descriptor")


@dataclass(frozen=True)
class GenerationResult:
    out_dir: Path
    compiled: CompiledGeometry | None
    sprite_path: Path | None
    validation: ValidationResult
    artifacts: list[Path]


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _save_masks(compiled: CompiledGeometry, out_dir: Path, *, entity_uv: bool = False) -> list[Path]:
    """Save compiler evidence without calling entity UV coverage an alpha mask."""
    mask_dir = out_dir / ("coverage" if entity_uv else "masks")
    part_dir = mask_dir / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    result = mask_dir / ("uv_coverage.png" if entity_uv else "mask.png")
    compiled.mask.save(result, "PNG")
    paths = [result]
    for part_id, mask in compiled.part_masks.items():
        path = part_dir / (part_id + ".png")
        mask.save(path, "PNG")
        paths.append(path)
    return paths


@dataclass(frozen=True)
class _ReferenceAlphaContract:
    """Separate source transparency from UV coverage for an unchanged model."""

    alpha: Image.Image
    source: Image.Image
    reference_path: str


# A same-size source may only act as the alpha authority when it stays inside
# this model's declared UV regions. Two armour layers are both 64x32, so size
# alone once handed layer 1's alpha to a layer 2 target and stamped a helmet
# and arms onto a pair of leggings.
_ALPHA_REGION_TOLERANCE = 0.1
# A model texture paints faces, so its opaque pixels form a few large blobs.
# An additive layer (a leather overlay, a trim, a dither) is instead hundreds
# of isolated pixels. Measured on 1.12.2 armour: every real layer scored 1-3
# components with 0% single-pixel fragments, while a leather overlay scored 338
# components of exactly one pixel each.
_ALPHA_MIN_SOLID_SHARE = 0.9
_ALPHA_SOLID_MINIMUM = 3
# ...and it must actually cover the model. A leather trim overlay is solid but
# sparse: it paints 19% of a leggings atlas where a real armour layer paints
# 41-49%, so accepting it left the plate mostly transparent. Both figures are
# measured from 1.12.2 armour, and a full layer keeps a comfortable margin.
_ALPHA_MIN_REGION_COVERAGE = 0.3


def _alpha_region_coverage(alpha: Image.Image, region_mask: set[tuple[int, int]]) -> float:
    """Share of the model's declared UV pixels this source actually paints."""
    if not region_mask:
        return 0.0
    covered = sum(1 for (x, y) in region_mask if alpha.getpixel((x, y)) >= 8)
    return covered / float(len(region_mask))


def _alpha_solid_share(alpha: Image.Image) -> float:
    """Share of opaque pixels that sit in a solid area rather than a dither."""
    points = {
        (x, y)
        for y in range(alpha.height)
        for x in range(alpha.width)
        if alpha.getpixel((x, y)) >= 8
    }
    if not points:
        return 0.0
    solid = 0
    seen: set[tuple[int, int]] = set()
    for start in points:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        group = 0
        while stack:
            x, y = stack.pop()
            group += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (x + dx, y + dy)
                if neighbour in points and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if group >= _ALPHA_SOLID_MINIMUM:
            solid += group
    return solid / float(len(points))


def _uv_region_mask(regions: list[UvRegionSpec], size: tuple[int, int]) -> set[tuple[int, int]]:
    """Every pixel the model's declared UV regions actually occupy."""
    mask: set[tuple[int, int]] = set()
    for region in regions:
        left, top, right, bottom = region.bbox
        for y in range(max(0, top), min(size[1], bottom)):
            for x in range(max(0, left), min(size[0], right)):
                mask.add((x, y))
    return mask


def _alpha_outside_ratio(alpha: Image.Image, region_mask: set[tuple[int, int]]) -> float | None:
    """Share of a source's opaque pixels that fall outside those regions."""
    opaque = 0
    outside = 0
    for y in range(alpha.height):
        for x in range(alpha.width):
            if alpha.getpixel((x, y)) >= 8:
                opaque += 1
                if (x, y) not in region_mask:
                    outside += 1
    if opaque == 0:
        return None
    return outside / float(opaque)


def _model_authored_the_paint_plan(plan: GenerationPlan) -> bool:
    """True when the model decided something about which cells to paint.

    This only matters under shape_policy "free", where a new design outranks
    conformance to the source model. Under "target_model_uv" the caller has
    asked for that model's exact contract, so the reference alpha still refines
    the silhouette and only a full, undecided base coat is left to it.

    Filling every declared region and carving nothing is not a decision. Any
    omitted region or any cutout is one, and an alpha authority must never paint
    over it: that is how a metal layer's grey once got inside a wooden helmet.
    """
    regions = list(plan.geometry.uv_regions)
    if not regions:
        return False
    if any(primitive.primitive == "cutout" for primitive in plan.geometry.primitives):
        return True
    filled: set[str] = set()
    for primitive in plan.geometry.primitives:
        if primitive.primitive != "uv_fill":
            continue
        selected = primitive.params.get("regions")
        if isinstance(selected, list):
            filled.update(str(item) for item in selected)
        else:
            filled.update(region.id for region in regions if region.part_id == primitive.part_id)
    return any(region.id not in filled for region in regions)


def _entity_reference_alpha_contract(plan: GenerationPlan) -> _ReferenceAlphaContract | None:
    """Load an exact source alpha only when the request keeps its target model.

    A material swatch must never unexpectedly become an entity alpha mask.
    `target_model_uv` is the explicit promise that the chosen source texture
    and model layout describe the same model, so the alpha authority is the
    matching-size source that actually agrees with this layout: a reference
    that paints outside the declared UV regions describes a different model and
    is rejected, leaving the compiled geometry mask in charge.
    """
    if (
        plan.request.form.value != "entity_uv"
        or plan.request.shape_policy != "target_model_uv"
    ):
        return None
    if plan.request.shape_policy == "free" and _model_authored_the_paint_plan(plan):
        # The model expressed a design decision about this atlas, so the
        # reference alpha must not undo it. That alpha exists to refine a plan
        # nobody made; an authority that paints over an authored opening is how
        # a metal layer's grey once appeared inside a wooden helmet.
        return None
    expected_size = (plan.request.width, plan.request.height)
    region_mask = _uv_region_mask(list(plan.geometry.uv_regions), expected_size)
    if not region_mask:
        return None
    ordered = sorted(
        enumerate(plan.references),
        key=lambda item: (
            ReferenceRole.SHAPE not in item[1].roles,
            ReferenceRole.UV_LAYOUT not in item[1].roles,
            item[0],
        ),
    )
    best: tuple[float, _ReferenceAlphaContract] | None = None
    for _index, reference in ordered:
        if ReferenceRole.NEGATIVE in reference.roles:
            continue
        try:
            with Image.open(reference.path) as loaded:
                source = loaded.convert("RGBA")
        except (OSError, ValueError):
            continue
        if source.size != expected_size:
            continue
        alpha = source.getchannel("A")
        if _alpha_solid_share(alpha) < _ALPHA_MIN_SOLID_SHARE:
            # A dither layer does not describe a model silhouette. Reusing one
            # as the alpha authority punched holes through solid armour plate.
            continue
        if _alpha_region_coverage(alpha, region_mask) < _ALPHA_MIN_REGION_COVERAGE:
            # Solid but sparse: a trim overlay describes part of an armour set,
            # not the model, so the compiled geometry mask stays authoritative.
            continue
        ratio = _alpha_outside_ratio(alpha, region_mask)
        if ratio is None:
            continue
        if best is None or ratio < best[0]:
            best = (
                ratio,
                _ReferenceAlphaContract(alpha=alpha, source=source, reference_path=reference.path),
            )
    if best is None or best[0] > _ALPHA_REGION_TOLERANCE:
        return None
    return best[1]


def _save_entity_alpha_diagnostics(
    compiled: CompiledGeometry,
    contract: _ReferenceAlphaContract,
    out_dir: Path,
) -> list[Path]:
    """Persist the three independent entity-atlas facts and their IoU."""
    alpha_dir = out_dir / "alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    reference_alpha_path = alpha_dir / "reference_alpha.png"
    contract.alpha.save(reference_alpha_path, "PNG")
    coverage = [pixel >= 8 for pixel in compiled.mask.convert("L").get_flattened_data()]
    reference = [pixel >= 8 for pixel in contract.alpha.get_flattened_data()]
    intersection = sum(first and second for first, second in zip(coverage, reference))
    union = sum(first or second for first, second in zip(coverage, reference))
    report = {
        "kind": "uv_coverage_vs_reference_alpha",
        "reference": contract.reference_path,
        "uv_coverage_opaque_pixels": sum(coverage),
        "reference_alpha_opaque_pixels": sum(reference),
        "intersection_pixels": intersection,
        "uv_coverage_extra_pixels": sum(first and not second for first, second in zip(coverage, reference)),
        "uv_coverage_missing_pixels": sum(second and not first for first, second in zip(coverage, reference)),
        "iou": intersection / float(max(union, 1)),
        "meaning": "UV coverage comes from the selected model layout; reference alpha comes from the source PNG. They are deliberately measured separately.",
    }
    report_path = _write_json(alpha_dir / "uv_coverage_vs_reference_alpha.json", report)
    return [reference_alpha_path, report_path]


class GenerationPipeline:
    def __init__(self, critic: SemanticCritic | None = None, repairer: GeometryRepairer | None = None,
                 max_geometry_repairs: int = 0) -> None:
        if max_geometry_repairs < 0:
            raise ValueError("max_geometry_repairs cannot be negative")
        self.critic = critic or StructuralCritic()
        self.repairer = repairer
        self.max_geometry_repairs = max_geometry_repairs

    @staticmethod
    def reference_comparison(
        out_dir: str | Path,
        sprite_path: str | Path,
        references: list[ReferenceAsset],
    ) -> dict[str, object] | None:
        """Write a human-readable reference/generated/change comparison.

        This is intentionally called by the quality loop *after* blind review.
        The blind reviewer receives only the generated preview, while this
        artifact is available to diagnose whether a miss came from the alpha
        contour, an intentional colour change, or lost local pixel structure.
        No target-specific object rules are used here.
        """
        sprite_file = Path(sprite_path)
        try:
            with Image.open(sprite_file) as loaded:
                generated = loaded.convert("RGBA")
        except (OSError, ValueError):
            return None
        match = _reference_for_comparison(references, generated.size)
        if match is None:
            return None
        source, reference_path, reference_mode = match
        data = _reference_comparison_data(source, generated, reference_path, reference_mode)
        diff_pixels = data.pop("_diff_pixels")
        assert isinstance(diff_pixels, list)
        out_root = Path(out_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        scale = max(1, min(16, 256 // max(generated.width, generated.height)))
        panel_width = generated.width * scale
        panel_height = generated.height * scale
        label_height = 18
        canvas = Image.new(
            "RGBA",
            (panel_width * 3, panel_height + label_height),
            (18, 18, 18, 255),
        )
        draw = ImageDraw.Draw(canvas)
        labels = ("REFERENCE", "GENERATED", "CHANGED")
        panels = (source, generated, Image.new("RGBA", generated.size))
        panels[2].putdata(diff_pixels)
        for panel_index, (label, image) in enumerate(zip(labels, panels)):
            left = panel_index * panel_width
            draw.text((left + 3, 3), label, fill=(240, 240, 240, 255))
            enlarged = image.resize((panel_width, panel_height), Image.Resampling.NEAREST)
            canvas.alpha_composite(enlarged, (left, label_height))
            if panel_index:
                draw.line((left, 0, left, canvas.height), fill=(110, 110, 110, 255), width=1)
        image_path = out_root / "reference_comparison.png"
        report_path = out_root / "reference_comparison.json"
        canvas.save(image_path, "PNG")
        data["generated"] = str(sprite_file.resolve())
        data["image"] = str(image_path)
        data["report"] = str(report_path)
        data["scale"] = scale
        data["panels"] = list(labels)
        _write_json(report_path, data)
        return data

    @staticmethod
    def _audit(
        root: Path,
        artifacts: list[Path],
        plan: GenerationPlan,
        validation: ValidationResult,
        sprite_path: Path | None,
        attempts: list[dict[str, object]],
    ) -> Path:
        audit = {
            "status": "PASS" if validation.passed else "FAIL",
            "artifact_count": len(artifacts) + 1,
            # Keep a rendered diagnostic even when a semantic critic rejects
            # the candidate.  Structural failures still stop before rendering;
            # a semantic score is useful feedback only when there is an image
            # for the blind reviewer and the user to inspect.
            "sprite": str(sprite_path) if sprite_path else None,
            "reference_count": len(plan.references),
            "shape_policy": plan.request.shape_policy,
            "geometry_attempts": attempts,
            "validation": validation,
        }
        return _write_json(root / "audit.json", audit)

    @staticmethod
    def _flow_audit(
        root: Path,
        plan: GenerationPlan,
        active_plan: GenerationPlan,
        compiled: CompiledGeometry | None,
        validation: ValidationResult,
        render_result: ValidationResult | None,
        alpha_contract: _ReferenceAlphaContract | None = None,
    ) -> Path:
        """Record the contracts crossing every stage of one generation.

        This is deliberately deterministic and does not claim that a model
        understood a reference.  It answers the narrower, testable question:
        was the same evidence and the same declared part set actually handed
        from descriptor to geometry to appearance and then to rendering?
        """
        descriptor_ids = {part.id for part in plan.descriptor.parts}
        geometry_ids = {part.id for part in active_plan.geometry.parts}
        primitive_part_ids = {primitive.part_id for primitive in active_plan.geometry.primitives}
        appearance_ids = set(active_plan.appearance.parts)
        reference_roles = {
            reference.name: [role.value for role in reference.roles]
            for reference in active_plan.references
        }
        reference_evidence = {
            reference.name: {
                "orientation_degrees": reference.features.get("orientation_degrees"),
                "stroke_profile": reference.features.get("stroke_profile", {}),
                "symmetry_profile": reference.features.get("symmetry_profile", {}),
            }
            for reference in active_plan.references
        }
        reference_candidates: list[str] = []
        for reference in active_plan.references:
            if ReferenceRole.NEGATIVE in reference.roles:
                continue
            if not any(role.value in {"shape", "pixel_style", "material", "palette"} for role in reference.roles):
                continue
            try:
                with Image.open(reference.path) as image:
                    if image.size == (active_plan.request.width, active_plan.request.height):
                        reference_candidates.append(reference.path)
            except (OSError, ValueError):
                continue
        selected_render_reference = _select_reference(
            active_plan.references,
            active_plan.request.width,
            active_plan.request.height,
            target_mask=compiled.mask if compiled is not None else None,
        )
        indexed_routing: dict[str, object] = {}
        # ``run_quality_loop`` writes routing.json at the quality-run root,
        # while this audit lives below round_XX/generated.  Look in the
        # current directory first for direct pipeline callers, then walk up
        # the small output tree so the audit reports the routing evidence that
        # actually fed this round.
        routing_paths = [root / "routing.json"]
        routing_paths.extend(parent / "routing.json" for parent in root.parents[:3])
        for routing_path in routing_paths:
            if not routing_path.exists():
                continue
            try:
                loaded_routing = json.loads(routing_path.read_text(encoding="utf-8"))
                if isinstance(loaded_routing, dict):
                    indexed_routing = loaded_routing
                    break
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        art_direction = to_jsonable(plan.descriptor.art_direction) if plan.descriptor.art_direction else None
        routed_art_direction = indexed_routing.get("art_direction")
        target_map_present = isinstance(plan.descriptor.target_part_map, dict)
        requires_silhouette_change = bool(
            plan.descriptor.art_direction
            and plan.descriptor.art_direction.requires_silhouette_change
        )
        flow = {
            "art_direction_to_router": {
                "present": art_direction is not None,
                "passed": art_direction is None or routed_art_direction == art_direction,
                "payload": art_direction,
            },
            "request_to_descriptor": {
                "query": plan.request.query,
                "form": plan.request.form.value,
                "dimensions": [plan.request.width, plan.request.height],
                "descriptor_parts": sorted(descriptor_ids),
                "reference_count": len(plan.references),
                "references": reference_roles,
                "art_direction_present": art_direction is not None,
                "art_direction_preserved": (
                    art_direction is None
                    or to_jsonable(active_plan.descriptor.art_direction) == art_direction
                ),
            },
            "descriptor_to_geometry": {
                "passed": descriptor_ids.issuperset(geometry_ids)
                and {part.id for part in plan.descriptor.parts if part.required and not part.paint_only}.issubset(geometry_ids),
                "geometry_parts": sorted(geometry_ids),
                "primitive_part_ids": sorted(primitive_part_ids),
                "missing_required_parts": sorted(
                    {part.id for part in plan.descriptor.parts if part.required and not part.paint_only} - geometry_ids
                ),
                "unknown_geometry_parts": sorted(geometry_ids - descriptor_ids),
                "references_available": bool(active_plan.references),
                "reference_evidence": reference_evidence,
                "art_direction_present": art_direction is not None,
                "target_ownership_map_present": target_map_present,
                "requires_silhouette_change": requires_silhouette_change,
                "repair_context_contract": (
                    ["art_direction", "target_part_map"]
                    if art_direction is not None else []
                ),
            },
            "geometry_to_appearance": {
                "passed": not (geometry_ids - appearance_ids) and not (appearance_ids - geometry_ids),
                "geometry_parts": sorted(geometry_ids),
                "styled_parts": sorted(appearance_ids),
                "fallback_parts": sorted(geometry_ids - appearance_ids),
                "unknown_appearance_parts": sorted(appearance_ids - geometry_ids),
                "reference_sampling": active_plan.appearance.reference_sampling,
                "references_available": bool(active_plan.references),
                "same_size_reference_candidates": reference_candidates,
                "selected_reference": selected_render_reference[1].path if selected_render_reference else None,
                "selected_reference_mode": selected_render_reference[2] if selected_render_reference else None,
                "selected_value_reference": selected_render_reference[1].path if selected_render_reference else None,
                },
            "geometry_to_render": {
                "compiled": compiled is not None,
                "validation_passed": validation.passed,
                "render_validation": render_result.metrics if render_result is not None else None,
                "alpha_locked": bool(render_result and render_result.metrics.get("alpha_exact", False)),
                "alpha_contract": "reference_alpha" if alpha_contract is not None else "compiled_geometry",
                "alpha_reference": alpha_contract.reference_path if alpha_contract is not None else None,
            },
            "reference_index_to_router": {
                "index": indexed_routing.get("index"),
                "local_recall": indexed_routing.get("local_recall", []),
                "router": indexed_routing.get("router"),
                "passed": not indexed_routing.get("index") or bool(indexed_routing.get("local_recall", [])),
            },
            "router_to_materialized_references": {
                "selected": indexed_routing.get("selected", []),
                "attached_images": indexed_routing.get("attached_images", [reference.path for reference in active_plan.references]),
                "passed": all(Path(reference.path).exists() for reference in active_plan.references),
                "reference_count": len(active_plan.references),
            },
        }
        flow["edges"] = [
            {
                "from": "art_direction",
                "to": "router",
                "payload": ["primary_subject", "composition", "change_actions", "preservation_rules"],
                "passed": flow["art_direction_to_router"]["passed"],
            },
            {
                "from": "art_direction",
                "to": "descriptor+geometry+appearance+repair",
                "payload": ["immutable design brief", "target ownership map when authored"],
                "passed": art_direction is None or (
                    flow["request_to_descriptor"]["art_direction_preserved"]
                    and flow["descriptor_to_geometry"]["art_direction_present"]
                ),
            },
            {
                "from": "request",
                "to": "descriptor",
                "payload": ["query", "form", "dimensions", "references"],
                "passed": bool(plan.request.query.strip()) and len(plan.references) == len(reference_roles),
            },
            {
                "from": "descriptor",
                "to": "geometry",
                "payload": ["parts", "orientation", "reference pixel maps", "reference images"],
                "passed": flow["descriptor_to_geometry"]["passed"] and bool(active_plan.references) == bool(plan.references),
            },
            {
                "from": "geometry",
                "to": "appearance",
                "payload": ["locked part masks", "part labels", "references"],
                "passed": flow["geometry_to_appearance"]["passed"],
            },
            {
                "from": "geometry+appearance",
                "to": "render",
                "payload": [
                    "UV coverage" if alpha_contract is not None else "compiled mask",
                    "reference alpha" if alpha_contract is not None else "compiled alpha",
                    "materials",
                    "reference value/pattern sampling",
                ],
                "passed": flow["geometry_to_render"]["alpha_locked"],
            },
        ]
        return _write_json(root / "flow_audit.json", flow)

    @staticmethod
    def _texture_audit(
        root: Path,
        plan: GenerationPlan,
        active_plan: GenerationPlan,
        sprite: Image.Image,
    ) -> Path:
        """Measure whether a same-sized reference's pixel evidence survived.

        This is deliberately descriptive: geometry remains the source of the
        generated mask, while the audit makes lost reference pixels visible to
        the quality loop instead of letting a low palette count look like a
        successful texture transfer.
        """
        result: dict[str, object] = {
            "reference_sampling": active_plan.appearance.reference_sampling,
            "reference": None,
            "source_unique_colors": None,
            "rendered_unique_colors": len(Counter(
                pixel[:3] for pixel in sprite.convert("RGBA").get_flattened_data() if pixel[3] >= 8
            )),
            "source_opaque_pixels": None,
            "rendered_opaque_pixels": sum(
                pixel[3] >= 8 for pixel in sprite.convert("RGBA").get_flattened_data()
            ),
            "alpha_extra_pixels": None,
            "alpha_missing_pixels": None,
            "regions": [],
        }
        source = None
        reference_mode = "exact"
        for reference in active_plan.references:
            try:
                with Image.open(reference.path) as loaded:
                    if loaded.size == sprite.size:
                        source = loaded.convert("RGBA")
                        result["reference"] = reference.path
                        break
                    if (
                        loaded.height == sprite.height
                        and loaded.width < sprite.width
                        and sprite.width % loaded.width == 0
                    ):
                        # A compact block face sample can be repeated across a
                        # wider face strip. Audit the expanded evidence with
                        # the same per-pixel metrics used for an exact atlas.
                        tile = loaded.convert("RGBA")
                        source = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
                        for left in range(0, sprite.width, tile.width):
                            source.alpha_composite(tile, (left, 0))
                        reference_mode = "face_tile"
                        result["reference"] = reference.path
                        break
            except (OSError, ValueError):
                continue
        if source is not None:
            result["reference_mode"] = reference_mode
            source_pixels = list(source.get_flattened_data())
            rendered_pixels = list(sprite.convert("RGBA").get_flattened_data())
            source_opaque = [pixel for pixel in source_pixels if pixel[3] >= 8]
            rendered_opaque = [pixel for pixel in rendered_pixels if pixel[3] >= 8]
            result["source_unique_colors"] = len(Counter(pixel[:3] for pixel in source_opaque))
            result["source_opaque_pixels"] = len(source_opaque)
            result["alpha_extra_pixels"] = sum(
                target[3] >= 8 and source_pixel[3] < 8
                for source_pixel, target in zip(source_pixels, rendered_pixels)
            )
            result["alpha_missing_pixels"] = sum(
                source_pixel[3] >= 8 and target[3] < 8
                for source_pixel, target in zip(source_pixels, rendered_pixels)
            )
            result["pattern"] = _texture_pattern_metrics(
                source,
                sprite.convert("RGBA"),
                (0, 0, source.width, source.height),
            )
            region_audits: list[dict[str, object]] = []
            for region in active_plan.geometry.uv_regions:
                left, top, right, bottom = region.bbox
                source_region = [
                    source.getpixel((x, y)) for y in range(top, bottom) for x in range(left, right)
                    if source.getpixel((x, y))[3] >= 8
                ]
                rendered_region = [
                    sprite.getpixel((x, y)) for y in range(top, bottom) for x in range(left, right)
                    if sprite.getpixel((x, y))[3] >= 8
                ]
                region_audits.append({
                    "id": region.id,
                    "source_unique_colors": len(Counter(pixel[:3] for pixel in source_region)),
                    "rendered_unique_colors": len(Counter(pixel[:3] for pixel in rendered_region)),
                    "source_opaque_pixels": len(source_region),
                    "rendered_opaque_pixels": len(rendered_region),
                    "pattern": _texture_pattern_metrics(source, sprite, (left, top, right, bottom)),
                })
            result["regions"] = region_audits
        return _write_json(root / "texture_audit.json", result)

    def run(self, plan: GenerationPlan, out_dir: str | Path, package: bool = True) -> GenerationResult:
        root = Path(out_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        artifacts: list[Path] = [
            _write_json(root / "request.json", plan.request),
            _write_json(root / "shape_descriptor.json", plan.descriptor),
            _write_json(root / "concept.json", plan.descriptor),
            _write_json(root / "geometry.initial.json", plan.geometry),
            _write_json(root / "appearance.json", plan.appearance),
            _write_json(root / "references.json", plan.references),
        ]
        active_plan = plan
        locked_reference_geometry = None
        if _reference_shape_is_locked(plan.descriptor):
            locked_reference_geometry = reference_partition_geometry(
                plan.descriptor,
                plan.references,
                plan.request.width,
                plan.request.height,
                plan.request.name,
            )
            if locked_reference_geometry is None:
                # The model rarely hand-writes the coarse ownership map, but
                # it always draws part masks. Those masks are the same
                # evidence at a better resolution, so clip them to the
                # source alpha instead of shipping the loose contour.
                locked_reference_geometry = reference_silhouette_partition(
                    plan.descriptor,
                    plan.geometry,
                    plan.references,
                    plan.request.width,
                    plan.request.height,
                    plan.request.name,
                )
            if locked_reference_geometry is not None:
                # The ownership map is model-authored evidence, not an
                # object-specific template. A variant that explicitly says
                # to preserve its silhouette must use that partition before
                # appearance mapping, even when the model's first geometry
                # draft happens to pass structural checks.
                active_plan = GenerationPlan(
                    request=plan.request,
                    descriptor=plan.descriptor,
                    geometry=locked_reference_geometry,
                    appearance=plan.appearance,
                    references=plan.references,
                )
        if plan.request.shape_policy == "reference":
            reference_geometry = reference_fallback_geometry(
                plan.descriptor,
                plan.references,
                plan.request.width,
                plan.request.height,
                plan.request.name,
            )
            if reference_geometry is not None:
                active_plan = GenerationPlan(
                    request=plan.request,
                    descriptor=plan.descriptor,
                    geometry=reference_geometry,
                    appearance=plan.appearance,
                    references=plan.references,
                )
        effective_repairer = self.repairer
        effective_max_repairs = self.max_geometry_repairs
        partition_recovery = locked_reference_geometry or reference_partition_geometry(
            plan.descriptor,
            plan.references,
            plan.request.width,
            plan.request.height,
            plan.request.name,
        )
        # For an explicit model-authored local overlay, use the ownership maps
        # as a conservative missing-part fallback.  They are useful evidence
        # for a support the geometry model forgot, but they are not reliable
        # enough to replace an existing support pixel-for-pixel: a low
        # resolution map can accidentally assign a continuation of a long
        # support to its neighbouring host (for example the lower blade next
        # to a guard).  Replacing the model draft with that map turns an
        # otherwise intact source silhouette into a truncated one.  Keep every
        # model-authored support that has pixels and add map evidence only for
        # missing supports; keep the model motif whenever it authored one.
        if (
            partition_recovery is not None
            and _target_overlay_descriptor(plan.descriptor)
            and active_plan.geometry is not partition_recovery
        ):
            try:
                draft_compiled = compile_geometry(active_plan.geometry)
                overlay_ids = _overlay_part_ids(plan.descriptor)
                draft_part_points = {
                    part.id: sum(
                        1
                        for y in range(draft_compiled.height)
                        for x in range(draft_compiled.width)
                        if draft_compiled.part_masks[part.id].getpixel((x, y)) > 0
                    )
                    for part in active_plan.geometry.parts
                }
                # Only append a source-map primitive for a declared support
                # that the draft genuinely omitted.  Existing support masks,
                # even when imperfect, remain available to the normal geometry
                # critic and repair loop instead of being silently replaced.
                missing_support_ids = {
                    part.id
                    for part in active_plan.geometry.parts
                    if part.id not in overlay_ids and draft_part_points.get(part.id, 0) == 0
                }
                fallback_support_primitives = [
                    primitive
                    for primitive in partition_recovery.primitives
                    if primitive.part_id in missing_support_ids
                ]
                draft_motif_primitives = [
                    primitive
                    for primitive in active_plan.geometry.primitives
                    if primitive.part_id in overlay_ids
                    and draft_part_points.get(primitive.part_id, 0) > 0
                ]
                fallback_motif_primitives = []
                for primitive in partition_recovery.primitives:
                    if primitive.part_id not in overlay_ids:
                        continue
                    if draft_part_points.get(primitive.part_id, 0) == 0:
                        fallback_motif_primitives.append(primitive)
                if fallback_support_primitives or fallback_motif_primitives:
                    recovered_geometry = GeometrySpec(
                        width=active_plan.geometry.width,
                        height=active_plan.geometry.height,
                        parts=list(active_plan.geometry.parts),
                        primitives=[
                            *active_plan.geometry.primitives,
                            *fallback_support_primitives,
                            *fallback_motif_primitives,
                        ],
                        connections=active_plan.geometry.connections,
                        constraints=active_plan.geometry.constraints,
                        uv_regions=active_plan.geometry.uv_regions,
                        background_transparent=active_plan.geometry.background_transparent,
                    )
                    active_plan = GenerationPlan(
                        request=active_plan.request,
                        descriptor=active_plan.descriptor,
                        geometry=recovered_geometry,
                        appearance=active_plan.appearance,
                        references=active_plan.references,
                    )
                    partition_recovery = recovered_geometry
            except (KeyError, TypeError, ValueError):
                # Keep the model draft when the optional map recovery cannot
                # be rasterized; the normal repair/quality loop remains the
                # source of truth for malformed plans.
                pass
        if effective_repairer is not None and partition_recovery is not None:
            effective_repairer = _ReferenceAwareRepairer(effective_repairer, plan.references)
        if effective_repairer is None and (
            plan.request.shape_policy == "auto" or partition_recovery is not None
        ):
            effective_repairer = _ReferenceFallbackRepairer(plan.references)
            effective_max_repairs = max(effective_max_repairs, 1)
        compiled: CompiledGeometry | None = None
        pre_render: ValidationResult | None = None
        geometry_passed = False
        attempts: list[dict[str, object]] = []

        for attempt_index in range(effective_max_repairs + 1):
            # Canvas margins are a model-authored choice. Inventory textures
            # may legitimately touch an edge, and forcing a one-pixel inset
            # changes the reference silhouette before review can inspect it.
            margin_fitted = False
            attempt_dir = root / "attempts" / ("%02d" % attempt_index)
            attempt_artifacts = [_write_json(attempt_dir / "geometry.json", active_plan.geometry)]
            try:
                compiled = compile_geometry(active_plan.geometry)
                attempt_artifacts.extend(_save_masks(
                    compiled, attempt_dir, entity_uv=active_plan.request.form.value == "entity_uv"
                ))
                geometry_result = validate_geometry(active_plan.geometry, compiled, active_plan.request.form)
                # Keep a structurally meaningful raster available for blind and
                # human review even when soft size/occupancy/orientation metrics
                # fail. The final validation remains failed and visible, while
                # the quality loop can now use the actual model contour instead
                # of substituting a fixed shape.
                structurally_renderable = _geometry_can_be_rendered(
                    active_plan.geometry, compiled
                )
                # A supplied repairer must get one chance to fix an explicit
                # geometry contract before a soft-failed mask is accepted for
                # visual feedback.  After the repair budget is exhausted, a
                # structurally meaningful mask is still rendered so the outer
                # blind-review loop can inspect the model's actual decision.
                geometry_passed = geometry_result.passed or (
                    structurally_renderable
                    and (
                        effective_repairer is None
                        or not getattr(effective_repairer, "repair_soft_failures", True)
                        or attempt_index >= effective_max_repairs
                    )
                )
                if active_plan.request.form.value in {"entity_uv", "block_multi"}:
                    # The compiled image is a UV atlas, not a model-facing
                    # silhouette. A semantic critic looking at that rectangle
                    # would judge the texture layout as a malformed icon;
                    # entity/block recognisability is checked after rendering
                    # via the appropriate front/isometric preview and blind review.
                    semantic_result = ValidationResult(
                        passed=True,
                        stage="semantic_model_skipped_atlas",
                        metrics={"atlas_semantic_review_skipped": True},
                        warnings=["UV atlas geometry; semantic review is deferred to the generated model preview"],
                    )
                elif is_material_only_descriptor(active_plan.descriptor):
                    semantic_result = ValidationResult(
                        passed=True,
                        stage="semantic_material_only",
                        metrics={"material_only_semantic_review": True},
                        warnings=["material-only request uses contour/material checks; object-noun classification is undefined"],
                    )
                else:
                    semantic_result = self.critic.review(active_plan.descriptor, compiled)
                pre_render = aggregate_results([geometry_result, semantic_result])
            except Exception as exc:  # noqa: BLE001 - planner output must still produce an audit trail
                compiled = None
                geometry_passed = False
                pre_render = ValidationResult(
                    passed=False,
                    stage="compile",
                    metrics={"attempt": attempt_index},
                    errors=["geometry compilation failed: %s" % exc],
                )
            attempt_artifacts.append(_write_json(attempt_dir / "validation_pre_render.json", pre_render))
            artifacts.extend(attempt_artifacts)
            attempt_record: dict[str, object] = {
                "index": attempt_index,
                "passed": pre_render.passed,
                "stage": pre_render.stage,
                "errors": pre_render.errors,
                "warnings": pre_render.warnings,
            }
            if margin_fitted:
                attempt_record["margin_fitted"] = True
            attempts.append(attempt_record)
            # Geometry and semantic identity are separate contracts. Once the
            # alpha/part geometry passes, keep that renderable candidate even
            # if the semantic critic dislikes it; feeding a valid silhouette
            # back through a geometry repairer often replaces it with a worse
            # polygon. The outer quality loop can then use blind review to
            # decide whether appearance or geometry needs another pass.
            if geometry_passed or pre_render.passed or effective_repairer is None or attempt_index >= effective_max_repairs:
                break
            try:
                repair_fn = effective_repairer.repair_geometry
                repair_args = (
                    active_plan.request,
                    active_plan.descriptor,
                    active_plan.geometry,
                    pre_render,
                    compiled,
                )
                if "references" in inspect.signature(repair_fn).parameters:
                    repaired = repair_fn(*repair_args, references=active_plan.references)
                else:
                    # Keep third-party/offline repairers written against the
                    # original five-argument protocol working unchanged.
                    repaired = repair_fn(*repair_args)
                active_plan = GenerationPlan(
                    request=active_plan.request,
                    descriptor=active_plan.descriptor,
                    geometry=repaired,
                    appearance=active_plan.appearance,
                    references=active_plan.references,
                )
            except Exception as exc:  # noqa: BLE001 - invalid model repairs must remain auditable
                repair_result = ValidationResult(
                    passed=False,
                    stage="repair",
                    metrics={"attempt": attempt_index},
                    errors=["geometry repair failed: %s" % exc],
                )
                pre_render = aggregate_results([pre_render, repair_result])
                attempts[-1]["repair_error"] = repair_result.errors[0]
                artifacts.append(_write_json(attempt_dir / "repair_failure.json", repair_result))
                break

        assert pre_render is not None  # the loop always has at least one iteration
        artifacts.append(_write_json(root / "geometry.json", active_plan.geometry))
        artifacts.append(_write_json(root / "validation_geometry.json", pre_render))
        # A semantic critic sees the pre-appearance silhouette, so it may reject
        # an appearance-dominated object (for example a recolored leather item)
        # even though the geometry is valid. Render those candidates anyway:
        # the resulting sprite is needed by blind review and by a human, while
        # hard geometry/compile failures still stop before rendering.
        if not geometry_passed or compiled is None:
            artifacts.append(self._flow_audit(root, plan, active_plan, compiled, pre_render, None))
            artifacts.append(self._audit(root, artifacts, active_plan, pre_render, None, attempts))
            return GenerationResult(root, compiled, None, pre_render, artifacts)

        entity_alpha_contract = _entity_reference_alpha_contract(active_plan)
        artifacts.extend(_save_masks(
            compiled, root, entity_uv=active_plan.request.form.value == "entity_uv"
        ))
        if entity_alpha_contract is not None:
            artifacts.extend(_save_entity_alpha_diagnostics(compiled, entity_alpha_contract, root))
        sprite = render_appearance(
            active_plan.geometry,
            compiled,
            active_plan.appearance,
            seed=active_plan.request.seed,
            references=active_plan.references,
            alpha_mask=entity_alpha_contract.alpha if entity_alpha_contract is not None else None,
            alpha_source=entity_alpha_contract.source if entity_alpha_contract is not None else None,
        )
        sprite_path = root / "sprite.png"
        sprite.save(sprite_path, "PNG")
        preview_path = root / "preview.png"
        scale_preview(sprite).save(preview_path, "PNG")
        checker_preview_path = root / "preview_checker.png"
        checkerboard_preview(sprite).save(checker_preview_path, "PNG")
        artifacts.extend([sprite_path, preview_path, checker_preview_path])
        artifacts.append(self._texture_audit(root, plan, active_plan, sprite))
        if active_plan.request.form.value == "entity_uv":
            try:
                entity_preview = render_front_preview(sprite, active_plan.geometry.uv_regions, scale=12)
            except ValueError:
                entity_preview = None
            if entity_preview is not None:
                entity_preview_path = root / "front_preview.png"
                entity_preview.save(entity_preview_path, "PNG")
                artifacts.append(entity_preview_path)
                # Vision models handle very dark vanilla palettes more
                # reliably against a neutral checker than against an implicit
                # black transparent canvas. Keep the transparent diagnostic and
                # add a separate review image with identical pixels.
                entity_checker_path = root / "front_preview_checker.png"
                checkerboard_preview(entity_preview, scale=1, tile=12).save(entity_checker_path, "PNG")
                artifacts.append(entity_checker_path)
                try:
                    entity_iso = render_entity_preview(sprite, active_plan.geometry.uv_regions, scale=12)
                except ValueError:
                    entity_iso = None
                if entity_iso is not None:
                    entity_iso_path = root / "entity_preview.png"
                    entity_iso.save(entity_iso_path, "PNG")
                    entity_iso_checker_path = root / "entity_preview_checker.png"
                    checkerboard_preview(entity_iso, scale=1, tile=12).save(entity_iso_checker_path, "PNG")
                    artifacts.extend([entity_iso_path, entity_iso_checker_path])
        if active_plan.request.form.value == "block_multi" and active_plan.geometry.uv_regions:
            try:
                block_preview = render_isometric_preview(sprite, active_plan.geometry.uv_regions, scale=8)
            except ValueError:
                block_preview = None
            if block_preview is not None:
                block_preview_path = root / "isometric_preview.png"
                block_preview.save(block_preview_path, "PNG")
                artifacts.append(block_preview_path)
        # UV coverage and source alpha are separate for an unchanged target
        # model.  A generated entity therefore validates against the selected
        # source PNG alpha, while the coverage-vs-alpha diagnostic records any
        # layout mismatch instead of silently treating it as geometry.
        render_result = validate_render_alpha(
            compiled,
            sprite,
            expected_mask=entity_alpha_contract.alpha if entity_alpha_contract is not None else compiled.mask,
        )
        final_result = aggregate_results([pre_render, render_result])
        artifacts.extend(
            [
                _write_json(root / "validation_render.json", render_result),
                _write_json(root / "validation.json", final_result),
            ]
        )
        artifacts.append(self._flow_audit(
            root, plan, active_plan, compiled, final_result, render_result, entity_alpha_contract
        ))
        if final_result.passed and package:
            artifacts.extend(
                build_resourcepack(
                    active_plan.request,
                    sprite_path,
                    root / "resourcepack",
                    pack_format=active_plan.request.pack_format,
                    uv_regions=active_plan.geometry.uv_regions,
                )
            )
        artifacts.append(self._audit(root, artifacts, active_plan, final_result, sprite_path, attempts))
        return GenerationResult(root, compiled, sprite_path, final_result, artifacts)
