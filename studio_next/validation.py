"""Deterministic geometry and render validation.

Validation does not need to know a closed vocabulary such as sword or dagger.
It checks the measurable requirements supplied by a runtime geometry plan:
parts, adjacency, margins, topology and planner-defined numeric constraints.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, degrees, sqrt
import re
from typing import Iterable, Protocol

from PIL import Image

from .contracts import AssetForm, GeometrySpec, ShapeDescriptor, ValidationResult
from .geometry import CompiledGeometry


def _points(mask: Image.Image, threshold: int = 1) -> set[tuple[int, int]]:
    image = mask.convert("L")
    return {
        (x, y)
        for y in range(image.height)
        for x in range(image.width)
        if image.getpixel((x, y)) >= threshold
    }


def _components(points: set[tuple[int, int]]) -> int:
    """Count visually connected sprite regions, including diagonal pixel steps.

    A one-pixel Minecraft diagonal is intentionally 8-connected. Treating it
    as a collection of separate objects would reject valid blades, branches and
    wings merely because their staircase edges have no orthogonal neighbour.
    """
    unseen = set(points)
    count = 0
    while unseen:
        count += 1
        queue: deque[tuple[int, int]] = deque([unseen.pop()])
        while queue:
            x, y = queue.popleft()
            for neighbour in (
                (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
                (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1),
            ):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return count


def _bbox(points: set[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _max_row_run(points: set[tuple[int, int]]) -> int:
    longest = 0
    by_row: dict[int, list[int]] = {}
    for x, y in points:
        by_row.setdefault(y, []).append(x)
    for xs in by_row.values():
        ordered = sorted(xs)
        run = 0
        previous = None
        for x in ordered:
            run = run + 1 if previous is not None and x == previous + 1 else 1
            longest = max(longest, run)
            previous = x
    return longest


def _principal_axis(points: set[tuple[int, int]]) -> tuple[float, float]:
    if len(points) < 2:
        return 0.0, 0.0
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    xx = yy = xy = 0.0
    for x, y in points:
        dx, dy = x - mean_x, y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    xx /= len(points)
    yy /= len(points)
    xy /= len(points)
    angle = 0.5 * atan2(2.0 * xy, xx - yy)
    trace = xx + yy
    determinant = max(xx * yy - xy * xy, 0.0)
    root = sqrt(max(trace * trace - 4.0 * determinant, 0.0))
    major, minor = (trace + root) / 2.0, (trace - root) / 2.0
    return degrees(angle), (major - minor) / max(major + minor, 1e-9)


def _touches(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    return any(
        (x + dx, y + dy) in b
        for x, y in a
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        )
    )


def geometry_metrics(compiled: CompiledGeometry) -> dict[str, float | int]:
    all_points = _points(compiled.mask)
    bbox = _bbox(all_points)
    metrics: dict[str, float | int] = {
        "opaque_pixels": len(all_points),
        "occupancy_ratio": len(all_points) / float(compiled.width * compiled.height),
        "components": _components(all_points),
    }
    orientation_degrees, axis_anisotropy = _principal_axis(all_points)
    metrics["orientation_degrees"] = orientation_degrees
    metrics["axis_anisotropy"] = axis_anisotropy
    if bbox is None:
        metrics.update(
            {
                "bbox_left": 0,
                "bbox_top": 0,
                "bbox_width": 0,
                "bbox_height": 0,
                "aspect_ratio": 0.0,
                "margin_left": 0,
                "margin_top": 0,
                "margin_right": 0,
                "margin_bottom": 0,
            }
        )
    else:
        left, top, right, bottom = bbox
        box_width, box_height = right - left, bottom - top
        metrics.update(
            {
                "bbox_left": left,
                "bbox_top": top,
                "bbox_width": box_width,
                "bbox_height": box_height,
                "bbox_width_ratio": box_width / float(compiled.width),
                "bbox_height_ratio": box_height / float(compiled.height),
                "aspect_ratio": box_width / float(max(box_height, 1)),
                "margin_left": left,
                "margin_top": top,
                "margin_right": compiled.width - right,
                "margin_bottom": compiled.height - bottom,
            }
        )
    for part_id, mask in compiled.part_masks.items():
        part_points = _points(mask)
        metrics["%s_pixels" % part_id] = len(part_points)
        metrics["%s_ratio" % part_id] = len(part_points) / float(max(len(all_points), 1))
        metrics["%s_components" % part_id] = _components(part_points)
        metrics["%s_max_row_run" % part_id] = _max_row_run(part_points)
        part_bbox = _bbox(part_points)
        if part_bbox is None:
            metrics["%s_bbox_width" % part_id] = 0
            metrics["%s_bbox_height" % part_id] = 0
            metrics["%s_bbox_width_ratio" % part_id] = 0.0
            metrics["%s_bbox_height_ratio" % part_id] = 0.0
            metrics["%s_aspect_ratio" % part_id] = 0.0
        else:
            left, top, right, bottom = part_bbox
            part_width, part_height = right - left, bottom - top
            metrics["%s_bbox_width" % part_id] = part_width
            metrics["%s_bbox_height" % part_id] = part_height
            metrics["%s_bbox_width_ratio" % part_id] = part_width / float(compiled.width)
            metrics["%s_bbox_height_ratio" % part_id] = part_height / float(compiled.height)
            metrics["%s_aspect_ratio" % part_id] = part_width / float(max(part_height, 1))
    return metrics


def validate_geometry(spec: GeometrySpec, compiled: CompiledGeometry, form: AssetForm,
                      minimum_margin: int | None = None) -> ValidationResult:
    metrics = geometry_metrics(compiled)
    errors: list[str] = []
    warnings: list[str] = []
    all_points = _points(compiled.mask)
    if not all_points:
        errors.append("geometry mask is empty")
    if metrics["components"] > 4:
        errors.append("geometry is fragmented into %s connected components" % metrics["components"])

    for part in spec.parts:
        points = _points(compiled.part_masks[part.id])
        if part.required and not part.paint_only and not points:
            errors.append("required part %s is empty" % part.id)
        elif points and _components(points) > 3:
            warnings.append("part %s has %s disconnected components" % (part.id, _components(points)))

    # Distinct labelled sections should not collapse into one almost-identical
    # mask. A small overlap is useful for a joint; near-total overlap means the
    # planner has lost the part boundary and usually produces a generic blob.
    required_points = {
        part.id: _points(compiled.part_masks[part.id])
        for part in spec.parts
        if part.required and not part.paint_only
    }
    part_by_id = {part.id: part for part in spec.parts}
    required_ids = list(required_points)
    for index, first_id in enumerate(required_ids):
        for second_id in required_ids[index + 1:]:
            first = required_points[first_id]
            second = required_points[second_id]
            overlap = len(first & second) / float(max(min(len(first), len(second)), 1))
            key = "overlap_%s_%s_ratio" % (first_id, second_id)
            metrics[key] = overlap
            detail_roles = {
                "highlight", "accent", "detail", "mark", "shadow", "edge", "texture",
                "motif", "inlaid", "embedded", "inlay", "overlay", "emblem",
                "eye", "gem", "socket", "boss",
            }
            first_role = part_by_id[first_id].style_role.lower()
            second_role = part_by_id[second_id].style_role.lower()
            # Match role words instead of substrings.  A host described as
            # "host support for an inlaid motif" is still a physical support;
            # treating the word ``inlaid`` as a detail role hid the very
            # overlap that made the eye consume the guard in earlier runs.
            first_words = set(re.findall(r"[a-z]+", first_role))
            second_words = set(re.findall(r"[a-z]+", second_role))
            first_text = (part_by_id[first_id].meaning + " " + first_role).lower()
            second_text = (part_by_id[second_id].meaning + " " + second_role).lower()
            # A local inlay is deliberately drawn over its host support, so
            # its alpha mask can overlap the host completely.  Only treat a
            # part as an overlay when its own role identifies it as one; a
            # host phrase such as "host support for the inlaid eye" must stay
            # a physical support and must not disable this check for unrelated
            # support pairs.
            first_is_host = "host support" in first_role or first_role.strip() in {"support", "primary support"}
            second_is_host = "host support" in second_role or second_role.strip() in {"support", "primary support"}
            first_is_overlay = bool((first_words & detail_roles) and not first_is_host)
            second_is_overlay = bool((second_words & detail_roles) and not second_is_host)
            if not first_is_overlay and any(token in first_text for token in ("local motif", "embedded", "inlaid", "inlay", "eyeball", "gem", "emblem")) and not first_is_host:
                first_is_overlay = True
            if not second_is_overlay and any(token in second_text for token in ("local motif", "embedded", "inlaid", "inlay", "eyeball", "gem", "emblem")) and not second_is_host:
                second_is_overlay = True
            is_detail = first_is_overlay or second_is_overlay
            if form in {AssetForm.ITEM, AssetForm.CROSS} and not is_detail and min(len(first), len(second)) >= 10 and overlap > 0.80:
                errors.append(
                    "required parts %s and %s overlap %.0f%%; keep labelled sections visibly distinct"
                    % (first_id, second_id, overlap * 100.0)
                )

    for connection in spec.connections:
        a = _points(compiled.part_masks[connection.a])
        b = _points(compiled.part_masks[connection.b])
        part_by_id = {part.id: part for part in spec.parts}
        if connection.required and not part_by_id[connection.a].paint_only and not part_by_id[connection.b].paint_only and not _touches(a, b):
            errors.append("required parts %s and %s do not touch" % (connection.a, connection.b))

    if form == AssetForm.ENTITY_UV:
        if not spec.uv_regions:
            errors.append("entity_uv geometry requires explicit uv_regions; do not guess a model layout")
        regions_by_part: dict[str, list[set[tuple[int, int]]]] = {}
        for region in spec.uv_regions:
            left, top, right, bottom = region.bbox
            region_points = {(x, y) for y in range(top, bottom) for x in range(left, right)}
            regions_by_part.setdefault(region.part_id, []).append(region_points)
            painted = _points(compiled.part_masks[region.part_id]) & region_points
            metrics["uv_%s_pixels" % region.id] = len(painted)
            if region.required and not painted:
                errors.append("required UV region %s has no pixels for part %s" % (region.id, region.part_id))
        for part in spec.parts:
            part_points = _points(compiled.part_masks[part.id])
            allowed = set().union(*regions_by_part.get(part.id, [])) if part.id in regions_by_part else set()
            outside = part_points - allowed
            if outside:
                errors.append(
                    "part %s paints %d pixel(s) outside its declared UV regions" % (part.id, len(outside))
                )
            if part.required and not part.paint_only and part_points and not allowed:
                errors.append("required entity part %s has no declared UV region" % part.id)

    if form == AssetForm.BLOCK_MULTI:
        if not spec.uv_regions:
            errors.append("block_multi geometry requires top/front/side UV regions")
        faces = {region.face.lower() for region in spec.uv_regions}
        if "top" not in faces:
            errors.append("block_multi layout is missing a top face")
        if not ({"front", "north", "south"} & faces):
            errors.append("block_multi layout is missing a front face")
        if not ({"right", "east", "west", "side", "left"} & faces):
            errors.append("block_multi layout is missing a side face")
        regions_by_part: dict[str, list[set[tuple[int, int]]]] = {}
        for region in spec.uv_regions:
            left, top, right, bottom = region.bbox
            region_points = {(x, y) for y in range(top, bottom) for x in range(left, right)}
            regions_by_part.setdefault(region.part_id, []).append(region_points)
            painted = _points(compiled.part_masks[region.part_id]) & region_points
            metrics["uv_%s_pixels" % region.id] = len(painted)
            if region.required and not painted:
                errors.append("required block face %s has no pixels for part %s" % (region.id, region.part_id))
        for part in spec.parts:
            part_points = _points(compiled.part_masks[part.id])
            allowed = set().union(*regions_by_part.get(part.id, [])) if part.id in regions_by_part else set()
            if part_points - allowed:
                errors.append("block part %s paints outside its declared face regions" % part.id)

    if minimum_margin is None:
        # Edge occupancy is part of the model-authored silhouette.  Inventory
        # sprites and crosses may legitimately touch a canvas edge (the
        # vanilla sword reference does), so a one-pixel inset is not a generic
        # validity rule.  Callers that truly need padding can still pass an
        # explicit ``minimum_margin`` or emit a model-authored constraint.
        minimum_margin = 0
    if all_points and minimum_margin > 0:
        for side in ("left", "top", "right", "bottom"):
            if int(metrics["margin_%s" % side]) < minimum_margin:
                errors.append("geometry violates %dpx %s transparent margin" % (minimum_margin, side))

    for constraint in spec.constraints:
        value = metrics.get(constraint.metric)
        if value is None:
            errors.append("constraint references unavailable metric: %s" % constraint.metric)
            continue
        numeric = float(value)
        # Principal-axis orientation has a sign because the image y-axis
        # points downward. Constraints express an undirected diagonal angle,
        # so compare its magnitude; otherwise a valid -50° blade fails a
        # 30–60° diagonal constraint while its mirrored +50° counterpart
        # passes.
        if (
            constraint.metric == "orientation_degrees"
            and (constraint.minimum is None or constraint.minimum >= 0)
            and (constraint.maximum is None or constraint.maximum >= 0)
        ):
            numeric = abs(numeric)
        if constraint.minimum is not None and numeric < constraint.minimum:
            errors.append(
                "%s: %s=%.3f is below %.3f" % (
                    constraint.message or constraint.metric,
                    constraint.metric,
                    numeric,
                    constraint.minimum,
                )
            )
        if constraint.maximum is not None and numeric > constraint.maximum:
            errors.append(
                "%s: %s=%.3f exceeds %.3f" % (
                    constraint.message or constraint.metric,
                    constraint.metric,
                    numeric,
                    constraint.maximum,
                )
            )
    return ValidationResult(
        passed=not errors,
        stage="geometry",
        metrics=metrics,
        errors=errors,
        warnings=warnings,
    )


def validate_render_alpha(
    compiled: CompiledGeometry,
    rendered: Image.Image,
    expected_mask: Image.Image | None = None,
) -> ValidationResult:
    expected = (expected_mask or compiled.mask).convert("L")
    actual = rendered.convert("RGBA").getchannel("A")
    if actual.size != expected.size:
        return ValidationResult(
            passed=False,
            stage="render_alpha",
            metrics={"expected_width": expected.width, "actual_width": actual.width},
            errors=["rendered image size differs from geometry mask"],
        )
    mismatch = sum(
        (expected.getpixel((x, y)) > 0) != (actual.getpixel((x, y)) > 0)
        for y in range(expected.height)
        for x in range(expected.width)
    )
    return ValidationResult(
        passed=mismatch == 0,
        stage="render_alpha",
        metrics={"alpha_exact": mismatch == 0, "alpha_mismatch_pixels": mismatch},
        errors=[] if mismatch == 0 else ["rendered alpha differs from the locked expected mask"],
    )


class SemanticCritic(Protocol):
    def review(self, descriptor: ShapeDescriptor, compiled: CompiledGeometry) -> ValidationResult:
        """Review semantic recognisability without altering the geometry."""


@dataclass
class StructuralCritic:
    """Offline critic used when an LLM critic is unavailable.

    This is deliberately conservative: it verifies descriptor parts are present
    and exposes metrics. It never pretends to recognise an arbitrary noun from
    pixels; an optional model critic can add that judgement later.
    """

    def review(self, descriptor: ShapeDescriptor, compiled: CompiledGeometry) -> ValidationResult:
        metrics = geometry_metrics(compiled)
        errors: list[str] = []
        warnings = ["offline structural critic: semantic noun recognition was not requested from a model"]
        for part in descriptor.parts:
            count = int(metrics.get("%s_pixels" % part.id, 0))
            if part.required and not part.paint_only and count == 0:
                errors.append("descriptor requires missing part: %s" % part.id)
        return ValidationResult(
            passed=not errors,
            stage="semantic",
            metrics=metrics,
            errors=errors,
            warnings=warnings,
        )


def aggregate_results(results: Iterable[ValidationResult]) -> ValidationResult:
    results = list(results)
    metrics: dict[str, float | int | str | bool] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for result in results:
        metrics.update({"%s.%s" % (result.stage, key): value for key, value in result.metrics.items()})
        errors.extend("[%s] %s" % (result.stage, item) for item in result.errors)
        warnings.extend("[%s] %s" % (result.stage, item) for item in result.warnings)
    return ValidationResult(
        passed=not errors,
        stage="aggregate",
        metrics=metrics,
        errors=errors,
        warnings=warnings,
    )
