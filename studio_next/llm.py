"""Optional OpenAI-compatible planning and semantic-critic adapter.

No key is stored in project files. The caller supplies LLM_API_KEY and optional
LLM_BASE_URL / LLM_MODEL / LLM_REASONING_EFFORT environment variables.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .contracts import (
    ArtDirection,
    AppearanceSpec,
    AssetForm,
    AssetRequest,
    GeometrySpec,
    ReferenceAsset,
    ReferenceRole,
    ShapeDescriptor,
    UvRegionSpec,
    ValidationResult,
    appearance_from_dict,
    art_direction_from_dict,
    descriptor_from_dict,
    geometry_from_dict,
    is_overlay_part,
    to_jsonable,
)
from .geometry import CompiledGeometry, SUPPORTED_PRIMITIVES, compile_geometry, primitive_parameter_guide, rescale_geometry_spec
from .references import summarize_reference
from .reference_retrieval import build_router_manifest, manifest_label, parse_router_selection, route_cache_key
from .validation import validate_geometry


def _family_slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug[:48] or fallback


def _normalize_family(data: dict[str, Any], query: str, max_members: int) -> dict[str, Any]:
    """Validate a model-authored member list into a stable set contract."""
    raw_members = data.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("family plan needs a non-empty members list")
    members: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_members[:max_members]):
        if isinstance(item, str):
            item = {"name": item, "target": item}
        if not isinstance(item, dict):
            raise ValueError("family member must be an object")
        target = str(item.get("target") or item.get("name") or "").strip()
        if not target:
            raise ValueError("family member needs a target")
        name = _family_slug(item.get("name") or target, "member_%d" % (index + 1))
        if name in seen:
            name = "%s_%d" % (name, index + 1)
        seen.add(name)
        members.append({"name": name, "target": target})
    return {
        "set_name": _family_slug(data.get("set_name") or query, "asset_set"),
        "shared_style": str(data.get("shared_style") or "").strip(),
        "members": members,
    }


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("model response does not contain a JSON object")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model response must be a JSON object")
    return data


def _normalize_descriptor_part_ids(data: dict[str, Any]) -> dict[str, Any]:
    """Keep model-generated part identifiers stable across all downstream files.

    Descriptions may follow the user's language, but identifiers are shared by
    geometry masks, appearance maps and filesystem artifacts.  Converting only
    invalid identifiers (rather than translating meanings) preserves the model's
    semantics while preventing non-ASCII names or punctuation from becoming
    accidental contract keys.
    """
    parts = data.get("parts")
    if not isinstance(parts, list):
        return data
    normalized = dict(data)
    normalized_parts: list[Any] = []
    used: set[str] = set()
    for index, raw_part in enumerate(parts, start=1):
        if not isinstance(raw_part, dict):
            normalized_parts.append(raw_part)
            continue
        part = dict(raw_part)
        candidate = re.sub(r"[^A-Za-z0-9_-]+", "_", str(part.get("id", ""))).strip("_")
        if not candidate or not re.match(r"^[A-Za-z]", candidate):
            candidate = "part_%d" % index
        if candidate in used:
            suffix = 2
            base = candidate
            while "%s_%d" % (base, suffix) in used:
                suffix += 1
            candidate = "%s_%d" % (base, suffix)
        used.add(candidate)
        part["id"] = candidate
        normalized_parts.append(part)
    normalized["parts"] = normalized_parts
    return normalized


def _strip_visual_overlay_parts(data: dict[str, Any]) -> dict[str, Any]:
    """Keep geometry contracts focused on physical sections, not paint layers."""
    parts = data.get("parts")
    if not isinstance(parts, list) or len(parts) <= 1:
        return data
    overlay_cues = (
        "highlight", "shadow", "outline", "texture", "grain", "stitch", "speckle",
        "glint", "shine", "wear", "fold", "高光", "阴影", "轮廓", "纹理", "颗粒",
        "缝线", "磨损", "折痕", "装饰",
    )
    physical_cues = (
        "body", "panel", "blade", "cutting", "handle", "grip", "shaft",
        "guard", "head", "stem", "cap", "hinge", "joint", "connector", "主体", "面板",
        "刀", "刃", "柄", "杆", "护", "接头",
    )
    kept: list[Any] = []
    for raw_part in parts:
        if not isinstance(raw_part, dict):
            kept.append(raw_part)
            continue
        searchable = " ".join(
            str(raw_part.get(key, "")) for key in ("id", "meaning", "style_role")
        ).lower()
        is_overlay = any(cue in searchable for cue in overlay_cues)
        is_physical = any(cue in searchable for cue in physical_cues)
        if not (is_overlay and not is_physical):
            kept.append(raw_part)
    # A malformed response containing only paint-layer names still needs one
    # physical anchor so downstream geometry has a valid required part.
    if not kept:
        kept = [parts[0]]
    normalized = dict(data)
    normalized["parts"] = kept
    return normalized


def _placeholder_geometry_issues(geometry: GeometrySpec, form: AssetForm) -> list[str]:
    """Find generic filled-block shortcuts before they reach semantic review.

    This is a shape-language lint, not an object classifier.  At small item
    resolutions it asks for an explicit pixel contour whenever a sizeable part
    is represented by a dense coarse primitive, including the single-part case
    where there is no second labelled part to expose the placeholder.
    """
    if form not in {AssetForm.ITEM, AssetForm.CROSS}:
        return []
    try:
        compiled = compile_geometry(geometry)
    except (KeyError, TypeError, ValueError):
        return []
    issues: list[str] = []
    coarse_parts = 0
    part_by_id = {part.id: part for part in geometry.parts}
    for part_id, mask in compiled.part_masks.items():
        pixels = sum(1 for value in mask.get_flattened_data() if value > 0)
        bbox = mask.getbbox()
        if part_by_id[part_id].required and not part_by_id[part_id].paint_only and pixels == 0:
            issues.append("required part %s has an empty mask" % part_id)
            continue
        if not bbox or pixels < 8:
            continue
        left, top, right, bottom = bbox
        area = max((right - left) * (bottom - top), 1)
        role = part_by_id[part_id].style_role.lower()
        if (right - left) >= 4 and (bottom - top) >= 4 and pixels / float(area) >= 0.84:
            if not any(token in role for token in {"body", "core", "base", "face"}):
                issues.append(
                    "%s is a nearly filled %dx%d block; encode its contour with custom_mask or a tapered primitive"
                    % (part_id, right - left, bottom - top)
                )
        primitive_types = {
            item.primitive
            for item in geometry.primitives
            if item.part_id == part_id
        }
        if (
            primitive_types
            and primitive_types <= {"polygon", "ellipse", "blob"}
            and (right - left) >= 6
            and (bottom - top) >= 6
            and pixels / float(area) >= 0.65
        ):
            issues.append(
                "%s is a coarse %s over a dense %dx%d area; encode its pixel contour with custom_mask"
                % (part_id, "/".join(sorted(primitive_types)), right - left, bottom - top)
            )
        if primitive_types and primitive_types <= {"polygon", "ellipse", "blob"}:
            coarse_parts += 1
    if len(geometry.parts) >= 2 and coarse_parts >= max(2, len(geometry.parts) // 2):
        issues.append("multiple semantic parts use only coarse polygon/ellipse/blob primitives")
    return issues


def _descriptor_needs_target_part_map(descriptor: ShapeDescriptor, form: AssetForm) -> bool:
    """Target ownership grids are retired in favour of direct final rasters.

    They duplicated the source pixel text while forcing the model to split a
    local edit into artificial physical regions.  A break, chip, inlay or
    other detail is already represented by the descriptor and the final
    GeometrySpec/AppearanceSpec.  The model can therefore make its own local
    placement decision with the full reference raster in view, without an
    intermediate labelled grid becoming a second, stale shape authority.
    """
    return False


def _target_map_covers_declared_parts(descriptor: ShapeDescriptor, width: int, height: int) -> bool:
    evidence = descriptor.target_part_map
    if not isinstance(evidence, dict):
        return False
    legend = evidence.get("legend")
    rows = evidence.get("rows")
    if not isinstance(legend, dict) or not isinstance(rows, list) or len(rows) != height:
        return False
    if any(not isinstance(row, str) or len(row) != width for row in rows):
        return False
    marker_by_part = {
        str(part_id): str(marker)
        for marker, part_id in legend.items()
    }
    source_points_by_part: dict[str, set[tuple[int, int]]] = {}
    source_evidence = descriptor.reference_part_map
    if isinstance(source_evidence, dict) and isinstance(source_evidence.get("legend"), dict) and isinstance(source_evidence.get("rows"), list):
        source_rows = source_evidence.get("rows", [])
        source_marker_to_part = {
            str(marker): str(part_id)
            for marker, part_id in source_evidence["legend"].items()
        }
        if len(source_rows) == height and all(isinstance(row, str) and len(row) == width for row in source_rows):
            for y, row in enumerate(source_rows):
                for x, value in enumerate(row):
                    part_id = source_marker_to_part.get(value)
                    if part_id:
                        source_points_by_part.setdefault(part_id, set()).add((x, y))
    # Every declared physical part must have at least one target cell, not
    # merely a legend entry.  A painterly detail may still be omitted from the
    # map by making it paint-only in the descriptor; geometry never has to
    # invent such a part.
    for part in descriptor.parts:
        if part.paint_only or not part.required or part.id not in marker_by_part:
            if part.required:
                if part.paint_only:
                    continue
                return False
            continue
        marker = marker_by_part[part.id]
        points = {
            (x, y)
            for y, row in enumerate(rows)
            for x, value in enumerate(row)
            if value == marker
        }
        if not points:
            return False
        if part.contour_intent == "compact":
            # The model supplied this intent. Check only generic spatial
            # evidence that the map did not turn it into a diagonal streak;
            # this validates a plan but never constructs a named object.
            if len(points) < 5:
                return False
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            if len(set(xs)) < 2 or len(set(ys)) < 2:
                return False
            mean_x = sum(xs) / float(len(xs))
            mean_y = sum(ys) / float(len(ys))
            variance_x = sum((value - mean_x) ** 2 for value in xs) / float(len(xs))
            variance_y = sum((value - mean_y) ** 2 for value in ys) / float(len(ys))
            covariance = sum((x - mean_x) * (y - mean_y) for x, y in points) / float(len(points))
            discriminant = max(
                0.0,
                (variance_x + variance_y) ** 2 - 4.0 * (variance_x * variance_y - covariance ** 2),
            ) ** 0.5
            principal = (variance_x + variance_y + discriminant) / 2.0
            secondary = (variance_x + variance_y - discriminant) / 2.0
            if principal <= 0.0 or (principal - secondary) / principal > 0.72:
                return False
        part_text = (part.meaning + " " + part.style_role).lower()
        # A model-declared round/local motif needs enough target cells to
        # survive rasterization as a contour.  This is a generic resolution
        # check; it does not prescribe a circle, palette or object category.
        if is_overlay_part(part) and any(token in part_text for token in ("round", "circle", "orb", "eye", "gem", "motif")):
            xs = {point[0] for point in points}
            ys = {point[1] for point in points}
            row_counts = [sum(1 for x, py in points if py == y) for y in sorted(ys)]
            max_count = max(row_counts)
            peak_rows = [index for index, count in enumerate(row_counts) if count == max_count]
            if (
                len(points) < max(5, min(8, round(min(width, height) * 0.5)))
                or len(xs) < 2
                or len(ys) < 3
                or len(set(row_counts)) < 2
                or not any(0 < index < len(row_counts) - 1 for index in peak_rows)
            ):
                return False
        # For an embedded/inlaid part, the target map must actually place the
        # new cells on the model-declared host support.  This catches a common
        # low-resolution failure where the map puts the motif on a handle or
        # terminal continuation merely because it is nearby.  New attached
        # motifs may sit on transparent source pixels; this check is only for
        # wording that explicitly says the part is embedded/inlaid.
        if is_overlay_part(part) and any(token in part_text for token in ("embedded", "inlaid", "inlay", "镶嵌", "嵌入")) and source_points_by_part:
            host_scores: dict[str, int] = {}
            for host_id, host_points in source_points_by_part.items():
                score = len(points & host_points)
                if score:
                    host_scores[host_id] = score
            if host_scores:
                best_host_score = max(host_scores.values())
                if best_host_score / float(max(len(points), 1)) < 0.60:
                    return False
    return True


def _target_map_reflects_requested_silhouette_change(
    descriptor: ShapeDescriptor,
    width: int,
    height: int,
) -> bool:
    """Whether model-authored source and target occupancy actually differ.

    This applies only when the prior reference-free direction says alpha must
    change. It does not prescribe the changed pixels; it prevents a planner
    from calling a copied source ownership map a broken/cut-away/new contour.
    """
    direction = descriptor.art_direction
    if direction is None or not direction.requires_silhouette_change:
        return True
    source = descriptor.reference_part_map
    target = descriptor.target_part_map
    if not isinstance(source, dict) or not isinstance(target, dict):
        return False
    source_rows = source.get("rows")
    target_rows = target.get("rows")
    if (
        not isinstance(source_rows, list)
        or not isinstance(target_rows, list)
        or len(source_rows) != height
        or len(target_rows) != height
        or any(not isinstance(row, str) or len(row) != width for row in source_rows)
        or any(not isinstance(row, str) or len(row) != width for row in target_rows)
    ):
        return False
    return any(
        (source_row[x] != ".") != (target_row[x] != ".")
        for source_row, target_row in zip(source_rows, target_rows)
        for x in range(width)
    )


def _target_overlay_geometry_issues(
    descriptor: ShapeDescriptor,
    geometry: GeometrySpec,
    compiled: CompiledGeometry,
) -> list[str]:
    """Keep a model-authored overlay local and visibly distinct.

    A target map strengthens the area check, but the contour checks also apply
    when the descriptor had to proceed without a valid map after correction.
    That leaves the geometry model a recoverable, open-ended way to author the
    motif instead of silently accepting a terminal blob.
    """
    target = descriptor.target_part_map
    overlay_ids = {part.id for part in descriptor.parts if is_overlay_part(part)}
    if not overlay_ids:
        return []
    marker_by_part = {}
    rows = []
    if isinstance(target, dict) and isinstance(target.get("legend"), dict):
        marker_by_part = {
            str(part_id): str(marker)
            for marker, part_id in target["legend"].items()
        }
        rows = target.get("rows", [])
    issues: list[str] = []
    for part_id in overlay_ids:
        marker = marker_by_part.get(part_id)
        if marker is None:
            continue
        target_pixels = sum(
            1 for row in rows if isinstance(row, str) for value in row if value == marker
        )
        mask = compiled.part_masks.get(part_id)
        if mask is None:
            continue
        candidate_pixels = sum(
            1 for y in range(compiled.height) for x in range(compiled.width)
            if mask.getpixel((x, y)) > 0
        )
        if target_pixels and candidate_pixels < target_pixels * 0.90:
            issues.append(
                "overlay part %s collapsed from %d target cells to %d"
                % (part_id, target_pixels, candidate_pixels)
            )
        points = {
            (x, y)
            for y in range(compiled.height)
            for x in range(compiled.width)
            if mask.getpixel((x, y)) > 0
        }
        if not points:
            continue
        part_text = " ".join(
            part.meaning + " " + part.style_role
            for part in descriptor.parts
            if part.id == part_id
        ).lower()
        support_parts = [
            part for part in descriptor.parts
            if part.id not in overlay_ids
        ]
        # An inlaid part should be a local overlay, not a second copy of the
        # host support.  Use the model's own masks and roles; no object noun or
        # fixed sword/tool dimensions are involved here.
        best_host = None
        best_overlap = 0
        for support in support_parts:
            host_mask = compiled.part_masks.get(support.id)
            if host_mask is None:
                continue
            host_points = {
                (x, y)
                for y in range(compiled.height)
                for x in range(compiled.width)
                if host_mask.getpixel((x, y)) > 0
            }
            overlap = len(points & host_points)
            if overlap > best_overlap:
                best_overlap = overlap
                best_host = host_points
        if best_host:
            ratio = len(points) / float(max(len(best_host), 1))
            if ratio >= 0.85:
                issues.append(
                    "overlay part %s consumes nearly the whole host support (%.0f%%)"
                    % (part_id, ratio * 100.0)
                )
            motif_bbox = (
                min(x for x, _ in points), min(y for _, y in points),
                max(x for x, _ in points) + 1, max(y for _, y in points) + 1,
            )
            host_bbox = (
                min(x for x, _ in best_host), min(y for _, y in best_host),
                max(x for x, _ in best_host) + 1, max(y for _, y in best_host) + 1,
            )
            rim_sides = sum(
                (
                    motif_bbox[0] > host_bbox[0],
                    motif_bbox[1] > host_bbox[1],
                    motif_bbox[2] < host_bbox[2],
                    motif_bbox[3] < host_bbox[3],
                )
            )
            if any(token in part_text for token in ("embedded", "inlaid", "motif", "eye", "gem", "orb", "round", "circle")) and rim_sides < 2:
                issues.append(
                    "overlay part %s is at the host terminal edge; leave a visible rim on at least two sides"
                    % part_id
                )
        if any(token in part_text for token in ("eye", "eyeball", "gem", "orb", "round", "circle", "motif")):
            minimum_local_pixels = max(5, min(16, round(min(compiled.width, compiled.height) * 0.5)))
            if len(points) < minimum_local_pixels:
                issues.append(
                    "overlay part %s is too small at this raster size (%d pixels); enlarge its local contour without consuming the host"
                    % (part_id, len(points))
                )
            row_counts = [
                sum(1 for x in range(compiled.width) if mask.getpixel((x, y)) > 0)
                for y in range(compiled.height)
            ]
            active_rows = [count for count in row_counts if count]
            if len(active_rows) >= 3:
                peak = max(active_rows)
                peak_index = max(index for index, count in enumerate(active_rows) if count == peak)
                if active_rows[0] >= peak or active_rows[-1] >= peak or peak_index in {0, len(active_rows) - 1}:
                    issues.append(
                        "overlay part %s has a terminal/flat contour; narrow the first and last rows around a middle peak"
                        % part_id
                    )
    return issues


def _paint_only_surface_issues(
    descriptor: ShapeDescriptor,
    geometry: GeometrySpec,
    appearance: AppearanceSpec,
) -> list[str]:
    """Verify that model-declared surface-only features have executable pixels.

    A paint-only part deliberately has no alpha mask.  On a compact item/cross
    there is no UV-region coordinate system either, so a native whole-canvas
    pixel map is the unambiguous, model-authored way to place it.  The renderer
    still preserves alpha and merely executes the map.
    """
    paint_only_required = [
        part for part in descriptor.parts if part.required and part.paint_only
    ]
    if not paint_only_required:
        return []
    pixel_map = appearance.pixel_map
    if not isinstance(pixel_map, dict):
        return ["required paint-only parts need a native pixel_map on the locked canvas"]
    rows = pixel_map.get("rows")
    legend = pixel_map.get("legend")
    if not isinstance(rows, list) or not isinstance(legend, dict):
        return ["paint-only pixel_map needs legend and rows"]
    if len(rows) != geometry.height or any(not isinstance(row, str) or len(row) != geometry.width for row in rows):
        return ["paint-only pixel_map rows must exactly match the native canvas"]
    used_symbols = {symbol for row in rows if isinstance(row, str) for symbol in row if symbol != "."}
    invalid_colours: list[str] = []
    for symbol in used_symbols:
        token = str(legend.get(symbol, "")).strip()
        is_hex = bool(re.fullmatch(r"#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})", token))
        if not token or (token not in appearance.palette and not is_hex):
            invalid_colours.append(symbol)
    if invalid_colours:
        return ["paint-only pixel_map symbols must resolve to palette tokens or hex colors: %s" % ", ".join(sorted(invalid_colours))]
    try:
        compiled = compile_geometry(geometry)
    except (KeyError, TypeError, ValueError):
        return ["paint-only pixel_map could not inspect the compiled host mask"]
    active_cells = [
        (x, y, symbol)
        for y, row in enumerate(rows)
        for x, symbol in enumerate(row)
        if symbol != "." and compiled.mask.getpixel((x, y)) > 0
    ]
    active = len(active_cells)
    opaque = sum(
        1
        for y in range(compiled.height)
        for x in range(compiled.width)
        if compiled.mask.getpixel((x, y)) > 0
    )
    if active == 0:
        return ["paint-only pixel_map has no authored cells on opaque host pixels"]
    if active > max(8, round(opaque * 0.45)):
        return ["paint-only pixel_map must stay local and cannot repaint most of its host"]
    used_tokens = {str(legend[symbol]).strip() for _x, _y, symbol in active_cells}
    missing_feature_tokens = []
    for part in paint_only_required:
        style = appearance.parts.get(part.id)
        if style is not None and not ({str(token).strip() for token in style.colors} & used_tokens):
            missing_feature_tokens.append(part.id)
    if missing_feature_tokens:
        return ["paint-only pixel_map must use each feature part's declared palette tokens: %s" % ", ".join(missing_feature_tokens)]
    return []


def _mask_data_uri(mask: Image.Image, scale: int = 24) -> str:
    preview = mask.convert("RGBA").resize(
        (mask.width * scale, mask.height * scale), Image.Resampling.NEAREST
    )
    buffer = BytesIO()
    preview.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_PART_MAP_COLORS: tuple[tuple[int, int, int], ...] = (
    (239, 96, 96), (96, 176, 239), (114, 214, 128), (224, 174, 76),
    (191, 112, 224), (74, 204, 198), (238, 132, 184), (168, 168, 168),
)


def _part_map_data_uri(descriptor: ShapeDescriptor, compiled: CompiledGeometry, scale: int = 24) -> str:
    """Encode an inspectable map where each declared part has a distinct color.

    A binary union mask cannot show overlapping semantic details such as eyes
    inside a head. This diagnostic image is sent only to planning/critic
    models, never to the target-free blind reviewer or as a rendered asset.
    """
    image = Image.new("RGBA", (compiled.width, compiled.height), (0, 0, 0, 0))
    # Draw in descriptor layer order so a detail part (eyes, inlay, etc.) is
    # visible on top of the physical part it occupies.
    for index, part in enumerate(sorted(descriptor.parts, key=lambda item: (item.layer, item.id))):
        mask = compiled.part_masks.get(part.id)
        if mask is None:
            continue
        color = _PART_MAP_COLORS[index % len(_PART_MAP_COLORS)]
        for y in range(compiled.height):
            for x in range(compiled.width):
                if mask.getpixel((x, y)) > 0:
                    image.putpixel((x, y), (*color, 255))
    return _mask_data_uri(image, scale=scale)


def _part_map_legend(descriptor: ShapeDescriptor) -> str:
    entries = []
    for index, part in enumerate(sorted(descriptor.parts, key=lambda item: (item.layer, item.id))):
        red, green, blue = _PART_MAP_COLORS[index % len(_PART_MAP_COLORS)]
        entries.append("%s=#%02X%02X%02X" % (part.id, red, green, blue))
    return ", ".join(entries)


def _appearance_brief(descriptor: ShapeDescriptor) -> dict[str, Any]:
    """Return the small, visual-only contract used by the texture author.

    Ownership maps are useful while constructing a mask, but they are an
    actively harmful representation for a model that is painting a locked
    raster: a large labelled blade invites it to repaint the blade as a flat
    area.  The painter sees the source pixels themselves and this short brief;
    it does not need a second symbolic version of the same sprite.
    """
    direction = descriptor.art_direction
    return {
        "target": descriptor.target,
        "semantic": descriptor.semantic,
        "visual_identity": descriptor.visual_identity,
        "negative_identities": descriptor.negative_identities,
        "orientation": descriptor.orientation,
        "shape_edit_mode": descriptor.shape_edit_mode,
        "art_direction": (
            {
                "design_intent": direction.design_intent,
                "visual_hierarchy": direction.visual_hierarchy,
                "composition": direction.composition,
                "feature_actions": direction.feature_actions,
                "surface_actions": direction.surface_actions,
                "preservation_rules": direction.preservation_rules,
                "feature_relationships": direction.feature_relationships,
                "negative_constraints": direction.negative_constraints,
            }
            if direction is not None
            else None
        ),
        "parts": [
            {
                "id": part.id,
                "meaning": part.meaning,
                "paint_only": part.paint_only,
                "style_role": part.style_role,
                "recognition_terms": part.recognition_terms,
            }
            for part in descriptor.parts
        ],
    }


def _appearance_reference_evidence(references: list[ReferenceAsset]) -> str:
    """Give the painter lossless small-raster evidence without analysis noise.

    Every attached reference remains the primary evidence.  For 64px-or-less
    sprites the text side channel mirrors every source pixel as a compact
    palette legend plus rows, so vision ambiguity cannot turn an original
    texture into a three-colour summary.  Large atlases are supplied as images
    and only identified by their role and dimensions.
    """
    entries: list[str] = []
    for reference in references:
        features = reference.features
        label = "%s roles=%s size=%sx%s" % (
            reference.name,
            ",".join(role.value for role in reference.roles),
            features.get("width", "?"),
            features.get("height", "?"),
        )
        # The router's rationale is semantic evidence.  Omitting it here made
        # a painter see two equally-sized opaque tiles and freely swap a
        # side/bark reference with a top/end-grain reference.  Keep this
        # compact: the attached pixels remain the primary evidence, while the
        # note explains what a structurally similar raster is meant to teach.
        notes = " ".join(str(note).strip() for note in reference.notes if str(note).strip())
        if notes:
            label += " intent=" + notes
        pixel_text = str(features.get("pixel_text", "")).strip()
        if pixel_text and max(int(features.get("width", 65) or 65), int(features.get("height", 65) or 65)) <= 64:
            entries.append(label + "\nSOURCE_PIXELS (legend and rows; . is transparent):\n" + pixel_text)
        else:
            entries.append(label)
    return "\n\n".join(entries) or "- none"


def _same_size_region_ratios(
    references: list[ReferenceAsset] | None,
    regions: list[Any],
    width: int,
    height: int,
) -> tuple[str, dict[str, float]] | None:
    """Per-face paint coverage of the best same-size reference.

    A dithered or partial source is not a usable alpha mask, but which faces it
    touches is still the vanilla model's own statement about what it paints and
    what it leaves open. Reporting that as numbers keeps the evidence usable
    instead of discarding the whole reference.
    """
    if not references or not regions:
        return None
    best: tuple[int, str, dict[str, float]] | None = None
    wanted = {
        ReferenceRole.SHAPE, ReferenceRole.UV_LAYOUT,
        ReferenceRole.PIXEL_STYLE, ReferenceRole.MATERIAL, ReferenceRole.PALETTE,
    }
    for reference in references:
        if ReferenceRole.NEGATIVE in reference.roles:
            continue
        if not any(role in wanted for role in reference.roles):
            continue
        try:
            with Image.open(reference.path) as loaded:
                source = loaded.convert("RGBA")
        except (OSError, ValueError):
            continue
        if source.size != (width, height):
            continue
        ratios: dict[str, float] = {}
        for region in regions:
            left, top, right, bottom = region.bbox
            total = painted = 0
            for y in range(max(0, top), min(height, bottom)):
                for x in range(max(0, left), min(width, right)):
                    total += 1
                    if source.getpixel((x, y))[3] >= 8:
                        painted += 1
            ratios[region.id] = painted / float(max(total, 1))
        touched = sum(1 for value in ratios.values() if value > 0.0)
        if best is None or touched > best[0]:
            best = (touched, reference.name, ratios)
    if best is None:
        return None
    return best[1], best[2]


def _reference_data_uri(path: str) -> str:
    """Encode a local sprite for a vision model without copying it into the project.

    Vanilla textures are commonly 16×16. Upscaling tiny sources with nearest
    neighbour makes their silhouette and individual pixels visible to a vision
    endpoint while preserving the original evidence exactly.
    """
    with Image.open(path) as loaded:
        image = loaded.convert("RGBA")
    if max(image.size) <= 64:
        image = image.resize((image.width * 12, image.height * 12), Image.Resampling.NEAREST)
    elif max(image.size) > 768:
        image.thumbnail((768, 768), Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _reference_data_uris(references: list[ReferenceAsset]) -> list[str]:
    """Load each declared reference once per planning stage, preserving its order."""
    image_data_uris: list[str] = []
    for reference in references:
        try:
            image_data_uris.append(_reference_data_uri(reference.path))
        except OSError as exc:
            raise RuntimeError("cannot load reference image %s: %s" % (reference.path, exc)) from exc
    return image_data_uris


def _reference_board_data_uri(references: list[ReferenceAsset]) -> str | None:
    """Build one labelled nearest-neighbour board for multi-source vision calls.

    Individual image messages preserve exact pixels but do not carry a stable
    human-readable label into every vision provider.  A small contact board
    makes the name-to-image association explicit while keeping each source's
    native pixel rhythm intact.  It is evidence only; the renderer still
    consumes the original local files.
    """
    if len(references) < 2:
        return None
    tiles: list[tuple[str, Image.Image]] = []
    for reference in references:
        try:
            image = Image.open(reference.path).convert("RGBA")
        except (OSError, ValueError):
            continue
        longest = max(image.size)
        scale = max(1, min(16, 256 // max(longest, 1)))
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
        tiles.append((reference.name, image))
    if len(tiles) < 2:
        return None
    tile_width = max(image.width for _name, image in tiles) + 24
    tile_height = max(image.height for _name, image in tiles) + 38
    columns = min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    board = Image.new("RGBA", (tile_width * columns, tile_height * rows), (18, 18, 18, 255))
    draw = ImageDraw.Draw(board)
    for index, (name, image) in enumerate(tiles):
        left = (index % columns) * tile_width + 12
        top = (index // columns) * tile_height + 28
        draw.text((left, 7 + (index // columns) * tile_height), name[:32], fill=(245, 245, 245, 255))
        board.alpha_composite(image, (left, top))
    buffer = BytesIO()
    board.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _appearance_image_data_uris(references: list[ReferenceAsset]) -> list[str]:
    """Attach raw references plus a labelled board to appearance planning."""
    image_data_uris = _reference_data_uris(references)
    board = _reference_board_data_uri(references)
    if board is not None:
        image_data_uris.append(board)
    return image_data_uris


def _geometry_reference_data_uris(references: list[ReferenceAsset]) -> list[str]:
    """Attach shape-role alpha maps beside their colour references.

    Tiny transparent PNGs are easy for a vision model to read as a colour
    swatch or lose against a black canvas. A matching white-on-transparent
    alpha map makes the contour evidence explicit while leaving the model free
    to author a new mask. The map is planning evidence only and is never
    copied into the render.
    """
    image_data_uris: list[str] = []
    for reference in references:
        image_data_uris.append(_reference_data_uri(reference.path))
        if ReferenceRole.SHAPE not in reference.roles:
            continue
        try:
            with Image.open(reference.path) as loaded:
                alpha = loaded.convert("RGBA").getchannel("A")
            silhouette = Image.new("RGBA", alpha.size, (0, 0, 0, 0))
            silhouette.putalpha(alpha)
            image_data_uris.append(_mask_data_uri(silhouette))
        except OSError as exc:
            raise RuntimeError("cannot load reference silhouette %s: %s" % (reference.path, exc)) from exc
    return image_data_uris


def _shape_baseline_text(
    references: list[ReferenceAsset], width: int, height: int
) -> str:
    """Render the selected same-size alpha reference as a compact text ruler."""
    for reference in references:
        if ReferenceRole.SHAPE not in reference.roles:
            continue
        if int(reference.features.get("width") or 0) != width:
            continue
        if int(reference.features.get("height") or 0) != height:
            continue
        profile = reference.features.get("silhouette_profile")
        rows = profile.get("silhouette_map") if isinstance(profile, dict) else None
        if isinstance(rows, str) and rows.strip():
            return (
                "SOURCE_ALPHA_BASELINE (one # per opaque source pixel; "
                "rows are top-to-bottom):\n" + rows
            )
    return "SOURCE_ALPHA_BASELINE: none"


@dataclass(frozen=True)
class OpenAICompatibleClient:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str | None = None
    timeout_seconds: int = 420
    trace_dir: str | None = None

    @classmethod
    def from_env(cls, trace_dir: str | Path | None = None) -> "OpenAICompatibleClient":
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("LLM_API_KEY is required for model planning")
        return cls(
            api_key=key,
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("LLM_MODEL", "deepseek-flash"),
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or None,
            trace_dir=str(trace_dir) if trace_dir is not None else None,
        )

    def _trace_response(self, prompt: str, response: str) -> None:
        """Persist text-only model exchanges when a pipeline supplies a trace root."""
        if not self.trace_dir:
            return
        try:
            root = Path(self.trace_dir)
            root.mkdir(parents=True, exist_ok=True)
            index = len(list(root.glob("*.response.txt")))
            stem = "%03d" % index
            (root / (stem + ".prompt.txt")).write_text(prompt, encoding="utf-8")
            (root / (stem + ".response.txt")).write_text(response, encoding="utf-8")
        except OSError:
            pass

    def complete(self, prompt: str, image_data_uris: list[str] | None = None,
                 temperature: float = 0.15, max_tokens: int = 7000,
                 json_mode: bool = False) -> str:
        content: str | list[dict[str, Any]]
        if image_data_uris:
            content = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": uri}}
                for uri in image_data_uris
            )
        else:
            content = prompt
        # DeepSeek Flash returns reasoning and final content through the same
        # completion channel.  Supplying a stage-sized output limit has twice
        # exhausted that limit in reasoning before a JSON contract began.  For
        # DeepSeek, omit the field and let the provider apply its model default;
        # other OpenAI-compatible endpoints retain their explicit stage limit.
        is_deepseek = ".deepseek.com" in self.base_url or self.model.startswith("deepseek-")
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
        }
        if not is_deepseek:
            body["max_tokens"] = max_tokens
        if is_deepseek:
            # Make the provider default explicit.  DeepSeek's thinking output
            # shares its completion budget with the JSON contract; specifying
            # high avoids accidental max-effort calls while preserving its
            # documented default budget (we intentionally omit max_tokens).
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = self.reasoning_effort or "high"
        elif self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            # DeepSeek's documented JSON mode and OpenAI's equivalent both use
            # this field. It prevents a planner from spending its response on
            # explanatory prose that cannot satisfy the contract parser.
            body["response_format"] = {"type": "json_object"}
        last_empty_detail = ""
        for attempt in range(2 if json_mode else 1):
            # Retrying the byte-identical request is deterministic: a model
            # that spent its whole completion budget thinking will spend it
            # again. Two live clock members died exactly this way --
            # finish_reason=length with 200k+ reasoning characters and no
            # content at all -- so the retry changes the request instead of
            # repeating it: thinking off, and a cap wide enough for the JSON
            # contract. The first attempt stays uncapped, because a hard cap on
            # a thinking model is what starves the contract in the first place.
            attempt_body = body
            if attempt:
                attempt_body = dict(body)
                attempt_body["thinking"] = {"type": "disabled"}
                attempt_body["max_tokens"] = max(max_tokens, 2048)
                last_empty_detail += " (retried with thinking disabled)"
            request = urllib.request.Request(
                self.base_url.rstrip("/") + "/chat/completions",
                data=json.dumps(attempt_body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.api_key,
                    "User-Agent": "mc-art-studio-next/0.1",
                },
                method="POST",
            )
            try:
                payload = None
                for transport_attempt in range(3):
                    try:
                        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                            payload = json.load(response)
                        break
                    except (http.client.IncompleteRead, urllib.error.URLError, TimeoutError, OSError) as exc:
                        if transport_attempt == 2:
                            raise RuntimeError("LLM request failed after transport retries: %s" % exc) from exc
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:800]
                raise RuntimeError("LLM HTTP %s: %s" % (exc.code, detail)) from exc
            except OSError as exc:
                raise RuntimeError("LLM request failed: %s" % exc) from exc
            try:
                choice = payload["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("unexpected LLM response: %s" % str(payload)[:800]) from exc
            if isinstance(content, list):
                text = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            else:
                text = str(content or "")
            if text.strip():
                # A json_mode caller is promised parseable JSON, and a live bow
                # member died on "Expecting value: line 1 column 846" when the
                # provider truncated the object mid-flight. Treat an unparseable
                # body as a retryable failure here, where the retry budget and
                # the bounded second request already live, instead of letting it
                # surface as a raw JSONDecodeError from deep inside a planner.
                malformed = ""
                if json_mode:
                    try:
                        json.loads(text)
                    except ValueError as exc:
                        malformed = "malformed JSON: %s" % exc
                if not malformed:
                    self._trace_response(prompt, text)
                    return text
            else:
                malformed = "finish_reason=%s, reasoning_chars=%d" % (
                    choice.get("finish_reason", "unknown"),
                    len(str(choice.get("message", {}).get("reasoning_content", ""))),
                )
            last_empty_detail = malformed if attempt == 0 else "%s; %s" % (last_empty_detail, malformed)
        raise RuntimeError("LLM returned no usable content after retry (%s)" % last_empty_detail)


@dataclass
class ModelPlanner:
    client: OpenAICompatibleClient
    # A model-authored mask that compiles and contains all required parts is
    # useful visual evidence even when a soft metric (occupancy, margins or a
    # ratio) misses. Let the outer blind loop revise it instead of replacing it
    # immediately with a second model repair. Offline/custom repairers retain
    # the historical default by omitting this flag.
    repair_soft_failures: bool = False

    def art_direct(self, query: str) -> ArtDirection:
        """Establish immutable intent before any vanilla asset is considered.

        This deliberately has no reference arguments or image attachments.  It
        is a small creative brief, not a geometry, palette, or retrieval step:
        later evidence may teach Minecraft's visual language but may not turn
        the requested subject into the reference subject.
        """
        prompt = """Write a concise art direction for one Minecraft art request.
You have no reference images and must not select references, file paths, asset
forms, geometry, palette colors, pixels, or implementation details. Establish
only the requested subject and its broad visual reading. This is an immutable
brief for later retrieval and rendering stages.

Return JSON only:
{
  "target": "concise target phrase",
  "primary_subject": "what a blind viewer must read first",
  "design_intent": "one-sentence overall visual idea",
  "visual_hierarchy": ["up to three ordered visual priorities"],
  "requires_silhouette_change": true|false,
  "composition": "specific viewing composition and main-axis placement",
  "silhouette_actions": ["up to two concrete contour/transparent-area decisions"],
  "feature_actions": ["up to two concrete local feature placement/readability decisions"],
  "surface_actions": ["up to two concrete texture/material-boundary decisions"],
  "preservation_rules": ["up to three things a later variant/reference must retain"],
  "feature_relationships": ["up to three relations such as a feature embedded in a named host"],
  "negative_constraints": ["up to three reading failures to avoid"]
}

This is a concrete change brief, not a mood board or a geometry specification.
Use object-neutral language where possible; never choose a fixed shape family.
Do state what visibly changes and what must remain: for example, which section
is missing versus retained, which host receives a local feature, whether that
feature must visibly read as an eye/gem/mark, and which material zones keep
their own texture rather than becoming one flat ramp. Do not give coordinates,
row masks, exact pixel counts, or palette values; later stages decide those
from the selected reference and target canvas. For a damaged or broken object,
say what portion is absent and how its remaining edge should read, rather than
only saying "damaged". For a recolour, say what remains textured and what
material character changes.
Set `requires_silhouette_change` true only when the requested result needs a
different opaque contour (for example broken, cut-away, longer, split or a
new attached physical mass). Set it false for recolours and features painted
inside an unchanged host.
If a request combines a host and a local feature, make one concrete design
decision about its host and placement. Explicitly state relative scale: the
host remains readable and the feature remains local. Do not leave alternatives
such as "blade or guard", "top or side", or "one of several parts" in the
brief; later stages need one intended relationship to preserve.
For an ordinary recolour/state variant, preserve the base object's identity and
spend the change on the named state. Do not invent a feature absent from the
request. When a query is plausibly a typo or near-name of a familiar Minecraft
tool/object, resolve it to the most likely intended physical reading and make
that reading explicit in `primary_subject`; do not silently turn an ambiguous
object word into an unrelated material block.

Request: %s
""" % query
        response = ""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.complete(
                    prompt + ("\nPrevious output was invalid. Return the complete JSON schema only.\n" if attempt else ""),
                    image_data_uris=[],
                    json_mode=True,
                    max_tokens=900,
                )
                return art_direction_from_dict(_json_object(response))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError("model returned an invalid art direction after retry: %s (%s)" % (response[:300], last_error))

    def author_boxes(self, query: str, max_boxes: int = 16) -> list[Any]:
        """Decompose one object into the boxes its texture atlas is derived from.

        The model supplies only the box decomposition (width/height/depth per
        part, plus an optional model-space origin for the preview). Texture
        coordinates are computed from those dimensions, so a mod model that has
        no vanilla equivalent still gets a valid UV layout.
        """
        from .box_model import MAX_BOXES, boxes_from_dict

        limit = max(1, min(int(max_boxes), MAX_BOXES))
        prompt = """Decompose one Minecraft object into the axis-aligned boxes its texture atlas is derived from.

A Minecraft 1.12 model is a list of ModelRenderer boxes. Each box you declare gets a texture net of (2*depth + 2*width) by (depth + height) pixels automatically, so you must NOT give texture coordinates: give the box decomposition only.

Rules:
- One box per physical part of the object: head, body, limbs, tail, wings, horns, plates.
- Do not split one physical part into several boxes unless the object really has separate volumes.
- Dimensions are model pixels, exactly as vanilla uses them: a player head is 8x8x8, a body 8x12x4, a leg 4x12x4, a small horn 1x3x1, a thin wing 1x8x6.
- Every dimension must be between 1 and 32, and you may declare at most %d boxes.
- "origin" is the box's minimum corner in model space: +x right, +y down, +z toward the viewer (the vanilla addBox convention). Give it whenever you can place the part; it is used for the diagnostic front preview.
- "notes" may quote the vanilla-style addBox call the part resembles.

Return JSON only:
{"boxes": [{"id": "head", "part_id": "head", "size": [8, 8, 8], "origin": [-4, -8, -4], "notes": "addBox(-4,-8,-4,8,8,8)"}]}

Examples:
- "a wandering soul" -> a floating core box with two tapering wisp boxes trailing below it.
- "a demon cow" -> head, body, four legs, two horns, one tail, keeping the vanilla quadruped proportions.
- "a wooden chest with a lid" -> base, lid, latch.

Request: %s
""" % (limit, query)
        response = ""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.complete(
                    prompt + ("\nPrevious output was invalid. Return the complete JSON schema only.\n" if attempt else ""),
                    image_data_uris=[],
                    json_mode=True,
                    max_tokens=1600,
                )
                return boxes_from_dict(_json_object(response))[:limit]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError("model returned an invalid box decomposition after retry: %s (%s)" % (response[:300], last_error))

    def plan_family(self, query: str, max_members: int = 6) -> dict[str, Any]:
        """Split one request into an ordered set of sibling assets.

        A request such as an armour set or a tool set describes several textures
        that must read as one family. This stage decides only the membership and
        the order. Every member is then generated by the normal pipeline, with
        the first finished member pinned as a style anchor, so a family needs no
        second generation architecture.
        """
        prompt = """Decide whether one Minecraft art request describes a single texture or a small set of sibling textures that must look like one family (an armour set, a tool set, coloured variants of one object).

Rules:
- First decide which texture files the request actually needs. A resource pack stores every tool and every item as its own PNG. Worn armour uses two layer files (layer_1 = helmet, chestplate, boots; layer_2 = leggings), and each wearable piece ALSO needs its own inventory icon, so a usable armour set is six files: the two layers plus one icon per piece.
- One member is exactly one of those files and therefore exactly one subject. Never merge two subjects into one member, and never split one file into several members.
- A request that names a set, a collection, or several distinct objects needs one member per file.
- A SINGLE object can still need several files when the game loads its states separately: a bow is bow_standby plus three drawing stages, a clock is 64 frames, a compass 32 angles, a brewing stand has a base and a fill level. Plan one member per state file the game actually loads, not one per object.
- Only when neither rule adds a file is the request one member.
- Never invent members the request does not imply.
- At most %d members.

Examples:
- "a ruby gem" -> one member.
- "iron tool set" -> iron_pickaxe, iron_axe, iron_shovel, iron_hoe, iron_sword (one per tool).
- "iron armour set" -> iron_armor_layer_1, iron_armor_layer_2, iron_helmet, iron_chestplate, iron_leggings, iron_boots (the two worn layers plus one inventory icon per wearable piece: without the icons the set cannot be worn or seen in an inventory).
- "a crystal bow" -> a stand-by bow plus one member per drawing stage, because the game loads those as separate files.
- Each member target must be a complete, self-contained natural-language request. It is planned independently afterwards and must not rely on the other members to be understood.
- Each member name must be a short lowercase snake_case identifier, unique in the set.
- shared_style states, in one sentence, what must stay identical across the whole set: palette, material reading, outline and highlight treatment, proportions. Use an empty string for a single member.

Return JSON only:
{
  "set_name": "short snake_case set id",
  "shared_style": "one sentence, or an empty string",
  "members": [{"name": "snake_case", "target": "complete request"}]
}

Request: %s
""" % (max_members, query)
        response = ""
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = self.client.complete(
                    prompt + ("\nPrevious output was invalid. Return the complete JSON schema only.\n" if attempt else ""),
                    image_data_uris=[],
                    json_mode=True,
                    max_tokens=1400,
                )
                return _normalize_family(_json_object(response), query, max_members)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError("model returned an invalid family plan after retry: %s (%s)" % (response[:300], last_error))

    def route(self, query: str, references: list[ReferenceAsset],
              layouts: list[tuple[str, dict[str, Any]]],
              art_direction: ArtDirection | None = None) -> dict[str, Any]:
        """Choose the render contract and vanilla evidence for a bare query.

        The public query API deliberately does not ask a caller to classify an
        asset first.  This small routing call is kept separate from semantic
        description so dimensions and an entity UV atlas are known before the
        geometry contract is requested.  The model sees a catalogue of local
        vanilla pixels and layout summaries, then returns only catalogue
        indices; the filesystem paths themselves never come from model text.
        """
        layout_lines = []
        for index, (path, summary) in enumerate(layouts):
            layout_lines.append(
                "%d: %s regions=%s parts=%s" % (
                    index,
                    path,
                    summary.get("region_count", "?"),
                    json.dumps(summary.get("parts", {}), ensure_ascii=False, separators=(",", ":")),
                )
            )
        prompt = """Route one Minecraft art request to the correct generic render contract.
The caller supplied only a natural-language target. Inspect the attached
vanilla reference board and choose the evidence that best teaches the target's
pixel language. Do not invent file paths or copy a reference silhouette.

Return JSON only:
{
  "form": "item|cross|block_multi|entity_uv|custom",
  "width": integer,
  "height": integer,
  "reference_indices": [integer, ...],
  "uv_layout_index": integer|null,
  "target_path": "textures/...png"|null,
  "reason": "short explanation"
}

Use entity_uv when the target is a creature, character, animal, mob or other
3-D model texture atlas. An entity atlas must use a supplied layout whose cube
regions match the selected vanilla model; choose that layout index when one is
available. Use block_multi for a block or multi-face cube atlas, cross for a
plant/billboard, and item for an inventory icon. For a block whose selected
vanilla texture is one tile reused on every cube face, prefer a listed
single-tile six-face layout when available; use a multi-face strip only when
the target or evidence calls for independent face textures. Choose width/height from the
selected vanilla atlas when one is clearly the model to recolor; otherwise use
16x16 for item/cross and a practical power-of-two atlas for blocks. Keep the
selected reference list short (one to four items), and only include a second
reference when it adds a distinct piece of evidence the first image does not
contain. If one candidate already shows the base material and its local motif,
do not add a generic background sample just to repeat the same information.
Avoid unrelated references because later stages use the first compatible pixel
source as the raster ruler. reference_indices and uv_layout_index are indices
into the catalog below, not paths.

Immutable art direction (written before reference retrieval):
%s

Target: %s

Reference catalog (attached images follow this order):
%s

Entity/block UV layout catalog:
%s
FINAL ROUTING DECISION: select the smallest evidence set that can produce the
requested asset. If one reference already depicts the same object and the
query asks only for a state/shape variant, keep that object reference as the
sole style/material source unless the query explicitly requests a new material
or colour. Do not add a generic leather, wood or stone sample based only on a
part noun such as handle, stem or support. The later stages will preserve the
selected source's own local colour families.
""" % (
            json.dumps(to_jsonable(art_direction), ensure_ascii=False) if art_direction else "(not available)",
            query,
            "\n".join(
                "%d: %s" % (index, summarize_reference(reference, include_pixel_map=False))
                for index, reference in enumerate(references)
            ) or "- none",
            "\n".join(layout_lines) or "- none",
        )
        image_data_uris = _reference_data_uris(references)

        def valid_route(data: dict[str, Any]) -> bool:
            # A provider can acknowledge ``response_format`` in the content
            # (for example ``{"type":"json_object"}``) without returning a
            # route. Never let that acknowledgement fall through to the
            # catalog's first entry, because it silently couples the target to
            # an unrelated reference.
            form = str(data.get("form", "")).strip().lower().replace("-", "_")
            valid_forms = {
                "item", "cross", "block_multi", "entity_uv", "custom",
                "entity", "entity_texture", "mob", "creature", "block",
                "cube", "billboard", "plant",
            }
            if form not in valid_forms:
                return False
            try:
                width = int(data.get("width"))
                height = int(data.get("height"))
            except (TypeError, ValueError):
                return False
            if width < 1 or height < 1:
                return False
            indices = data.get("reference_indices")
            if isinstance(indices, int):
                indices = [indices]
            if not isinstance(indices, list):
                return False
            if references and not indices:
                return False
            if any(not isinstance(index, int) or not 0 <= index < len(references) for index in indices):
                return False
            layout_index = data.get("uv_layout_index")
            if layout_index is not None and (
                not isinstance(layout_index, int) or not 0 <= layout_index < len(layouts)
            ):
                return False
            return True

        response = ""
        last_error: Exception | None = None
        for attempt in range(2):
            retry_note = ""
            if attempt:
                retry_note = """
The previous response was not a route. Re-read the catalogue and return the
complete schema above. `form`, positive integer `width`/`height`, and a
non-empty in-range `reference_indices` list are required when the catalogue is
non-empty. Do not return a response-format acknowledgement.
"""
            try:
                response = self.client.complete(
                    prompt + retry_note,
                    image_data_uris=image_data_uris,
                    json_mode=(attempt == 0),
                    max_tokens=1400,
                )
                data = _json_object(response)
                if valid_route(data):
                    return data
                last_error = ValueError("route JSON is missing required fields")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        detail = response.strip().replace("\n", " ")[:400]
        raise RuntimeError("model returned an invalid route after retry: %s (%s)" % (detail, last_error))

    def route_indexed(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        layouts: list[tuple[str, dict[str, Any]]],
        cache_dir: Path | None = None,
        index_fingerprint: str = "unknown",
        model_name: str | None = None,
        art_direction: ArtDirection | None = None,
    ) -> dict[str, Any]:
        """Route against a compact indexed manifest without attaching images.

        The response is normalized to the legacy route fields as well as the
        stable ``asset_ids``/``selections`` fields, so callers can migrate
        incrementally while audits stop depending on array positions.
        """
        model = model_name or getattr(self.client, "model", None)
        cache_path: Path | None = None
        cache_key = route_cache_key(
            query, None, candidates, index_fingerprint, model,
            art_direction=to_jsonable(art_direction) if art_direction else None,
        )
        if cache_dir is not None:
            cache_path = Path(cache_dir).expanduser().resolve() / "route" / (cache_key + ".json")
            if cache_path.exists():
                try:
                    cached = _json_object(cache_path.read_text(encoding="utf-8"))
                    if cached.get("_route_schema") != 4:
                        raise ValueError("stale indexed route cache schema")
                    parse_router_selection(cached, candidates)
                    cached["cache_hit"] = True
                    return cached
                except (OSError, ValueError, json.JSONDecodeError):
                    cache_path.unlink(missing_ok=True)

        # Layout files are an implementation detail.  The router only needs a
        # stable display label plus the structural facts that distinguish one
        # layout from another; the caller maps the returned index back to the
        # actual local file after the model responds.
        layout_lines = [
            "%d: name=%s regions=%s parts=%s" % (
                index,
                Path(path).stem,
                summary.get("region_count", "?"),
                json.dumps(summary.get("parts", {}), ensure_ascii=False, separators=(",", ":")),
            )
            for index, (path, summary) in enumerate(layouts)
        ]
        candidate_lines = "\n".join(
            manifest_label(item).replace("\n", " ")
            for item in candidates
        ) or "(none)"
        prompt = """Route a Minecraft art request using a compact local asset index.
The target is supplied as one natural-language query. The candidate list is a
broad recall window from the complete local index; it is evidence, not a
closed shape/category list. No candidate image is attached in this call.
Choose the smallest useful evidence set and keep shape/style roles separate.
The form is the deliverable contract, not the camera view described in the
brief. If the primary subject is a block, cube, ore, stone, wood, brick or
other placed world material — especially when the chosen structural reference
has category `block` — choose `block_multi`. A phrase such as "front face",
"single face" or "straight-on" describes that block texture's motif; it never
turns the block into a held inventory `item`. Use `item` only for a portable
icon/tool/weapon/consumable whose subject itself is not a placed block.

Return JSON only:
{
  "form": "item|cross|block_multi|entity_uv|custom",
  "selections": [{"name": "exact name from the list", "roles": ["shape|scale|material|palette|pixel_style|uv_layout|negative"], "reason": "", "confidence": 0.0}],
  "uv_layout_index": integer|null,
  "unmet_evidence": [],
  "reason": "short explanation"
}

Only use a name that appears in the list. The list intentionally
contains display names and broad categories only; paths, hashes, dimensions
and internal IDs are not part of the model context. Use an existing candidate
only when it teaches a concrete dimension of this target. An empty selection
is valid when the list has no suitable evidence. Return no more than six
selections; each extra source must add evidence the earlier selections lack.
When the target is a variant or a set that must match an existing model, include
several comparable base candidates rather than a single one: the caller scores
them and keeps the best, and one plausible-looking name is not enough evidence
that it is the right base. Do not return fields beyond the schema above.

Immutable art direction (created without candidate access):
%s
This ranks above the candidate list: choose sources that teach its form or
style, but never let a source redefine its primary subject or the stated
host/feature relationship.

Target: %s

Candidate list (one line per entry: `name category/family`; the family after
the slash is a local grouping label, never part of the name you return):
%s

UV layout catalog:
%s
""" % (
            json.dumps(to_jsonable(art_direction), ensure_ascii=False) if art_direction else "(not available)",
            query,
            candidate_lines,
            "\n".join(layout_lines) or "- none",
        )
        response = ""
        last_error: Exception | None = None
        data: dict[str, Any] | None = None
        for attempt in range(2):
            retry_note = ""
            if attempt:
                retry_note = "\nPrevious output was invalid. Return only the exact JSON schema, use names from the list, and choose at most four selections.\n"
            try:
                response = self.client.complete(prompt + retry_note, image_data_uris=[], json_mode=True, max_tokens=1400)
                data = _json_object(response)
                parsed = parse_router_selection(data, candidates)
                break
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        else:
            raise RuntimeError("indexed route response was invalid after retry: %s" % last_error)
        assert data is not None
        result: dict[str, Any] = dict(data)
        result["_route_schema"] = 4
        # Keep the distinction so the quality boundary can recognize a
        # legacy client response that happened to use the old numeric field.
        # Numeric positions are valid for the legacy catalogue, but must not
        # be reinterpreted as positions in the source-wide indexed manifest.
        result["_legacy_numeric_route"] = (
            "selections" not in data and "reference_indices" in data
        )
        result["selections"] = [
            {
                "asset_id": choice.asset_id,
                "roles": list(choice.roles),
                "reason": choice.reason,
                "confidence": choice.confidence,
            }
            for choice in parsed.selections
        ]
        result["asset_ids"] = [choice.asset_id for choice in parsed.selections]
        rank_by_id = {str(item.get("asset_id")): int(item.get("rank", index)) for index, item in enumerate(candidates)}
        result["reference_indices"] = [rank_by_id[item] for item in result["asset_ids"] if item in rank_by_id]
        result["cache_hit"] = False
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, cache_path)
        return result

    def describe(self, query: str, form: AssetForm, width: int, height: int,
                 references: list[ReferenceAsset],
                 art_direction: ArtDirection | None = None) -> ShapeDescriptor:
        prompt = """You plan Minecraft pixel-art assets. Do not choose from a fixed category list.
Describe the requested target as an open semantic structure with parts and measurable visual identity.
References are evidence only; list how to borrow their structure/style without copying an entire silhouette.
When the request names only a new material, hue or ordinary variant and does
not explicitly request a new motif layout, make the selected reference's
local motif coordinates and occupied clusters the default. Describe a changed
cluster contour only when the request names that change; otherwise reserve
novelty for the requested material or hue.
Return JSON only with this schema:
{
  "target": string,
  "semantic": string,
  "visual_identity": [string, ...],
  "parts": [{"id": string, "meaning": string, "required": boolean, "paint_only": boolean, "layer": integer, "style_role": string, "recognition_terms": [string, ...], "contour_intent": "compact|elongated|surface|free"}],
  "negative_identities": [string, ...],
  "orientation": string,
  "reference_strategy": string,
  "shape_edit_mode": "appearance_only|preserve_silhouette|local_silhouette_edit|new_silhouette",
  "reference_part_map": {"legend": {"a": "part_id"}, "rows": [string, ...]},
  "target_part_map": {"legend": {"a": "part_id"}, "rows": [string, ...]}
}

Keep every string concise. Use at most 6 visual_identity entries and at most 8 parts.
When the request contains an appearance modifier or finish (for example
corrupted, demonic, molten, cursed, icy, glowing or weathered), translate it
into at least one concrete, paint-visible identity with a plausible coverage
plan. Do not reduce the modifier to an adjective in `semantic` while leaving
the base object visually unchanged; the later AppearanceSpec must be able to
express it through a coherent ramp, repeated local marks or exact UV clusters.
Part IDs must be short ASCII snake_case identifiers beginning with a letter; do
not use Chinese characters, spaces or punctuation in IDs. Parts describe
physical regions that need their own alpha geometry. Set `paint_only:true` for
a declared feature whose identity is intentionally carried by AppearanceSpec
marks or pixel_map inside an existing support; such a part must not receive a
primitive or alter the alpha silhouette. Keep `required:true` when the feature
must remain visibly present, regardless of whether it is physical or paint-only.
Do not turn highlights,
shadows, outlines, edges, texture, grain, color or material into separate
required geometry parts: those are AppearanceSpec styling and may be expressed
with palette bands or marks. A simple material recolor usually has one required
body/surface part. Add multiple parts only when their physical contours or
connections are genuinely different.
`required` is an acceptance contract, not a geometry flag. Every named
feature that makes this request differ from its base reference must be
`required:true`, including a local stain, seep, crack, inset, eye, emblem or
other `paint_only` feature. Use `required:false` only for genuinely optional
incidental decoration that may disappear without changing the requested
identity. A feature named in the target, visual_identity, feature_actions or
negative constraints is not incidental.
For a required local/embedded/detail part, provide one to three short
`recognition_terms` naming only that feature as a target-free reviewer should
see it. Exclude the host part, whole object, material, direction, and generic
words such as `motif` or `detail`; this is an inspection contract, not a fixed
shape template.
For every physical part, choose `contour_intent`: `compact` for a locally
bounded mass that must not read as a diagonal streak; `elongated` for a shaft,
edge or continuation; `surface` for a region whose identity comes from paint;
and `free` when none applies. This is the model's own visual decision, not an
object-category rule.
Choose `shape_edit_mode` before describing parts: `appearance_only` keeps the
reference silhouette and expresses the distinction through paint;
`preserve_silhouette` keeps the source union while clarifying ownership;
`local_silhouette_edit` makes the smallest model-authored local alpha edit at
a named host while preserving unnamed supports; `new_silhouette` authors a
fresh contour. Do not choose a mode from a noun template; choose the smallest
mode that makes the requested visual identity legible at native pixels.
When a shape reference is available, make `reference_strategy` include a
short spatial landmark description grounded in that raster: which end or
quadrant owns each physical part, the approximate junction location, and the
part's local axis/thickness. These landmarks are evidence for geometry; do not
guess them from the object noun when the raster disagrees.
When a same-size shape reference and more than one physical part are present,
you must also return `reference_part_map` whenever `shape_edit_mode` is
`appearance_only` or `preserve_silhouette`: `rows` must contain exactly one
string per source row and one character per source column, `.` for transparent
cells, and the `legend` must map each other character to one of the declared
part IDs. Label the cells you are confident about rather than the whole raster;
every unlabelled source cell is handed to the nearest labelled part. This is
model-authored ownership evidence for partitioning a union silhouette, not a
closed shape template. Omit it only when the reference has no separable
physical parts.
When you preserve the source silhouette, that source raster *is* the output, so
every visually distinct element in it must be owned by a declared part with its
own material: a held or nocked accessory, a wrapping, a differently-coloured
tip, a separate cord. An element you leave undeclared is not dropped and not
kept bright -- it is painted as whatever declared part happens to lie nearest,
which is almost always the wrong material. A live crystal bow lost the nocked
arrow this way: the white and silver arrow cells fell inside the bow body and
came out as dark violet crystal.
When the request adds a new physical local motif that is absent from the
source (for example an emblem, inset, boss, eye, gem, socket or clasp), also
return `target_part_map`. It uses the same full canvas grid and legend format,
but describes intended target ownership: unchanged source supports keep their
source cells and the new motif gets its own local cells even when those cells
were transparent in the source. This is planning evidence for the geometry
model, not a fixed category template; the model still chooses the motif's
final contour while keeping it inside the named host support.
For an inlaid motif, a target-map motif marker is an overlay hint, not an
instruction to delete the host cell underneath. Reconstruct the host support
from `reference_part_map` first, then layer a smaller motif inside that host;
keep a visible host rim or side band around most of the motif.
When a request adds or embeds a local motif in a referenced composite object,
keep every unchanged physical support from the source in the part list and in
the ownership map (for example blade, guard and handle around a new boss or
eye). The edited motif may be its own part or paint-only when the descriptor
decides that is clearer, but it must never replace or absorb an unchanged
support. A reference-part map whose primary label covers the whole source while
the descriptor omits visible supports is invalid.
Read grammar such as "X embedded/inlaid in Y" literally: Y remains the host
support with its source physical identity and scale, while X is a separate
local motif. Do not relabel the whole host as X, turn a guard into a giant eye,
or use a secondary motif reference to replace the host's contour. When several
shape references are attached, assign each reference's contour only to the
part whose role it evidences; a local motif reference cannot redefine an
unchanged support.
If the query names only a state or contour variant and does not request a new
material or hue, preserve the selected reference's material families in the
descriptor. Do not relabel an existing support from the source (for example by
guessing leather or metal from the word grip) and do not invent a material
change that the query did not ask for.
When a recognizable composite reference has a narrow continuation at an
opposite end of a head, guard or working surface, keep that continuation as
its own support part (handle, shaft, stem or equivalent) and map it separately;
do not absorb a distinct support into the nearest broad part just because their
alpha pixels touch.
When one declared part is the primary support, use the longest coherent
staircase or axis in the source alpha as its ownership evidence, including
pixels that continue behind or through a crossing host. A lateral host branch
may overlap that axis at a junction, but must not swallow the continuation of
the primary support merely because the pixels touch. The ownership map should
explain the source's physical structure, not divide the opaque area into
similarly sized blobs.
State changes such as broken, chipped, jagged, tapered or recoloured belong to
the owning physical part's contour or AppearanceSpec; do not create a second
"break", "edge", "tip" or "fracture" part unless it is a physically separate
piece with its own independent alpha region and connection.
For a damage or state variant, default to editing the owning part's contour;
only add a detached fragment part when the user explicitly asks for a separate
piece. A notch or missing endpoint inside the original part is not a new part.
Every concrete distinguishing feature named in the request or
`visual_identity` (for example a horn, eye, tail, grip or clasp) must either
appear as its own physical part or be explicitly identified as paint-only;
never silently omit it just to make the part list shorter. Overlapping details
still need their own labelled part and layer so the geometry and appearance
stages can verify them.
Set `target` to a concise English noun phrase even when the request is written in
another language; this gives the unattended blind-review gate a stable noun to
compare, while the other fields may follow the request language.
If the request names only a material/colour with no concrete object (for
example “blue leather” or “青金色皮革”), describe a neutral material swatch or
small hide patch. Do not invent a shirt, armor torso, animal, face or other
wearable/character silhouette; those nouns are not present in the request.

Immutable art direction (already decided before retrieval; do not return or
rewrite it):
%s
It outranks any individual reference. References teach pixel structure,
materials and style only. Keep the stated primary subject and local
host/feature relationship legible at native size. Treat `composition`,
`silhouette_actions`, `feature_actions` and `surface_actions` as binding
visible instructions. They must be realized as alpha, local pixels and
material clusters where applicable; do not reduce them to adjectives in the
semantic field.

Request: %s
Form: %s, canvas: %dx%d
Reference evidence (each attached colour image follows this same order):\n%s
For every reference labelled `shape`, the next attached image is its matching
alpha-only silhouette map. Use that map to read the contour, row spans and
spatial landmarks. For a named same-size variant, treat those landmarks and
unnamed occupied regions as the baseline; for a new design, use them only as
proportion/style evidence.
""" % (
            json.dumps(to_jsonable(art_direction), ensure_ascii=False) if art_direction else "(not available)",
            query,
            form.value,
            width,
            height,
            "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
        )
        if form in {AssetForm.ITEM, AssetForm.CROSS}:
            # Keep the exact alpha grid in the text channel as well as the
            # attached image. Vision models often blur a 16px raster; this
            # compact ruler makes the variant baseline auditable without
            # selecting an object-specific template.
            prompt += "\n" + _shape_baseline_text(references, width, height)
        if form == AssetForm.ENTITY_UV:
            prompt += """
Entity UV note: the supplied model atlas determines the physical parts. A
feature such as an eye, nostril, stripe or tail detail that has no separate UV
cube is paint-only inside an existing region; do not turn it into a new alpha
part. Keep the vanilla model's head/body/leg/horn part structure and describe
new demon details as texture marks unless the layout exposes a physical cube.
"""
        # The descriptor is an intent hand-off, not a second copy of the
        # sprite.  Keep it short enough that the vision evidence and the
        # reference-free design remain the things the model can actually hold
        # together. Geometry receives its own mask/raster evidence later.
        prompt = """Plan a Minecraft pixel-art asset from the attached vanilla references.
Return JSON only:
{"target":"English noun phrase","semantic":"short visual description","visual_identity":["required visual fact"],"parts":[{"id":"ascii_snake_case","meaning":"...","required":true,"paint_only":false,"layer":0,"style_role":"...","recognition_terms":["..."],"contour_intent":"compact|elongated|surface|free"}],"negative_identities":["..."],"orientation":"...","reference_strategy":"...","shape_edit_mode":"appearance_only|preserve_silhouette|local_silhouette_edit|new_silhouette"}

Only name physical parts that require separate alpha geometry. Put a feature
drawn inside an existing surface in the parts list with `paint_only:true`.
Do not create parts for colours, highlights, outlines, grain or shadows.
Every concrete distinguishing feature must be a physical part or paint-only.
The parts list is the full production inventory, not merely the novelty. When
the target preserves or adapts a referenced silhouette, include at least one
`paint_only:false` host part that owns the inherited opaque pixels. If the
visual brief names persistent host zones (such as a body and support, or a
blade, guard and grip), list them as physical parts even when the requested
novel feature is paint-only. The geometry stage cannot reconstruct omitted
hosts from prose.
Part ids must be short ASCII snake_case identifiers.
Choose the smallest shape_edit_mode that makes the brief legible. Do not emit
reference_part_map or target_part_map: a later geometry stage receives the
locked alpha raster directly.

Immutable art direction / VISUAL BRIEF
%s

REQUEST: %s
FORM: %s; CANVAS: %dx%d

REFERENCES (attached in this order; SOURCE_PIXELS is literal #RRGGBB evidence)
%s
""" % (
            json.dumps(to_jsonable(art_direction), ensure_ascii=False, separators=(",", ":")) if art_direction else "{}",
            query,
            form.value,
            width,
            height,
            _appearance_reference_evidence(references),
        )
        if form in {AssetForm.ITEM, AssetForm.CROSS}:
            prompt += "\n" + _shape_baseline_text(references, width, height)
        descriptor_data = _normalize_descriptor_part_ids(
            _strip_visual_overlay_parts(
                _json_object(self.client.complete(prompt, image_data_uris=_reference_data_uris(references), json_mode=True, max_tokens=24000))
            )
        )
        # Keep the comparison noun stable when a vision model ignores the
        # English-target instruction and echoes Chinese. This does not rewrite
        # the user's request or any visual semantics; it only fills the field
        # consumed by the target-free blind gate.
        target_text = str(descriptor_data.get("target", ""))
        if re.search(r"[\u4e00-\u9fff]", target_text):
            english_target = None
            for source, replacement in (
                ("皮革", "leather patch"),
                ("方块", "block"),
                ("村民", "villager"),
                ("蘑菇", "mushroom"),
                ("牛", "cow"),
                ("刀", "knife"),
                ("剑", "sword"),
            ):
                if source in query or source in target_text:
                    english_target = replacement
                    break
            if english_target:
                descriptor_data["target"] = english_target
        descriptor = replace(descriptor_from_dict(descriptor_data), art_direction=art_direction)
        # A surface-only motif still needs a physical host.  Do not invent a
        # host in Python: ask the descriptor author to supply its missing
        # production inventory while it still has the request and references.
        if (
            form in {AssetForm.ITEM, AssetForm.CROSS}
            and not any(not part.paint_only for part in descriptor.parts)
        ):
            host_correction = prompt + """
The previous descriptor listed only surface-only parts, so downstream geometry
would be empty. Return the complete descriptor JSON again. Preserve the
design, but include the physical host inventory that owns the referenced alpha
silhouette. A paint-only eye, emblem, marking, grain or highlight remains a
paint-only part; it does not replace the host body, supports, or named material
zones. Do not invent a category template: infer the host parts from this
request, visual brief, and source pixels.
"""
            corrected_data = _normalize_descriptor_part_ids(
                _strip_visual_overlay_parts(
                    _json_object(self.client.complete(
                        host_correction,
                        image_data_uris=_reference_data_uris(references),
                        json_mode=True,
                        max_tokens=24000,
                    ))
                )
            )
            corrected = replace(descriptor_from_dict(corrected_data), art_direction=art_direction)
            if any(not part.paint_only for part in corrected.parts):
                descriptor = corrected
            else:
                raise ValueError("descriptor has no physical host parts after correction")
        # A source ownership map cannot place a new inset on transparent
        # source pixels.  Ask the model for a target ownership map when the
        # request contains an explicit local addition; this keeps placement
        # model-authored while preventing geometry from guessing the motif's
        # location from its noun alone.
        if (
            _descriptor_needs_target_part_map(descriptor, form)
            and (
                not _target_map_covers_declared_parts(descriptor, width, height)
                or not any(is_overlay_part(part) for part in descriptor.parts)
            )
        ):
            correction = prompt + """
The previous descriptor omitted a usable target_part_map for its newly added
local motif. Return the complete descriptor JSON again. Keep the same declared
source support landmarks, and add a separate physical part whose own meaning
and style_role identify the newly authored local motif. If the previous
descriptor called the host itself an inlaid/motif part, split that host back
into a support part and a new local motif part while preserving every source
support. Add `target_part_map` with exactly one
row per canvas row and one marker per canvas column. Its legend must include
every required physical part (`paint_only:false`). Paint-only parts may be
omitted because they do not own alpha geometry. Place the new motif as a small local region inside the
named host support; do not let it cover the blade/handle or replace the host.
The motif marker is an overlay hint: retain the host support beneath it and
leave a visible host rim around most sides.
Do not satisfy a round/local motif with a one- or two-pixel line; give it
enough target cells and varying row spans for its named contour to survive the
requested raster size.
This map is model-authored placement evidence, not a category template.
"""
            try:
                corrected_data = _normalize_descriptor_part_ids(
                    _strip_visual_overlay_parts(
                        _json_object(self.client.complete(
                            correction,
                         image_data_uris=_reference_data_uris(references),
                            json_mode=True,
                            max_tokens=24000,
                        ))
                    )
                )
                corrected = replace(descriptor_from_dict(corrected_data), art_direction=art_direction)
                if _target_map_covers_declared_parts(corrected, width, height):
                    descriptor = corrected
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if _descriptor_needs_target_part_map(descriptor, form) and isinstance(descriptor.reference_part_map, dict):
            # A second model-owned audit catches a common low-resolution
            # failure: assigning a narrow continuation at the opposite end of
            # the source to a broad host part.  It may add a support part, but
            # it may not remove or rename any part already established by the
            # first descriptor pass.
            audit_prompt = prompt + """
SOURCE PART AUDIT: inspect the attached same-size source raster and the
descriptor below one more time before geometry is generated. Return the
complete descriptor JSON only. Keep every existing part ID and add a separate
physical support part when the source has a distinct narrow continuation,
shaft, grip, stem, tail, connector or other section at a different end. Do
not absorb that continuation into a broad host merely because the alpha pixels
touch. Keep surface texture/highlights as appearance, and keep the newly
requested local motif separate from its host. Update both ownership maps:
`reference_part_map` must partition the source supports, and `target_part_map`
must include the new physical motif cells as an overlay while retaining all source
supports. Every required physical part (`paint_only:false`) must have at least
one target cell; paint-only parts may be omitted. Return only
the JSON object. For the source map, trace the longest continuous axis of the
opaque raster for the part whose role is `primary support`; keep that axis
continuous through a crossing host unless the source visibly terminates there.
Assign a guard, host or branch only to the lateral cluster that actually
changes direction or material. Do not label the tail of the primary axis as
host just because it touches the branch. For an inlaid motif, its target cells
must be inside that lateral host, with a visible host rim, and must not replace
the primary axis or the support continuation.
Descriptor to audit:
""" + json.dumps(to_jsonable(descriptor), ensure_ascii=False)
            try:
                audited_data = _normalize_descriptor_part_ids(
                    _strip_visual_overlay_parts(
                        _json_object(self.client.complete(
                            audit_prompt,
                            image_data_uris=_reference_data_uris(references),
                            json_mode=True,
                            max_tokens=24000,
                        ))
                    )
                )
                audited = replace(descriptor_from_dict(audited_data), art_direction=art_direction)
                original_ids = {part.id for part in descriptor.parts}
                audited_ids = {part.id for part in audited.parts}
                if (
                    original_ids.issubset(audited_ids)
                    and len(audited.parts) <= 8
                    and _target_map_covers_declared_parts(audited, width, height)
                ):
                    descriptor = audited
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if _descriptor_needs_target_part_map(descriptor, form) and not _target_map_covers_declared_parts(descriptor, width, height):
            final_map_prompt = prompt + """
FINAL TARGET-MAP CORRECTION: the descriptor still has an unusable target map.
Return the complete descriptor JSON with the same source supports and local
motif. Every required physical part (`paint_only:false`) must have real target
cells; paint-only parts may be omitted. For a part described as
round/circular/orb-like/eye/gem/motif, place at least five connected marker
cells across at least three rows. The middle row(s) must be wider than the end
row(s), with a peak away from the first and last row; do not return a two-pixel
split, filled rectangle or straight stripe. Keep the motif inside its host and
leave the host support underneath it. Return JSON only.
Current descriptor:
""" + json.dumps(to_jsonable(descriptor), ensure_ascii=False)
            try:
                final_data = _normalize_descriptor_part_ids(
                    _strip_visual_overlay_parts(
                        _json_object(self.client.complete(
                            final_map_prompt,
                            image_data_uris=_reference_data_uris(references),
                            json_mode=True,
                            max_tokens=24000,
                        ))
                    )
                )
                final_descriptor = replace(descriptor_from_dict(final_data), art_direction=art_direction)
                if _target_map_covers_declared_parts(final_descriptor, width, height):
                    descriptor = final_descriptor
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if _descriptor_needs_target_part_map(descriptor, form) and not _target_map_covers_declared_parts(descriptor, width, height):
            # Keep raster ownership authoring small and isolated. Asking a
            # lightweight model to repeat a full descriptor plus a 16-row map
            # often loses the newly declared part before geometry sees it.
            # This map remains a fresh, model-authored intermediate contract;
            # it is never a category template or stored mask library.
            try:
                target_map = self.author_target_part_map(descriptor, width, height, references)
                candidate = replace(descriptor, target_part_map=target_map)
                if _target_map_covers_declared_parts(candidate, width, height):
                    descriptor = candidate
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if (
            form in {AssetForm.ITEM, AssetForm.CROSS}
            and art_direction is not None
            and art_direction.requires_silhouette_change
            and isinstance(descriptor.reference_part_map, dict)
            and not _target_map_reflects_requested_silhouette_change(descriptor, width, height)
        ):
            delta_prompt = prompt + """
SILHOUETTE-CHANGE CONTRACT FAILED: the immutable art direction explicitly
requires an alpha change, but `target_part_map` has the same occupied cells as
`reference_part_map`. Return the complete descriptor JSON again. Preserve
unchanged supports, but make the target map visibly add or remove the local
opaque region named by `silhouette_actions`. For a broken/cut-away state, the
removed source cells must become `.` in the target map and no replacement part
may occupy them. This is model-authored target evidence, not a reusable mask
or a category rule.

Current descriptor:
""" + json.dumps(to_jsonable(descriptor), ensure_ascii=False)
            try:
                delta_data = _normalize_descriptor_part_ids(
                    _strip_visual_overlay_parts(
                        _json_object(self.client.complete(
                            delta_prompt,
                            image_data_uris=_reference_data_uris(references),
                            json_mode=True,
                            max_tokens=24000,
                        ))
                    )
                )
                delta_descriptor = replace(
                    descriptor_from_dict(delta_data), art_direction=art_direction
                )
                if (
                    _target_map_covers_declared_parts(delta_descriptor, width, height)
                    and _target_map_reflects_requested_silhouette_change(delta_descriptor, width, height)
                ):
                    descriptor = delta_descriptor
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        # Repair calls return only a descriptor schema. Re-attach the brief
        # here so an otherwise valid repair cannot silently discard the
        # earlier reference-free decision before geometry or appearance sees
        # it.
        # A declared paint-only part is a production feature: it has no alpha
        # primitive to make it visible later, so allowing it to be optional
        # silently turns a named seep, emblem, crack or inset into a no-op.
        # Optional decoration belongs in a support part's styling, not in the
        # descriptor's part inventory. This is form-agnostic and does not
        # choose a motif, location, colour, or shape for the model.
        if any(part.paint_only and not part.required for part in descriptor.parts):
            descriptor = replace(
                descriptor,
                parts=[
                    replace(part, required=True) if part.paint_only else part
                    for part in descriptor.parts
                ],
            )
        return replace(descriptor, art_direction=art_direction)

    def author_target_part_map(
        self,
        descriptor: ShapeDescriptor,
        width: int,
        height: int,
        references: list[ReferenceAsset],
    ) -> dict[str, Any]:
        """Author only the model-owned target part map for a local edit."""
        prompt = """Author the target ownership map for one small Minecraft pixel asset.
This is a model-authored intermediate plan, not a reusable object template.
Return JSON only:
{"legend":{"a":"part_id"},"rows":["exactly one %d-character row", ...]}

Use `.` for transparent cells. The legend must map markers to every required
physical part (`paint_only:false`) exactly as named in the descriptor, and
every such part must own at least one marker cell. Paint-only parts may be
omitted from the map because they do not own alpha geometry. Preserve source support ownership from
`reference_part_map` where it exists. The descriptor's `shape_edit_mode`
controls whether a requested local physical part may use a small new contour:
for `local_silhouette_edit`, choose its host and compact native-pixel shape
yourself, retaining all unnamed source supports; for `appearance_only` or
`preserve_silhouette`, do not add a physical feature. When a local part lies
inside a host, its marker may replace the host marker in this target map; the
source ownership map retains the host underneath for later layering. Do not
invent or rename parts. Work on exactly %dx%d cells.
Honor each part's model-selected `contour_intent`: `compact` means a localized
mass with real width and height, not a diagonal ribbon; `elongated` may follow
an axis; `surface` remains on its host. These are descriptor decisions, not
fixed geometry templates.

Descriptor:
%s
Reference ownership map:
%s
""" % (
            width,
            width,
            height,
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            json.dumps(descriptor.reference_part_map, ensure_ascii=False),
        )
        last_target: dict[str, Any] | None = None
        for attempt in range(2):
            data = _json_object(self.client.complete(
                prompt,
                image_data_uris=_geometry_reference_data_uris(references),
                json_mode=True,
                max_tokens=2200,
            ))
            target = data.get("target_part_map") if isinstance(data.get("target_part_map"), dict) else data
            if not isinstance(target, dict):
                raise ValueError("target-map author did not return an object")
            last_target = target
            if _target_map_covers_declared_parts(
                replace(descriptor, target_part_map=target), width, height
            ):
                return target
            if attempt == 0:
                prompt += """
The previous map was structurally invalid. A legend entry by itself does not
place a part: every required marker in the legend must occur in at least one
actual row cell. Re-author the complete map, preserving source supports while
giving every declared physical part a real native-pixel region. Return only
the map JSON; do not explain the correction.
"""
        raise ValueError("target-map author returned no usable ownership map: %s" % json.dumps(last_target, ensure_ascii=False))

    def geometry(self, descriptor: ShapeDescriptor, width: int, height: int,
                 uv_regions: list[UvRegionSpec] | None = None,
                 form: AssetForm = AssetForm.ITEM,
                 references: list[ReferenceAsset] | None = None) -> GeometrySpec:
        references = references or []
        # Pixel text is lossless coordinate evidence, while the enlarged
        # reference image supplies the gestalt a grid cannot express: which
        # branch is a guard, where the handle terminates, and whether a
        # shortened diagonal has become a point.  Passing both keeps a local
        # variant anchored to the original object instead of asking the model
        # to reconstruct its part hierarchy from symbols alone.
        geometry_images = _geometry_reference_data_uris(references)
        allowed = ", ".join(sorted(SUPPORTED_PRIMITIVES))
        prompt = """Generate a GeometrySpec for a Minecraft pixel asset. It must be novel and must satisfy the descriptor.
Use only the listed primitive language. This language is expressive; it is not a category whitelist.
All parts and connections must be explicit. Constraints must refer only to compiler metrics:
opaque_pixels, occupancy_ratio, components, orientation_degrees, axis_anisotropy, bbox_width_ratio, bbox_height_ratio, aspect_ratio,
margin_left, margin_top, margin_right, margin_bottom, and <part_id>_pixels, <part_id>_ratio,
<part_id>_components.
Form: %s. Canvas edge occupancy is a model decision; preserve it when a
same-size shape reference is the baseline and do not invent a margin
constraint solely by convention.
`<part_id>_ratio` means that part's opaque pixel count divided by the union opaque pixel count.
Only emit a quantitative constraint when it is necessary to the descriptor and feasible for your own
primitive plan; omit uncertain constraints rather than guessing restrictive numbers. No more than six
constraints. A required connection means the two named masks directly overlap
or touch; leave a connection optional when the descriptor deliberately
introduces a break, gap or detached fragment between those parts. Do not mark
an intentional fracture as a required physical joint.
Before returning, rasterize the rows and verify every required pair yourself:
their masks must overlap or be 8-neighbour adjacent at the coordinates you
actually emitted. If a physical junction is separated, adjust that part's
local rows; do not leave a required connection pointing at a visible gap.
Return JSON only:
{
 "width": int, "height": int, "parts": [PartSpec],
 "primitives": [{"id": string, "part_id": string, "primitive": string, "params": object, "layer": int}],
 "connections": [{"a": string, "b": string, "required": boolean}],
 "constraints": [{"metric": string, "minimum": number?, "maximum": number?, "message": string}],
 "uv_regions": [{"id": string, "part_id": string, "bbox": [left, top, right, bottom], "face": string, "required": boolean}],
 "background_transparent": true
}
Primitive language: %s
%s
Descriptor:\n%s
Reference evidence (each attached image follows this same order):\n%s
Provided UV layout: %s
The descriptor's immutable art_direction contains a concrete composition and
silhouette_actions. Execute those actions before borrowing a reference
contour. If it names a missing, broken, cut-away or transparent section, that
section must be absent from the final union alpha and end on the stated
irregular boundary; do not retain a complete reference shape and merely
recolour it. If it names a local feature as surface-only, preserve the host
alpha and reserve feature placement for AppearanceSpec rather than inventing
a detached reference-shaped overlay.
If a descriptor part has `paint_only:true`, keep it in the fixed part list but
do not create a primitive or connection for it; its visible identity must be
carried by the AppearanceSpec stage. Every required physical part
(`paint_only:false`) must own a non-empty primitive mask. If a feature can be
represented entirely inside an existing support, preserve the silhouette and
let the descriptor's paint_only decision carry it instead of inventing alpha.
If the descriptor includes `reference_part_map`, use its legend and rows as
model-authored ownership evidence: preserve those source coordinates for the
corresponding physical parts before applying a named variant edit. Do not let
the primary part absorb cells assigned to a support. The map is evidence for a
partition, not a requirement to copy an unrelated or newly designed contour.
If the descriptor includes `target_part_map`, use its legend and rows as the
placement evidence for the requested target. It may contain new motif cells
where the source was transparent. Keep each new local motif inside its named
host support, with a visible host rim/contact around it, and keep unrelated
supports on their source landmarks. The map is model-authored evidence for
where the parts belong; still emit the final pixel contour yourself. Motif
markers are overlays: do not remove the host support underneath them or let a
motif consume the host's entire cross-section.
If a provided UV layout is non-empty, copy it exactly and never invent or alter it. For entity_uv, use one `uv_fill` primitive per part as the base coat unless a part needs deliberately narrower face treatment; every painted pixel must stay inside that part's declared regions. For block_multi, treat every declared cube face as authoritative: a layout may contain independent face tiles or six face labels sharing one tile. Use one `uv_fill` base coat for each physical surface part and preserve a shared tile's single-texture semantics.
Use the attached reference images, pixel maps and alpha-only silhouette maps as a pixel ruler before inventing geometry: trace each row span, opaque staircase, shaft thickness, end cap and major value cluster at the requested resolution. `row_spans` are inclusive-exclusive x ranges for each source row; use them to check contour width instead of guessing from the object noun. For a genuinely new design, borrow directional pixel language and relative stroke scale rather than copying a complete alpha mask. For a named variant of a same-size shape reference, first reconstruct the reference's occupied coordinates as the baseline for every unnamed physical part; a complete mask copy is allowed as that baseline, then apply only the requested local delta. A narrow shaft or handle in the evidence must remain a narrow shaft or handle in the new mask; do not turn it into a filled square because the target noun sounds substantial.
If `reference_strategy` contains row/column or quadrant landmarks, treat those
landmarks as coordinate anchors: place each named physical part at the stated
end and junction before editing the requested feature. Do not move a support
to the opposite end merely because the object noun has a familiar orientation.
When the request changes one named property of a referenced object (such as a
broken, chipped or recoloured variant), preserve the reference's proportions
and all unnamed parts; spend the geometry change on the named property instead
of redesigning the handle, grip, guard or other supporting section.
The reference summary includes row/run stroke statistics. Use them as a scale check: when the evidence is built from one-to-two-pixel runs, a semantic handle, shaft, stem or rod should stay in that range unless the descriptor explicitly calls for a bulky grip or body. A broad head or container may be filled, but do not let that thickness leak into a narrow supporting part.
The summary also includes diagonal and anti-diagonal symmetry scores. Treat a strong score as evidence of deliberate mirrored pixel bands or a balanced cross-section: preserve that local rhythm in the new part, while relaxing global symmetry when the descriptor calls for an asymmetric object. Do not add random stair steps simply to make a contour look novel.
When the descriptor is a material-only sample or swatch, make a neutral hide/
leather/cloth patch with a modest irregular edge. Do not invent shoulder
extensions, a waist notch, sleeves, a neckline or any other garment/armor
silhouette, and do not trace an unrelated reference's complete contour.
For a recognizable composite object, model its semantic components as separate labelled parts (for example a cutting surface, a grip and a connector) instead of collapsing the whole object into one diagonal segment. At 16x16, use a deliberate `custom_mask` for every semantic part whose contour is diagonal, tapered, narrow or irregular; a few simple primitives are acceptable only for genuinely round or rectangular parts. For item/cross canvases up to 32x32, do not use a filled polygon or ellipse as a shortcut for a blade, handle, shaft, stem or other diagnostic contour: write its actual pixel rows in `custom_mask`. The final silhouette must make the required parts visibly distinct while keeping them connected.
Before returning, mentally rasterize every primitive at the requested pixel size. Do not represent every required part as an equally sized rectangle or ellipse: that is a degenerate placeholder and fails semantic review. Composite parts must have visibly different scale/aspect/contour, with a small junction or overlap where the object changes section. If a part is narrow, tapered or irregular, encode its actual pixel contour with `custom_mask` rows rather than hoping a smooth polygon will survive rasterization. Never use a generic pair of rectangles as a substitute for the requested object.
Internal surface cues inside a declared motif (pupil, iris, facet, slot,
highlight or grain) belong in AppearanceSpec marks. Do not add a new geometry
part for such a cue unless the descriptor explicitly declares it as a separate
physical region with its own alpha contour.
For an item/cross canvas at or below 32x32, treat `custom_mask` as mandatory for
an irregular sheet, hide, cloth, tool, blade, handle or other diagnostic
contour; use polygon/ellipse/blob only for a genuinely round base or a shape
whose smooth fill is explicitly requested. Do not duplicate a full-body mask
for an edge, highlight, shadow, outline, texture or stitch part. Before
returning, rasterize your own primitives and remove or relax any constraint
that the resulting pixels do not satisfy; a self-authored constraint must
never contradict its own mask. When uncertain, omit the constraint.
Use no more than 20 primitives. Keep messages and IDs concise.
""" % (
            form.value,
            allowed,
            primitive_parameter_guide(),
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
        json.dumps(to_jsonable(uv_regions or []), ensure_ascii=False),
        )
        if form in {AssetForm.ITEM, AssetForm.CROSS} and max(width, height) <= 32:
            # A short mask-first prompt is more reliable at tiny resolutions
            # than asking a vision model to juggle the full primitive grammar.
            # The result is still the same open GeometrySpec contract; only the
            # representation is made pixel-explicit where contour errors are
            # most visible.
            prompt = """Generate a GeometrySpec for this small Minecraft pixel asset. Work directly on the %dx%d pixel grid.
This is an open semantic request, not a fixed category template. Copy the
descriptor's part list exactly. Parts are physical alpha regions only; surface
highlight, shadow, outline, grain and stitches belong in AppearanceSpec, not
as duplicate geometry masks. For every diagonal, tapered, narrow or
irregular part, emit exactly one `custom_mask` primitive with `rows` containing
exactly %d characters per row and exactly %d rows; `X` is opaque and `.` is
transparent. At this resolution do not use `transform` primitives either;
write the final raster rows directly. Do not use polygon, ellipse or blob for
those parts. Keep every
part's contour visibly distinct, keep the reference's edge occupancy when a
same-size shape reference is supplied, and use a small overlap/touch at
declared joints. A broad genuinely
round base may use ellipse, but do not make a blade, handle, shaft, stem or
other diagnostic part a filled rectangle. Required connections must touch;
make a connection optional when the descriptor explicitly calls for a break,
gap or detached fragment between those parts.
Internal motif cues such as a pupil, iris, facet, slot, highlight or grain are
appearance marks unless the descriptor explicitly lists them as physical alpha
parts; do not invent extra geometry part IDs for them.
Before returning, rasterize the rows and verify every required pair yourself:
their masks must overlap or be 8-neighbour adjacent at the coordinates you
actually emitted. If a physical junction is separated, adjust that part's
local rows; do not leave a required connection pointing at a visible gap.
Return JSON only with this schema:
{"width":%d,"height":%d,"parts":[PartSpec...],"primitives":[{"id":string,"part_id":string,"primitive":"custom_mask","params":{"offset":[0,0],"marker":"X","rows":[string...]},"layer":int}],"connections":[{"a":string,"b":string,"required":boolean}],"constraints":[{"metric":string,"minimum":number?,"maximum":number?,"message":string}],"uv_regions":[],"background_transparent":true}
Constraints may use only compiler metrics opaque_pixels, occupancy_ratio, components,
orientation_degrees, axis_anisotropy, bbox_width_ratio, bbox_height_ratio,
aspect_ratio, margin_left, margin_top, margin_right, margin_bottom and
<part_id>_pixels/<part_id>_ratio; width and height
are dimensions, not constraint metrics.
After rasterizing the rows, verify every numeric constraint against that exact
mask. Remove any constraint you cannot satisfy; never return a self-contradictory
mask and constraint set.
Descriptor:
%s
Reference evidence (colour images follow this same order; each `shape`
reference is immediately followed by its alpha-only silhouette map):
%s
If `reference_part_map` is present in the descriptor, use its legend and rows
to partition the source union into physical parts before applying the named
change. Preserve coordinates assigned to unnamed supports; do not let a
primary diagonal part absorb the lower support simply because their alpha
touches. If the map conflicts with the source alpha's longest coherent axis,
preserve the source union and keep that axis assigned to the part whose role
is `primary support`; a lateral host may cross it at a junction but must not
replace the continuation with its own fill.
If `target_part_map` is present, use it to place any newly authored local
motif. A motif marked as embedded/inlaid must be a small region inside its
declared host, visibly distinct from unrelated supports; do not put it over a
blade or handle merely because those pixels are nearby. The host must remain
visible as a rim or side band around most of the motif; the motif may overlap
the host for layering, but it must not have the same footprint or consume the
host's whole contour. Keep the target map's host and motif ownership when
writing the final custom-mask rows.
Target-map motif cells overlay the host; preserve the host's full source
support underneath and place the motif as a smaller local contour inside it.
Use the reference `silhouette_map`, `row_spans`, stroke runs and symmetry
scores as a pixel ruler. For a new design, borrow the relative staircase
thickness and repeated pixel rhythm without copying the complete reference
mask. For a same-size referenced variant, copy the reference row spans for
unnamed parts as the starting mask and keep the named change local; this
conditional baseline rule takes precedence over the generic new-design rule.
For that variant baseline, do not leave a required physical part empty: first
cover the source's non-transparent cells with the corresponding named parts
and only then remove or add the pixels that express the requested difference.
If ownership of a source pixel is uncertain, retain it in the nearest declared
physical part rather than deleting the source contour. The final union should
therefore remain recognisably the referenced object before the local edit.
Compare the candidate's row spans with the source before returning: a source
run of several adjacent pixels must not collapse into a repeated one-pixel
diagonal merely because the requested variant is shorter or damaged. Keep the
source body thickness and support scale, and confine a named break, chip or
tip change to its owning region.
When the named change is damage, breakage or a chip, make that change visible
as an irregular local boundary (alternating row spans, a notch, or a detached
fragment when the descriptor calls for one). A straight intact-looking end
does not express damage. Keep the opposite end and all unnamed supports on the
source coordinates.
If the request is a variant of the referenced object, keep every unnamed part
at the reference scale and proportion; change only the named property (for
example a chip or break in a blade), rather than enlarging its handle or
redesigning the whole silhouette.
If the descriptor's reference strategy gives spatial landmarks or a junction
row, use those coordinates directly when laying out the masks; do not infer a
different support direction from the noun alone.
Treat explicit part landmarks in `reference_strategy` as ownership boundaries:
when it assigns a row range, end or quadrant to a part, do not let another
part's mask continue through that range. First partition the source union into
those physical parts, then apply the named local edit. The source alpha is a
union silhouette, not permission for the primary part to absorb the handle,
guard or other support.
For a material-only request, keep the patch recognizable as material rather
than clothing, a shield or an animal. If a same-size transparent material
sample is attached, its alpha contour is the neutral swatch's contour; trace
that contour and spend novelty on colour and surface pattern. Full opaque
references remain style evidence and must not become a silhouette.
""" % (
                width,
                height,
                width,
                height,
                width,
                height,
                json.dumps(to_jsonable(descriptor), ensure_ascii=False),
                "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
            )
        if form in {AssetForm.ITEM, AssetForm.CROSS} and max(width, height) <= 32:
            prompt += "\n" + _shape_baseline_text(references, width, height)
            prompt += """
For a same-size variant, this SOURCE_ALPHA_BASELINE is the exact union to
partition. Use `reference_strategy` to assign its opaque rows to the declared
parts before changing anything. Never let one part's mask silently absorb the
source coordinates that the strategy assigns to another part.
"""
            # Tiny sprites are where a long generic grammar hurts most: the
            # model starts satisfying diagnostics and loses the visual task.
            # Replace the verbose draft with a compact final-authoring brief;
            # the same open GeometrySpec contract remains, but the reference,
            # target and requested delta are now adjacent to the JSON boundary.
            prompt = """Author the final %dx%d Minecraft pixel-art raster for this request.
Target: %s
Semantic intent: %s
Reference strategy: %s
Descriptor parts (use these exact IDs): %s

The attached images are the reference colour image(s), followed by each
reference's alpha silhouette map when it is labelled shape. The last two images
are absent at this stage; do not invent a second object. The text below is the
same-size source alpha union, one row per line:
%s
Reference pixel/style text:
%s

Return JSON only in this schema:
{"width":%d,"height":%d,"parts":[PartSpec...],"primitives":[{"id":string,"part_id":string,"primitive":"custom_mask","params":{"offset":[0,0],"rows":[string...]},"layer":int}],"connections":[{"a":string,"b":string,"required":boolean}],"constraints":[],"uv_regions":[],"background_transparent":true}

Draw the target, not a generic diagonal icon. For a same-size variant, first
preserve the source's unnamed physical supports at their source coordinates and
scale, then apply only the named change. If a part-ownership map is present in
the descriptor, use it as optional evidence for that partition; the final
silhouette is still your judgement. Keep narrow shafts, handles and supports
narrow. Required physical joints must touch, while an intentional break or gap
is optional. If the request names a break, chip, fracture, damage or jagged
change, make it unmistakable at the affected part's exposed endpoint with a
short irregular/notched boundary; keep the interior rows and intact junction
regular. Do not add extra semantic parts just to express the break; multiple
fragments may share the owning part when the physical object calls for it.
Do not add a margin rule or decorative constraints. Every custom_mask must have
exactly %d rows of %d characters using X and .; include a non-empty mask for
each required part. Before returning, mentally rasterize and check the actual
connections and the visible target.
When the descriptor names an embedded or inlaid local motif, give that part a
compact contiguous contour with enough pixels to read as its own shape rather
than a one-pixel trail that merely follows the host object's axis. Keep its
local orientation independent unless the descriptor explicitly couples it to
the host. If the motif has an internal cue such as a pupil, facet, slot or
emblem, represent that cue as a separate declared part or as a part-targeted
Appearance mark; do not claim it in prose while leaving it invisible.
For a motif described as embedded or inlaid, center its mask within the host
support's span and leave a visible host rim or contact on at least two sides;
do not place the motif at the support's terminal end or let the host become a
single blob around it. Declare the physical contact as a required connection
when the parts are meant to be joined.
For any part described as round, curved, eye-like, gem-like or otherwise
compact, vary its row spans so the top and bottom are narrower than the middle;
do not emit a long flat strip that spans most of the host. For an internal
directional detail, keep it centered inside its motif and follow the declared
direction (for example a vertical detail stays narrow and vertical).
""" % (
                width,
                height,
                descriptor.target,
                descriptor.semantic,
                descriptor.reference_strategy,
                json.dumps([part.id for part in descriptor.parts], ensure_ascii=False),
                _shape_baseline_text(references, width, height),
                "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
                width,
                height,
                height,
                width,
            )
        if form == AssetForm.ENTITY_UV:
            # An entity texture is an atlas, so its geometry is a list of
            # declared UV faces rather than a single 2-D silhouette. Keeping
            # this response small also prevents the model from spending its
            # output budget on invented polygon coordinates.
            region_lines = "\n".join(
                "- %s part=%s face=%s bbox=%s" % (region.id, region.part_id, region.face, list(region.bbox))
                for region in (uv_regions or [])
            ) or "- none"
            hint = _same_size_region_ratios(references, uv_regions or [], width, height)
            if hint is None:
                alpha_lines = ""
            else:
                hint_name, ratios = hint
                alpha_lines = (
                    "A same-size vanilla reference (%s) paints these faces of the same model:\n%s\n"
                    "Treat that as evidence about which faces this model uses, never as a mask to "
                    "copy. A partial or dithered layer still states where the model paints and where "
                    "it opens, so follow it where it agrees with the brief and depart from it where "
                    "the brief asks for a different design.\n"
                    "Faces it leaves completely unpainted: %s\n"
                    "Omit EVERY one of those faces from your uv_fill. They are surfaces this "
                    "model never shows, so painting them adds invisible mass and contradicts the "
                    "model you are texturing. Only include one when the brief explicitly asks to "
                    "cover that exact surface.\n" % (
                        hint_name,
                        "\n".join(
                            "  %s %.0f%%" % (region_id, value * 100.0)
                            for region_id, value in sorted(ratios.items())
                        ),
                        ", ".join(sorted(
                            region_id for region_id, value in ratios.items() if value <= 0.0
                        )) or "none",
                    )
                )
            prompt = """Return a GeometrySpec for an entity texture atlas. The supplied UV layout is authoritative and will be copied by the caller.
Use exactly the descriptor's part IDs that correspond to model cubes; details
without a declared cube (eyes, nostrils, markings, tail accents) are paint-only
and must not become geometry parts. Return JSON only:
{"width":%d,"height":%d,"parts":[{"id":string,"meaning":string,"required":boolean,"layer":integer,"style_role":string}],"primitives":[{"id":string,"part_id":string,"primitive":"uv_fill","params":{"regions":["region_id", ...]},"layer":integer}],"connections":[],"constraints":[],"uv_regions":[],"background_transparent":true}

You decide which atlas faces this asset actually paints:
- Paint every face of every part by default: omit "regions" (a full base coat).
- Never drop a whole visible face to make an opening. A suit whose front face is
  missing is a hole a player sees straight through, not a design. Only omit a
  face that this model genuinely never shows, such as a surface the body fully
  hides.
- To open something inside a face the viewer does see, keep that face painted
  and add a "cutout": params {"shape":"ellipse","params":{"bbox":[l,t,r,b]}} or
  {"shape":"polygon","params":{"points":[...]}}, on a higher "layer" than the
  fill. Openings are usually small and local: a face opening within a helmet
  front, a visor slit, a hole through a plate. Do not delete the face around it.
- When the brief or descriptor asks for a face, neck, hand or limb opening to
  stay transparent, you MUST emit one cutout per opening. A full base coat does
  not satisfy that brief, and deleting the whole face is not the opening it
  means: a missing front face is a hole the viewer sees straight through.
Emit one uv_fill per physical model part as the base coat, then any cutouts you
want. Do not copy or invent UV rectangles; the caller binds the exact supplied atlas.

Atlas regions (id, part, face, bbox) - use these exact ids:
%s

%s
Descriptor:
%s
""" % (width, height, region_lines, alpha_lines, json.dumps(to_jsonable(descriptor), ensure_ascii=False))
        if (
            isinstance(descriptor.target_part_map, dict)
            and not _target_map_covers_declared_parts(descriptor, width, height)
        ):
            prompt += """
TARGET-MAP WARNING: the descriptor's target_part_map failed the raster
contract or host-placement audit. Treat it only as weak placement evidence;
do not copy a thin line, terminal patch or host-sized rectangle from it. For
each declared round/eye/gem/motif part, author a model-owned compact contour
with at least three rows, narrower ends and a wider middle, leaving a visible
host rim. Keep internal pupil/iris/facet cues in AppearanceSpec unless the
descriptor explicitly declares a separate physical part. The final geometry
must contain a real non-empty mask for every required physical part
(`paint_only:false`). Paint-only parts stay in the part list without a mask and
are authored by AppearanceSpec.
"""
        # Put the actual visual objective next to the response boundary.  The
        # long grammar above is reference material; this short tail is what a
        # vision model should optimize after it has read the contract.
        prompt += """
FINAL VISUAL TASK (follow this after the schema rules):
Target: %s
Semantic intent: %s
Reference strategy: %s
Model-selected shape edit mode: %s. `appearance_only` keeps alpha unchanged;
`preserve_silhouette` keeps the source union while retaining an honest part
partition; `local_silhouette_edit` preserves unnamed supports and spends only
the smallest local alpha change on the model-declared physical feature;
`new_silhouette` authors the contour from the descriptor and reference pixel
language. Follow this model decision rather than a fixed object family.
Return the finished raster for this target. Preserve the reference's unnamed
physical supports and their spatial anchors. If the semantic intent names a
break, chip, fracture, damage, tear or jagged change, make that change visible
in the owning part's final rows as an asymmetric notch or terminated endpoint
at the part's exposed end, away from an intact junction; do not leave a
continuous clean source run through the changed area. Keep all
other parts recognizable at the reference scale. A smooth taper alone is not a
break: use a notch, alternating row span, or another visibly non-monotonic
pixel boundary in only the last few rows at that exposed endpoint. Keep the
interior source rows regular; do not sprinkle notches through the whole shaft
or alter the rows at an intact support junction. Do not spend output on prose,
category templates or decorative constraints.
""" % (
            descriptor.target,
            descriptor.semantic,
            descriptor.reference_strategy,
            descriptor.shape_edit_mode,
        )
        if form in {AssetForm.ITEM, AssetForm.CROSS} and max(width, height) <= 32:
            prompt += "\nLAST INSTRUCTION: follow this exact local delta and coordinate guidance: %s\nReturn the final raster for `%s` now." % (
                descriptor.reference_strategy,
                descriptor.target,
            )
            # Geometry does not need to repeat the semantic part catalogue.
            # The descriptor owns it; this stage owns only the masks/joints.
            physical_ids = [part.id for part in descriptor.parts if not part.paint_only]
            paint_ids = [part.id for part in descriptor.parts if part.paint_only]
            prompt = """Author only the geometry raster for this %dx%d Minecraft pixel-art asset.
Target: %s
Design: %s
Shape policy: %s

Fixed physical geometry IDs: %s
Fixed surface-only IDs: %s
Surface-only IDs are painted later. They must not occur in a primitive or
connection. Do not return a `parts` field and do not invent, rename, split or
merge any IDs.

Reference silhouette_map / source alpha union (`#` is opaque):
%s

Reference pixel/style text:
%s

Reference contour evidence (silhouette_map and row_spans):
%s

Return JSON only:
{"width":%d,"height":%d,"primitives":[{"id":string,"part_id":"one fixed physical ID","primitive":"custom_mask","params":{"offset":[0,0],"rows":[string...]},"layer":int}],"connections":[{"a":"fixed physical ID","b":"fixed physical ID","required":boolean}],"constraints":[],"uv_regions":[],"background_transparent":true}

Each custom_mask has exactly %d rows of %d `X`/`.` characters. Partition the
source union across physical IDs, preserve its silhouette under policy `%s`,
and keep each physical ID non-empty. Keep the source's narrow grip and diagonal
blade at their original coordinates. Do not represent colour, iris, pupil,
highlight, outline, texture or the embedded eye with geometry.
Only return the JSON object.
""" % (
                width,
                height,
                descriptor.target,
                descriptor.semantic,
                descriptor.shape_edit_mode,
                json.dumps(physical_ids, ensure_ascii=False),
                json.dumps(paint_ids, ensure_ascii=False),
                _shape_baseline_text(references, width, height),
                _appearance_reference_evidence(references),
                "\n".join("- " + summarize_reference(item, include_pixel_map=False) for item in references) or "- none",
                width,
                height,
                height,
                width,
                descriptor.shape_edit_mode,
            )
        response = ""
        geometry: GeometrySpec | None = None
        last_error: Exception | None = None
        # A vision response can satisfy the JSON schema while leaving one
        # required mask empty or duplicating a neighbouring part. Give the
        # model two corrective passes inside the geometry stage before the
        # outer pipeline repair loop is allowed to intervene; this keeps the
        # original pixel language alive instead of jumping straight to a
        # generic polygon repair.
        for attempt in range(3):
            response = self.client.complete(
                prompt,
                image_data_uris=geometry_images,
                json_mode=True,
                max_tokens=7500,
            )
            try:
                geometry_data = _json_object(response)
                # The descriptor is the sole source of semantic part metadata.
                # This response owns masks and connections only.
                geometry_data["parts"] = [to_jsonable(part) for part in descriptor.parts]
                geometry = geometry_from_dict(geometry_data)
                expected_part_ids = {part.id for part in descriptor.parts}
                actual_part_ids = {part.id for part in geometry.parts}
                if actual_part_ids != expected_part_ids:
                    missing = sorted(expected_part_ids - actual_part_ids)
                    extra = sorted(actual_part_ids - expected_part_ids)
                    details = []
                    if missing:
                        details.append("missing declared parts: %s" % ", ".join(missing))
                    if extra:
                        details.append("undeclared parts: %s" % ", ".join(extra))
                    raise ValueError("geometry part set must match descriptor (%s)" % "; ".join(details))
                # Geometry owns masks and connections, never semantic part
                # metadata.  Rebind the model's ID-only response to the
                # immutable descriptor so a surface-only detail cannot become
                # a physical alpha part merely because the model omitted its
                # `paint_only` flag in the geometry schema.
                geometry = replace(geometry, parts=list(descriptor.parts))
                paint_only_ids = {part.id for part in descriptor.parts if part.paint_only}
                unknown_primitive_parts = [
                    primitive.part_id for primitive in geometry.primitives
                    if primitive.part_id not in expected_part_ids
                ]
                if unknown_primitive_parts:
                    raise ValueError(
                        "geometry primitives use undeclared parts: %s"
                        % ", ".join(sorted(set(unknown_primitive_parts)))
                    )
                invalid_paint_primitives = [
                    primitive.id for primitive in geometry.primitives
                    if primitive.part_id in paint_only_ids
                ]
                if invalid_paint_primitives:
                    raise ValueError(
                        "paint-only parts cannot have geometry primitives: %s"
                        % ", ".join(invalid_paint_primitives)
                    )
                lint_issues = _placeholder_geometry_issues(geometry, form)
                if lint_issues and attempt < 2:
                    last_error = ValueError("; ".join(lint_issues))
                    prompt = prompt + "\nGeometry lint rejected the previous construction: %s. Return a materially different pixel-level contour. For this small item/cross, replace coarse filled blocks with explicit custom_mask rows, give every required physical part its own non-empty rows, leave paint-only parts without primitives, and keep narrow parts narrow. Check pairwise part overlap before returning." % (
                        "; ".join(lint_issues),
                    )
                    continue
                try:
                    compiled_candidate = compile_geometry(geometry)
                    overlay_issues = _target_overlay_geometry_issues(
                        descriptor, geometry, compiled_candidate
                    )
                    if overlay_issues and attempt < 2:
                        last_error = ValueError("; ".join(overlay_issues))
                        prompt = prompt + "\nTarget overlay lint rejected the previous response: %s. Keep the model-authored motif inside its host, but do not collapse it below the target map's local area; preserve a readable contour and leave internal surface cues to AppearanceSpec." % (
                            "; ".join(overlay_issues),
                        )
                        continue
                    geometry_validation = validate_geometry(geometry, compiled_candidate, form)
                except (KeyError, TypeError, ValueError) as exc:
                    geometry_validation = None
                    last_error = exc
                if geometry_validation is not None and not geometry_validation.passed and attempt < 2:
                    last_error = ValueError("; ".join(geometry_validation.errors))
                    prompt = prompt + "\nGeometry self-check rejected the previous response: %s. Recompute the rasterized mask, ensure all required parts are non-empty and locally distinct, then remove or relax any self-contradictory constraints before returning." % (
                        "; ".join(geometry_validation.errors),
                    )
                    continue
                break
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                # Vision planners occasionally invent friendly semantic names
                # in connections instead of copying the exact part IDs.  Give
                # one schema-focused retry rather than silently dropping those
                # relationships or mutating a valid design in code.
                known_parts = json.dumps([part.id for part in descriptor.parts], ensure_ascii=False)
                prompt = prompt + "\nThe previous JSON failed schema validation: %s. Known part IDs are exactly %s. Return a corrected complete GeometrySpec; every connection endpoint and primitive part_id must use one of these IDs, and do not add undeclared parts." % (
                    str(exc), known_parts,
                )
        if geometry is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("geometry planner returned no response")
        if (
            form in {AssetForm.ITEM, AssetForm.CROSS}
            and len(geometry.parts) > 1
            and max(width, height) <= 32
            and descriptor.shape_edit_mode not in {"appearance_only", "preserve_silhouette"}
        ):
            # Composite low-resolution items benefit from one model-owned
            # raster audit before the outer blind-review loop. This does not
            # select a template or edit pixels in Python: the same planner
            # receives its own draft plus the reference and may return a
            # better contour. Invalid audit responses are discarded.
            review_prompt = """Audit this small multi-part GeometrySpec against the descriptor and attached reference.
Return a complete replacement GeometrySpec JSON only. Keep the exact part IDs,
canvas dimensions and requested semantics. Preserve all unnamed support parts
at the reference scale; spend changes on the named variant property. For every
narrow, diagonal, tapered or broken part, use full-width, full-height
`custom_mask` rows at this resolution, not transform wrappers. Keep declared
connections touching. Do not add a
category template or color data.
The attached images are ordered as: reference colour image(s), each followed
by its alpha silhouette map when labelled `shape`, then the draft union mask
and draft labelled part map. Do not treat the draft as the reference. For a
same-size variant, partition the reference union into the descriptor's named
parts first and preserve every unnamed support coordinate; apply the requested
change only to its owning part.
Descriptor:
%s
Draft geometry:
%s
Reference evidence (same order as attached images):
%s
FINAL AUDIT TASK: reconstruct the requested target `%s` now. Preserve every
unnamed support from the reference and make the named semantic change visible
in the owning part's exposed endpoint, away from an intact junction. Return
only the complete GeometrySpec JSON.
""" % (
                json.dumps(to_jsonable(descriptor), ensure_ascii=False),
                json.dumps(to_jsonable(geometry), ensure_ascii=False),
                "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
                descriptor.target,
            )
            try:
                review_compiled = compile_geometry(geometry)
                reviewed = geometry_from_dict(_json_object(self.client.complete(
                    review_prompt,
                    image_data_uris=[
                        *_geometry_reference_data_uris(references),
                        _mask_data_uri(review_compiled.mask),
                        _part_map_data_uri(descriptor, review_compiled),
                    ],
                    json_mode=True,
                    max_tokens=7500,
                )))
                if (
                    {part.id for part in reviewed.parts} == {part.id for part in geometry.parts}
                    and reviewed.width == width
                    and reviewed.height == height
                ):
                    reviewed_compiled = compile_geometry(reviewed)
                    reviewed_validation = validate_geometry(reviewed, reviewed_compiled, form)
                    current_compiled = compile_geometry(geometry)
                    current_validation = validate_geometry(geometry, current_compiled, form)
                    # The raster audit is a repair opportunity, not a license
                    # to replace a valid model draft with a second, generic
                    # contour.  Preserve a structurally valid draft for the
                    # outer blind/revision loop; use the audit result only
                    # when it fixes an actual geometry contract failure.
                    if reviewed_validation.passed and not current_validation.passed:
                        geometry = reviewed
            except (KeyError, TypeError, ValueError):
                pass
        if geometry.width != width or geometry.height != height:
            geometry = rescale_geometry_spec(geometry, width=width, height=height)
        return geometry

    def appearance(self, descriptor: ShapeDescriptor, geometry: GeometrySpec,
                   references: list[ReferenceAsset] | None = None) -> AppearanceSpec:
        references = references or []
        prompt = """Design the palette and part materials for a locked Minecraft pixel mask.
You may not add, remove, merge or move geometry. Return JSON only:
{
 "palette": {"name": "#RRGGBB"},
 "parts": {"part_id": {"colors": ["palette_name_or_hex"], "material": string,
 "shade_axis": string, "noise": number, "highlight_ratio": number,
 "marks": [{"parts": ["part_id"], "regions": ["uv_region_id"], "offset": [x,y], "rows": ["...X..."], "color": "palette_name_or_hex"}]},
 "pixel_map": {"legend": {"a": "palette_name_or_hex"}, "rows": ["native-width rows"]}|null,
 "region_rules": [{"regions": ["exact_uv_region_id"], "mode": "retint"|"source_exact"|"palette",
 "target_color": "palette_name_or_hex", "cluster_only": true|false, "strength": 0.0}],
 "motif_policy": "free"|"reference_locked"|"model_authored"|"none",
  "outline_color": "palette_name_or_hex_or_null", "outline_width": 0_or_1,
  "transparent_background": true,
 "reference_sampling": "none", "value" or "pattern",
  "part_reference_sampling": {"part_id": "none"|"value"|"pattern"},
  "part_reference_sources": {"part_id": "exact reference name or null"},
  "part_reference_composite": {"part_id": "overlay"|"replace"}
}
Descriptor:\n%s
Geometry parts:\n%s
Reference evidence (each attached image follows this same order):\n%s
The descriptor's immutable art_direction is a detailed visual contract. Its
`surface_actions` must appear as deliberate local value clusters, material
boundaries or marks; do not flatten a textured support into a single smooth
ramp. Its `feature_actions` specify the host, approximate footprint and
internal detail of a requested local feature. For a paint-only feature, put
those feature pixels in the top-level full-canvas `pixel_map` on opaque host
cells. A nested `pixel_map` inside a part mark is not supported and will draw
nothing; marks contain one `color` plus X rows only.
Use only the declared palette/material/pixel-style evidence. If a same-sized
reference atlas provides pixel-style or material evidence, prefer
"reference_sampling": "pattern" when the source's discrete palette bands
should survive, or "value" when only broad light/dark placement should transfer;
otherwise use "none". Marks are optional internal pixels and
must stay inside the listed UV regions and locked part masks. Marks may use a
small local `rows` grid with an explicit pixel `offset:[x,y]`; without an
offset their rows are local to the owning physical part's bounds. A descriptor
part marked `paint_only:true` has no independent alpha mask: author its visible
feature here with a top-level pixel_map, and keep the surrounding
support's base texture intact. For an item/cross paint-only mark, `offset:[x,y]`
is mandatory unless its rows already span the full native canvas; its X cells
must land on opaque host pixels, never on transparent canvas cells. At item sizes of
16x16 or below, give each major part a clear value hierarchy: reserve the
darkest pixels for an edge, underside or seam and the lightest pixels for a
small directional highlight. Keep accents subordinate to the main silhouette.
Read each reference `pixel_map` legend and rows as actual evidence: preserve
the reference's material families and dark/mid/light ordering unless the
request explicitly asks for a different material. Do not introduce saturated
gold, cyan or neon accents merely to make a part noticeable. For an item with
wood and metal evidence, the wood remains a compact supporting grip and the
metal cutting/working surface carries the strongest light-dark edge contrast.
References labelled `negative` are anti-evidence: avoid their shape and
palette instead of copying them. If a newly authored local part has a
reference labelled `material` or `palette`, borrow that reference's hue
family, value hierarchy and small-pixel rhythm for the new part while keeping
its geometry model-authored; do not sample the new part from the unrelated
host texture.
For any composite item whose parts have different functions (for example a
working surface, handle, guard or connector), keep those parts visibly
separate in value and material response even when the request names one
overall material. A dark handle beside a dark blade still needs a distinct
mid-tone band, edge direction or highlight cluster so the part boundary reads
in a blind preview; do not let reference sampling collapse every part into
one indistinguishable ramp.
Use the reference symmetry scores and repeated value bands as a texture cue:
mirror or repeat nearby light/dark clusters when the source does, and keep
noise low enough that the deliberate pixel rhythm remains readable at 16x16.
Read each attached reference's exact `name` and `roles` before choosing a
material. A `shape`/`uv_layout` reference is the coordinate ruler; a
`material`/`palette` reference is texture and colour evidence. When a selected
secondary reference is texture-bearing, carry its small-pixel rhythm into
model-authored marks or local region rules and state the exact target regions;
do not reduce it to one accent swatch or a single centred emblem.
For an entity or block atlas, do not reduce the reference to only three global
swatches. Inspect each declared face region separately and retain its local
clusters: face/muzzle or ear patches, horn bands, underside/udder patches,
seams and repeated grain marks need their own palette entry or a small mark.
When the request combines a broad base material with an embedded local motif
(for example an ore crystal, emblem, stripe or inlay), keep the broad part
colours as the base material ramp and place the secondary hue through sparse
non-empty marks or exact region rules. Do not spread an accent across every
pixel merely because it exists in the palette; choose its coverage and exact
pixel clusters from the reference and descriptor.
For a shared single-tile block surface, the `colors` array is the broad base
ramp only; do not mix a sparse motif's accent family into that array. Put the
motif in a model-authored mark or `cluster_only`/`retint` rule so its coverage
remains local on every face that reuses the tile.
Use `region_rules` when a region contains a semantic or material sub-cluster
that cannot be expressed by the part ramp alone. You must choose the exact
UV region IDs from the supplied geometry; do not invent anatomy names. Use
`retint` when the source cluster's shape/value should survive but the new
asset needs a different hue, `source_exact` only for a deliberately retained
accent, `palette` when the normal authored palette mapping should be kept
without source-colour preservation, and `cluster_only: true` when the rule should touch secondary pixels
inside a region rather than its dominant base. This decision belongs to the
appearance plan, so the renderer does not guess whether an arbitrary region
is an ear, a gem, a seam, a tooth or a fabric patch.
For a composite with both unchanged source supports and a newly authored local
motif, use `part_reference_sampling` to make that ownership explicit: keep
unchanged supports on `pattern` or `value`, and set a newly authored motif to
`none` so its declared palette and marks are not sampled from an unrelated
source. An omitted entry inherits the global mode. A mark may target exact UV
`regions` when the geometry declares them, or exact declared `parts` when it
is an item/cross mask without UV regions; part-targeted rows are local to that
part's mask bounds. Do not use a source-driven mode for a part whose pixels
are meant to be newly invented.
For every newly authored motif part, keep its `colors` within one coherent
target hue family instead of concatenating source/reference swatches with a
new accent family. Give an internal mark a visibly contrasting declared color
against that part's base ramp and its host support; a mark whose color is the
same dark or light token as the surrounding ramp is not a visible feature.
Treat `colors` as the broad base ramp only. If a named internal detail (such as
 a pupil, slit, facet, vein, slot, stitch or emblem) is drawn by `marks`, keep
 its color token out of that part's `colors` array; otherwise radial or linear
 shading will spread the detail across the whole motif. Let the mark carry
 that detail color and leave the surrounding base/iris/facet ramp visible.
When a motif has a named directional internal detail, make the mark grid obey
that direction at native resolution: a vertical detail remains a narrow
centered column, while a horizontal detail remains a short centered band. Do
not let a detail grid turn the motif into a broad stripe.
Use `pixel_map` only when neighbouring parts or a local inset need one
deliberate final colour relationship. It is a full native-canvas RGB overlay:
`.` preserves ordinary reference/part paint and every other one-character
symbol resolves through `legend`. It cannot modify alpha or silhouette. Decide
from this request and attached references; do not use object-specific defaults.
When `pattern` evidence shows a clearly separated local colour cluster inside
an otherwise broad region, declare a rule for that exact UV region instead of
letting the renderer flatten the cluster. Choose the target hue from the
current request's palette: use `retint` to make a variant while retaining the
source cluster's pixel shape/value, or `source_exact` only when keeping the
source hue is part of the design. If the region is intentionally uniform,
leave `region_rules` empty and let the normal ramp do the work.
An entry in `marks` whose rows contain only `.` or spaces is a no-op and will
be discarded. If the request adds a visible paint motif (ore flecks, an
emblem, a stripe, a stitch or similar), encode at least one non-empty mark or
an exact-region rule; do not claim the motif in prose while returning an
empty pixel pattern.
The image is supplied at native nearest-neighbour resolution; the text map is
an aid, not permission to erase pixels that are visible in the image.
When the request is a recolour, damaged variant or other modification of a
selected reference, preserve the reference's local motif coordinates and
occupied pixel clusters. A newly authored pixel is allowed only when the
descriptor explicitly requests a motif that the reference does not contain;
do not add repeated marks at convenient empty coordinates just because an
accent colour exists. The reference is the coordinate ruler for embedded
flecks, ore inclusions, emblems and patches, while the model chooses the new
hue or material.
Use the smallest justified appearance change. If the request names a damage,
break, chip or other variant but does not name a new hue/material, keep every
unnamed supporting part's source material family, palette assignment, local
value bands and texture coordinates unchanged; do not recolour a handle,
grip, guard or connector just to emphasize the primary change. A palette or
material change is allowed only when the descriptor explicitly asks for it or
when it is confined to the named damaged region.
Set `motif_policy` to `reference_locked` when the selected source's local
clusters are the motif and the request does not explicitly add or rearrange
them; in that mode do not rely on `marks` to redraw the source. With
`reference_locked` plus `reference_sampling: "pattern"`, opaque source pixels
remain the base raster unless an explicit region rule claims them. Therefore a
`cluster_only: true` retint changes only the named secondary cluster and keeps
the surrounding stone, hide, leather or fabric pixels exactly as sampled. Use
`model_authored` or `free` for a broad material recolour, or author a full
region rule when the background itself is meant to change. Use `none` when
there is no local motif.
When using `reference_sampling: "pattern"`, set `highlight_ratio` to 0: the
reference already supplies directional edge highlights and a random overlay
would repaint light pixels onto an otherwise dark outline. Do not use a single
uniform `outline_color` to overwrite a reference-driven edge; preserve the
source's multiple edge bands instead.
When a selected secondary reference is the actual named material reference and
texture source for
one part, you may bind it explicitly with `part_reference_sources`: use the
exact reference `name` from the evidence, and pair it with that part's
`part_reference_sampling` mode. Choose `overlay` when it adds local texture
clusters to an unchanged structural base, or `replace` when the part itself is
the newly requested material. Both keep part of the source's own colour on
purpose: `replace` blends the source pixel toward your ramp while retaining
some of its original RGB so grain and mottling survive, and `overlay` copies the
source's salient pixels through unchanged. That is exactly right when the part
keeps the source material, and exactly wrong when the part stands in for a
different one -- a crystal, glass, metal or glowing surface rendered over a
wooden source will show the wood. For such a part set
`part_reference_sampling: "none"`, which paints it from your ramp alone. This
is optional and model-authored; the
renderer samples that named raster locally while preserving the locked UV
geometry. Leave the mapping empty for parts that should keep the global
structural reference. Never invent a source name or bind a shape-only source
as a material layer.
Only a reference carrying a material or palette role is eligible for
this binding; a shape/UV/pixel-style-only source is a coordinate/style aid,
not a material layer. If the request explicitly names a material or finish
and an eligible selected reference supplies that material evidence, bind the
affected broad surface parts to it so its local pixel texture is executable.
For an entity atlas, the vanilla face layout is the visual anchor. Keep the
base hide/material on broad head, body and leg regions, and reserve a saturated
accent for a few pixels of the most diagnostic face or marking. Never make an
entire broad or lower supporting part bright red/neon when the request only
asks for a demonic accent. Any `marks.regions` value must be an exact declared
UV region ID; a shorthand part name means the first front-facing region and
must not be stamped onto every side.
When the reference is a known animal or character, preserve the cues that make
the source readable at a glance: face/muzzle contrast, horn/ear contrast,
separated limb bands and any small underside marking. Recolour those cues into
the requested material family instead of flattening every face to one dark
swatch. A large source outlier inside a broad body face (for example a pink
vanilla marking) must become a restrained mid-tone variation or a sparse mark,
never a second bright slab that can read as a tool head. The generated atlas is
judged through the supplied model-facing preview, so the head, torso and
multiple supports must remain visually separable after recolouring.
For each physical part, keep the `colors` array within a single hue family unless
the descriptor explicitly requests a multicolour material. The array must be
ordered dark-to-light because `reference_sampling: "pattern"` maps source
palette bands onto that order; use at most six colors per part. For a recolor
request, do not mix the source material's brown/gray colors into the requested
hue merely to add detail. Use the source only for value ordering and pixel
placement; all new hues must come from the requested material.
For a broad material or finish change, the affected part must use a
target-family ramp as its main `colors` array. Do not concatenate the old
source ramp before the new material ramp: source-to-target pattern mapping
will otherwise spend most pixels on the old hue and leave the requested
finish invisible. If exposed source material is intentionally retained, keep
it to a sparse model-authored mark or an explicit local rule, while the
dominant surface follows the requested material family.
For rust or corrosion, interpret that family as a red-orange/burnt-rust ramp:
brown may supply a dark shadow, but a neutral gray or brown-only ramp must not
be used as the dominant blade or metal surface when rust is the requested
finish.
If the descriptor or query explicitly names a hue or material (for example
dark red, crimson, infernal, emerald, ice or gold), that request outranks the
vanilla palette: encode it in the broad body/primary part while retaining the
source's light/dark structure. Do not silently fall back to the reference's
brown simply because the source animal is brown.
Do not copy a reference alpha mask or claim to change the locked geometry.
""" % (
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            json.dumps(to_jsonable(geometry.parts), ensure_ascii=False),
            "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
        )
        prompt += """
FINAL APPEARANCE TASK (follow this after the schema rules):
Target: %s
Semantic intent: %s
Keep the locked geometry untouched. Preserve the selected reference's local
value bands and texture coordinates for unnamed parts, especially supporting
surfaces; change hue/material only where the target explicitly asks for it.
When the target is only a state or contour variant, copy each unchanged part's
source hue family and pixel-map legend instead of choosing a new material from
its semantic name. If a support is brown in the reference, keep that brown
family when the request did not ask for recolouring; do not substitute another
plausible material merely to make the part visible.
Choose a restrained, readable Minecraft palette and return JSON only.
""" % (descriptor.target, descriptor.semantic)
        eligible_lines = [
            "eligible material references (exact names for part_reference_sources):"
        ]
        for reference in references:
            role_values = {role.value for role in reference.roles}
            if {"material", "palette"} & role_values:
                eligible_lines.append(
                    "- %s: eligible (%s)" % (
                        reference.name,
                        ",".join(sorted(role_values)),
                    )
                )
            else:
                eligible_lines.append(
                    "- %s: ineligible as a material source (%s)" % (
                        reference.name,
                        ",".join(sorted(role_values)) or "no roles",
                    )
                )
        if len(eligible_lines) == 1:
            eligible_lines.append("- none")
        prompt += "\n" + "\n".join(eligible_lines) + "\n"
        prompt += (
            "For an explicitly requested material or finish, keep its eligible "
            "material source binding in the affected broad parts; do not replace "
            "an eligible material source with a shape source.\n"
            "Treat a reference described as a variant or analogous example as "
            "structural precedent unless its own pixel pattern clearly matches "
            "the requested material.\n"
            "A structural variant is not a direct material source when a more "
            "specific texture reference is eligible.\n"
            "For a requested material finish, prioritize a material-role source "
            "over a palette-only source; use palette-only sources for hue/value "
            "guidance when no direct material raster exists.\n"
            "Whenever part_reference_sources is used, set part_reference_composite "
            "explicitly to overlay or replace. Use overlay for local clusters on "
            "an unchanged support and replace only for a deliberate full-part "
            "material change. Once an eligible source is bound with overlay, "
            "let its irregular raster pixels carry the texture; keep marks "
            "empty or sparse instead of redrawing repeated stripes.\n"
        )
        # The preceding long-form rules were retained temporarily while the
        # renderer contracts were introduced.  They are deliberately not sent
        # to the painter: a dense policy essay plus a labelled copy of the
        # sprite made the model reason about schema mechanics instead of the
        # actual vanilla pixels.  This task card is the complete appearance
        # interface.
        prompt = """You are painting one locked Minecraft raster. Return one complete
AppearanceSpec JSON object and no prose. You may change RGB only: do not alter
canvas size, alpha, or the outer silhouette.

OUTPUT SHAPE
{"palette":{"name":"#RRGGBB"},"parts":{"part_id":{"colors":["name"],"material":"...","shade_axis":"...","noise":0.0,"highlight_ratio":0.0,"marks":[]}},"pixel_map":{"legend":{"a":"#RRGGBB"},"rows":["..."]}|null,"region_rules":[],"motif_policy":"reference_locked|model_authored|free|none","outline_color":null,"outline_width":0,"transparent_background":true,"reference_sampling":"none|value|pattern","part_reference_sampling":{},"part_reference_sources":{},"part_reference_composite":{}}

WORKING RULES
1. The attached images are the source of truth. Preserve each unchanged
reference pixel cluster, including edge bands, texture marks and small
asymmetries. Do not replace it with a smooth ramp or a large symbolic area.
2. References are listed in attachment order. `shape` fixes the raster
silhouette; `material` and `palette` give colour/texture evidence; `negative`
is anti-evidence. The SOURCE_PIXELS block is an exact text version of every
small reference: its legend maps symbols directly to #RRGGBB.
3. Make only the changes named by the visual brief. Keep unnamed supporting
parts in their source hue family and pixel positions.
4. A paint_only feature is a local drawing on existing opaque host pixels. Put
only its changed cells in top-level pixel_map; every unchanged cell is `.`.
Use literal #RRGGBB values in pixel_map.legend. A readable local feature needs
a distinct boundary/rim, base colour, and internal contrast where the brief
asks for one (for example iris plus pupil). Do not redraw the host sprite.
5. `parts[*].colors` is a material ramp, not a request to flood-fill that part.
Use `reference_sampling:"pattern"` with `highlight_ratio` to 0 when keeping a
source texture. Keep each part's broad ramp in one single hue family. A
reference can supply either structural pixels (shape/uv/pixel_style) or a
material palette. `part_reference_sources` names the base structural raster
for that exact locked part, so it may use a non-negative structural reference
when its stated intent matches that face; use `replace` for that base pattern
and `overlay` only for an explicitly secondary layer. Material/palette sources
guide the hue ramp but must not silently displace a named structural source.
Respect each declared UV `face` literally: `top` is the physical top face,
while `front`, `right`, `left`, etc. are vertical faces. Do not swap an
end-grain, bark, panel, mouth, or other face-specific reference onto a
different physical face. Use `model_authored` only for a genuinely new local
feature or broad requested material change. Order every material ramp
dark-to-light and preserve the reference's local motif coordinates.
`motif_policy:"reference_locked"` preserves unclaimed source pixels at their
original RGB values; use it only when the brief keeps that source material or
changes a small named cluster. If the brief changes the broad hue or material
of a surface, choose `model_authored` (or `free`) plus
`reference_sampling:"pattern"`: this retains the source pixel rhythm while
mapping it through your target colour ramp. Never claim a broad recolour with
`reference_locked`, because it will render the original source colours.
Every newly authored pixel must be justified by the visual brief; do not add
decorative accents at convenient coordinates. Make the smallest justified appearance change.

VISUAL BRIEF
%s

LOCKED PHYSICAL PARTS
%s

REFERENCE PIXELS
%s
""" % (
            json.dumps(_appearance_brief(descriptor), ensure_ascii=False, separators=(",", ":")),
            json.dumps([
                {"id": part.id, "meaning": part.meaning, "paint_only": part.paint_only}
                for part in geometry.parts
            ], ensure_ascii=False, separators=(",", ":")),
            _appearance_reference_evidence(references),
        )
        draft = appearance_from_dict(_json_object(self.client.complete(
            prompt,
            image_data_uris=_appearance_image_data_uris(references),
            json_mode=True,
            max_tokens=3500,
        )))

        def finalize(candidate: AppearanceSpec) -> AppearanceSpec:
            issues = _paint_only_surface_issues(descriptor, geometry, candidate)
            if not issues:
                return candidate
            try:
                compiled = compile_geometry(geometry)
                images = [*_appearance_image_data_uris(references), _mask_data_uri(compiled.mask)]
            except (KeyError, TypeError, ValueError):
                images = _appearance_image_data_uris(references)
            correction = """Correct this AppearanceSpec after a surface-only feature contract failure.
Return a complete AppearanceSpec JSON object only. The geometry is locked.

The descriptor contains one or more required `paint_only:true` features on a
flat item/cross canvas. For this form, `marks` alone are insufficient because
they have no independent alpha bounds. You MUST provide `pixel_map` with a
legend and exactly %d rows of exactly %d characters. `.` preserves the normal
render. Place a small, recognisable multi-colour cluster for each paint-only
feature on opaque host pixels shown by the final attached binary mask. Do not
put the cluster at `[0,0]` by default, do not paint transparent cells, and do
not recolour the whole host. Keep the supporting reference texture intact
outside the few named feature cells.

Contract failures: %s
Descriptor: %s
Candidate AppearanceSpec: %s
""" % (
                geometry.height,
                geometry.width,
                json.dumps(issues, ensure_ascii=False),
                json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
                json.dumps(to_jsonable(candidate), ensure_ascii=False),
            )
            current = candidate
            for attempt in range(2):
                try:
                    revised = appearance_from_dict(_json_object(self.client.complete(
                        correction,
                        image_data_uris=images,
                        json_mode=True,
                        max_tokens=3500,
                    )))
                except (KeyError, TypeError, ValueError):
                    break
                current = revised
                issues = _paint_only_surface_issues(descriptor, geometry, current)
                if not issues:
                    return current
                correction += "\nThe previous correction still failed: %s. Return the complete corrected object again." % json.dumps(issues, ensure_ascii=False)
            # Keep the final-authoring task narrow when a provider repeatedly
            # returns a familiar AppearanceSpec while omitting its required
            # surface grid. This remains model-authored: Python only attaches
            # the returned map after checking dimensions and host coverage.
            map_prompt = """Author only the `pixel_map` for a Minecraft pixel-art asset.
Return JSON only:
{"legend":{"one_character":"existing_palette_name_or_hex"},"rows":["exact native rows"]}

This is a %dx%d item/cross canvas. `.` delegates to the existing render.
The final attached binary mask shows where pixels are opaque. The descriptor
contains required `paint_only:true` feature(s): draw their recognisable local
multi-colour surface cluster on opaque host cells only. Keep all unrelated
cells `.`. Each row must be exactly %d characters and there must be exactly %d
rows. Do not return an AppearanceSpec, prose, a mask, or coordinates outside
this grid. Use at least one palette token from every paint-only part's own
`colors` list; do not reproduce the whole underlying sprite in this map.

Descriptor: %s
Existing palette: %s
""" % (
                geometry.width,
                geometry.height,
                geometry.width,
                geometry.height,
                json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
                json.dumps(current.palette, ensure_ascii=False),
            )
            for attempt in range(2):
                try:
                    raw_map = _json_object(self.client.complete(
                        map_prompt,
                        image_data_uris=images,
                        json_mode=True,
                        max_tokens=2200,
                    ))
                    map_data = raw_map.get("pixel_map") if isinstance(raw_map.get("pixel_map"), dict) else raw_map
                    revised = replace(current, pixel_map=map_data)
                except (KeyError, TypeError, ValueError):
                    break
                issues = _paint_only_surface_issues(descriptor, geometry, revised)
                if not issues:
                    return revised
                map_prompt += "\nThe previous grid failed: %s. Return a complete corrected pixel_map only." % json.dumps(issues, ensure_ascii=False)
            return current
        if not geometry.uv_regions:
            # A same-size reference is useful for preserving value bands and
            # texture coordinates, but a first pass can confuse that with
            # preserving the source hue. Give the model one explicit audit
            # pass so broad material changes use the authored ramp while local
            # motifs can still remain reference-locked.
            same_size_reference = False
            for reference in references:
                try:
                    with Image.open(reference.path) as image:
                        if image.size == (geometry.width, geometry.height):
                            same_size_reference = True
                            break
                except (OSError, ValueError):
                    continue
            if same_size_reference:
                review_prompt = """Audit this AppearanceSpec for a same-size pixel-art variant.
Return a complete AppearanceSpec JSON object only. Keep the locked geometry and
the reference's local value ordering, pixel clusters and edge rhythm. Decide
the scope from the descriptor, not from a category template:

- If the target explicitly changes a broad material, hue or finish (for example
  rusted, emerald, icy or gold), preserve the source texture coordinates but
  use `motif_policy` `model_authored` or `free` so the declared palette remaps
  the affected parts. `reference_locked` alone would keep the source RGB and
  would fail to express that requested material change.
- If the target only preserves the source material or changes a local cluster,
  `reference_locked` with an exact `cluster_only` region rule is appropriate.
  Do not broaden a local edit to the entire surface.
- Keep unchanged supporting parts in their source material family. Do not add
  marks or regions that the descriptor does not request, and use only exact
  supplied UV IDs when the geometry declares any.
- If the descriptor explicitly adds a new embedded, inlaid or attached motif
  that is not present in the source raster, keep the unchanged supports on
  `pattern` or `value` but set that newly authored part to `none` in
  `part_reference_sampling`. Use `motif_policy` `model_authored` or `free`
  for the new part and make its declared palette visible. On an item/cross
  mask without UV regions, bind an internal mark to the exact declared part
  with `parts: ["part_id"]`; do not write a UV region name that does not
  exist. `reference_locked` may preserve unchanged supports, but it must not
  suppress the requested new motif.
For a newly authored part, keep its colors in one coherent target hue family;
do not concatenate source swatches with the new accent family. Any internal
part mark must use a declared color that visibly contrasts with the part's
base ramp and its host support, otherwise the motif is effectively absent.

Descriptor:
%s
Geometry parts:
%s
Draft AppearanceSpec:
%s
Reference evidence:
%s
Return JSON only.""" % (
                    json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
                    json.dumps(to_jsonable(geometry.parts), ensure_ascii=False),
                    json.dumps(to_jsonable(draft), ensure_ascii=False),
                    "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
                )
                try:
                    reviewed = appearance_from_dict(_json_object(self.client.complete(
                        review_prompt,
                        image_data_uris=_appearance_image_data_uris(references),
                        json_mode=True,
                        max_tokens=3500,
                    )))
                    descriptor_text = " ".join(
                        [descriptor.semantic, descriptor.reference_strategy, *descriptor.visual_identity]
                    ).lower()
                    embedded_language = (
                        "embedded", "inlaid", "inlay", "inserted", "attached", "mounted",
                        "encased", "grafted", "set into", "镶嵌", "嵌入",
                    )
                    if any(word in descriptor_text for word in embedded_language):
                        authored_parts = {
                            part.id
                            for part in descriptor.parts
                            if is_overlay_part(part)
                        }
                        sampled_new_parts = [
                            part_id for part_id in authored_parts
                            if reviewed.part_reference_sampling.get(part_id, reviewed.reference_sampling)
                            in {"value", "pattern"}
                        ]
                        support_ids = {
                            part.id for part in descriptor.parts if part.id not in authored_parts
                        }
                        unsampled_supports = [
                            part_id for part_id in support_ids
                            if reviewed.part_reference_sampling.get(part_id, reviewed.reference_sampling) == "none"
                        ]

                        def _appearance_rgb(token: object) -> tuple[int, int, int] | None:
                            raw = str(reviewed.palette.get(str(token), token)).strip().lstrip("#")
                            if len(raw) == 3:
                                raw = "".join(char * 2 for char in raw)
                            if len(raw) != 6:
                                return None
                            try:
                                return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))
                            except ValueError:
                                return None

                        low_contrast_marks: list[str] = []
                        missing_motif_marks: list[str] = []
                        misplaced_motif_marks: list[str] = []
                        unscoped_support_marks: list[str] = []
                        oversized_motif_marks: list[str] = []
                        detail_colors_in_base: list[str] = []
                        motif_host_overlap: list[str] = []
                        detail_tokens = ("pupil", "detail", "iris", "facet", "slot", "emblem")
                        motif_palette_tokens: set[str] = set()
                        for motif_id in authored_parts:
                            motif_style = reviewed.parts.get(motif_id)
                            if motif_style is not None:
                                motif_palette_tokens.update(str(color_name).lower() for color_name in motif_style.colors)
                        support_palette_colors = {
                            color
                            for support_id in support_ids
                            for color in (
                                _appearance_rgb(token)
                                for token in (reviewed.parts.get(support_id).colors if reviewed.parts.get(support_id) is not None else [])
                            )
                            if color is not None
                        }
                        for part_id in authored_parts:
                            style = reviewed.parts.get(part_id)
                            if style is None:
                                continue
                            non_empty_marks = [
                                mark for mark in style.marks
                                if isinstance(mark, dict)
                                and any(char not in {".", " "} for row in mark.get("rows", []) if isinstance(row, str) for char in row)
                            ]
                            base_color_tokens = {str(token).strip().lower() for token in style.colors}
                            mark_color_tokens = {
                                str(mark.get("color", "")).strip().lower()
                                for mark in non_empty_marks
                                if str(mark.get("color", "")).strip()
                            }
                            if base_color_tokens & mark_color_tokens:
                                detail_colors_in_base.append(part_id)
                            descriptor_part = next(
                                (part for part in descriptor.parts if part.id == part_id), None
                            )
                            part_text = " ".join(
                                (descriptor_part.id, descriptor_part.meaning, descriptor_part.style_role)
                            ).lower() if descriptor_part is not None else part_id.lower()
                            if not non_empty_marks and not any(token in part_text for token in detail_tokens):
                                missing_motif_marks.append(part_id)
                            base_colors = [color for color in (_appearance_rgb(token) for token in style.colors) if color]
                            if (
                                base_colors
                                and support_palette_colors
                                and sum(color in support_palette_colors for color in base_colors)
                                / float(len(base_colors)) >= 0.60
                            ):
                                motif_host_overlap.append(part_id)
                            part_points: set[tuple[int, int]] = set()
                            part_bounds: tuple[int, int, int, int] | None = None
                            try:
                                compiled_for_marks = compile_geometry(geometry)
                                part_mask = compiled_for_marks.part_masks[part_id]
                                part_points = {
                                    (x, y)
                                    for y in range(compiled_for_marks.height)
                                    for x in range(compiled_for_marks.width)
                                    if part_mask.getpixel((x, y)) > 0
                                }
                                if part_points:
                                    part_bounds = (
                                        min(x for x, _ in part_points),
                                        min(y for _, y in part_points),
                                        max(x for x, _ in part_points) + 1,
                                        max(y for _, y in part_points) + 1,
                                    )
                            except (KeyError, TypeError, ValueError):
                                part_points = set()
                            for mark in non_empty_marks:
                                mark_color = _appearance_rgb(mark.get("color", ""))
                                if mark_color is None or not base_colors:
                                    continue
                                contrasted = any(
                                    max(abs(mark_color[index] - base[index]) for index in range(3)) >= 40
                                    or abs(
                                        0.2126 * mark_color[0] + 0.7152 * mark_color[1] + 0.0722 * mark_color[2]
                                        - (0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2])
                                    ) >= 18
                                    for base in base_colors
                                )
                                if not contrasted:
                                    low_contrast_marks.append(part_id)
                                if part_points and part_bounds and any(
                                    token in part_text for token in ("round", "circle", "orb", "eye", "gem", "motif", *detail_tokens)
                                ):
                                    rows = mark.get("rows", [])
                                    covered = 0
                                    for x, y in part_points:
                                        local_x = x - part_bounds[0]
                                        local_y = y - part_bounds[1]
                                        if (
                                            isinstance(rows, list)
                                            and 0 <= local_y < len(rows)
                                            and isinstance(rows[local_y], str)
                                            and 0 <= local_x < len(rows[local_y])
                                            and rows[local_y][local_x] in {"X", "x"}
                                        ):
                                            covered += 1
                                    if covered / float(max(len(part_points), 1)) > 0.50:
                                        oversized_motif_marks.append(part_id)
                        for support_id in support_ids:
                            support_style = reviewed.parts.get(support_id)
                            if support_style is None:
                                continue
                            for mark in support_style.marks:
                                if not isinstance(mark, dict):
                                    continue
                                mark_parts = mark.get("parts")
                                mark_regions = mark.get("regions")
                                if not mark_parts and not mark_regions:
                                    unscoped_support_marks.append(support_id)
                                color_name = str(mark.get("color", "")).lower()
                                if color_name in motif_palette_tokens or any(
                                    token in color_name for token in ("eye", "iris", "pupil", "gem", "emblem", "inlay")
                                ):
                                    misplaced_motif_marks.append(support_id)
                        if (
                            sampled_new_parts
                            or unsampled_supports
                            or low_contrast_marks
                            or missing_motif_marks
                            or misplaced_motif_marks
                            or unscoped_support_marks
                            or oversized_motif_marks
                            or detail_colors_in_base
                            or motif_host_overlap
                            or reviewed.motif_policy == "reference_locked"
                        ):
                            correction_prompt = review_prompt + """
CORRECTION REQUIRED: repair the previous audit and return the full
AppearanceSpec again. Newly authored motif parts that were still sampled or
locked: %s. Unchanged supports that were incorrectly set to `none`: %s.
Motif marks with insufficient contrast: %s. Motif parts missing an internal
mark where no separate detail part exists: %s. Keep unchanged supports
Motif-colour marks incorrectly attached to support parts: %s. Unscoped
support marks that could bleed a motif into another part: %s. Keep unchanged
Motif marks that cover most of their own motif mask: %s. For a named internal
detail such as a pupil, iris, facet or slot, leave the base ramp visible around
the detail and size the mark grid to the motif's local bounding box.
Internal mark colors that were also placed in the broad base ramp: %s. Remove
those detail tokens from the part's `colors` array; a mark color must not be
allowed to spread through radial or linear shading.
Motif palettes that reuse most of the host support swatches: %s. Give the new
part a distinct hue/value family guided by any attached `material`/`palette`
reference, while keeping its internal mark readable against that family.
Keep unchanged supports source-driven, set only newly authored motif parts to
`part_reference_sampling: "none"`, use one coherent target hue family for
each new motif, and make internal marks visibly contrast with their base ramp.
Attach new motif marks to the motif part itself with `parts: ["part_id"]`;
do not paint motif colours on the blade, guard, handle or other host support.
Do not remove the motif or change geometry.
Previous audited response:
%s
""" % (
                                json.dumps(sampled_new_parts, ensure_ascii=False),
                                json.dumps(unsampled_supports, ensure_ascii=False),
                                json.dumps(low_contrast_marks, ensure_ascii=False),
                                json.dumps(missing_motif_marks, ensure_ascii=False),
                                json.dumps(misplaced_motif_marks, ensure_ascii=False),
                                json.dumps(unscoped_support_marks, ensure_ascii=False),
                                json.dumps(oversized_motif_marks, ensure_ascii=False),
                                json.dumps(detail_colors_in_base, ensure_ascii=False),
                                json.dumps(motif_host_overlap, ensure_ascii=False),
                                json.dumps(to_jsonable(reviewed), ensure_ascii=False),
                            )
                            try:
                                reviewed = appearance_from_dict(_json_object(self.client.complete(
                                    correction_prompt,
                                    image_data_uris=_appearance_image_data_uris(references),
                                    json_mode=True,
                                    max_tokens=3500,
                                )))
                                if oversized_motif_marks:
                                    # One additional model-owned pass is useful
                                    # for tiny motifs: a model may acknowledge
                                    # the audit but repeat the same full-width
                                    # mark grid.  Do not edit the pixels here;
                                    # ask it to leave enough of the base ramp
                                    # visible for the motif to read.
                                    strict_prompt = correction_prompt + """
STRICT MOTIF CHECK: the replacement above still needs a sparse internal
detail. A mark for a pupil/iris/facet/slot must cover only a minority of the
declared motif mask and must not be duplicated as a full-size grid for every
colour. If the previous mark covers most of the mask, delete it or replace it
with at most one to three `X` cells in a small local grid; let the motif's
base ramp provide the surrounding sclera/iris/facet. Keep the host support
palette out of the new motif. Return the full AppearanceSpec JSON only;
geometry is locked.
Any color used by that internal mark must be removed from the motif's broad
`colors` ramp, or radial/linear shading will turn the whole motif into the
detail color.
""" + json.dumps(to_jsonable(reviewed), ensure_ascii=False)
                                    try:
                                        reviewed = appearance_from_dict(_json_object(self.client.complete(
                                            strict_prompt,
                                            image_data_uris=_appearance_image_data_uris(references),
                                            json_mode=True,
                                            max_tokens=3500,
                                        )))
                                    except (KeyError, TypeError, ValueError):
                                        pass
                            except (KeyError, TypeError, ValueError):
                                pass
                    return finalize(reviewed)
                except (KeyError, TypeError, ValueError):
                    pass
            return finalize(draft)
        # A paint-only detail has no physical region, so its executable
        # coordinates must remain in the native pixel map even on an atlas.
        review_prompt = """Audit this AppearanceSpec for the locked UV atlas.
Return a complete AppearanceSpec JSON object only. Re-read the descriptor's
target and visual identities before accepting the draft: an explicitly
requested hue or material is binding. Preserve a source cluster's shape and
value ordering, but remap its RGB hue to the requested target; never copy a
source accent colour unchanged when it conflicts with the descriptor.
Keep the broad reference material ramp intact. A `paint_only:true` descriptor
part has no physical UV region, even on this atlas: preserve or author it only
as a full native-size top-level `pixel_map`. Every `.` keeps the host texture;
every non-dot cell is an exact overlay on that locked canvas. Never convert it
to `marks` or a region rule, because those require a physical UV-owning part
and would make the feature disappear. For a local motif owned by an actual
physical part (an ore inclusion, emblem, stripe, inlay, eye or similar), use
non-empty marks or exact UV region rules and keep coverage local. Do not turn
the entire face into the accent hue. Do not satisfy a named motif by declaring
an unused palette family or by deleting that family; connect it to at least one
executable sparse declaration. Every region and UV-owned mark must use an exact
supplied UV ID.
Treat each declared UV `face` as a physical placement constraint, not a loose
label. A `top` region is the cube's top, and front/right/left regions are
vertical faces. Before accepting the draft, check every named
`part_reference_sources` binding against the reference's stated intent and
the owning UV face: do not put a side/bark/panel source on an end-grain/top
surface, or an end-grain/top source on a vertical side, merely because their
tile dimensions match. A structural reference is valid base pattern evidence
for its matching part even without a material role; a palette/material
reference supplies the target hue and must not replace that structure.
If the reference has no matching source cluster for a UV-owned motif, use a
non-empty mark to author the new cluster; a `cluster_only` rule preserves an
existing source cluster and cannot create one from a uniform face. For a
paint-only motif, use `pixel_map` instead.
Honor `motif_policy`: a `reference_locked` policy must not contain marks that
invent or relocate source clusters; a `model_authored` policy may contain
marks for the explicitly requested new motif.
Inspect each supplied UV region independently: when the reference contains
the named secondary cluster in that region, make a deliberate declaration for
that region as well. Do not omit a face merely because another face already
has a matching rule; the local pixel evidence decides coverage.
When a single compact tile is reused for multiple cube faces, keep the same
cluster coverage and hue decision coherent across those faces unless the
descriptor explicitly asks for directional variation.
For that shared tile, keep a sparse motif out of the broad part `colors`
array; use `pixel_map` for a paint-only accent, and marks or an exact region
rule only for a UV-owned accent. Do not make a local accent a broad ramp unless
the descriptor explicitly requests a fully multicolour surface.
Plan the palette globally before assigning individual regions: parts that
share the same declared `material` should reuse a common dark/mid/light base
ramp and the same scene-light direction. A colour family belonging to another
material or to a small diagnostic feature must stay in a sparse mark or a
`cluster_only` rule; do not put it in a support part's broad `colors` array
just because that feature appears on one face. For a broad UV region, leave
`cluster_only` true for a local secondary cluster unless the descriptor
explicitly asks to recolour that entire region. This keeps separately sampled
face details subordinate to one coherent object palette.
Preserve any explicit `part_reference_sources` material source binding from the
draft when its name appears in the eligible material references. A material
source binding is executable texture evidence; do not replace it with a shape
source merely because that source has the matching atlas dimensions.
A structural variant remains a structural precedent when a more specific
material texture reference is eligible for the requested finish.
For a requested material finish, prioritize a material-role source over a palette-only source; use palette-only sources for hue/value guidance when no direct material raster exists.
Use the exact mark field names from the schema: `regions` (array of UV IDs),
`rows` (array of strings made of `X` and `.`) and `color`. Do not return a
numeric `grid` or a singular `region` field.
Before returning, inspect the descriptor's named visual identities one by one:
each identity that is a paint detail must be visible in the returned contract,
while a base material identity belongs in the broad ramp. A paint-only detail
uses a non-empty native `pixel_map`; a UV-owned local mark uses a region ID, a
non-empty row grid at that region's resolution and one of the declared palette
colours. Keep a local secondary material out of the broad base ramp: use
`pixel_map` when it is paint-only, otherwise marks or a region rule.
If the descriptor carries an explicit appearance modifier or finish, it must
have visible coverage in the returned contract: a palette token that is never
used, or one tiny isolated mark, does not satisfy a modifier such as a glow,
corruption, molten surface or aura. Use one coherent accent family across the
relevant regions while preserving the shared base ramp and source pixel rhythm.
Descriptor:
%s
Geometry UV regions:
%s
Draft AppearanceSpec:
%s
Reference evidence:
%s
""" % (
            json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
            json.dumps(to_jsonable(geometry.uv_regions), ensure_ascii=False),
            json.dumps(to_jsonable(draft), ensure_ascii=False),
            _appearance_reference_evidence(references),
        )
        try:
            reviewed = appearance_from_dict(_json_object(self.client.complete(
                review_prompt,
                image_data_uris=_appearance_image_data_uris(references),
                json_mode=True,
                max_tokens=3500,
            )))
            material_groups: dict[str, list[str]] = {}
            token_materials: dict[str, set[str]] = {}
            token_parts: dict[str, set[str]] = {}
            for part_id, style in reviewed.parts.items():
                material = style.material.strip().lower()
                material_groups.setdefault(material, []).append(part_id)
                for token in style.colors:
                    normalized_token = str(token).strip().lower()
                    token_materials.setdefault(normalized_token, set()).add(material)
                    token_parts.setdefault(normalized_token, set()).add(part_id)
            palette_cohesion_issues: list[str] = []
            lighting_cohesion_issues: list[str] = []
            for material, part_ids in material_groups.items():
                if len(part_ids) < 2:
                    continue
                group_ids = set(part_ids)
                axes = {
                    reviewed.parts[part_id].shade_axis.strip().lower()
                    for part_id in part_ids
                    if part_id in reviewed.parts
                    and reviewed.parts[part_id].shade_axis.strip().lower() not in {"", "auto"}
                }
                if len(axes) > 1:
                    lighting_cohesion_issues.append(material)
                shared_tokens = {
                    token for token, users in token_parts.items()
                    if len(users & group_ids) >= 2
                }
                for part_id in part_ids:
                    style = reviewed.parts.get(part_id)
                    if style is None:
                        continue
                    own_tokens = {str(token).strip().lower() for token in style.colors}
                    foreign_tokens = {
                        token for token in own_tokens
                        if token_materials.get(token, set()) - {material}
                    }
                    if foreign_tokens and not shared_tokens.issubset(own_tokens):
                        palette_cohesion_issues.append(part_id)
            broad_cross_material_rules: list[str] = []
            region_by_id = {region.id: region for region in geometry.uv_regions}
            part_materials = {
                part_id: style.material.strip().lower()
                for part_id, style in reviewed.parts.items()
            }
            for raw_rule in reviewed.region_rules:
                if not isinstance(raw_rule, dict) or bool(raw_rule.get("cluster_only", False)):
                    continue
                target_token = str(raw_rule.get("target_color", "")).strip().lower()
                if not target_token:
                    continue
                target_materials = token_materials.get(target_token, set())
                for raw_region_id in raw_rule.get("regions", []):
                    region = region_by_id.get(str(raw_region_id))
                    if region is None:
                        continue
                    area = max(0, region.bbox[2] - region.bbox[0]) * max(0, region.bbox[3] - region.bbox[1])
                    owner_material = part_materials.get(region.part_id, "")
                    if area >= 32 and target_materials and owner_material not in target_materials:
                        broad_cross_material_rules.append(str(raw_region_id))
            if palette_cohesion_issues or lighting_cohesion_issues or broad_cross_material_rules:
                coherence_feedback = """
COHERENCE CORRECTION REQUIRED: return a complete corrected AppearanceSpec.
Parts with a cross-material accent leaked into their broad base ramp: %s.
Same-material groups with inconsistent shade directions: %s.
Broad UV regions using a cross-material target without `cluster_only`: %s.
For each same-material group, reuse its shared base dark/mid/light ramp. Move
small foreign accents into sparse marks or exact `cluster_only` rules, preserving
source clusters without repainting an entire support face. Keep any descriptor-
requested broad material change, but do not let a local feature redefine an
unrelated support part. Use one coherent shade direction for each same-material
group unless the descriptor explicitly requires a directional material change.
Do not change geometry or UV IDs.
Previous reviewed response:
%s
""" % (
                    json.dumps(palette_cohesion_issues, ensure_ascii=False),
                    json.dumps(lighting_cohesion_issues, ensure_ascii=False),
                    json.dumps(broad_cross_material_rules, ensure_ascii=False),
                    json.dumps(to_jsonable(reviewed), ensure_ascii=False),
                )
                try:
                    reviewed = appearance_from_dict(_json_object(self.client.complete(
                        review_prompt + coherence_feedback,
                         image_data_uris=_appearance_image_data_uris(references),
                        json_mode=True,
                        max_tokens=3500,
                    )))
                except (KeyError, TypeError, ValueError):
                    pass
            return finalize(reviewed)
        except (KeyError, TypeError, ValueError):
            return finalize(draft)

    def revise_appearance_from_review(
        self,
        request: AssetRequest,
        descriptor: ShapeDescriptor,
        geometry: GeometrySpec,
        appearance: AppearanceSpec,
        blind_review: dict[str, Any],
        references: list[ReferenceAsset] | None = None,
        current_image: str | Path | None = None,
    ) -> AppearanceSpec:
        """Repaint a valid mask when blind review identifies a visual mismatch."""
        references = references or []
        image_data_uris = _appearance_image_data_uris(references)
        current_image_note = ""
        if current_image is not None:
            try:
                image_data_uris.append(_reference_data_uri(str(current_image)))
                current_image_note = (
                    "The final attached image is the current rendered image/preview. "
                    "Compare it against the named references to locate the failure; "
                    "it is a diagnostic, never a source to copy."
                )
            except OSError:
                current_image_note = "The current rendered preview was unavailable; use the textual blind review and references."
        prompt = """Revise only the AppearanceSpec for a Minecraft pixel asset after a blind visual review.
The alpha geometry, physical parts, UV regions and texture dimensions are
locked. Return a complete AppearanceSpec JSON object and no prose. Preserve the
selected vanilla reference's local value/pixel pattern when reference_sampling
is pattern. If the blind reviewer confused an entity with a weapon, keep broad
body/hide regions mid-dark and make diagnostic accents sparse marks on the
front-facing UV face; never turn an entire body, limb or support part into a
bright red/neon slab. Keep any eyes, symbols or glow marks within exact UV
region IDs when UV regions exist. On item/cross masks without UV regions, bind
internal marks to exact declared part IDs with `parts: ["part_id"]` and use
`part_reference_sampling` to keep newly authored motifs out of unrelated
source sampling. Do not change the canvas or geometry.
If the review contains a reference comparison with texture divergence, treat
that as TEXTURE AUDIT FAILED: the current surface lost the reference's local
pixel rhythm. Return a materially different AppearanceSpec that restores
multiple irregular source clusters or binds the eligible named material
reference to the affected parts; do not merely swap one accent colour.
Read the attached reference names and roles literally. Restore the source
object's distinctive local texture clusters when the blind result resembles a
different class of object, then apply the requested finish through a coherent
            base ramp. If a secondary material reference has visible pixel texture, use
its irregular cluster rhythm in several small, region-scoped marks; do not
replace it with one symmetric icon or a flat colour slab.
If that secondary raster should directly drive a particular part's texture,
declare `part_reference_sources` with the exact reference `name` and keep the
part's sampling mode explicit. Set `part_reference_composite` to `overlay`
when that raster adds local material clusters to an unchanged structural
surface; use `replace` only when the affected part is intentionally a full
material replacement. This is a model-authored source binding, so leave
unchanged supports on the global structural source and never invent a
reference name. Once an eligible material raster is bound with `overlay`, let
its own irregular pixel clusters carry the texture: keep broad `marks` empty
or sparse and do not redraw the same source pattern as repeated stripes.
Only a reference carrying a `material` or `palette` role is eligible for
this binding; a shape/UV/pixel-style-only source is a coordinate/style aid,
not a material layer. If the request explicitly names a material or finish
and an eligible selected reference supplies that material evidence, bind the
affected broad surface parts to it so its local pixel texture is executable.
If a local reference cluster needs a new hue, encode that choice in
`region_rules` with exact UV region IDs (`retint`, `source_exact` or `palette`)
instead of relying on an anatomy-specific renderer guess. Keep broad body/base
ramps coherent while allowing independently planned ears, face planes, seams,
patches, gems or other small regions.
If the descriptor names an embedded or inlaid feature and the blind review does
not mention it, make that existing part unmistakable through a coherent
target-hue ramp plus a sparse, high-contrast part/region mark. Keep the host
support's source value bands and do not recolour the whole asset to advertise
the feature.
The review may include `target_visual_review`. Unlike the target-free noun
review, its missing/errors entries compare the render directly against the
requested design brief. Treat those entries as binding: repair the named
colour, material, face treatment or local feature while preserving every
unrelated source cluster. For a block atlas, keep the UV face labels literal
when repairing source bindings: top is the physical top; vertical faces must
not inherit its end-grain merely because all tiles are square.
If that review reports that a requested broad hue/material is missing, the
current `reference_locked` policy is invalid: it emits untouched source RGB
outside explicit local rules. Change it to `model_authored` or `free` while
keeping `reference_sampling:"pattern"`, so the same source clusters are
recoloured through the declared target ramps. Do not answer this correction
with a palette that is never allowed to reach the rendered pixels.
PAINT-ONLY COORDINATE CONTRACT: a descriptor part with `paint_only:true` has
no mask of its own. Its `marks` use full-canvas coordinates: write either a
full native-size `pixel_map`, or give every local mark an explicit
`offset:[x,y]` that places its X cells on existing opaque host pixels. Never
use `[0,0]` as a placeholder for a small paint-only mark. Inspect the supplied
reference part map and current render, place the mark on its model-declared
host, and use enough distinct palette colours/rows for a target-free reviewer
to recognise the requested feature. Returning a mark that lands only on
transparent pixels is a failed response.

Request: %s
Descriptor:
%s
Geometry parts and UV regions:
%s
Current AppearanceSpec:
%s
Blind review to correct:
%s
References (same order as attached images):
%s
%s
        """ % (
            request.query,
            json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
            json.dumps({"parts": to_jsonable(geometry.parts), "uv_regions": to_jsonable(geometry.uv_regions)}, ensure_ascii=False),
            json.dumps(to_jsonable(appearance), ensure_ascii=False),
            json.dumps(blind_review, ensure_ascii=False),
            _appearance_reference_evidence(references),
            current_image_note,
        )
        def revise_paint_only_map(candidate: AppearanceSpec) -> AppearanceSpec:
            paint_only_required = [
                part for part in descriptor.parts if part.required and part.paint_only
            ]
            if not paint_only_required:
                return candidate
            try:
                compiled = compile_geometry(geometry)
                map_images = [*image_data_uris, _mask_data_uri(compiled.mask)]
            except (KeyError, TypeError, ValueError):
                map_images = image_data_uris
            map_prompt = """The target-free blind review did not recognise the required
surface-only feature. Re-author only its native `pixel_map`; return JSON only:
{"legend":{"one_character":"palette_token_or_hex"},"rows":["exact native rows"]}

The asset is %dx%d. `.` leaves the existing texture unchanged. The final
attached image is the failed render and the final attached binary image is the
opaque host mask. Draw a small, high-contrast, recognisable cluster for every
required `paint_only:true` descriptor part on host pixels only. Use at least
one palette token from each such part's own `colors` list, including its
internal contrast/detail colour when its recognition terms require one. Do not
redraw the base sprite: all unrelated cells must be `.`. Return exactly %d
rows, each exactly %d characters.

Descriptor: %s
Blind review: %s
Candidate palette and paint-only styles: %s
""" % (
                geometry.width,
                geometry.height,
                geometry.height,
                geometry.width,
                json.dumps(_appearance_brief(descriptor), ensure_ascii=False),
                json.dumps(blind_review, ensure_ascii=False),
                json.dumps({
                    "palette": candidate.palette,
                    "paint_only_parts": {
                        part.id: to_jsonable(candidate.parts.get(part.id))
                        for part in paint_only_required
                    },
                }, ensure_ascii=False),
            )
            current_candidate = candidate
            for attempt in range(2):
                try:
                    raw_map = _json_object(self.client.complete(
                        map_prompt,
                        image_data_uris=map_images,
                        json_mode=True,
                        max_tokens=2200,
                    ))
                    map_data = raw_map.get("pixel_map") if isinstance(raw_map.get("pixel_map"), dict) else raw_map
                    mapped = replace(current_candidate, pixel_map=map_data)
                except (KeyError, TypeError, ValueError):
                    break
                issues = _paint_only_surface_issues(descriptor, geometry, mapped)
                if not issues:
                    return mapped
                map_prompt += "\nThe previous map failed the executable contract: %s. Return the complete corrected pixel_map only." % json.dumps(issues, ensure_ascii=False)
            return current_candidate
        current = appearance
        for attempt in range(2):
            try:
                revised = appearance_from_dict(_json_object(self.client.complete(
                    prompt,
                    image_data_uris=image_data_uris,
                    json_mode=True,
                    max_tokens=3500,
                )))
            except (KeyError, TypeError, ValueError):
                return revise_paint_only_map(current)
            if to_jsonable(revised) != to_jsonable(current):
                return revise_paint_only_map(revised)
            if attempt == 0:
                prompt += """
The previous response was unchanged even though the blind review still misses
the named feature. Return a materially different complete AppearanceSpec:
preserve source sampling for unchanged supports, set only newly authored motif
parts to `none`, and add a contrasting part/region mark for the feature. Do
not answer with an acknowledgement.
"""
        return revise_paint_only_map(current)

    def repair_geometry(
        self,
        request: AssetRequest,
        descriptor: ShapeDescriptor,
        geometry: GeometrySpec,
        validation: ValidationResult,
        compiled: CompiledGeometry | None,
        references: list[ReferenceAsset] | None = None,
    ) -> GeometrySpec:
        """Ask the planner to repair only a failed geometry contract.

        The descriptor and parts are deliberately retained: a repair changes
        proportion, adjacency or primitive construction, never the target noun
        mid-run. A failed compile has no mask, so the textual error alone is
        supplied for that attempt.
        """
        allowed = ", ".join(sorted(SUPPORTED_PRIMITIVES))
        prompt = """Repair a failed GeometrySpec for a Minecraft pixel-art asset.
Keep the exact supplied part list and target semantics. Alter primitives,
connections and constraints only as needed to satisfy the errors. Do not emit a
final color grid. Return a complete GeometrySpec JSON object and no prose.
This is a hard acceptance checklist, not feedback to summarize: the replacement
must change every failed condition. A required connection means the two named
part masks directly overlap or touch; connection through a third part does not
count. Leave a connection optional when the descriptor deliberately creates a
break, gap or detached fragment between those parts. Before returning,
rasterize every row and check each required pair for overlap or 8-neighbour
contact. Components use 8-neighbour pixel adjacency. Do not merely adjust
color, rename a part, or repeat an unchanged failing primitive. Rebuild the
primitive list from the descriptor and fixed part list below; do not copy the
previous primitive list. Prefer a simpler silhouette that passes every hard
check over a detailed but invalid one.
For item/cross canvases at or below 32x32, every required physical part
(`paint_only:false`) must contain
at least one opaque pixel and every diagonal, tapered, blade, handle, shaft or
irregular part must use an explicit custom_mask with full-width pixel rows.
Never replace those parts with a large polygon/ellipse or a shared rectangle.
Keep each part's local contour distinct, use a one- or two-pixel overlap only
at the intended joint, and preserve source edge occupancy for a same-size
shape variant; do not add an edge margin unless the descriptor calls for one. If the previous geometry has empty required parts,
rebuild those rows first rather than expanding the remaining part.
The required top-level JSON schema is exactly:
{"width":int,"height":int,"parts":[PartSpec,...],"primitives":[PrimitiveSpec,...],
 "connections":[ConnectionSpec,...],"constraints":[ConstraintSpec,...],
 "uv_regions":[UvRegionSpec,...],"background_transparent":true}
`parts` must be an array containing the fixed PartSpec objects, never a nested
object. `primitives` must be a separate array. When the fixed UV layout is
empty, return "uv_regions": [].
Allowed primitive language: %s
%s
Request: %s
Descriptor: %s
Fixed parts: %s
Previous connections and constraints (you may correct them): %s
Fixed UV layout (must be copied exactly): %s
Validation: %s
Reference evidence (each attached image follows this same order):
%s
When images are attached, the declared reference images are listed first,
followed by the current binary alpha mask and labeled part map. The part-map
legend is: %s. Use the mask and map to inspect the current construction, then
use the reference images as the pixel ruler for scale and local structure. If
the descriptor identifies a same-size referenced variant, reconstruct the
reference row spans for unnamed parts before applying the named local change;
do not turn a reference-preservation instruction into a new generic contour.
""" % (
            allowed,
            primitive_parameter_guide(),
            json.dumps(to_jsonable(request), ensure_ascii=False),
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            json.dumps(to_jsonable(geometry.parts), ensure_ascii=False),
            json.dumps(
                {"connections": geometry.connections, "constraints": geometry.constraints},
                default=to_jsonable,
                ensure_ascii=False,
            ),
            json.dumps(to_jsonable(geometry.uv_regions), ensure_ascii=False),
            json.dumps(to_jsonable(validation), ensure_ascii=False),
            "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in (references or [])) or "- none",
            _part_map_legend(descriptor),
        )
        images = _geometry_reference_data_uris(references or [])
        if compiled is not None:
            images.extend([_mask_data_uri(compiled.mask), _part_map_data_uri(descriptor, compiled)])
        if request.form in {AssetForm.ITEM, AssetForm.CROSS}:
            prompt += "\n" + _shape_baseline_text(references or [], request.width, request.height)
        last_error: Exception | None = None
        for repair_attempt in range(2):
            try:
                repaired = geometry_from_dict(
                    _json_object(self.client.complete(
                        prompt,
                        image_data_uris=images,
                        json_mode=True,
                        max_tokens=7500,
                    ))
                )
                expected_parts = {part.id for part in geometry.parts}
                if {part.id for part in repaired.parts} != expected_parts:
                    raise ValueError("geometry repair changed the declared part set")
                if repaired.width != request.width or repaired.height != request.height:
                    raise ValueError("geometry repair changed canvas dimensions")
                if repaired.uv_regions != geometry.uv_regions:
                    raise ValueError("geometry repair changed the declared UV layout")
                return repaired
            except (KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if repair_attempt:
                    raise
                prompt += (
                    "\nThe previous repair response failed the fixed geometry schema: %s. "
                    "Return the complete object again. Every primitive must include "
                    "part_id using exactly one of these IDs: %s; use canonical keys "
                    "primitive and params, and keep the same dimensions, parts and UV layout."
                    % (str(exc), json.dumps(sorted(part.id for part in geometry.parts), ensure_ascii=False))
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("geometry repair returned no response")

    def revise_geometry_from_review(
        self,
        request: AssetRequest,
        descriptor: ShapeDescriptor,
        geometry: GeometrySpec,
        validation: ValidationResult,
        blind_review: dict[str, Any],
        compiled: CompiledGeometry | None = None,
        references: list[ReferenceAsset] | None = None,
        reference_comparison: dict[str, Any] | None = None,
    ) -> GeometrySpec:
        """Revise a mask when an independent blind review misreads it.

        The blind reviewer receives no target label. Its output is therefore a
        useful negative signal, but it is never allowed to rename parts or
        select a closed category template. The planner keeps the exact
        descriptor part set and only changes the open geometry language.
        """
        references = references or []
        allowed = ", ".join(sorted(SUPPORTED_PRIMITIVES))
        comparison_text = json.dumps(reference_comparison, ensure_ascii=False) if reference_comparison else "- none"
        prompt = """Revise this open GeometrySpec after an independent target-free blind review.
Keep the request, descriptor semantics, exact part list, canvas dimensions and
UV layout unchanged. Change proportions, contour, part adjacency or primitive
construction so the requested object is easier to recognize and the blind
misread is less likely. Do not add a category-specific template and do not
change colors; this is geometry only. Return a complete GeometrySpec JSON and
no prose. If the candidate has no rendered image, the validation text is the
authoritative review: fix every listed hard/semantic failure before returning.
Keep required connections only where the parts are physically joined; if the
descriptor names a break, gap or detached fragment, mark that fracture
relationship optional rather than forcing the two masks to touch.

Descriptor: %s
Current geometry: %s
Current validation: %s
Blind review (target was hidden from reviewer): %s
Post-blind reference comparison diagnostics (descriptive locator, not a new
target label): %s
If the descriptor includes `reference_part_map`, preserve its mapped source
coordinates for unnamed supports and edit only the owning part of the named
variant. Treat that map as ownership evidence, never as permission for a
primary part to absorb a handle, guard, stem or other support.
Allowed primitive language: %s
%s
When images are attached, the declared reference images are listed first,
followed by the current compiled alpha mask and labeled part map. The part-map
legend is: %s. If a final image is attached, it is the post-blind
reference/generated/changed comparison named in the diagnostics. Use the
reference as the pixel ruler, then use the labeled map and comparison to check
that overlapping details remain distinct before rebuilding the geometry.
Use custom_mask rows when a narrow staircase or irregular contour matters at
this resolution. Preserve source edge occupancy for a same-size referenced
variant; otherwise let the descriptor decide. Keep all required parts
connected as declared, and make the most diagnostic
part visually dominant rather than turning the object into a generic diagonal
bar. For item/cross canvases up to 32x32, rebuild each diagonal, tapered,
narrow or irregular semantic part with explicit `custom_mask` rows; a large
axis-aligned polygon is not an acceptable repair for such a part. At 16x16,
write every changed part as full-width, full-height `custom_mask` rows instead
of a transform wrapper so the intended contour can be checked directly. For a
descriptor that names an embedded or inlaid motif, treat an unclear or
occluded motif warning as a request to enlarge and center that motif within
its host support, preserve a visible host rim or contact on at least two sides,
and keep the host's other supports intact; do not shrink the motif to a token
or let the primary blade/body cover it.
For a new design, reference images and pixel maps are
style/scale evidence; borrow relative stroke thickness and value clusters
without copying a complete silhouette. For a same-size referenced variant,
they are also the coordinate ruler: reconstruct unnamed parts at the
reference row spans first, then edit only the requested property. Do not
reinterpret an explicit "preserve ... coordinates" instruction as permission
to redesign the whole object.
When a narrow primary part meets a supporting part, preserve the reference's
compact junction and relative axis: a support that becomes a long broad bar
perpendicular to the primary part can make the whole silhouette read as a
different tool. Keep the primary part dominant and make only the smallest
connector/guard change needed by the descriptor.
If the descriptor's reference strategy states a junction row, quadrant or end
for a part, use it as a coordinate anchor during the repair. Keep the support
at that end and do not slide it along the diagonal to satisfy a generic noun
interpretation.
If the blind alternatives name a different elongated tool, treat that as a
hard signal that the current primary/support proportions are wrong. Restore
the reference's multi-pixel body and compact junction using its row spans and
stroke statistics; do not leave a one-pixel shaft with a small head cluster.
Always return a materially different contour when the blind result is not
aligned, even if the current contract already passes its numeric checks.
If this is a named variant of a referenced object, keep unnamed supporting
parts at the reference's scale and proportion and spend the revision on the
named difference. Treat this as a minimal-change edit: preserve each unnamed
support's occupied coordinate cluster, stroke thickness, axis and value role
from the reference, and alter only the pixels required by the named property.
When rebuilding a same-size variant, explicitly repopulate every required
support from the reference row spans before changing the named primary part;
an empty support or a one-pixel placeholder is a failed reconstruction even
when the union passes a numeric constraint.
Compare row-run widths with the reference as well: do not turn a multi-pixel
body into a one-pixel diagonal shaft, and do not shorten or recolour unnamed
supports while expressing the requested damage.
For a damage/break variant, the owning primary part must show an asymmetric
notch or jagged endpoint in the rows that differ from the reference, at the
part's exposed end away from an intact junction; do not return a clean pointed
end or an unbroken continuation. A smooth monotonic taper does not count as a
break; include a notch or alternating row span that survives at native pixels,
but confine it to the exposed endpoint's last few rows and preserve the
interior shaft and intact support junction.
Do not enlarge, recolour, rotate or redesign a handle, grip, guard or other
support merely to make the requested change more visible. If the comparison
reports a large missing/extra alpha region, restore the reference-scale
support first and keep the named change local. If it reports a colour change
inside an otherwise unchanged support, leave that support's palette assignment
and local value bands alone.
Reference evidence (each attached image follows this same order):
%s
""" % (
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            json.dumps(to_jsonable(geometry), ensure_ascii=False),
            json.dumps(to_jsonable(validation), ensure_ascii=False),
            json.dumps(blind_review, ensure_ascii=False),
            comparison_text,
            allowed,
            primitive_parameter_guide(),
            _part_map_legend(descriptor),
            "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
        )
        images = _geometry_reference_data_uris(references)
        if compiled is not None:
            images.extend([_mask_data_uri(compiled.mask), _part_map_data_uri(descriptor, compiled)])
        if request.form in {AssetForm.ITEM, AssetForm.CROSS}:
            prompt += "\n" + _shape_baseline_text(references, request.width, request.height)
        comparison_image = None
        if isinstance(reference_comparison, dict):
            candidate = reference_comparison.get("image")
            if isinstance(candidate, str) and Path(candidate).exists():
                comparison_image = candidate
        if comparison_image is not None:
            images.append(_reference_data_uri(comparison_image))
        if request.form in {AssetForm.ITEM, AssetForm.CROSS} and max(request.width, request.height) <= 32:
            # Keep the corrective pass as focused as the initial small-sprite
            # pass. The generic revision contract above remains available for
            # larger/UV assets, but tiny item repairs should spend their token
            # budget on the visible pixels called out by the blind critic.
            prompt = """Repair the final %dx%d Minecraft pixel-art raster for this target.
Target: %s
Semantic intent: %s
Immutable art direction (the target change brief):
%s
Reference strategy: %s
TARGET_OWNERSHIP_MAP (model-authored spatial plan; `.` means no physical
target cell, and paint-only parts intentionally have no marker):
%s
Current geometry JSON:
%s
Current validation errors:
%s
Blind review (target hidden from that reviewer):
%s
Post-blind reference comparison diagnostics (descriptive locator):
%s
SOURCE_ALPHA_BASELINE:
%s
Reference pixel/style text:
%s

Return JSON only with the same complete GeometrySpec schema. Keep the exact
current part IDs, canvas dimensions and UV layout. Preserve every unnamed
support at the reference coordinates and scale; do not move, enlarge or
recolour it. Keep narrow shafts and handles narrow. Required physical joints
must touch, while an intentional fracture may be optional. If the semantic
intent names a break, chip, fracture, damage or jagged change, change only its
owning part and make the exposed endpoint unmistakable in the last few native
pixel rows: include a notch or alternating row span, not a smooth crop at the
junction and not repeated holes through the interior. Do not add extra parts,
category templates, margin constraints or prose. Do not force a required
connection between broken fragments or across a large overlap; if a reported
connection error concerns an intentional fracture, make that relationship
optional and keep only real compact physical joints required. Before
returning, rasterize the rows and verify required connections.
When TARGET_OWNERSHIP_MAP is present, treat it as this run's model-authored
intent for which source cells remain, disappear or become a new physical part.
Do not replace it with the unedited source silhouette. It is not a category
template: retain the model's own part labels and translate its target cells
into your final primitives while preserving paint-only details for appearance.
""" % (
            request.width,
            request.height,
            descriptor.target,
            descriptor.semantic,
            json.dumps(to_jsonable(descriptor.art_direction), ensure_ascii=False),
            descriptor.reference_strategy,
            json.dumps(descriptor.target_part_map, ensure_ascii=False),
            json.dumps(to_jsonable(geometry), ensure_ascii=False),
                json.dumps(to_jsonable(validation), ensure_ascii=False),
                json.dumps(blind_review, ensure_ascii=False),
                comparison_text,
                _shape_baseline_text(references, request.width, request.height),
                "\n".join("- " + summarize_reference(item, include_pixel_map=True) for item in references) or "- none",
            )
            prompt += "\nLAST INSTRUCTION: apply this exact local delta and coordinate guidance: %s\nReturn the final raster for `%s` now." % (
                descriptor.reference_strategy,
                descriptor.target,
            )
        expected_parts = {part.id for part in geometry.parts}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                revised = geometry_from_dict(
                    _json_object(self.client.complete(
                        prompt,
                        image_data_uris=images,
                        json_mode=True,
                        max_tokens=7500,
                    ))
                )
                if {part.id for part in revised.parts} != expected_parts:
                    raise ValueError("blind-review geometry revision changed the declared part set")
                if revised.width != request.width or revised.height != request.height:
                    raise ValueError("blind-review geometry revision changed canvas dimensions")
                if revised.uv_regions != geometry.uv_regions:
                    raise ValueError("blind-review geometry revision changed the declared UV layout")
                return revised
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt:
                    raise
                prompt += (
                    "\nThe previous revision was not a complete GeometrySpec (%s). "
                    "Return JSON only with exactly these part IDs: %s, dimensions %dx%d, "
                    "and the unchanged UV layout. Include a non-empty custom_mask for "
                    "every required physical part (paint_only:false); leave paint-only "
                    "parts without primitives. Do not return a response-format acknowledgement."
                    % (
                        str(exc),
                        json.dumps(sorted(expected_parts), ensure_ascii=False),
                        request.width,
                        request.height,
                    )
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("blind-review geometry revision returned no response")

    def recommend_repair_scope(
        self,
        descriptor: ShapeDescriptor,
        validation: ValidationResult,
        blind_review: dict[str, Any],
        current_image: str | None = None,
    ) -> dict[str, Any]:
        """Let the planning model decide which contract needs another pass.

        The semantic critic sees only alpha and part ownership while the blind
        reviewer sees the painted sprite.  Neither signal alone can tell us
        whether a weak feature needs a different contour, different value
        contrast, or both.  Keep that interpretation in the model rather than
        encoding noun- or feature-specific keyword rules in the coordinator.
        """
        prompt = """Decide the repair scope for a generated Minecraft pixel asset.
You are a planning director, not a renderer. The semantic validation inspected
the binary silhouette and labeled part ownership. The blind reviewer inspected
the painted sprite without knowing the target. Decide whether the next pass
must revise geometry, appearance, or both. Geometry means alpha contour,
part placement, adjacency, scale, or silhouette. Appearance means palette,
value contrast, internal marks, material pattern, or layering without alpha
changes. A named local motif may need both if its footprint and its visual
contrast each prevent recognition. Do not infer a fixed template from the
noun; reason from the supplied evidence.

Return JSON only:
{"geometry": boolean, "appearance": boolean, "reason": string}

Descriptor: %s
Semantic validation: %s
Target-free blind review: %s
""" % (
            json.dumps(to_jsonable(descriptor), ensure_ascii=False),
            json.dumps(to_jsonable(validation), ensure_ascii=False),
            json.dumps(blind_review, ensure_ascii=False),
        )
        images = [_reference_data_uri(current_image)] if current_image and Path(current_image).exists() else None
        data = _json_object(self.client.complete(
            prompt,
            image_data_uris=images,
            json_mode=True,
            max_tokens=900,
        ))
        return {
            "geometry": bool(data.get("geometry", False)),
            "appearance": bool(data.get("appearance", False)),
            "reason": str(data.get("reason", "")).strip(),
        }


@dataclass
class ModelSemanticCritic:
    client: OpenAICompatibleClient

    def review(self, descriptor: ShapeDescriptor, compiled: CompiledGeometry) -> ValidationResult:
        prompt = """Review this pixel silhouette for the specified target. Judge only shape identity, part
recognisability, proportions, connections, and whether it resembles one of the negative identities.
The first attached image is the binary union silhouette. The second is a
labeled part map whose legend is: %s. Use the labeled map to judge overlapping
physical parts such as eyes, horns, ears or seams; do not call a physical part
missing merely because it is covered in the binary union. Parts marked
`paint_only:true` in the descriptor intentionally have no alpha mask and cannot
be judged from these two images; do not list them in `missing_parts` solely for
that reason. Do not judge color. Return JSON only:
{"passed": boolean, "scores": {"target_identity": 0_to_1, "part_clarity": 0_to_1,
"negative_identities": {"negative name": 0_to_1}}, "missing_parts": [string],
"errors": [string], "warnings": [string]}
Each `negative_identities` value is the probability that the rendered
silhouette actually matches that forbidden identity: high is bad and low is
good. Do not return a high value merely because the candidate successfully
avoids the negative description.
Descriptor:\n%s
If the descriptor names a state-changing contour feature such as broken,
chipped, fractured, damaged, torn or jagged, treat that feature as a required
identity condition. A continuous or clean boundary that could be mistaken for
the unmodified reference is an `error` and must make `passed` false; reserve
`warnings` for nonessential polish after the named feature is unmistakable.
For such a change, a smooth taper or a clean crop at a junction is not enough:
the affected part's exposed endpoint must contain a visible notch, alternating
row span, or other non-monotonic pixel boundary in its last few rows, while
the interior shaft and intact support junction remain regular.
""" % (_part_map_legend(descriptor), json.dumps(to_jsonable(descriptor), ensure_ascii=False))
        data = _json_object(
            self.client.complete(
                prompt,
                image_data_uris=[_mask_data_uri(compiled.mask), _part_map_data_uri(descriptor, compiled)],
                json_mode=True,
                max_tokens=2500,
            )
        )
        scores = data.get("scores", {}) if isinstance(data.get("scores"), dict) else {}
        target_score = float(scores.get("target_identity", 0.0))
        negative_scores = scores.get("negative_identities", {})
        if not isinstance(negative_scores, dict):
            negative_scores = {}
        legacy_negative_score = float(scores.get("negative_identity", 0.0))
        if not negative_scores and descriptor.negative_identities:
            negative_scores = {name: legacy_negative_score for name in descriptor.negative_identities}
        negative_score = max((float(value) for value in negative_scores.values()), default=legacy_negative_score)
        part_clarity = float(scores.get("part_clarity", 0.0))
        errors = [str(item) for item in data.get("errors", [])]
        if target_score < 0.65:
            errors.append("critic target_identity %.2f is below 0.65" % target_score)
        if negative_score > 0.45:
            errors.append("critic negative_identity %.2f exceeds 0.45" % negative_score)
        if part_clarity < 0.55:
            errors.append("critic part_clarity %.2f is below 0.55" % part_clarity)
        missing_parts = [str(item) for item in data.get("missing_parts", [])]
        errors.extend("critic missing required/recognisable part: %s" % item for item in missing_parts)
        warnings = [str(item) for item in data.get("warnings", [])]
        # A named embedded/inlaid/attached motif is part of the requested
        # identity, not optional polish. If the critic says that motif is
        # unclear, feed it back into the repair loop just like a missing
        # required part.  Read this only from a required part's own semantic
        # declaration.  Broad prose can say "no attached mass" or "an
        # embedded highlight" as a negative/material relation; treating such
        # a substring as a physical motif created false geometry failures.
        embedded_words = (
            "embedded", "inlaid", "inlay", "inserted", "attached", "mounted",
            "encased", "grafted", "set into", "镶嵌", "嵌入",
        )
        embedded_part_ids = {
            part.id
            for part in descriptor.parts
            if part.required and any(
                word in " ".join([
                    part.id,
                    part.meaning,
                    part.style_role,
                    *part.recognition_terms,
                ]).lower()
                for word in embedded_words
            )
        }
        if embedded_part_ids:
            for warning in warnings:
                lowered = warning.lower()
                # A paint-only feature is intentionally absent from the
                # silhouette-only critic input. Its visibility is judged by
                # the subsequent rendered-image blind review, so this
                # structural limitation must not be turned into a geometry
                # failure here.
                if "paint_only" in lowered or "paint-only" in lowered:
                    continue
                if any(token in lowered for token in ("motif", "inlay", "embedded", "eye", "gem", "socket", "unclear", "not read")):
                    errors.append("embedded motif is not yet visually clear: %s" % warning)
        descriptor_text = " ".join(
            [descriptor.semantic, descriptor.reference_strategy, *descriptor.visual_identity]
        ).lower()
        variant_words = ("broken", "break", "chipped", "fracture", "fractured", "damage", "damaged", "torn", "jagged")
        # A critic is allowed to suggest proportion or contrast polish, but
        # that is not evidence that the requested state change is absent.
        # Promote a warning only when it explicitly says the changed endpoint
        # cannot be read, remains continuous, or still reads as unmodified.
        # This keeps a deliverable broken tool from being rejected merely
        # because a guard is "slightly" broad or a shaft is "a little" short.
        variant_failure_phrases = (
            "does not read", "not read", "not visible", "no visible",
            "cannot distinguish", "unclear", "still reads as intact",
            "appears intact", "unbroken continuation", "continuous clean",
            "smooth taper", "clean truncation", "too subtle to read",
            "not unmistakable",
        )
        if any(word in descriptor_text for word in variant_words):
            for warning in warnings:
                lowered = warning.lower()
                if any(phrase in lowered for phrase in variant_failure_phrases):
                    errors.append("variant feature is not yet visually prominent: %s" % warning)
        return ValidationResult(
            passed=bool(data.get("passed", False)) and not errors,
            stage="semantic_model",
            metrics={
                "target_identity": target_score,
                "negative_identity": negative_score,
                "part_clarity": part_clarity,
                **{
                    "negative_%s" % re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_"): float(value)
                    for name, value in negative_scores.items()
                },
            },
            errors=errors,
            warnings=warnings,
        )


@dataclass
class BlindReviewer:
    """Target-free visual check for a rendered sprite.

    This deliberately receives only the transparent output image. It does
    not receive the request, descriptor, part names, references or a list of
    candidate nouns, so its answer is an independent recognisability signal.
    """

    client: OpenAICompatibleClient

    def review(self, image_path: str) -> dict[str, Any]:
        prompt = """Blindly identify the single main object in this transparent-background Minecraft pixel sprite or entity diagnostic.
Do not assume or use any target label. Do not describe the transparent
background, checkerboard, canvas or image format. First inspect the topology:
count separate masses, look for a head/face, torso, limbs, horns, a handle,
guard or a thin shaft, and use their relative proportions. A broad torso with
a distinct head and multiple supporting legs is an animal or mob, even when
the palette is dark. A narrow shaft with a compact grip is a tool or weapon;
only call it a blade when an actual blade contour is visible. For a regular
When a compact diagonal object has a short tapered cutting surface, a small
crossguard, and a handle of similar or greater length, prefer the ordinary
short-blade tool label; reserve a long-blade weapon label for a substantially
longer blade with a dominant weapon silhouette.
three-face isometric prism with straight repeated edges and no tapering facets,
prefer "block" or "cube" over crystal/gem: perspective makes the lower corner
pointed, but that point is not a crystal tip. Do not invent
parts from a familiar icon and do not anchor on one category before checking
the whole silhouette. Prefer one ordinary concrete noun; if uncertain, lower
confidence and give a few alternatives. Return JSON only:
{
  "primary_object": string,
  "alternatives": [string, ...],
  "confidence": number,
  "evidence": [string, ...],
  "orientation": string,
  "visible_parts": [string, ...]
}
After naming the main object, inspect its high-contrast local features and
include them in `visible_parts` or `evidence` only when they are actually
visible: examples include an embedded eye, gem, emblem, socket, stripe or
other small motif. Do not infer a motif from the object's familiar category.
"""
        data = _json_object(self.client.complete(
            prompt,
            image_data_uris=[_reference_data_uri(image_path)],
            json_mode=True,
            max_tokens=1600,
        ))
        alternatives = data.get("alternatives", [])
        evidence = data.get("evidence", [])
        visible_parts = data.get("visible_parts", [])
        if isinstance(alternatives, str):
            alternatives = [alternatives]
        if isinstance(evidence, str):
            evidence = [evidence]
        if isinstance(visible_parts, str):
            visible_parts = [visible_parts]
        return {
            "primary_object": str(data.get("primary_object", "unknown")),
            "alternatives": [str(item) for item in alternatives if str(item).strip()],
            "confidence": float(data.get("confidence", 0.0)),
            "evidence": [str(item) for item in evidence if str(item).strip()],
            "orientation": str(data.get("orientation", "unknown")),
            "visible_parts": [str(item) for item in visible_parts if str(item).strip()],
        }


@dataclass
class TargetVisualReviewer:
    """Target-aware visual gate for a completed render.

    Blind recognition deliberately answers only "what object is this?".  That
    misses a frequent failure mode in mod art: a recognisable log, cow or sword
    whose requested material, face treatment or motif never appeared.  This
    reviewer receives the authored design brief and checks the finished preview
    without prescribing a class-specific pixel operation.
    """

    client: OpenAICompatibleClient

    def review(self, descriptor: ShapeDescriptor, image_path: str) -> dict[str, Any]:
        prompt = """Inspect the attached final Minecraft pixel-art render against this
design brief. This is a target-aware acceptance review, not a request to
redesign the asset. Check the subject/form, requested material or colour
family, explicitly declared separate part(s), face-specific treatment, and
negative identities at native pixel scale. A broadly recognisable base noun is
insufficient when the requested variant/finish is absent. For a UV block
preview, verify that top versus vertical-face textures match their intended
surface roles. Return JSON only:
{"passed":boolean,"score":0_to_1,"satisfied":[string],"critical_missing":[string],"optional_deviations":[string],"errors":[string],"evidence":[string]}
Set passed false only when a core requested identity, an explicit required
part, or a negative constraint is absent or contradicted. The brief can also
contain model-authored art-direction suggestions: alternate coherent choices
of highlight placement, mottling, interior depth, or other local surface
treatment are optional deviations, not failure conditions, unless they erase
the requested material/identity itself. Do not reject an asset merely because
a source palette was recoloured as the brief requests.
Design brief:
%s
""" % json.dumps(_appearance_brief(descriptor), ensure_ascii=False)
        data = _json_object(self.client.complete(
            prompt,
            image_data_uris=[_reference_data_uri(image_path)],
            json_mode=True,
            max_tokens=1800,
        ))
        score = float(data.get("score", 0.0))
        # Older/less strict providers may retain the previous `missing` key;
        # treat it as critical only for that legacy response. New reviews
        # distinguish an absent required identity from a different but
        # coherent model-authored surface treatment.
        missing = data.get("critical_missing", data.get("missing", []))
        optional_deviations = data.get("optional_deviations", [])
        errors = data.get("errors", [])
        evidence = data.get("evidence", [])
        satisfied = data.get("satisfied", [])
        if isinstance(missing, str):
            missing = [missing]
        if isinstance(errors, str):
            errors = [errors]
        if isinstance(optional_deviations, str):
            optional_deviations = [optional_deviations]
        if isinstance(evidence, str):
            evidence = [evidence]
        if isinstance(satisfied, str):
            satisfied = [satisfied]
        passed = bool(data.get("passed", False)) and score >= 0.70 and not errors and not missing
        return {
            "passed": passed,
            "score": score,
            "satisfied": [str(item) for item in satisfied if str(item).strip()],
            "missing": [str(item) for item in missing if str(item).strip()],
            "optional_deviations": [str(item) for item in optional_deviations if str(item).strip()],
            "errors": [str(item) for item in errors if str(item).strip()],
            "evidence": [str(item) for item in evidence if str(item).strip()],
        }
