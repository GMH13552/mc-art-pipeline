"""Render deterministic pixel material inside an already approved geometry mask."""

from __future__ import annotations

import colorsys
from hashlib import sha256
from collections import Counter
from random import Random

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .contracts import AppearanceSpec, GeometrySpec, PartAppearance, ReferenceAsset, ReferenceRole
from .geometry import CompiledGeometry


def _hex_to_rgb(token: str) -> tuple[int, int, int]:
    clean = token.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(char * 2 for char in clean)
    if len(clean) != 6 or any(char not in "0123456789abcdefABCDEF" for char in clean):
        raise ValueError("invalid color: %r" % token)
    return int(clean[:2], 16), int(clean[2:4], 16), int(clean[4:], 16)


# A palette token the model never defined used to abort a whole generation:
# an item form reaches colour resolution on the very first painted part. The
# renderer now degrades to a neutral grey -- the same fallback the planner-side
# resolver already uses -- while the undefined token stays visible in the
# asset's own appearance.json for a reviewer to find.
_UNRESOLVED_COLOR = (96, 96, 96)

# How far a "replace" reference is pulled toward the authored palette. The rest
# keeps the source's own values, which is what preserves its grain. 1.0 would
# snap every pixel onto the palette and erase the texture; 0.0 would ignore the
# palette entirely and lose family control. The value below keeps both readable.
_REPLACE_RETINT_STRENGTH = 0.6


def _palette_color(palette: dict[str, str], token: object) -> tuple[int, int, int]:
    """Resolve a palette token or literal hex colour without ever aborting."""
    if not isinstance(token, str):
        return _UNRESOLVED_COLOR
    try:
        return _hex_to_rgb(palette.get(token, token))
    except (TypeError, ValueError):
        return _UNRESOLVED_COLOR


def _resolve_colors(style: PartAppearance, palette: dict[str, str]) -> list[tuple[int, int, int]]:
    resolved: list[tuple[int, int, int]] = []
    for color in style.colors:
        resolved.append(_palette_color(palette, color))
    if not resolved:
        raise ValueError("part appearance needs at least one color")
    return resolved


def _luma(color: tuple[int, int, int]) -> float:
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]


def _chroma(color: tuple[int, int, int]) -> float:
    return float(max(color) - min(color))


def _hue_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    first_hue = colorsys.rgb_to_hsv(*(channel / 255.0 for channel in first))[0]
    second_hue = colorsys.rgb_to_hsv(*(channel / 255.0 for channel in second))[0]
    distance = abs(first_hue - second_hue)
    return min(distance, 1.0 - distance)


def _target_hue_ramp(
    palette: dict[str, str],
    target: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Collect the model-authored colours that belong to one target hue family."""
    target_chroma = _chroma(target)
    candidates: list[tuple[int, int, int]] = [target]
    for raw in palette.values():
        try:
            color = _hex_to_rgb(raw)
        except (TypeError, ValueError):
            continue
        if color in candidates:
            continue
        if target_chroma < 12.0:
            compatible = _chroma(color) < 24.0
        else:
            compatible = (
                _chroma(color) >= max(10.0, target_chroma * 0.35)
                and _hue_distance(color, target) <= 0.10
            )
        if compatible:
            candidates.append(color)
    return sorted(candidates, key=_luma)


def _cluster_only_pixel_is_secondary(
    source_pixel: tuple[int, int, int, int] | None,
    dominant: tuple[int, int, int] | None,
    source_count: int,
    region_count: int,
) -> bool:
    """Decide whether a rare source colour is a material cluster.

    Frequency alone mistakes an isolated dark/light shading band for a local
    motif (a common pattern in vanilla stone and ore tiles).  Keep the model's
    ``cluster_only`` intent generic by requiring either a clear chroma
    separation or an extreme value excursion from the dominant material.
    Neutral mid-value shades therefore remain part of the base ramp, while
    saturated deposits and deliberately extreme highlights can still be
    retinted.
    """
    if (
        source_pixel is None
        or source_pixel[3] < 8
        or dominant is None
        or region_count <= 0
        or source_count > max(4, region_count * 0.18)
    ):
        return False
    source = source_pixel[:3]
    dominant_chroma = _chroma(dominant)
    source_chroma = _chroma(source)
    chroma_separation = source_chroma - dominant_chroma
    luma_gap = abs(_luma(source) - _luma(dominant))
    return (
        source_chroma >= 24.0 and chroma_separation >= 10.0
    ) or (
        source_chroma >= 16.0 and chroma_separation >= 18.0
    ) or luma_gap >= 72.0


def _feature_reference_color(
    source_pixel: tuple[int, int, int, int] | None,
    dominant: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """Preserve an explicit local source outlier when no region rule overrides it.

    This is a colour-statistics fallback only. It does not inspect anatomy names
    or palette token names; a model that wants a new hue declares ``retint`` in
    ``AppearanceSpec.region_rules`` and therefore remains in control.
    """
    if source_pixel is None or source_pixel[3] < 8 or dominant is None:
        return None
    source = source_pixel[:3]
    if abs(_luma(source) - _luma(dominant)) < 30.0 and abs(_chroma(source) - _chroma(dominant)) < 22.0:
        return None
    return source

def _colorful_pattern_outlier(
    source_pixel: tuple[int, int, int, int] | None,
    dominant: tuple[int, int, int] | None,
) -> tuple[int, int, int] | None:
    """Keep saturated authored clusters on otherwise broad UV surfaces.

    Entity atlases sometimes place a small pink/colored patch in a body
    face's UV rectangle (the vanilla cow's underside is one example).  The
    body still needs recolouring as a material, so neutral lighting pixels are
    mapped through the target ramp; only a clearly colourful local outlier is
    copied verbatim.
    """
    if source_pixel is None or source_pixel[3] < 8 or dominant is None:
        return None
    source = source_pixel[:3]
    source_chroma = _chroma(source)
    if source_chroma < 24.0:
        return None
    dominant_chroma = _chroma(dominant)
    luma_gap = abs(_luma(source) - _luma(dominant))
    if source_chroma - dominant_chroma >= 18.0 or luma_gap >= 45.0:
        return source
    return None


def _cluster_detail_color(
    source_pixel: tuple[int, int, int, int] | None,
    dominant: tuple[int, int, int] | None,
    source_count: int,
    region_count: int,
    feature_region: bool,
) -> tuple[int, int, int] | None:
    """Recover a small local colour cluster without knowing its semantic name.

    Custom models frequently call an ear, cheek or seam ``part_3``.  The
    region name then gives us no feature cue, but the texture still contains a
    compact secondary cluster.  This detector uses cluster size plus value /
    chroma distance, so it remains generic and does not turn a broad material
    into a copy of the source palette.
    """
    if source_pixel is None or source_pixel[3] < 8 or dominant is None or region_count <= 0:
        return None
    fraction = source_count / float(region_count)
    if fraction > 0.45:
        return None
    source = source_pixel[:3]
    luma_gap = abs(_luma(source) - _luma(dominant))
    chroma_gap = abs(_chroma(source) - _chroma(dominant))
    colorful = _chroma(source) >= 24.0
    if feature_region and (luma_gap >= 22.0 or chroma_gap >= 18.0):
        return source
    if colorful and (luma_gap >= 35.0 or chroma_gap >= 18.0):
        return source
    return None


def _ensure_texture_ramp(
    colors: list[tuple[int, int, int]],
    material: str,
    width: int,
    height: int,
) -> list[tuple[int, int, int]]:
    """Give reference-free surfaces enough value bands to carry pixel detail."""
    if len(colors) >= 2 or material.lower() in {"accent", "glow", "eye", "symbol"}:
        return colors
    if not colors or width * height < 4:
        return colors
    base = colors[0]
    spread = max(base) - min(base)
    dark_factor = 0.62 if max(base) > 150 else 0.72
    dark = tuple(max(0, min(255, round(channel * dark_factor))) for channel in base)
    light = tuple(
        max(0, min(255, round(channel + max(16, (255 - channel) * 0.24))))
        for channel in base
    )
    if spread < 8 and dark == base and light == base:
        return colors
    return [dark, base, light]


def _ensure_contrast_ramp(
    colors: list[tuple[int, int, int]],
    material: str,
) -> list[tuple[int, int, int]]:
    """Expand a collapsed material ramp while leaving authored accents untouched.

    The model's material label is the only semantic input here.  A renderer
    should not need to know whether a part is called an ear, muzzle, blade or
    sleeve in order to preserve a usable value hierarchy.
    """
    if not colors or material.lower() in {"accent", "glow", "eye", "symbol", "flesh"}:
        return colors
    if len(colors) == 1:
        base = colors[0]
        dark = tuple(max(0, round(channel * 0.68)) for channel in base)
        light = tuple(min(255, round(channel * 1.24 + 4)) for channel in base)
        return [dark, base, light]
    lumas = [_luma(color) for color in colors]
    if max(lumas) - min(lumas) >= 34.0:
        return colors
    low_source = colors[0]
    high_source = colors[-1]
    low = tuple(max(0, min(255, round(channel * 0.68))) for channel in low_source)
    high = tuple(
        max(0, min(255, round(channel + max(24, (255 - channel) * 0.20))))
        for channel in high_source
    )
    count = len(colors)
    ramp: list[tuple[int, int, int]] = []
    for index in range(count):
        ratio = index / float(max(count - 1, 1))
        ramp.append(tuple(round(low[channel] + (high[channel] - low[channel]) * ratio) for channel in range(3)))
    return ramp


def _region_rule_for(rules: list[dict[str, object]], region_id: str) -> dict[str, object] | None:
    """Return the model-authored rule for one exact UV region ID."""
    for raw in rules:
        if not isinstance(raw, dict):
            continue
        regions = raw.get("regions", [])
        if isinstance(regions, str):
            regions = [regions]
        if isinstance(regions, list) and region_id in {str(item) for item in regions}:
            return raw
    return None


def _retint_reference_pixel(
    source_pixel: tuple[int, int, int, int] | None,
    target: tuple[int, int, int],
    dominant: tuple[int, int, int] | None,
    strength: float,
    target_ramp: list[tuple[int, int, int]] | None = None,
) -> tuple[int, int, int] | None:
    """Move a source cluster into a model-selected hue while retaining value."""
    if source_pixel is None or source_pixel[3] < 8:
        return None
    source_luma = _luma(source_pixel[:3])
    dominant_luma = _luma(dominant) if dominant is not None else 128.0
    relative_value = source_luma / float(max(dominant_luma, 1.0))
    # Compress the source-to-base ratio before applying the target hue. A
    # bright accent can be several times lighter than a stone/wood base; a
    # linear factor with a hard upper clamp collapses every accent band into
    # one neon colour. The square-root curve keeps the ordering visible while
    # preventing extreme source highlights from blowing out the target ramp.
    factor = max(0.55, min(1.10, 0.70 + 0.20 * min(max(relative_value, 0.0), 4.0) ** 0.5))
    target_for_value = target
    if target_ramp:
        # Keep the target hue selected by the model, but choose its authored
        # dark/mid/light band that best matches the source value. This avoids
        # flattening a bright reference highlight into the rule's mid colour.
        target_for_value = min(target_ramp, key=lambda color: abs(_luma(color) - source_luma))
    retinted = tuple(max(0, min(255, round(channel * factor))) for channel in target_for_value)
    blend = max(0.0, min(1.0, strength))
    return tuple(round(source_pixel[channel] * (1.0 - blend) + retinted[channel] * blend) for channel in range(3))


def _part_has_interior(mask: Image.Image, width: int, height: int) -> bool:
    """Whether a part is thick enough for a one-pixel inner edge to read.

    The inner-edge rule exists so a small part such as an ear, a nose or a
    handle still gets a readable contour when the planner authored no outline.
    On a part only one or two pixels thick every pixel *is* an edge, so the
    rule painted the whole part in its darkest stop: a five-pixel nocked arrow
    came out entirely in the ramp's darkest colour instead of the icy accent
    the planner had just authored for it.
    """
    for y in range(height):
        for x in range(width):
            if mask.getpixel((x, y)) == 0:
                continue
            if (
                (x > 0 and mask.getpixel((x - 1, y)) == 0)
                or (x + 1 < width and mask.getpixel((x + 1, y)) == 0)
                or (y > 0 and mask.getpixel((x, y - 1)) == 0)
                or (y + 1 < height and mask.getpixel((x, y + 1)) == 0)
            ):
                continue
            return True
    return False


def _shade_value(x: int, y: int, width: int, height: int, axis: str,
                 bounds: tuple[int, int, int, int] | None = None) -> float:
    if bounds is None:
        left, top, right, bottom = 0, 0, width, height
    else:
        left, top, right, bottom = bounds
    x_ratio = (x - left) / float(max(right - left - 1, 1))
    y_ratio = (y - top) / float(max(bottom - top - 1, 1))
    if axis in {"top", "bottom"}:
        value = 1.0 - y_ratio if axis == "top" else y_ratio
    elif axis in {"left", "right"}:
        value = 1.0 - x_ratio if axis == "left" else x_ratio
    elif axis in {"bottom_left_to_top_right", "diagonal_up"}:
        value = (x_ratio + (1.0 - y_ratio)) / 2.0
    elif axis in {"top_left_to_bottom_right", "diagonal_down"}:
        value = (x_ratio + y_ratio) / 2.0
    elif axis in {"radial", "center", "centre"}:
        # A radial part (eye, gem, orb, bead, etc.) should get its value from
        # distance to its own mask/UV bounds rather than inheriting the canvas
        # diagonal.  The model chooses the axis; the renderer only evaluates
        # that declared shading intent.
        center_x = (left + right - 1) / 2.0
        center_y = (top + bottom - 1) / 2.0
        half_width = max((right - left - 1) / 2.0, 1.0)
        half_height = max((bottom - top - 1) / 2.0, 1.0)
        distance = ((x - center_x) / half_width) ** 2 + ((y - center_y) / half_height) ** 2
        value = max(0.0, 1.0 - (distance ** 0.5) / (2.0 ** 0.5))
    elif axis in {"none", "flat", "uniform"}:
        value = 0.5
    else:
        value = (x_ratio + (1.0 - y_ratio)) / 2.0
    return max(0.0, min(1.0, value))


def _color_for_pixel(
    colors: list[tuple[int, int, int]],
    shade: float,
    noise: float,
    rng: Random,
    continuous: bool = False,
) -> tuple[int, int, int]:
    if len(colors) == 1:
        return colors[0]
    jitter = (rng.random() - 0.5) * noise
    # Equal-width bins keep a reference's dark/mid/light clusters visible at
    # 8-16 pixel scales; banker's rounding otherwise collapses the upper bin.
    normalized = max(0.0, min(1.0, shade + jitter))
    if continuous:
        position = normalized * (len(colors) - 1)
        lower = int(position)
        upper = min(len(colors) - 1, lower + 1)
        ratio = position - lower
        return tuple(
            round(colors[lower][channel] + (colors[upper][channel] - colors[lower][channel]) * ratio)
            for channel in range(3)
        )
    index = min(len(colors) - 1, int(normalized * len(colors)))
    return colors[index]


def _part_seed(seed: int, part_id: str) -> int:
    digest = sha256((str(seed) + ":" + part_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _alpha_iou(reference: Image.Image, target_mask: Image.Image | None) -> float | None:
    """Return alpha overlap with the already locked geometry, when available."""
    if target_mask is None or reference.size != target_mask.size:
        return None
    target = target_mask.convert("L")
    candidate = reference.getchannel("A")
    target_points = {
        (x, y)
        for y in range(target.height)
        for x in range(target.width)
        if target.getpixel((x, y)) >= 8
    }
    candidate_points = {
        (x, y)
        for y in range(candidate.height)
        for x in range(candidate.width)
        if candidate.getpixel((x, y)) >= 8
    }
    union = target_points | candidate_points
    return len(target_points & candidate_points) / float(max(len(union), 1))


def _select_reference(
    references: list[ReferenceAsset] | None,
    width: int,
    height: int,
    target_mask: Image.Image | None = None,
) -> tuple[Image.Image, ReferenceAsset, str] | None:
    """Select the raster belonging to the locked geometry.

    Routing may attach a structural source together with a same-sized local
    material/motif reference.  Choosing the first material reference makes its
    pixel bands control unrelated support parts.  Alpha agreement with the
    approved geometry is a generic disambiguator; roles only break ties.
    """
    if not references:
        return None
    candidates: list[tuple[float, int, Image.Image, ReferenceAsset, str]] = []
    for index, reference in enumerate(references):
        if ReferenceRole.NEGATIVE in reference.roles:
            continue
        if not any(role in {
            ReferenceRole.SHAPE,
            ReferenceRole.PIXEL_STYLE,
            ReferenceRole.MATERIAL,
            ReferenceRole.PALETTE,
        } for role in reference.roles):
            continue
        try:
            loaded = Image.open(reference.path).convert("RGBA")
        except (OSError, ValueError):
            continue
        image = loaded
        scoring_image = loaded
        mode = "exact"
        if loaded.size != (width, height):
            if loaded.height != height or loaded.width >= width or width % max(loaded.width, 1) != 0:
                continue
            scoring_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            for left in range(0, width, loaded.width):
                scoring_image.alpha_composite(loaded, (left, 0))
            mode = "face_tile"
        overlap = _alpha_iou(scoring_image, target_mask)
        role_score = 0
        if ReferenceRole.SHAPE in reference.roles:
            role_score += 3
        if ReferenceRole.PIXEL_STYLE in reference.roles:
            role_score += 2
        if ReferenceRole.MATERIAL in reference.roles:
            role_score += 1
        score = (overlap if overlap is not None else 0.0) * 100.0 + role_score
        candidates.append((score, -index, image, reference, mode))
    if not candidates:
        return None
    _score, _order, image, reference, mode = max(candidates, key=lambda item: (item[0], item[1]))
    return image, reference, mode


def _reference_image(
    references: list[ReferenceAsset] | None,
    width: int,
    height: int,
    target_mask: Image.Image | None = None,
) -> Image.Image | None:
    """Load a pixel-style/material reference suitable for this canvas.

    A reference is evidence, never an alpha source.  Matching dimensions are
    required so a cube-UV atlas is sampled at its declared UV coordinates and
    an item reference cannot silently distort a different canvas.

    A compact horizontal face strip is the one useful exception: a 16x16
    material sample can be sampled in normalized coordinates for each face of
    a wider block atlas. This is a generic size relationship, not a named
    block/entity template; all other mismatched canvases are ignored.
    """
    selected = _select_reference(references, width, height, target_mask=target_mask)
    return selected[0] if selected is not None else None


def _reference_value(pixel: tuple[int, int, int, int]) -> float | None:
    red, green, blue, alpha = pixel
    if alpha < 8:
        return None
    # Minecraft's deliberately small palettes are better represented by
    # perceptual luma than by hue; hue is supplied by the new palette.
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0


def _reference_palette(reference: Image.Image) -> list[tuple[int, int, int]]:
    colors = Counter(
        (red, green, blue)
        for red, green, blue, alpha in reference.get_flattened_data()
        if alpha >= 8
    )
    return [color for color, _ in sorted(
        colors.items(),
        key=lambda item: (0.2126 * item[0][0] + 0.7152 * item[0][1] + 0.0722 * item[0][2], -item[1]),
    )]


def _reference_pattern_value(pixel: tuple[int, int, int, int], palette: list[tuple[int, int, int]]) -> float | None:
    if pixel[3] < 8 or not palette:
        return None
    source = pixel[:3]
    index = min(
        range(len(palette)),
        key=lambda candidate: sum((source[channel] - palette[candidate][channel]) ** 2 for channel in range(3)),
    )
    return index / float(max(len(palette) - 1, 1))


def _reference_sampling_bbox(reference: Image.Image) -> tuple[int, int, int, int]:
    """Choose the native tile used when a material raster is resampled.

    Minecraft animated textures are commonly stored as a vertical stack of
    equal square frames (for example a 16x48 lava strip). Treating that whole
    strip as one image stretches three unrelated frames across a cow face,
    block face or item part. A compact, aspect-ratio-only heuristic keeps the
    first native tile; ordinary rasters continue to use their opaque bounds.
    This is independent of object names and works for any stacked material.
    """
    width, height = reference.size
    if width > 0 and height > width and height % width == 0 and height // width <= 8:
        return (0, 0, width, width)
    if height > 0 and width > height and width % height == 0 and width // height <= 8:
        return (0, 0, height, height)
    return _reference_opaque_bbox(reference)


def _covered_block(
    x: int, y: int,
    target_bounds: tuple[int, int, int, int],
    source_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """The source rectangle one target pixel covers."""
    left, top, right, bottom = target_bounds
    source_left, source_top, source_right, source_bottom = source_bbox
    width = float(max(right - left, 1))
    height = float(max(bottom - top, 1))
    span_x = float(max(source_right - source_left, 1))
    span_y = float(max(source_bottom - source_top, 1))
    x0 = source_left + int((x - left) * span_x / width)
    x1 = source_left + int((x - left + 1) * span_x / width)
    y0 = source_top + int((y - top) * span_y / height)
    y1 = source_top + int((y - top + 1) * span_y / height)
    return (x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))


def _block_average(
    reference: Image.Image,
    block: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Mean colour of the opaque pixels in a source block, or None if empty."""
    left, top, right, bottom = block
    totals = [0, 0, 0]
    counted = 0
    for y in range(max(0, top), min(reference.height, bottom)):
        for x in range(max(0, left), min(reference.width, right)):
            pixel = reference.getpixel((x, y))
            if pixel[3] < 8:
                continue
            totals[0] += pixel[0]
            totals[1] += pixel[1]
            totals[2] += pixel[2]
            counted += 1
    if not counted:
        return None
    return (
        int(round(totals[0] / counted)),
        int(round(totals[1] / counted)),
        int(round(totals[2] / counted)),
        255,
    )


def _sample_reference_pixel(reference: Image.Image, x: int, y: int,
                            target_bounds: tuple[int, int, int, int],
                            source_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = target_bounds
    source_left, source_top, source_right, source_bottom = source_bbox
    # Downscaling must average the covered source block. Point-sampling its
    # corner aliases a material: a 16x16 plank tile pressed into an 8x8 face
    # sampled source rows 0,2,4,6,9,11,13,15, which skips two of the tile's four
    # plank separators (rows 3 and 7) and keeps rows 11 and 15, so the regular
    # grain came out as an irregular checker.
    block = _covered_block(x, y, target_bounds, source_bbox)
    if (block[2] - block[0]) * (block[3] - block[1]) > 1:
        averaged = _block_average(reference, block)
        if averaged is not None:
            return averaged
    x_ratio = (x - left) / float(max(right - left - 1, 1))
    y_ratio = (y - top) / float(max(bottom - top - 1, 1))
    source_x = source_left + round(max(0.0, min(1.0, x_ratio)) * max(source_right - source_left - 1, 0))
    source_y = source_top + round(max(0.0, min(1.0, y_ratio)) * max(source_bottom - source_top - 1, 0))
    pixel = reference.getpixel((source_x, source_y))
    if pixel[3] >= 8:
        return pixel
    # Item silhouettes have transparent gaps. Find the nearest opaque source
    # pixel instead of replacing every gap with the same flat mean value.
    for radius in range(1, max(reference.width, reference.height)):
        candidates = {
            (source_x - radius, source_y), (source_x + radius, source_y),
            (source_x, source_y - radius), (source_x, source_y + radius),
            (source_x - radius, source_y - radius), (source_x + radius, source_y - radius),
            (source_x - radius, source_y + radius), (source_x + radius, source_y + radius),
        }
        for candidate_x, candidate_y in sorted(candidates, key=lambda point: (point[1], point[0])):
            if 0 <= candidate_x < reference.width and 0 <= candidate_y < reference.height:
                candidate = reference.getpixel((candidate_x, candidate_y))
                if candidate[3] >= 8:
                    return candidate
    return None


def _sample_item_reference_pixel(reference: Image.Image, x: int, y: int,
                                 target_bounds: tuple[int, int, int, int],
                                 source_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    """Prefer authored canvas coordinates for a same-sized item reference.

    A generated item may move or reshape a part, so relative sampling remains
    the fallback. When a generated pixel overlaps an opaque source pixel at
    the same canvas coordinate, that pixel is stronger evidence for the
    preserved part's material and local value band. New pixels still inherit a
    nearby source rhythm through the relative sampler.
    """
    if 0 <= x < reference.width and 0 <= y < reference.height:
        pixel = reference.getpixel((x, y))
        if pixel[3] >= 8:
            return pixel
    return _sample_reference_pixel(reference, x, y, target_bounds, source_bbox)


def _reference_opaque_bbox(reference: Image.Image) -> tuple[int, int, int, int]:
    """Return the useful source area for relative item-style sampling.

    Item references often have a different silhouette from the new geometry.
    Sampling the same canvas coordinate then puts a sword's hilt shading on a
    new blade (or transparent pixels on a handle). Relative sampling keeps the
    reference's pixel-level value rhythm while allowing the new alpha contour
    to remain independent. UV atlases keep their exact coordinates below.
    """
    return reference.getchannel("A").getbbox() or (0, 0, reference.width, reference.height)


def _reference_global_range(reference: Image.Image,
                            bbox: tuple[int, int, int, int]) -> tuple[float, float, float]:
    source = reference.load()
    values: list[float] = []
    left, top, right, bottom = bbox
    for y in range(max(0, top), min(reference.height, bottom)):
        for x in range(max(0, left), min(reference.width, right)):
            value = _reference_value(source[x, y])
            if value is not None:
                values.append(value)
    if not values:
        return 0.0, 1.0, 0.5
    return min(values), max(values), sum(values) / len(values)


def _sample_reference_value(reference: Image.Image, x: int, y: int,
                            target_bounds: tuple[int, int, int, int],
                            source_bbox: tuple[int, int, int, int]) -> float | None:
    """Sample a same-sized item reference in local normalized coordinates."""
    pixel = _sample_reference_pixel(reference, x, y, target_bounds, source_bbox)
    return _reference_value(pixel) if pixel is not None else None


def _reference_ranges(reference: Image.Image | None,
                      regions: dict[str, tuple[int, int, int, int]]) -> dict[str, tuple[float, float, float]]:
    """Get local luma ranges so a dark-green reference still drives a full palette.

    Vanilla atlases often keep an entire face inside a narrow absolute luma
    band. Normalising per UV face preserves its pixel pattern while allowing a
    new material palette to use its dark/mid/light entries.
    """
    if reference is None:
        return {}
    source = reference.load()
    result: dict[str, tuple[float, float, float]] = {}
    for region_id, (left, top, right, bottom) in regions.items():
        values: list[float] = []
        for y in range(max(0, top), min(reference.height, bottom)):
            for x in range(max(0, left), min(reference.width, right)):
                value = _reference_value(source[x, y])
                if value is not None:
                    values.append(value)
        if values:
            result[region_id] = (min(values), max(values), sum(values) / len(values))
    return result


def _named_reference_image(
    references: list[ReferenceAsset] | None,
    name: str,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image | None:
    """Load a model-selected reference by its manifest name.

    A named material source is allowed to have a different raster size from
    the locked target atlas.  Its pixels are sampled in local normalized
    coordinates below; the source never contributes alpha or geometry.
    """
    wanted = str(name).strip().casefold()
    if not wanted or not references:
        return None
    selected: ReferenceAsset | None = next(
        (
            candidate for candidate in references
            if candidate.name.strip().casefold() == wanted
            and ReferenceRole.NEGATIVE not in candidate.roles
        ),
        None,
    )
    if selected is None:
        return None
    # A model can accidentally bind a structural variant or palette-only
    # example even though the same routed evidence contains a direct material
    # raster.  Prefer the source carrying both material and palette roles only
    # in that conflict; otherwise preserve the model's named choice.
    roles = set(selected.roles)
    notes = " ".join(selected.notes).casefold()
    direct_materials = [
        candidate for candidate in references
        if ReferenceRole.NEGATIVE not in candidate.roles
        and ReferenceRole.MATERIAL in candidate.roles
        and ReferenceRole.PALETTE in candidate.roles
    ]
    variant_language = any(token in notes for token in ("variant", "analogous", "precedent", "layering"))
    selected_is_direct = (
        ReferenceRole.MATERIAL in roles and ReferenceRole.PALETTE in roles
    )
    # An explicit part binding is a model-authored face/material decision.
    # Only a source labelled as a loose variant/precedent may yield to a
    # direct material sample. Otherwise a separate palette reference would
    # silently replace the selected grain, ring, seam, or UV-face evidence.
    if direct_materials and variant_language:
        selected = direct_materials[0]
    try:
        return Image.open(selected.path).convert("RGBA")
    except (OSError, ValueError):
        return None
    return None


def _reference_context(
    reference: Image.Image | None,
    width: int,
    height: int,
    region_bounds: dict[str, tuple[int, int, int, int]],
) -> tuple[
    bool,
    bool,
    object,
    tuple[int, int, int, int] | None,
    tuple[float, float, float],
    list[tuple[int, int, int]],
    bool,
    dict[str, tuple[float, float, float]],
    dict[str, tuple[int, int, int]],
    dict[str, Counter[tuple[int, int, int]]],
]:
    """Prepare sampling metadata for one source without semantic inference."""
    if reference is None:
        return (False, False, None, None, (0.0, 1.0, 0.5), [], False, {}, {}, {})
    exact_size = reference.size == (width, height)
    face_tile = (
        not exact_size
        and reference.height == height
        and width % max(reference.width, 1) == 0
    )
    bbox = _reference_sampling_bbox(reference)
    global_range = _reference_global_range(reference, bbox)
    palette = _reference_palette(reference)
    shared_uv_tile = bool(
        len(region_bounds) >= 3
        and len(set(region_bounds.values())) == 1
        and next(iter(region_bounds.values())) == (0, 0, width, height)
    )
    # A named material may be a compact animated strip or another native
    # raster.  It still has a discrete pixel palette, so use palette ranks
    # after local sampling instead of stretching its absolute luma range.
    discrete_tile = face_tile or shared_uv_tile or not exact_size
    ranges = {} if discrete_tile else _reference_ranges(reference, region_bounds)
    dominants: dict[str, tuple[int, int, int]] = {}
    counts: dict[str, Counter[tuple[int, int, int]]] = {}
    for region_id, (left, top, right, bottom) in region_bounds.items():
        if discrete_tile:
            source_pixels = (
                reference.getpixel((x, y))
                for y in range(reference.height)
                for x in range(reference.width)
            )
        else:
            source_pixels = (
                reference.getpixel((x, y))
                for y in range(top, min(bottom, reference.height))
                for x in range(left, min(right, reference.width))
            )
        region_counts = Counter(pixel[:3] for pixel in source_pixels if pixel[3] >= 8)
        if region_counts:
            counts[region_id] = region_counts
            dominants[region_id] = region_counts.most_common(1)[0][0]
    if discrete_tile and palette and dominants:
        dominant = next(iter(dominants.values()))
        dominant_luma = _luma(dominant)
        dominant_chroma = _chroma(dominant)
        material_palette = [
            color for color in palette
            if _chroma(color) <= max(18.0, dominant_chroma + 12.0)
            and abs(_luma(color) - dominant_luma) <= 64.0
        ]
        if len(material_palette) >= 2:
            palette = material_palette
    return (
        face_tile,
        exact_size,
        reference.load(),
        bbox,
        global_range,
        palette,
        discrete_tile,
        ranges,
        dominants,
        counts,
    )


def _overlay_reference_is_salient(
    source_pixel: tuple[int, int, int, int] | None,
    dominant: tuple[int, int, int] | None,
    source_count: int = 0,
    region_count: int = 0,
) -> bool:
    """Decide whether a secondary source contributes a visible pixel cluster.

    A composite reference is evidence for local material detail, rather than
    a second silhouette or a full replacement image.  Keep the decision
    entirely raster based: a source colour is eligible when it is a minority
    outlier from that source's own dominant material ramp and has a meaningful
    luma or chroma separation.  No part or object names are consulted here.
    """
    if source_pixel is None or source_pixel[3] < 8 or dominant is None:
        return False
    source = source_pixel[:3]
    if source == dominant:
        return False
    # A broad colour band is part of the source's base material.  The model
    # can request ``replace`` when that broad band should cover the whole
    # target; overlay only transfers the source's compact, salient clusters.
    if region_count > 0 and source_count / float(region_count) > 0.55:
        return False
    luma_gap = abs(_luma(source) - _luma(dominant))
    chroma_gap = _chroma(source) - _chroma(dominant)
    source_chroma = _chroma(source)
    return (
        # Require a clear hue excursion for a chromatic deposit. Near-base
        # brown/grey shades still belong to the structural material ramp and
        # should remain underneath the overlay.
        (source_chroma >= 60.0 and chroma_gap >= 40.0)
        or (source_chroma >= 48.0 and chroma_gap >= 36.0 and luma_gap >= 40.0)
        or (luma_gap >= 64.0 and (source_count <= max(4, region_count * 0.35) if region_count else True))
    )


def _reference_context_dominant(
    context: tuple[object, ...] | None,
    reference: Image.Image | None = None,
) -> tuple[int, int, int] | None:
    """Return a source-wide dominant colour from prepared context metadata."""
    if context is None:
        return None
    # ``_reference_context`` stores region dominants/counts at the tail.  Use
    # those when available, then fall back to a source-wide count for item
    # references that have no declared UV regions.
    dominants = context[8] if len(context) > 8 else {}
    if isinstance(dominants, dict) and dominants:
        return next(iter(dominants.values()))
    reference_pixels = reference.load() if reference is not None else None
    if reference_pixels is None or reference is None:
        return None
    counts = Counter(
        reference_pixels[x, y][:3]
        for y in range(reference.height)
        for x in range(reference.width)
        if reference_pixels[x, y][3] >= 8
    )
    return counts.most_common(1)[0][0] if counts else None


def _pattern_shade(pixel: tuple[int, int, int, int] | None,
                   region_id: str,
                   ranges: dict[str, tuple[float, float, float]],
                   global_range: tuple[float, float, float],
                   palette: list[tuple[int, int, int]]) -> float | None:
    """Turn a source pixel into a stable dark-to-light pattern value.

    Entity UV faces are often tiny and use a much narrower value range than a
    complete atlas.  A single global palette therefore collapses a face's
    eyes, muzzle or edge band into one target swatch.  Keep item references on
    their global discrete palette, but normalize atlas faces against their own
    local range so their authored pixel rhythm survives recolouring.
    """
    if pixel is None or pixel[3] < 8:
        return None
    source_value = _reference_value(pixel)
    if source_value is None:
        return None
    if region_id:
        low, high, _mean = ranges.get(region_id, global_range)
        if high > low:
            return max(0.0, min(1.0, (source_value - low) / (high - low)))
        return 0.5
    return _reference_pattern_value(pixel, palette)


def render_appearance(geometry: GeometrySpec, compiled: CompiledGeometry,
                      appearance: AppearanceSpec, seed: int = 0,
                      references: list[ReferenceAsset] | None = None,
                      alpha_mask: Image.Image | None = None,
                      alpha_source: Image.Image | None = None) -> Image.Image:
    """Paint a geometry coverage map under an explicit output-alpha contract.

    For normal 2-D assets the compiled geometry remains both the paint coverage
    and alpha contract.  A target-model entity atlas is different: its UV
    coverage says which model faces sample the texture, while its reference
    alpha says which atlas cells are intentionally transparent.  The caller
    can therefore supply a separate alpha mask without pretending that a
    hand-written UV layout is the source PNG's silhouette.
    """
    result = Image.new("RGBA", (compiled.width, compiled.height), (0, 0, 0, 0))
    pixels = result.load()
    # Both sampling modes need the source image.  `value` transfers its local
    # luminance, while `pattern` transfers the ordering of its discrete
    # palette bands.  Previously only `value` loaded the reference, so a
    # planner selecting the more faithful pattern mode silently rendered as if
    # no reference had been supplied.
    reference = _reference_image(references, compiled.width, compiled.height, target_mask=compiled.mask) \
        if appearance.reference_sampling in {"value", "pattern"} else None
    reference_face_tile = bool(
        reference is not None
        and reference.size != (compiled.width, compiled.height)
        and reference.height == compiled.height
        and compiled.width % max(reference.width, 1) == 0
    )
    reference_pixels = reference.load() if reference is not None else None
    global_reference = reference
    reference_exact_size = bool(reference is not None and reference.size == (compiled.width, compiled.height))
    reference_bbox = _reference_opaque_bbox(reference) if reference is not None else None
    reference_global_range = (
        _reference_global_range(reference, reference_bbox)
        if reference is not None and reference_bbox is not None else (0.0, 1.0, 0.5)
    )
    reference_palette = _reference_palette(reference) if reference is not None else []
    region_rules = appearance.region_rules
    ordered_parts = sorted(geometry.parts, key=lambda part: part.layer)
    uv_regions_by_part: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for region in geometry.uv_regions:
        uv_regions_by_part.setdefault(region.part_id, []).append((region.id, tuple(region.bbox)))
    region_bounds = {
        region_id: bounds
        for items in uv_regions_by_part.values()
        for region_id, bounds in items
    }
    # Keep a stable alias because the per-pixel region loop uses the concise
    # name ``region_bounds`` for one candidate rectangle.
    all_region_bounds = region_bounds
    # A single 16x16 block texture can be bound to all six cube faces. Such a
    # layout has overlapping region boxes that cover the whole canvas; treat it
    # like the compact face-tile case so the same discrete source bands are
    # reused instead of re-normalizing the shared tile once per region.
    shared_uv_tile = bool(
        reference is not None
        and len(region_bounds) >= 3
        and len(set(region_bounds.values())) == 1
        and next(iter(region_bounds.values())) == (0, 0, compiled.width, compiled.height)
    )
    discrete_tile = reference_face_tile or shared_uv_tile
    reference_ranges = {} if discrete_tile else _reference_ranges(reference, region_bounds)
    reference_region_dominants: dict[str, tuple[int, int, int]] = {}
    reference_region_counts: dict[str, Counter[tuple[int, int, int]]] = {}
    if reference is not None:
        for region_id, (left, top, right, bottom) in region_bounds.items():
            if discrete_tile:
                colors_in_region = Counter(
                    reference.getpixel((x, y))[:3]
                    for y in range(reference.height)
                    for x in range(reference.width)
                    if reference.getpixel((x, y))[3] >= 8
                )
            else:
                colors_in_region = Counter(
                    reference.getpixel((x, y))[:3]
                    for y in range(top, bottom)
                    for x in range(left, right)
                    if reference.getpixel((x, y))[3] >= 8
                )
            if colors_in_region:
                reference_region_counts[region_id] = colors_in_region
                reference_region_dominants[region_id] = colors_in_region.most_common(1)[0][0]
    if discrete_tile and reference_palette and reference_region_dominants:
        # Shared block tiles often combine a neutral base ramp with a much
        # brighter coloured deposit.  Ranking against all source colours
        # makes the four neutral bands occupy only the first half of the
        # target ramp.  When the dominant material is near-neutral, use its
        # nearby neutral palette for base-band ranking; the region rule still
        # handles chromatic/defining clusters using the original source pixel.
        dominant = next(iter(reference_region_dominants.values()))
        dominant_luma = _luma(dominant)
        dominant_chroma = _chroma(dominant)
        material_palette = [
            color for color in reference_palette
            if _chroma(color) <= max(18.0, dominant_chroma + 12.0)
            and abs(_luma(color) - dominant_luma) <= 64.0
        ]
        if len(material_palette) >= 2:
            reference_palette = material_palette
    global_reference_context = (
        reference,
        reference_face_tile,
        reference_exact_size,
        reference_pixels,
        reference_bbox,
        reference_global_range,
        reference_palette,
        discrete_tile,
        reference_ranges,
        reference_region_dominants,
        reference_region_counts,
    )
    for part in ordered_parts:
        mask = compiled.part_masks[part.id]
        part_bounds = mask.getbbox() or (0, 0, compiled.width, compiled.height)
        style = appearance.parts.get(part.id)
        if style is None:
            style = PartAppearance(colors=["#8B8B8B"], material="fallback")
        # The model may bind a secondary material/palette reference to this
        # part.  Keep the default global source for all unbound parts, and use
        # local normalized sampling for a source whose native dimensions do
        # not match the locked atlas.
        overlay_reference: Image.Image | None = None
        overlay_context: tuple[object, ...] | None = None
        overlay_global_dominant: tuple[int, int, int] | None = None
        overlay_global_counts: Counter[tuple[int, int, int]] = Counter()
        # Declared once: the painting loop below reads it for every part, not
        # only for a part that managed to bind a named reference.
        composite_mode = ""
        part_source_name = appearance.part_reference_sources.get(part.id)
        if part_source_name:
            named_reference = _named_reference_image(
                references,
                part_source_name,
                compiled.width,
                compiled.height,
            )
            if named_reference is not None:
                composite_mode = str(
                    appearance.part_reference_composite.get(part.id, "")
                ).strip().lower()
                # A named source is an explicit face/material choice.  It must
                # therefore supply the base pixels unless the model explicitly
                # asks to layer it over another source.  Treating it as a
                # default overlay made a requested log end-grain inherit bark
                # from a global reference, and loses the meaning of named
                # per-part evidence in any asset class.
                if composite_mode not in {"overlay", "replace"}:
                    composite_mode = "replace"
                if composite_mode == "overlay" and global_reference is not None:
                    overlay_reference = named_reference
                    overlay_context = _reference_context(
                        named_reference,
                        compiled.width,
                        compiled.height,
                        all_region_bounds,
                    )
                    overlay_global_dominant = _reference_context_dominant(
                        overlay_context,
                        named_reference,
                    )
                    overlay_global_counts = Counter(
                        pixel[:3]
                        for pixel in named_reference.get_flattened_data()
                        if pixel[3] >= 8
                    )
                    (
                        reference,
                        reference_face_tile,
                        reference_exact_size,
                        reference_pixels,
                        reference_bbox,
                        reference_global_range,
                        reference_palette,
                        discrete_tile,
                        reference_ranges,
                        reference_region_dominants,
                        reference_region_counts,
                    ) = global_reference_context
                    named_reference = None
                if named_reference is not None:
                    (
                        reference_face_tile,
                        reference_exact_size,
                        reference_pixels,
                        reference_bbox,
                        reference_global_range,
                        reference_palette,
                        discrete_tile,
                        reference_ranges,
                        reference_region_dominants,
                        reference_region_counts,
                    ) = _reference_context(
                        named_reference,
                        compiled.width,
                        compiled.height,
                        all_region_bounds,
                    )
                    reference = named_reference
            else:
                (
                    reference,
                    reference_face_tile,
                    reference_exact_size,
                    reference_pixels,
                    reference_bbox,
                    reference_global_range,
                    reference_palette,
                    discrete_tile,
                    reference_ranges,
                    reference_region_dominants,
                    reference_region_counts,
                ) = global_reference_context
        else:
            (
                reference,
                reference_face_tile,
                reference_exact_size,
                reference_pixels,
                reference_bbox,
                reference_global_range,
                reference_palette,
                discrete_tile,
                reference_ranges,
                reference_region_dominants,
                reference_region_counts,
                ) = global_reference_context
        overlay_reference_pixels = None
        overlay_reference_face_tile = False
        overlay_reference_exact_size = False
        overlay_reference_region_dominants: dict[str, tuple[int, int, int]] = {}
        overlay_reference_region_counts: dict[str, Counter[tuple[int, int, int]]] = {}
        if overlay_context is not None:
            (
                overlay_reference_face_tile,
                overlay_reference_exact_size,
                overlay_reference_pixels,
                _overlay_reference_bbox,
                _overlay_reference_global_range,
                _overlay_reference_palette,
                _overlay_discrete_tile,
                _overlay_reference_ranges,
                overlay_reference_region_dominants,
                overlay_reference_region_counts,
            ) = overlay_context
        sampling_mode = appearance.part_reference_sampling.get(
            part.id, appearance.reference_sampling
        )
        colors = _resolve_colors(style, appearance.palette)
        if reference_pixels is None or sampling_mode == "none":
            colors = _ensure_texture_ramp(colors, style.material, compiled.width, compiled.height)
        elif sampling_mode == "pattern":
            colors = _ensure_contrast_ramp(colors, style.material)
        # Pattern references describe discrete source bands from dark to
        # light.  Models often name colors semantically ("edge", "highlight")
        # and return them in that semantic order, so relying on list position
        # can map a bright source pixel to a dark edge color.  Sort only for
        # reference-driven pattern transfer; normal shading keeps the explicit
        # authored order for cases where no reference controls the value ramp.
        if sampling_mode == "pattern":
            colors.sort(key=_luma)
        rng = Random(_part_seed(seed, part.id))
        has_interior = _part_has_interior(mask, compiled.width, compiled.height)
        for y in range(compiled.height):
            for x in range(compiled.width):
                if mask.getpixel((x, y)) == 0:
                    continue
                bounds = part_bounds
                region_id = ""
                for candidate_id, region_bounds in uv_regions_by_part.get(part.id, []):
                    if region_bounds[0] <= x < region_bounds[2] and region_bounds[1] <= y < region_bounds[3]:
                        region_id = candidate_id
                        bounds = region_bounds
                        break
                shade = _shade_value(x, y, compiled.width, compiled.height, style.shade_axis, bounds)
                source_pixel: tuple[int, int, int, int] | None = None
                reference_shade: float | None = None
                if reference_pixels is not None and sampling_mode in {"value", "pattern"}:
                    if region_id:
                        # Entity/block UV regions are real atlas coordinates;
                        # never remap an exact atlas through an item
                        # silhouette. A compact face tile is the explicit
                        # size-compatible fallback and is sampled locally.
                        source_pixel = (
                            _sample_reference_pixel(
                                reference,
                                x,
                                y,
                                next(
                                    bounds
                                    for candidate_id, bounds in uv_regions_by_part.get(part.id, [])
                                    if candidate_id == region_id
                                ),
                                (0, 0, reference.width, reference.height),
                            )
                            if reference_face_tile or not reference_exact_size
                            else reference_pixels[x, y]
                        )
                        reference_shade = (
                            # A compact face tile is already a discrete native
                            # Minecraft raster. Keep its palette rank exactly;
                            # local continuous normalization is reserved for
                            # real UV faces whose value range is authored per
                            # region.
                            _reference_pattern_value(source_pixel, reference_palette)
                            if sampling_mode == "pattern" and discrete_tile
                            else _pattern_shade(
                                source_pixel,
                                region_id,
                                reference_ranges,
                                reference_global_range,
                                reference_palette,
                            )
                            if sampling_mode == "pattern"
                            else _reference_value(reference_pixels[x, y])
                        )
                    else:
                        source_bbox = reference_bbox or (0, 0, reference.width, reference.height)
                        if sampling_mode == "pattern" and reference.size == (compiled.width, compiled.height):
                            source_pixel = _sample_item_reference_pixel(
                                reference, x, y, part_bounds, source_bbox
                            )
                        else:
                            source_pixel = _sample_reference_pixel(
                                reference, x, y, part_bounds, source_bbox
                            )
                        reference_shade = (
                            _reference_pattern_value(source_pixel, reference_palette)
                            if sampling_mode == "pattern" and source_pixel is not None
                            else _reference_value(source_pixel) if source_pixel is not None else None
                        )
                    range_key = region_id or part.id
                    low, high, mean = reference_ranges.get(range_key, reference_global_range)
                    # Atlas gaps are common in legacy Minecraft sheets. They
                    # still need a base coat, so use the face's mean value
                    # instead of falling back to a canvas-wide bright ramp.
                    if reference_shade is None:
                        reference_shade = (
                            (mean - low) / (high - low)
                            if sampling_mode == "pattern" and high > low
                            else mean
                        )
                    if sampling_mode != "pattern" and high > low:
                        # ``value`` sampling starts from an absolute source
                        # luminance. ``pattern`` sampling already returns a
                        # local 0..1 band from ``_pattern_shade``; applying
                        # this conversion to it again collapses all low/mid
                        # source bands into the darkest target swatch.
                        local_shade = (reference_shade - low) / (high - low)
                        reference_shade = reference_shade * 0.7 + local_shade * 0.3
                    shade = reference_shade
                # Pattern mode is an explicit request to preserve the source
                # pixel bands. The source already supplies the texture's
                # variation, so model-authored random noise would invent new
                # intermediate colours and blur the native pixel language.
                pattern_noise = 0.0 if sampling_mode == "pattern" and reference_shade is not None else style.noise
                rgb = _color_for_pixel(
                    colors,
                    shade,
                    pattern_noise,
                    rng,
                    continuous=(
                        sampling_mode == "pattern"
                        and not discrete_tile
                    ),
                )
                # A named secondary reference can augment the locked/global
                # raster.  Transfer only its own salient colour clusters;
                # transparent pixels and the source's dominant base stay
                # untouched, so local fur, face, grain or edge pixels from the
                # structural reference remain visible.  The model chooses the
                # source and composite mode; this pass only applies generic
                # pixel statistics to the already approved mask.
                if (
                    overlay_reference is not None
                    and overlay_reference_pixels is not None
                    and sampling_mode in {"value", "pattern"}
                ):
                    overlay_pixel: tuple[int, int, int, int] | None
                    if region_id:
                        overlay_pixel = (
                            _sample_reference_pixel(
                                overlay_reference,
                                x,
                                y,
                                bounds,
                                (0, 0, overlay_reference.width, overlay_reference.height),
                            )
                            if overlay_reference_face_tile or not overlay_reference_exact_size
                            else overlay_reference_pixels[x, y]
                        )
                    else:
                        overlay_bbox = _reference_sampling_bbox(overlay_reference)
                        overlay_pixel = _sample_reference_pixel(
                            overlay_reference,
                            x,
                            y,
                            part_bounds,
                            overlay_bbox,
                        )
                    overlay_cluster = overlay_reference_region_counts.get(region_id)
                    overlay_dominant = (
                        overlay_reference_region_dominants.get(region_id)
                        if region_id else None
                    ) or overlay_global_dominant
                    overlay_count = (
                        overlay_cluster.get(overlay_pixel[:3], 0)
                        if overlay_cluster and overlay_pixel is not None
                        else overlay_global_counts.get(overlay_pixel[:3], 0)
                        if overlay_pixel is not None
                        else 0
                    )
                    overlay_total = (
                        sum(overlay_cluster.values())
                        if overlay_cluster
                        else sum(overlay_global_counts.values())
                    )
                    if _overlay_reference_is_salient(
                        overlay_pixel,
                        overlay_dominant,
                        overlay_count,
                        overlay_total,
                    ):
                        rgb = overlay_pixel[:3]
                region_rule = _region_rule_for(region_rules, region_id) if region_id else None
                rule_applied = False
                if region_rule is not None and sampling_mode in {"value", "pattern"}:
                    mode = str(region_rule.get("mode", "palette")).strip().lower()
                    target_token = region_rule.get("target_color")
                    raw_target = appearance.palette.get(str(target_token), str(target_token or ""))
                    try:
                        target_color = _hex_to_rgb(raw_target)
                    except (TypeError, ValueError):
                        target_color = None
                    cluster = reference_region_counts.get(region_id)
                    cluster_only = bool(region_rule.get("cluster_only", False))
                    is_secondary = _cluster_only_pixel_is_secondary(
                        source_pixel,
                        reference_region_dominants.get(region_id),
                        cluster.get(source_pixel[:3], 0) if cluster and source_pixel is not None else 0,
                        sum(cluster.values()) if cluster else 0,
                    )
                    if not cluster_only or is_secondary:
                        if mode in {"source_exact", "exact"} and source_pixel is not None and source_pixel[3] >= 8:
                            rgb = source_pixel[:3]
                            rule_applied = True
                        elif mode in {"retint", "recolor", "tint", "palette"} and target_color is not None:
                            try:
                                strength = float(region_rule.get("strength", 1.0))
                            except (TypeError, ValueError):
                                strength = 1.0
                            retinted = _retint_reference_pixel(
                                source_pixel,
                                target_color,
                                reference_region_dominants.get(region_id),
                                strength,
                                target_ramp=_target_hue_ramp(appearance.palette, target_color),
                            )
                            if retinted is not None:
                                rgb = retinted
                                rule_applied = True
                        elif mode == "palette":
                            # An explicit palette rule says to keep the
                            # normal source-to-authored-ramp mapping. Mark it
                            # as handled so the generic source-outlier safety
                            # net does not silently override the model's
                            # choice when ``cluster_only`` is false.
                            rule_applied = True
                # ``reference_locked`` means the source raster is the base
                # layer of a variant.  A cluster-only rule may then repaint a
                # named local feature, while every opaque source pixel that
                # the rule did not claim remains byte-for-byte unchanged.
                # This keeps stone grain, fur bands, leather pixels and other
                # unmentioned material detail from being re-authored by the
                # generic target ramp.  Models that want a broad recolour can
                # opt into an explicit non-locked motif policy or a full
                # region rule.
                if (
                    not rule_applied
                    and appearance.motif_policy == "reference_locked"
                    and sampling_mode == "pattern"
                    and source_pixel is not None
                    and source_pixel[3] >= 8
                ):
                    rgb = source_pixel[:3]
                    rule_applied = True
                # A model-authored rule has priority.  Without one, use only
                # generic evidence: material labels and the size/contrast of
                # a local UV colour cluster.  Region names are identifiers,
                # not an anatomy taxonomy, so ``part_3`` works the same as
                # ``ear`` or ``blade_front``.
                cluster = reference_region_counts.get(region_id) if region_id else None
                small_region = bool(cluster) and sum(cluster.values()) <= 64
                feature_region = bool(region_id) and (
                    style.material.lower() in {"bone", "horn", "flesh", "accent", "eye", "tooth"}
                    or small_region
                )
                # A "replace" composite is documented as letting the named
                # raster provide the full ramp, but it used to fall through to
                # the generic authored-swatch mapping. A 16x16 plank tile
                # supplies 43 distinct browns and that mapping collapsed them
                # onto 4, erasing exactly the grain the model asked to keep.
                # Retint the source instead, so its own value rhythm survives
                # while the hue still comes from the authored palette.
                if (
                    not rule_applied
                    and composite_mode == "replace"
                    and sampling_mode == "pattern"
                    and source_pixel is not None
                    and source_pixel[3] >= 8
                ):
                    retinted = _retint_reference_pixel(
                        source_pixel,
                        colors[len(colors) // 2],
                        reference_region_dominants.get(region_id) if region_id else None,
                        _REPLACE_RETINT_STRENGTH,
                        target_ramp=_target_hue_ramp(appearance.palette, colors[len(colors) // 2]),
                    )
                    if retinted is not None:
                        rgb = retinted
                        rule_applied = True
                if not rule_applied and sampling_mode == "pattern" and region_id and cluster:
                    preserved = _cluster_detail_color(
                        source_pixel,
                        reference_region_dominants.get(region_id),
                        cluster.get(source_pixel[:3], 0) if source_pixel is not None else 0,
                        sum(cluster.values()),
                        feature_region,
                    )
                    if preserved is not None:
                        rgb = preserved
                if not rule_applied and sampling_mode == "pattern" and feature_region:
                    preserved = _feature_reference_color(
                        source_pixel,
                        reference_region_dominants.get(region_id),
                    )
                    if preserved is not None:
                        rgb = preserved
                if not rule_applied and sampling_mode == "pattern" and region_id and small_region:
                    # A broad face may still contain a compact saturated patch
                    # (for example the pink underside cluster in the vanilla
                    # cow atlas). Preserve that local colour while keeping
                    # neutral body shading in the requested material ramp.
                    preserved = _colorful_pattern_outlier(
                        source_pixel,
                        reference_region_dominants.get(region_id),
                    )
                    if preserved is not None:
                        rgb = preserved
                if (
                    source_pixel is None
                    and has_interior
                    and len(colors) > 1
                    and style.material.lower() not in {"glow", "accent", "eye", "symbol"}
                ):
                    # A one-pixel inner edge gives ears, noses, handles and
                    # other small parts a readable contour even when the LLM
                    # did not provide an explicit outline or mark.
                    edge_pixel = (
                        mask.getpixel((x - 1, y)) == 0 if x > 0 else True
                    ) or (
                        mask.getpixel((x + 1, y)) == 0 if x + 1 < compiled.width else True
                    ) or (
                        mask.getpixel((x, y - 1)) == 0 if y > 0 else True
                    ) or (
                        mask.getpixel((x, y + 1)) == 0 if y + 1 < compiled.height else True
                    )
                    if edge_pixel:
                        rgb = colors[0]
                if len(colors) > 1 and style.highlight_ratio > 0 and source_pixel is None:
                    upper_left_edge = (
                        mask.getpixel((x - 1, y)) == 0 if x > 0 else True
                    ) or (
                        mask.getpixel((x, y - 1)) == 0 if y > 0 else True
                    )
                    if upper_left_edge and rng.random() < style.highlight_ratio:
                        rgb = colors[-1]
                if appearance.motif_policy in {"free", "model_authored"}:
                    marks = style.marks
                else:
                    marks = []
                for mark in marks:
                    if not isinstance(mark, dict):
                        continue
                    mark_parts = mark.get("parts", [])
                    if isinstance(mark_parts, str):
                        mark_parts = [mark_parts]
                    if mark_parts and part.id not in {str(item) for item in mark_parts}:
                        continue
                    mark_regions = mark.get("regions", [])
                    if isinstance(mark_regions, str):
                        mark_regions = [mark_regions]
                    if mark_regions and region_id not in mark_regions:
                        continue
                    rows = mark.get("rows", [])
                    raw_offset = mark.get("offset", [0, 0])
                    try:
                        offset_x = int(raw_offset[0]) if isinstance(raw_offset, (list, tuple)) else 0
                        offset_y = int(raw_offset[1]) if isinstance(raw_offset, (list, tuple)) else 0
                    except (TypeError, ValueError, IndexError):
                        offset_x = offset_y = 0
                    local_x = x - bounds[0] - offset_x
                    local_y = y - bounds[1] - offset_y
                    if (
                        isinstance(rows, list)
                        and 0 <= local_y < len(rows)
                        and isinstance(rows[local_y], str)
                        and 0 <= local_x < len(rows[local_y])
                        and rows[local_y][local_x] in {"X", "x"}
                    ):
                        color_token = str(mark.get("color", "")).strip()
                        if not color_token:
                            continue
                        rgb = _palette_color(appearance.palette, color_token)
                        break
                pixels[x, y] = (*rgb, 255)

    # Paint-only parts have no alpha geometry by contract.  Their marks still
    # need a way to land on the existing support, so apply model-authored mark
    # rows against the union mask after physical parts are painted.  This is a
    # generic coordinate bridge: the model supplies the rows and colour token;
    # the renderer only clips them to existing opaque pixels.
    for part in ordered_parts:
        if not part.paint_only:
            continue
        style = appearance.parts.get(part.id)
        if style is None or appearance.motif_policy not in {"free", "model_authored"}:
            continue
        for mark in style.marks:
            if not isinstance(mark, dict):
                continue
            rows = mark.get("rows", [])
            if not isinstance(rows, list):
                continue
            color_token = str(mark.get("color", "")).strip()
            if not color_token:
                continue
            try:
                mark_rgb = _hex_to_rgb(appearance.palette.get(color_token, color_token))
            except (TypeError, ValueError):
                continue
            raw_offset = mark.get("offset", [0, 0])
            try:
                offset_x = int(raw_offset[0]) if isinstance(raw_offset, (list, tuple)) else 0
                offset_y = int(raw_offset[1]) if isinstance(raw_offset, (list, tuple)) else 0
            except (TypeError, ValueError, IndexError):
                offset_x = offset_y = 0
            for local_y, row in enumerate(rows):
                if not isinstance(row, str):
                    continue
                y = offset_y + local_y
                if y < 0 or y >= compiled.height:
                    continue
                for local_x, value in enumerate(row):
                    x = offset_x + local_x
                    if 0 <= x < compiled.width and value in {"X", "x"} and compiled.mask.getpixel((x, y)) > 0:
                        pixels[x, y] = (*mark_rgb, 255)

    # A reference already contains the edge pixels, including light rims and
    # broken outlines.  A uniform post-pass would erase that information, so
    # apply authored outlines only when the material is not reference-driven.
    if (appearance.outline_color and appearance.outline_width > 0
            and reference_pixels is None):
        if appearance.outline_width > 2:
            raise ValueError("outline_width above 2 is not appropriate for low-resolution assets")
        size = appearance.outline_width * 2 + 1
        eroded = compiled.mask.filter(ImageFilter.MinFilter(size=size))
        inner_edge = ImageChops.subtract(compiled.mask, eroded)
        outline = _palette_color(appearance.palette, appearance.outline_color)
        for y in range(compiled.height):
            for x in range(compiled.width):
                if inner_edge.getpixel((x, y)) > 0:
                    pixels[x, y] = (*outline, 255)

    # An optional full-canvas model map aligns colour decisions across parts.
    # It never creates alpha and contains no object-specific interpretation.
    pixel_map = appearance.pixel_map
    if pixel_map is not None:
        rows, legend = pixel_map["rows"], pixel_map["legend"]
        if len(rows) != compiled.height or any(len(row) != compiled.width for row in rows):
            raise ValueError("pixel_map rows must match the native asset canvas")
        for y, row in enumerate(rows):
            for x, symbol in enumerate(row):
                if symbol == ".":
                    continue
                if symbol not in legend:
                    raise ValueError("pixel_map uses a symbol absent from its legend")
                if compiled.mask.getpixel((x, y)) == 0:
                    continue
                token = str(legend[symbol])
                pixels[x, y] = (*_palette_color(appearance.palette, token), 255)

    output_alpha = (alpha_mask or compiled.mask).convert("L")
    if output_alpha.size != result.size:
        raise ValueError("output alpha mask size differs from geometry canvas")
    if alpha_source is not None:
        source = alpha_source.convert("RGBA")
        if source.size != result.size:
            raise ValueError("alpha source size differs from geometry canvas")
        # An alpha authority owns transparency, never colour. Copying its own
        # pixels here painted a metal layer's grey into a wooden helmet, exactly
        # where the authority was opaque but the model had opened a face. Borrow
        # the nearest painted pixel instead: the silhouette still follows the
        # authority while the material stays the asset's own, and an uncovered
        # cell is still visible rather than becoming an opaque black hole.
        painted = [
            (x, y)
            for y in range(compiled.height)
            for x in range(compiled.width)
            if compiled.mask.getpixel((x, y)) > 0 and output_alpha.getpixel((x, y)) > 0
        ]
        if painted:
            for y in range(compiled.height):
                for x in range(compiled.width):
                    if output_alpha.getpixel((x, y)) > 0 and compiled.mask.getpixel((x, y)) == 0:
                        nearest_x, nearest_y = min(
                            painted, key=lambda point: (point[0] - x) ** 2 + (point[1] - y) ** 2
                        )
                        pixels[x, y] = pixels[nearest_x, nearest_y]
    result.putalpha(output_alpha)
    return result


def scale_preview(image: Image.Image, scale: int = 24) -> Image.Image:
    if scale < 1:
        raise ValueError("preview scale must be positive")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def checkerboard_preview(image: Image.Image, scale: int = 24, tile: int = 2) -> Image.Image:
    """Show transparent pixel art against a neutral checkerboard for human review.

    The deliverable sprite remains RGBA and untouched.  This separate diagnostic
    preview prevents dark outlines and transparent margins from disappearing
    against a viewer's black canvas.
    """
    if tile < 1:
        raise ValueError("checkerboard tile must be positive")
    base = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    # Keep both very dark outlines and white specular pixels visible during
    # human/vision review.  Near-white checker tiles make vanilla highlights
    # disappear and can lead the reviewer to blame the texture for a preview
    # contrast problem.
    light, dark = (202, 202, 202, 255), (154, 154, 154, 255)
    for top in range(0, image.height, tile):
        for left in range(0, image.width, tile):
            color = light if ((left // tile + top // tile) % 2 == 0) else dark
            draw.rectangle((left, top, min(image.width, left + tile) - 1,
                           min(image.height, top + tile) - 1), fill=color)
    base.alpha_composite(image.convert("RGBA"))
    return scale_preview(base, scale=scale)
