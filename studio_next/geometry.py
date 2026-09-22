"""Compile an open, part-labelled geometry DSL into pixel masks.

The primitive vocabulary is intentionally small and stable, like SVG. It is a
language for arbitrary shapes, not a list of allowed item types. A planner can
combine primitives freely or provide a custom labelled mask for unusual forms.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, cos, sin
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .contracts import AssetForm, GeometrySpec, PrimitiveSpec


SUPPORTED_PRIMITIVES = frozenset(
    {
        "polygon",
        "ellipse",
        "blob",
        "segment",
        "thick_segment",
        "tapered_polygon",
        "ring",
        "custom_mask",
        "uv_fill",
        "repeat",
        "cutout",
        "mirror",
        "transform",
    }
)


def primitive_parameter_guide() -> str:
    """Concise, machine-facing grammar supplied to a planner with every request."""
    return """Parameter grammar (all coordinates are pixels unless normalized=true):
- polygon: {points:[[x,y],[x,y],[x,y],...]}
- ellipse/blob: {bbox:[left,top,right,bottom]} or {center:[x,y], radius:n} / {center:[x,y], radius_x:n, radius_y:n}
- segment/thick_segment: {start:[x,y], end:[x,y], width:n, round_caps:true}
- tapered_polygon: {start:[x,y], end:[x,y], width_start:n, width_end:n}
- ring: {bbox:[left,top,right,bottom], inset:n}
- custom_mask: {offset:[x,y], rows:["X..", ".XX"], marker:"X"}; when marker is omitted, every non-`.`/space token is opaque (useful for palette-index pixel maps)
- uv_fill: {regions?:["uv_region_id", ...]} fills all declared UV face regions for this part; valid only when a GeometrySpec supplies those regions
- repeat: {count:n, delta:[x,y], child:{primitive:<basic primitive>, params:{...}}}
- mirror: {axis:"vertical"|"horizontal", include_original:true, child:{primitive:<basic primitive>, params:{...}}}
- transform: {flip:"none"|"horizontal"|"vertical"|"both", rotate:multiple_of_90, scale:n, scale_x:n, scale_y:n, translate:[x,y], child:{primitive:<basic primitive>, params:{...}}}
- cutout: {target_part:"part_id", shape:<basic primitive>, params:{...}}
Use the field names shown above. ellipse may also use cx/cy/rx/ry, and line primitives may also use x1/y1/x2/y2/thickness; the compiler canonicalizes those aliases."""


@dataclass(frozen=True)
class CompiledGeometry:
    width: int
    height: int
    mask: Image.Image
    part_masks: dict[str, Image.Image]
    primitive_order: list[str]

    def opaque_pixels(self) -> int:
        return sum(1 for alpha in self.mask.get_flattened_data() if alpha > 0)


def _is_normalized(value: Any) -> bool:
    return isinstance(value, float) and 0.0 <= value <= 1.0


def _coord(value: float | int, length: int, normalized: bool) -> int:
    if normalized or _is_normalized(value):
        return round(float(value) * (length - 1))
    return round(float(value))


def _point(raw: list[float | int] | tuple[float | int, float | int], width: int, height: int,
           normalized: bool = False) -> tuple[int, int]:
    if len(raw) != 2:
        raise ValueError("point must contain [x, y]")
    return _coord(raw[0], width, normalized), _coord(raw[1], height, normalized)


def _bbox(raw: list[float | int], width: int, height: int, normalized: bool) -> tuple[int, int, int, int]:
    if len(raw) != 4:
        raise ValueError("bbox must contain [left, top, right, bottom]")
    left = _coord(raw[0], width, normalized)
    top = _coord(raw[1], height, normalized)
    right = _coord(raw[2], width, normalized)
    bottom = _coord(raw[3], height, normalized)
    if right < left or bottom < top:
        raise ValueError("bbox right/bottom must not precede left/top")
    return left, top, right, bottom


def _tapered_points(start: tuple[int, int], end: tuple[int, int], width_start: float,
                    width_end: float) -> list[tuple[float, float]]:
    angle = atan2(end[1] - start[1], end[0] - start[0])
    nx, ny = -sin(angle), cos(angle)
    a = width_start / 2.0
    b = width_end / 2.0
    return [
        (start[0] + nx * a, start[1] + ny * a),
        (end[0] + nx * b, end[1] + ny * b),
        (end[0] - nx * b, end[1] - ny * b),
        (start[0] - nx * a, start[1] - ny * a),
    ]


def _draw_custom_mask(draw: ImageDraw.ImageDraw, params: dict[str, Any], width: int, height: int) -> None:
    rows = params.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("custom_mask requires non-empty rows")
    marker = params.get("marker")
    # Pixel maps in reference text often use palette symbols (`0`, `1`, `A`)
    # instead of a single geometry marker. When a planner omits `marker`, keep
    # that text useful by treating every non-transparent token as opaque. An
    # explicit marker remains strict so callers can still reserve other
    # characters for holes or annotations.
    opaque_tokens = None if marker is None else {str(marker)}
    offset = params.get("offset", [0, 0])
    ox, oy = _point(offset, width, height, bool(params.get("normalized", False)))
    for y, row in enumerate(rows):
        if not isinstance(row, str):
            raise ValueError("custom_mask rows must be strings")
        for x, token in enumerate(row):
            if token in {".", " "}:
                continue
            if opaque_tokens is None or token in opaque_tokens:
                draw.point((ox + x, oy + y), fill=255)


def _canonical_params(primitive: str, params: dict[str, Any]) -> dict[str, Any]:
    """Accept conventional geometric aliases while preserving one internal DSL.

    Vision models reliably know centre/radius and x1/y1/x2/y2 notation. They
    describe the same open geometry as the native fields, so accepting those
    names reduces format-only failures without adding an object category or a
    target-specific template.
    """
    result = dict(params)
    if primitive in {"ellipse", "blob", "ring"} and "bbox" not in result:
        center = result.get("center")
        center_x = result.get("cx")
        center_y = result.get("cy")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            center_x = center[0] if center_x is None else center_x
            center_y = center[1] if center_y is None else center_y
        radius = result.get("radius")
        radius_x = result.get("rx", result.get("radius_x", radius))
        radius_y = result.get("ry", result.get("radius_y", radius))
        if center_x is not None and center_y is not None and radius_x is not None and radius_y is not None:
            result["bbox"] = [
                float(center_x) - float(radius_x),
                float(center_y) - float(radius_y),
                float(center_x) + float(radius_x),
                float(center_y) + float(radius_y),
            ]
    if primitive in {"segment", "thick_segment", "tapered_polygon"} and "start" not in result:
        aliases = ("x1", "y1", "x2", "y2")
        if all(name in result for name in aliases):
            result["start"] = [result["x1"], result["y1"]]
            result["end"] = [result["x2"], result["y2"]]
    if primitive in {"segment", "thick_segment"} and "width" not in result and "thickness" in result:
        result["width"] = result["thickness"]
    if primitive == "ring" and "inset" not in result and "thickness" in result:
        result["inset"] = result["thickness"]
    return result


def _render_shape(draw: ImageDraw.ImageDraw, primitive: str, params: dict[str, Any], width: int,
                  height: int) -> None:
    params = _canonical_params(primitive, params)
    normalized = bool(params.get("normalized", False))
    if primitive == "polygon":
        points = [_point(point, width, height, normalized) for point in params["points"]]
        if len(points) < 3:
            raise ValueError("polygon requires at least 3 points")
        draw.polygon(points, fill=255)
    elif primitive in {"ellipse", "blob"}:
        if primitive == "blob" and "bbox" not in params:
            center = _point(params["center"], width, height, normalized)
            rx = _coord(params.get("radius_x", params.get("radius", 1)), width, normalized)
            ry = _coord(params.get("radius_y", params.get("radius", 1)), height, normalized)
            bbox = (center[0] - rx, center[1] - ry, center[0] + rx, center[1] + ry)
        else:
            bbox = _bbox(params["bbox"], width, height, normalized)
        draw.ellipse(bbox, fill=255)
    elif primitive in {"segment", "thick_segment"}:
        start = _point(params["start"], width, height, normalized)
        end = _point(params["end"], width, height, normalized)
        line_width = max(1, round(float(params.get("width", 1))))
        draw.line((start, end), fill=255, width=line_width)
        if primitive == "thick_segment" and params.get("round_caps", True):
            radius = max(0, (line_width - 1) // 2)
            for x, y in (start, end):
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    elif primitive == "tapered_polygon":
        start = _point(params["start"], width, height, normalized)
        end = _point(params["end"], width, height, normalized)
        points = _tapered_points(
            start,
            end,
            float(params.get("width_start", 2)),
            float(params.get("width_end", 1)),
        )
        draw.polygon(points, fill=255)
    elif primitive == "ring":
        outer = _bbox(params["bbox"], width, height, normalized)
        draw.ellipse(outer, fill=255)
        inset = max(1, round(float(params.get("inset", 1))))
        inner = (outer[0] + inset, outer[1] + inset, outer[2] - inset, outer[3] - inset)
        if inner[2] >= inner[0] and inner[3] >= inner[1]:
            draw.ellipse(inner, fill=0)
    elif primitive == "custom_mask":
        _draw_custom_mask(draw, params, width, height)
    else:
        raise ValueError("unsupported primitive: %s" % primitive)


def _child_mask(child: dict[str, Any], width: int, height: int, *, context: str) -> Image.Image:
    """Render one non-meta primitive used by a combinator to an isolated mask."""
    if not isinstance(child, dict):
        raise ValueError("%s requires child={primitive, params}" % context)
    primitive = str(child.get("primitive", ""))
    allowed = SUPPORTED_PRIMITIVES - {"repeat", "cutout", "mirror", "transform", "uv_fill"}
    if primitive not in allowed:
        raise ValueError("%s child uses unsupported primitive: %s" % (context, primitive))
    mask = Image.new("L", (width, height), 0)
    _render_shape(ImageDraw.Draw(mask), primitive, dict(child.get("params", {})), width, height)
    return mask


def _repeat_into(mask: Image.Image, spec: PrimitiveSpec, width: int, height: int) -> None:
    params = spec.params
    child = params.get("child")
    if not isinstance(child, dict):
        raise ValueError("repeat requires child={primitive, params}")
    primitive = str(child.get("primitive", ""))
    if primitive not in SUPPORTED_PRIMITIVES - {"repeat", "cutout", "mirror", "transform", "uv_fill"}:
        raise ValueError("repeat child uses unsupported primitive: %s" % primitive)
    count = int(params.get("count", 0))
    if not 1 <= count <= 64:
        raise ValueError("repeat count must be in 1..64")
    delta = params.get("delta", [0, 0])
    if not isinstance(delta, list) or len(delta) != 2:
        raise ValueError("repeat delta must be [x, y]")
    for index in range(count):
        child_params = dict(child.get("params", {}))
        dx, dy = float(delta[0]) * index, float(delta[1]) * index
        _translate_params(child_params, dx, dy)
        _render_shape(ImageDraw.Draw(mask), primitive, child_params, width, height)


def _paste_translated(source: Image.Image, width: int, height: int, dx: int, dy: int) -> Image.Image:
    target = Image.new("L", (width, height), 0)
    target.paste(source, (dx, dy))
    return target


def _transform_child(child: dict[str, Any], params: dict[str, Any], width: int, height: int,
                     *, context: str) -> Image.Image:
    """Apply discrete pixel-safe transforms without expanding a canvas.

    The coordinate system remains the target canvas. Scaling is around the child
    bounding-box centre unless an explicit `anchor: [x, y]` is supplied. This
    keeps transformations composable while avoiding image-library interpolation.
    """
    source = _child_mask(child, width, height, context=context)
    flip = str(params.get("flip", "none"))
    if flip not in {"none", "horizontal", "vertical", "both"}:
        raise ValueError("%s flip must be none/horizontal/vertical/both" % context)
    if flip in {"horizontal", "both"}:
        source = ImageOps.mirror(source)
    if flip in {"vertical", "both"}:
        source = ImageOps.flip(source)

    rotation = int(params.get("rotate", 0))
    if rotation % 90:
        raise ValueError("%s rotate must be a multiple of 90" % context)
    if rotation:
        source = source.rotate(-rotation, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)

    scale_x = float(params.get("scale_x", params.get("scale", 1.0)))
    scale_y = float(params.get("scale_y", params.get("scale", 1.0)))
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("%s scale values must be positive" % context)
    bbox = source.getbbox()
    if bbox and (scale_x != 1.0 or scale_y != 1.0):
        left, top, right, bottom = bbox
        cropped = source.crop(bbox)
        resized = cropped.resize(
            (max(1, round(cropped.width * scale_x)), max(1, round(cropped.height * scale_y))),
            Image.Resampling.NEAREST,
        )
        anchor = params.get("anchor")
        if anchor is None:
            center_x, center_y = (left + right - 1) / 2.0, (top + bottom - 1) / 2.0
        else:
            center_x, center_y = _point(anchor, width, height, bool(params.get("normalized", False)))
        source = Image.new("L", (width, height), 0)
        source.paste(resized, (round(center_x - (resized.width - 1) / 2.0),
                               round(center_y - (resized.height - 1) / 2.0)))

    translate = params.get("translate", [0, 0])
    if not isinstance(translate, list) or len(translate) != 2:
        raise ValueError("%s translate must be [x, y]" % context)
    return _paste_translated(source, width, height, round(float(translate[0])), round(float(translate[1])))


def _mirror_into(mask: Image.Image, spec: PrimitiveSpec, width: int, height: int) -> None:
    params = spec.params
    child = params.get("child")
    source = _child_mask(child, width, height, context="mirror")
    axis = str(params.get("axis", "vertical"))
    if axis not in {"vertical", "horizontal"}:
        raise ValueError("mirror axis must be vertical or horizontal")
    reflected = ImageOps.mirror(source) if axis == "vertical" else ImageOps.flip(source)
    if bool(params.get("include_original", True)):
        reflected = ImageChops.lighter(source, reflected)
    mask.paste(ImageChops.lighter(mask, reflected))


def _translate_params(params: dict[str, Any], dx: float, dy: float) -> None:
    """Translate known coordinate fields in-place for repeat.

    It remains intentionally small; complicated repeated structures can be
    emitted as multiple ordinary primitives by the planner.
    """
    for name in ("start", "end", "center", "offset"):
        value = params.get(name)
        if isinstance(value, list) and len(value) == 2:
            params[name] = [value[0] + dx, value[1] + dy]
    if isinstance(params.get("bbox"), list) and len(params["bbox"]) == 4:
        left, top, right, bottom = params["bbox"]
        params["bbox"] = [left + dx, top + dy, right + dx, bottom + dy]
    if isinstance(params.get("points"), list):
        params["points"] = [[point[0] + dx, point[1] + dy] for point in params["points"]]


def _rescale_coordinate(value: float | int, source: int, target: int) -> int:
    if source <= 1 or target <= 1:
        return 0
    # Coordinate values address pixels, so map the first and last source pixels
    # to the first and last target pixels. Clamp model overrun (for example a
    # bbox right edge given as the source width) to a valid target pixel.
    return min(target - 1, max(0, round(float(value) * (target - 1) / (source - 1))))


def _rescale_params(params: dict[str, Any], source_width: int, source_height: int,
                    target_width: int, target_height: int) -> dict[str, Any]:
    """Rescale coordinate-bearing DSL fields after a model chooses another grid.

    A model may reason on 32×32 while the request requires 16×16. Keeping its
    raw coordinates after merely replacing the canvas would silently drop parts.
    This conversion retains the geometry's relative construction and lets the
    normal validators decide whether the downsized result is usable.
    """
    result = dict(params)
    scale_x = (target_width - 1) / float(max(source_width - 1, 1))
    scale_y = (target_height - 1) / float(max(source_height - 1, 1))

    def point(value: Any) -> list[int] | Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [
                _rescale_coordinate(value[0], source_width, target_width),
                _rescale_coordinate(value[1], source_height, target_height),
            ]
        return value

    for key in ("start", "end", "center", "offset", "translate", "delta", "anchor"):
        if key in result:
            result[key] = point(result[key])
    if isinstance(result.get("points"), list):
        result["points"] = [point(item) for item in result["points"]]
    if isinstance(result.get("bbox"), (list, tuple)) and len(result["bbox"]) == 4:
        left, top, right, bottom = result["bbox"]
        result["bbox"] = [
            _rescale_coordinate(left, source_width, target_width),
            _rescale_coordinate(top, source_height, target_height),
            _rescale_coordinate(right, source_width, target_width),
            _rescale_coordinate(bottom, source_height, target_height),
        ]
    for key in ("cx", "x1", "x2"):
        if key in result:
            result[key] = _rescale_coordinate(result[key], source_width, target_width)
    for key in ("cy", "y1", "y2"):
        if key in result:
            result[key] = _rescale_coordinate(result[key], source_height, target_height)
    for key in ("radius_x", "rx"):
        if key in result:
            result[key] = max(1, round(float(result[key]) * scale_x))
    for key in ("radius_y", "ry"):
        if key in result:
            result[key] = max(1, round(float(result[key]) * scale_y))
    for key in ("radius", "width", "thickness", "width_start", "width_end", "inset"):
        if key in result:
            result[key] = max(1, round(float(result[key]) * min(scale_x, scale_y)))
    if isinstance(result.get("child"), dict):
        child = dict(result["child"])
        child["params"] = _rescale_params(
            dict(child.get("params", {})), source_width, source_height, target_width, target_height
        )
        result["child"] = child
    if isinstance(result.get("params"), dict):
        result["params"] = _rescale_params(
            dict(result["params"]), source_width, source_height, target_width, target_height
        )
    return result


def rescale_geometry_spec(spec: GeometrySpec, width: int, height: int) -> GeometrySpec:
    """Map every primitive in a geometry spec to a requested canvas size."""
    if spec.width == width and spec.height == height:
        return spec
    primitives = [
        replace(
            primitive,
            params=_rescale_params(primitive.params, spec.width, spec.height, width, height),
        )
        for primitive in spec.primitives
    ]
    return replace(spec, width=width, height=height, primitives=primitives)


def _shift_geometry_params(params: dict[str, Any], dx: int, dy: int) -> dict[str, Any]:
    """Shift absolute pixel coordinates while keeping combinator deltas intact."""
    if params.get("normalized") is True:
        # Normalized coordinates need a scale-aware rewrite rather than a pixel
        # offset. Leave them for the model repair path if they cannot fit.
        return dict(params)
    result = dict(params)

    def point(value: Any) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
        ):
            return [value[0] + dx, value[1] + dy]
        return value

    # A transform's ``translate`` moves the already-rasterized child. Shift
    # that operation as a whole; shifting both the child coordinates and the
    # transform would move it twice and can turn a feasible margin correction
    # into a new edge violation after nearest-neighbour scaling.
    operation_translation = "translate" in result
    for name in ("start", "end", "center", "offset", "anchor", "translate"):
        if name in result:
            result[name] = point(result[name])
    if isinstance(result.get("points"), list):
        result["points"] = [point(item) for item in result["points"]]
    if isinstance(result.get("bbox"), (list, tuple)) and len(result["bbox"]) == 4:
        left, top, right, bottom = result["bbox"]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool)
               for item in (left, top, right, bottom)):
            result["bbox"] = [left + dx, top + dy, right + dx, bottom + dy]
    for x_name, y_name in (("x1", "y1"), ("x2", "y2"), ("cx", "cy")):
        if x_name in result and y_name in result:
            if isinstance(result[x_name], (int, float)) and isinstance(result[y_name], (int, float)):
                result[x_name] += dx
                result[y_name] += dy

    # Recurse into child/shape parameter objects. Operation fields such as a
    # repeat delta or transform translate remain unchanged; shifting the child
    # itself moves the complete constructed primitive exactly once.
    if not operation_translation:
        for name in ("child", "params"):
            nested = result.get(name)
            if isinstance(nested, dict):
                result[name] = _shift_geometry_params(nested, dx, dy)
    return result


def fit_geometry_to_margins(
    spec: GeometrySpec,
    form: AssetForm,
    minimum_margin: int = 1,
) -> GeometrySpec:
    """Move a feasible item silhouette inward when it only misses a margin.

    This is a mechanical canvas fit, not an object-specific shape correction:
    all part masks, proportions and connections remain unchanged. If the
    silhouette is too large to fit, or uses normalized coordinates that cannot
    be shifted safely, the original spec is returned for model repair.
    """
    if form not in {AssetForm.ITEM, AssetForm.CROSS} or minimum_margin <= 0:
        return spec
    try:
        compiled = compile_geometry(spec)
    except (KeyError, TypeError, ValueError):
        return spec
    bbox = compiled.mask.getbbox()
    if bbox is None:
        return spec
    left, top, right, bottom = bbox
    available_width = spec.width - 2 * minimum_margin
    available_height = spec.height - 2 * minimum_margin
    if right - left > available_width or bottom - top > available_height:
        # A tiny asset can legitimately be one pixel too large after a model
        # rasterizes a diagonal custom mask. Preserve the model's labelled
        # construction and proportions by applying one nearest-neighbour
        # scale around each primitive's own centre before asking the model to
        # redesign it. This avoids a repair turn replacing a good dagger/tool
        # contour with a generic polygon. UV atlases never enter this helper.
        scale = min(
            available_width / float(max(right - left, 1)),
            available_height / float(max(bottom - top, 1)),
            1.0,
        )
        if scale <= 0.0:
            return spec
        scaled: list[PrimitiveSpec] = []
        for primitive in spec.primitives:
            scaled.append(
                replace(
                    primitive,
                    primitive="transform",
                    params={
                        "scale_x": scale,
                        "scale_y": scale,
                        "translate": [0, 0],
                        "child": {
                            "primitive": primitive.primitive,
                            "params": dict(primitive.params),
                        },
                    },
                )
            )
        adjusted = replace(spec, primitives=scaled)
        try:
            adjusted_bbox = compile_geometry(adjusted).mask.getbbox()
        except (KeyError, TypeError, ValueError):
            return spec
        if adjusted_bbox is not None:
            left, top, right, bottom = adjusted_bbox
            if (
                right - left <= available_width
                and bottom - top <= available_height
            ):
                spec = adjusted
                bbox = adjusted_bbox
            else:
                return spec

    def inward_shift(low_edge: int, high_edge: int, canvas: int) -> int:
        lower = minimum_margin - low_edge
        upper = (canvas - minimum_margin) - high_edge
        if lower > upper:
            return 0
        if lower > 0:
            return lower
        if upper < 0:
            return upper
        return 0

    dx = inward_shift(left, right, spec.width)
    dy = inward_shift(top, bottom, spec.height)
    if dx == 0 and dy == 0:
        return spec
    primitives = [
        replace(primitive, params=_shift_geometry_params(primitive.params, dx, dy))
        for primitive in spec.primitives
    ]
    adjusted = replace(spec, primitives=primitives)
    try:
        adjusted_bbox = compile_geometry(adjusted).mask.getbbox()
    except (KeyError, TypeError, ValueError):
        return spec
    if adjusted_bbox is None:
        return spec
    if (
        adjusted_bbox[0] < minimum_margin
        or adjusted_bbox[1] < minimum_margin
        or adjusted_bbox[2] > spec.width - minimum_margin
        or adjusted_bbox[3] > spec.height - minimum_margin
    ):
        return spec
    return adjusted


def compile_geometry(spec: GeometrySpec) -> CompiledGeometry:
    """Compile labelled primitive geometry to a union mask and per-part masks."""
    part_masks = {
        part.id: Image.new("L", (spec.width, spec.height), 0) for part in spec.parts
    }
    # A cutout only means anything on top of a base coat. Model-authored ids
    # are free-form, and "cut_face_opening" sorts before "fill_head", so
    # ordering by id alone carved an empty mask and then painted straight over
    # it. Fills always run first within a layer.
    ordered = sorted(
        spec.primitives,
        key=lambda item: (item.layer, 0 if item.primitive == "uv_fill" else 1, item.id),
    )
    for primitive in ordered:
        if primitive.primitive not in SUPPORTED_PRIMITIVES:
            raise ValueError(
                "primitive %s uses %r; supported: %s"
                % (primitive.id, primitive.primitive, ", ".join(sorted(SUPPORTED_PRIMITIVES)))
            )
        target = part_masks[primitive.part_id]
        if primitive.primitive == "uv_fill":
            requested = primitive.params.get("regions")
            if requested is None:
                regions = [region for region in spec.uv_regions if region.part_id == primitive.part_id]
            else:
                if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
                    raise ValueError("uv_fill regions must be a list of UV region ids")
                by_id = {region.id: region for region in spec.uv_regions}
                missing = [region_id for region_id in requested if region_id not in by_id]
                if missing:
                    raise ValueError("uv_fill references unknown UV region(s): %s" % ", ".join(missing))
                regions = [by_id[region_id] for region_id in requested]
                foreign = [region.id for region in regions if region.part_id != primitive.part_id]
                if foreign:
                    raise ValueError("uv_fill may only paint its own part; foreign region(s): %s" % ", ".join(foreign))
            if not regions:
                raise ValueError("uv_fill part %s has no declared UV regions" % primitive.part_id)
            draw = ImageDraw.Draw(target)
            for region in regions:
                left, top, right, bottom = region.bbox
                draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
        elif primitive.primitive == "cutout":
            target_name = str(primitive.params.get("target_part", primitive.part_id))
            if target_name not in part_masks:
                raise ValueError("cutout target_part does not exist: %s" % target_name)
            cut = _child_mask(
                {
                    "primitive": str(primitive.params.get("shape", "polygon")),
                    "params": dict(primitive.params.get("params", {})),
                },
                spec.width,
                spec.height,
                context="cutout",
            )
            part_masks[target_name] = ImageChops.subtract(part_masks[target_name], cut)
        elif primitive.primitive == "repeat":
            _repeat_into(target, primitive, spec.width, spec.height)
        elif primitive.primitive == "mirror":
            _mirror_into(target, primitive, spec.width, spec.height)
        elif primitive.primitive == "transform":
            transformed = _transform_child(
                primitive.params.get("child"), primitive.params, spec.width, spec.height, context="transform"
            )
            target.paste(ImageChops.lighter(target, transformed))
        else:
            # Every ordinary primitive contributes a positive mask. Rendering it
            # first into a scratch image matters for a ring: its internal hole
            # must not erase a core contributed by an earlier primitive.
            fragment = Image.new("L", (spec.width, spec.height), 0)
            _render_shape(ImageDraw.Draw(fragment), primitive.primitive, primitive.params, spec.width, spec.height)
            target.paste(ImageChops.lighter(target, fragment))

    union = Image.new("L", (spec.width, spec.height), 0)
    for part in sorted(spec.parts, key=lambda item: item.layer):
        union = ImageChops.lighter(union, part_masks[part.id])
    return CompiledGeometry(
        width=spec.width,
        height=spec.height,
        mask=union,
        part_masks=part_masks,
        primitive_order=[item.id for item in ordered],
    )
