"""Generic cube-UV expansion for legacy Minecraft ModelRenderer layouts.

The format describes a model's texture offsets and cube dimensions, then
derives the six actual face rectangles. It is useful for vanilla 1.12-style
models and custom Forge models that use the same ModelRenderer convention; it
does not assume a player, villager, or any other entity category.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import UvRegionSpec


@dataclass(frozen=True)
class CubeUVSpec:
    id: str
    part_id: str
    u: int
    v: int
    width: int
    height: int
    depth: int
    required: bool = True
    notes: str = ""
    preview_origin: list[int] | None = None
    preview_size: list[int] | None = None
    preview_layer: int = 0
    preview_instances: list[list[int]] | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.part_id:
            raise ValueError("cube UV spec needs id and part_id")
        if min(self.u, self.v) < 0 or min(self.width, self.height, self.depth) <= 0:
            raise ValueError("cube UV offsets must be non-negative and dimensions positive")


def cube_to_regions(cube: CubeUVSpec) -> list[UvRegionSpec]:
    """Expand one `ModelRenderer.addBox` texture offset into six UV faces.

    Minecraft's legacy cube net places top and bottom above the side strip:
    `side-z, front-x, side-z, back-x` below it. Bboxes are half-open to match
    Python/Pillow crop conventions and the rest of this project's UV contract.
    """
    u, v, x, y, z = cube.u, cube.v, cube.width, cube.height, cube.depth
    faces = {
        "top": [u + z, v, u + z + x, v + z],
        "bottom": [u + z + x, v, u + z + 2 * x, v + z],
        "left": [u, v + z, u + z, v + z + y],
        "front": [u + z, v + z, u + z + x, v + z + y],
        "right": [u + z + x, v + z, u + 2 * z + x, v + z + y],
        "back": [u + 2 * z + x, v + z, u + 2 * z + 2 * x, v + z + y],
    }
    return [
        UvRegionSpec(
            id="%s_%s" % (cube.id, face),
            part_id=cube.part_id,
            bbox=bbox,
            face=face,
            required=cube.required,
            notes=cube.notes,
            preview_origin=cube.preview_origin,
            preview_size=cube.preview_size,
            preview_layer=cube.preview_layer,
            preview_instances=cube.preview_instances,
        )
        for face, bbox in faces.items()
    ]


def layout_regions(data: object) -> list[UvRegionSpec]:
    """Load either explicit regions or compact cube declarations from JSON data."""
    if isinstance(data, list):
        return [UvRegionSpec(**region) for region in data]
    if not isinstance(data, dict):
        raise ValueError("UV layout must be a regions list or an object")
    regions = data.get("regions")
    cubes = data.get("cubes")
    if regions is not None and cubes is not None:
        raise ValueError("UV layout must use either regions or cubes, not both")
    if regions is not None:
        if not isinstance(regions, list) or not regions:
            raise ValueError("UV layout regions must be a non-empty list")
        return [UvRegionSpec(**region) for region in regions]
    if not isinstance(cubes, list) or not cubes:
        raise ValueError("UV layout must contain a non-empty regions or cubes list")
    result: list[UvRegionSpec] = []
    for raw in cubes:
        result.extend(cube_to_regions(CubeUVSpec(**raw)))
    return result


def layout_declared_size(data: object) -> tuple[int, int] | None:
    """The atlas size a layout declares, when it declares one."""
    if not isinstance(data, dict):
        return None
    try:
        width = int(data.get("texture_width"))
        height = int(data.get("texture_height"))
    except (TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def layout_canvas_extent(data: object) -> tuple[int, int]:
    """The canvas a layout needs: its declared atlas size, else its regions.

    A layout that declares texture_width/height knows the real atlas it targets
    -- a villager atlas is 64x64 even though its cubes stop at 62, and a leggings
    atlas is 64x32 even though its content stops at 40. The declaration therefore
    wins, while the region box still guarantees no region is ever clipped.
    """
    regions = layout_regions(data)
    box_width = max(region.bbox[2] for region in regions)
    box_height = max(region.bbox[3] for region in regions)
    declared = layout_declared_size(data)
    if declared is None:
        return (box_width, box_height)
    return (max(box_width, declared[0]), max(box_height, declared[1]))


def layout_summary(regions: list[UvRegionSpec]) -> dict[str, Any]:
    by_part: dict[str, int] = {}
    for region in regions:
        by_part[region.part_id] = by_part.get(region.part_id, 0) + 1
    return {"region_count": len(regions), "parts": by_part}


def render_front_preview(texture: Any, regions: list[UvRegionSpec], scale: int = 8) -> Any:
    """Render declared front faces into a generic model-facing preview.

    This is a diagnostic view, not a replacement renderer: it requires layouts
    to provide `preview_origin` (and optionally `preview_size`) for front faces.
    The same function can therefore preview any cube-UV model without knowing
    whether it is a villager, machine, animal or custom entity.
    """
    from PIL import Image

    if scale <= 0:
        raise ValueError("preview scale must be positive")
    front = [
        region for region in regions
        if region.face == "front" and (region.preview_origin is not None or region.preview_instances is not None)
    ]
    if not front:
        raise ValueError("UV layout has no front regions with preview_origin metadata")
    bounds = []
    for region in front:
        instances = region.preview_instances
        if instances is None:
            origin = region.preview_origin
            size = region.preview_size or [region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]]
            instances = [[origin[0], origin[1], size[0], size[1]]]
        bounds.extend((item[0], item[1], item[0] + item[2], item[1] + item[3]) for item in instances)
    left = min(item[0] for item in bounds)
    top = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    bottom = max(item[3] for item in bounds)
    output = Image.new("RGBA", ((right - left) * scale, (bottom - top) * scale), (0, 0, 0, 0))
    source = texture.convert("RGBA")
    for region in sorted(front, key=lambda item: item.preview_layer):
        instances = region.preview_instances
        if instances is None:
            origin = region.preview_origin
            size = region.preview_size or [region.bbox[2] - region.bbox[0], region.bbox[3] - region.bbox[1]]
            instances = [[origin[0], origin[1], size[0], size[1]]]
        for origin_x, origin_y, size_x, size_y in instances:
            crop = source.crop(tuple(region.bbox))
            crop = crop.resize((size_x * scale, size_y * scale), Image.Resampling.NEAREST)
            output.alpha_composite(crop, ((origin_x - left) * scale, (origin_y - top) * scale))
    return output


def render_entity_preview(texture: Any, regions: list[UvRegionSpec], scale: int = 8) -> Any:
    """Compose a compact orthographic/isometric entity diagnostic from an atlas.

    Entity textures are not readable as a flat 64x32 sheet. This renderer uses
    each cube's declared front face plus its right and top faces, placing the
    latter beside/above the front rectangle using the same preview instances.
    It remains layout-driven and works for any legacy cube UV atlas; no cow,
    villager or player silhouette is embedded here.
    """
    from PIL import Image

    if scale <= 0:
        raise ValueError("preview scale must be positive")
    source = texture.convert("RGBA")
    by_part: dict[str, dict[str, UvRegionSpec]] = {}
    for region in regions:
        by_part.setdefault(region.part_id, {})[region.face.lower()] = region
    placements: list[tuple[int, Image.Image, int, int]] = []
    bounds: list[tuple[int, int, int, int]] = []
    for part_id, faces in by_part.items():
        front = next((faces[name] for name in ("front", "north", "south") if name in faces), None)
        if front is None or (front.preview_origin is None and front.preview_instances is None):
            continue
        instances = front.preview_instances
        if instances is None:
            origin = front.preview_origin
            size = front.preview_size or [front.bbox[2] - front.bbox[0], front.bbox[3] - front.bbox[1]]
            instances = [[origin[0], origin[1], size[0], size[1]]]
        side = next((faces[name] for name in ("right", "east", "left", "west") if name in faces), None)
        top = faces.get("top")
        for origin_x, origin_y, size_x, size_y in instances:
            front_image = source.crop(tuple(front.bbox)).resize((size_x * scale, size_y * scale), Image.Resampling.NEAREST)
            placements.append((front.preview_layer + 2, front_image, origin_x, origin_y))
            bounds.append((origin_x, origin_y, origin_x + size_x, origin_y + size_y))
            if side is not None:
                side_image = source.crop(tuple(side.bbox)).resize(((side.bbox[2] - side.bbox[0]) * scale, size_y * scale), Image.Resampling.NEAREST)
                side_x = origin_x + size_x
                placements.append((front.preview_layer, side_image, side_x, origin_y))
                bounds.append((side_x, origin_y, side_x + (side.bbox[2] - side.bbox[0]), origin_y + size_y))
            if top is not None:
                top_image = source.crop(tuple(top.bbox)).resize((size_x * scale, (top.bbox[3] - top.bbox[1]) * scale), Image.Resampling.NEAREST)
                top_y = origin_y - (top.bbox[3] - top.bbox[1])
                placements.append((front.preview_layer + 1, top_image, origin_x, top_y))
                bounds.append((origin_x, origin_y - (top.bbox[3] - top.bbox[1]), origin_x + size_x, origin_y))
    if not bounds:
        raise ValueError("UV layout has no front regions with preview metadata")
    left = min(item[0] for item in bounds)
    top_bound = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    bottom = max(item[3] for item in bounds)
    output = Image.new("RGBA", ((right - left) * scale, (bottom - top_bound) * scale), (0, 0, 0, 0))
    for _layer, image, x, y in sorted(placements, key=lambda item: item[0]):
        output.alpha_composite(image, ((x - left) * scale, (y - top_bound) * scale))
    return output


def render_isometric_preview(texture: Any, regions: list[UvRegionSpec], scale: int = 8) -> Any:
    """Compose top/front/side UV faces into a small pixel-isometric preview.

    This is intentionally driven by face labels and rectangles, so it works
    for any block or cube atlas rather than a hard-coded block type. Each
    source pixel becomes a nearest-neighbour parallelogram; no smoothing or
    3-D texture synthesis is introduced.
    """
    from math import ceil
    from PIL import Image, ImageDraw

    if scale <= 0:
        raise ValueError("preview scale must be positive")
    def choose(*names: str) -> UvRegionSpec | None:
        wanted = {name.lower() for name in names}
        return next((region for region in regions if region.face.lower() in wanted), None)

    top = choose("top")
    front = choose("front", "north", "south")
    side = choose("right", "east", "west", "side", "left")
    if top is None or front is None or side is None:
        raise ValueError("isometric preview requires top, front and side UV regions")
    source = texture.convert("RGBA")
    top_image = source.crop(tuple(top.bbox))
    front_image = source.crop(tuple(front.bbox))
    side_image = source.crop(tuple(side.bbox))
    width = max(top_image.width, front_image.width, 1)
    depth = max(top_image.height, side_image.width, 1)
    height = max(front_image.height, side_image.height, 1)
    half = scale / 2.0
    margin = scale * 2
    top_origin = (margin + depth * half, margin)
    top_u = (half, half)
    top_v = (-half, half)
    front_origin = (top_origin[0] + depth * top_v[0], top_origin[1] + depth * top_v[1])
    side_origin = (top_origin[0] + width * top_u[0], top_origin[1] + width * top_u[1])
    max_x = top_origin[0] + width * top_u[0]
    min_x = top_origin[0] - depth * half
    max_y = top_origin[1] + (width + depth) * half + height * scale
    output = Image.new("RGBA", (ceil(max_x - min_x + margin * 2), ceil(max_y + margin)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(output)

    def paint(face: Image.Image, origin: tuple[float, float], basis_u: tuple[float, float], basis_v: tuple[float, float]) -> None:
        for py in range(face.height):
            for px in range(face.width):
                color = face.getpixel((px, py))
                if color[3] < 8:
                    continue
                x0 = origin[0] - min_x + px * basis_u[0] + py * basis_v[0]
                y0 = origin[1] + px * basis_u[1] + py * basis_v[1]
                points = [
                    (round(x0), round(y0)),
                    (round(x0 + basis_u[0]), round(y0 + basis_u[1])),
                    (round(x0 + basis_u[0] + basis_v[0]), round(y0 + basis_u[1] + basis_v[1])),
                    (round(x0 + basis_v[0]), round(y0 + basis_v[1])),
                ]
                draw.polygon(points, fill=color)

    # Paint order follows a cube: top first, then the far side, then the front.
    paint(top_image, top_origin, top_u, top_v)
    paint(side_image, side_origin, top_v, (0, scale))
    paint(front_image, front_origin, top_u, (0, scale))
    return output
