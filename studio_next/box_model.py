"""Derive a Minecraft UV layout from a model-authored box decomposition.

A mod's model does not exist in vanilla, so no shipped layout under layouts/
can describe it. This module closes that gap: the planner only has to say which
axis-aligned boxes the object is made of (width/height/depth per part), and the
texture atlas follows deterministically.

Minecraft's legacy ModelRenderer net for one box is fully determined by its
dimensions: the six faces tile a (2*depth + 2*width) by (depth + height)
rectangle. So packing the boxes is enough -- no texture coordinates are asked
of the model, and the result is reproducible rather than guessed.

This is deliberately not an attempt to reverse a 2D silhouette into a UV
unwrapping. That is not well posed: a silhouette is a projection, so many box
decompositions produce the same outline. Declaring the boxes removes the
ambiguity instead of trying to invert it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from .contracts import UvRegionSpec
from .uv_layout import CubeUVSpec, cube_to_regions


MAX_BOXES = 24
MAX_DIMENSION = 32
LAYOUT_FORMAT = "minecraft_1_12_modelrenderer_cube_uv"


def _identifier(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    cleaned = "".join(char if (char.isalnum() or char in "_-") else "_" for char in text).strip("_")
    return cleaned or fallback


def _positive_int(value: Any, label: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a whole number" % label) from None
    if not 1 <= number <= MAX_DIMENSION:
        raise ValueError("%s must be within 1..%d" % (label, MAX_DIMENSION))
    return number


def _origin(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("origin must be three whole numbers")
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError):
        raise ValueError("origin must be three whole numbers") from None


@dataclass(frozen=True)
class BoxSpec:
    """One physical part of the object, as an axis-aligned box."""

    id: str
    part_id: str
    width: int
    height: int
    depth: int
    origin: tuple[int, int, int] | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for label, value in (("width", self.width), ("height", self.height), ("depth", self.depth)):
            if not 1 <= value <= MAX_DIMENSION:
                raise ValueError("box %s must be within 1..%d" % (label, MAX_DIMENSION))

    @property
    def net_size(self) -> tuple[int, int]:
        """The (width, height) of this box's six-face texture net."""
        return (2 * self.depth + 2 * self.width, self.depth + self.height)


def boxes_from_dict(payload: Any) -> list[BoxSpec]:
    """Validate a model-authored decomposition into BoxSpec objects."""
    if isinstance(payload, list):
        raw_boxes = payload
    elif isinstance(payload, dict):
        raw_boxes = payload.get("boxes")
    else:
        raise ValueError("a box decomposition must be an object or a list")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("a box decomposition needs a non-empty boxes list")
    if len(raw_boxes) > MAX_BOXES:
        raise ValueError("at most %d boxes are supported" % MAX_BOXES)
    boxes: list[BoxSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_boxes):
        if not isinstance(item, dict):
            raise ValueError("each box must be an object")
        size = item.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 3:
            width, height, depth = size
        else:
            width = item.get("width")
            height = item.get("height")
            depth = item.get("depth")
        box_id = _identifier(item.get("id") or item.get("part_id"), "box_%d" % (index + 1))
        if box_id in seen:
            box_id = "%s_%d" % (box_id, index + 1)
        seen.add(box_id)
        boxes.append(BoxSpec(
            id=box_id,
            part_id=_identifier(item.get("part_id") or box_id, box_id),
            width=_positive_int(width, "box width"),
            height=_positive_int(height, "box height"),
            depth=_positive_int(depth, "box depth"),
            origin=_origin(item.get("origin")),
            notes=str(item.get("notes") or "").strip(),
        ))
    return boxes


def _next_power_of_two(value: int) -> int:
    size = 1
    while size < value:
        size *= 2
    return size


def _preview_placements(boxes: list[BoxSpec]) -> list[tuple[list[list[int]], int]]:
    """Front-view placement per box, plus its draw order.

    A front view of an axis-aligned box is just its (x, y) rectangle, so a
    declared model-space origin projects directly. When the model gave no
    origins the boxes are stacked in declaration order, which still yields a
    readable diagnostic instead of failing the run.
    """
    if all(box.origin is not None for box in boxes):
        min_x = min(box.origin[0] for box in boxes if box.origin)
        min_y = min(box.origin[1] for box in boxes if box.origin)
        min_z = min(box.origin[2] for box in boxes if box.origin)
        return [
            ([[box.origin[0] - min_x, box.origin[1] - min_y, box.width, box.height]], box.origin[2] - min_z)
            for box in boxes
            if box.origin
        ]
    placements: list[tuple[list[list[int]], int]] = []
    cursor = 0
    for index, box in enumerate(boxes):
        placements.append(([[0, cursor, box.width, box.height]], index))
        cursor += box.height + 1
    return placements


@dataclass(frozen=True)
class AuthoredLayout:
    """A UV layout derived from a box decomposition, ready to serialise."""

    cubes: tuple[CubeUVSpec, ...]
    regions: tuple[UvRegionSpec, ...]
    canvas: tuple[int, int]
    boxes: tuple[BoxSpec, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": LAYOUT_FORMAT,
            "source": "model-authored box decomposition",
            "texture_width": self.canvas[0],
            "texture_height": self.canvas[1],
            "notes": self.notes or (
                "Derived from the object's own box decomposition, not a shipped vanilla model. "
                "Front preview coordinates are a diagnostic orthographic projection."
            ),
            "cubes": [
                {
                    "id": cube.id,
                    "part_id": cube.part_id,
                    "u": cube.u,
                    "v": cube.v,
                    "width": cube.width,
                    "height": cube.height,
                    "depth": cube.depth,
                    "preview_instances": cube.preview_instances,
                    "preview_layer": cube.preview_layer,
                    "notes": cube.notes,
                }
                for cube in self.cubes
            ],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target


def pack_boxes(
    boxes: Iterable[BoxSpec],
    *,
    canvas_width: int | None = None,
    margin: int = 0,
) -> AuthoredLayout:
    """Lay the box nets out on a power-of-two atlas and derive the UV regions.

    Boxes are packed in declaration order into shelves. Nothing overlaps: a
    shelf advances by each net's exact width, so every box owns a disjoint
    rectangle and the canvas is simply grown until it fits.
    """
    ordered = list(boxes)
    if not ordered:
        raise ValueError("a layout needs at least one box")
    if margin < 0:
        raise ValueError("margin cannot be negative")

    widest = max(box.net_size[0] for box in ordered)
    width = canvas_width or max(64, widest)
    if width < widest:
        width = widest

    placements: list[tuple[BoxSpec, int, int]] = []
    x = y = row_height = 0
    for box in ordered:
        net_width, net_height = box.net_size
        if x > 0 and x + net_width > width:
            y += row_height + margin
            x = 0
            row_height = 0
        placements.append((box, x, y))
        x += net_width + margin
        row_height = max(row_height, net_height)

    used_height = y + row_height
    canvas = (_next_power_of_two(width), _next_power_of_two(max(16, used_height)))

    previews = _preview_placements(ordered)
    preview_by_id = {box.id: placement for box, placement in zip(ordered, previews)}

    cubes: list[CubeUVSpec] = []
    for box, u, v in placements:
        instances, layer = preview_by_id[box.id]
        cubes.append(CubeUVSpec(
            id=box.id,
            part_id=box.part_id,
            u=u,
            v=v,
            width=box.width,
            height=box.height,
            depth=box.depth,
            preview_instances=[list(item) for item in instances],
            preview_layer=layer,
            notes=box.notes or "box %dx%dx%d" % (box.width, box.height, box.depth),
        ))

    regions: list[UvRegionSpec] = []
    for cube in cubes:
        regions.extend(cube_to_regions(cube))

    overflow = [region.id for region in regions if region.bbox[2] > canvas[0] or region.bbox[3] > canvas[1]]
    if overflow:
        raise ValueError("packed UV regions exceed the canvas: %s" % ", ".join(overflow[:4]))

    return AuthoredLayout(
        cubes=tuple(cubes),
        regions=tuple(regions),
        canvas=canvas,
        boxes=tuple(ordered),
        notes="",
    )
