"""Tests for deriving a UV layout from a model-authored box decomposition."""

from __future__ import annotations

from pathlib import Path

import pytest

from studio_next.box_model import BoxSpec, boxes_from_dict, pack_boxes
from studio_next.contracts import AssetForm
from studio_next.plans import uv_layout_from_file
from studio_next.quality import _layout_extent, _layout_matches_form

BOXES = [
    BoxSpec(id="head", part_id="head", width=8, height=8, depth=8, origin=(-4, -8, -4)),
    BoxSpec(id="body", part_id="body", width=8, height=12, depth=4, origin=(-4, 0, -2)),
    BoxSpec(id="tail", part_id="tail", width=1, height=6, depth=1, origin=(0, 2, 3)),
]


def test_regions_are_disjoint_and_inside_the_canvas() -> None:
    layout = pack_boxes(BOXES)
    seen: set[tuple[int, int]] = set()
    for region in layout.regions:
        left, top, right, bottom = region.bbox
        assert 0 <= left < right <= layout.canvas[0]
        assert 0 <= top < bottom <= layout.canvas[1]
        for y in range(top, bottom):
            for x in range(left, right):
                assert (x, y) not in seen, "two faces overlap at %s" % ((x, y),)
                seen.add((x, y))


def test_canvas_is_a_power_of_two_and_covers_every_region() -> None:
    layout = pack_boxes(BOXES)
    width, height = layout.canvas
    assert width & (width - 1) == 0 and height & (height - 1) == 0
    assert max(region.bbox[2] for region in layout.regions) <= width
    assert max(region.bbox[3] for region in layout.regions) <= height


def test_a_different_decomposition_produces_a_different_atlas() -> None:
    """The atlas follows the object's own shape, which is the whole point."""
    single = pack_boxes([BoxSpec(id="blob", part_id="blob", width=8, height=8, depth=8)])
    stacked = pack_boxes([
        BoxSpec(id="top", part_id="top", width=8, height=8, depth=8),
        BoxSpec(id="bottom", part_id="bottom", width=8, height=8, depth=8, origin=(0, 8, 0)),
    ])
    assert single.regions != stacked.regions
    assert {region.part_id for region in single.regions} == {"blob"}
    assert {region.part_id for region in stacked.regions} == {"top", "bottom"}


def test_the_derived_layout_satisfies_the_entity_uv_contract(tmp_path: Path) -> None:
    layout = pack_boxes(BOXES)
    path = layout.write(tmp_path / "authored.json")
    assert _layout_matches_form(path, AssetForm.ENTITY_UV) is True
    assert _layout_extent(path) == layout.canvas
    assert len(uv_layout_from_file(path)) == 6 * len(BOXES)


def test_boxes_without_origins_still_get_a_usable_preview() -> None:
    layout = pack_boxes([
        BoxSpec(id="a", part_id="a", width=4, height=4, depth=4),
        BoxSpec(id="b", part_id="b", width=4, height=4, depth=4),
    ])
    instances = [cube.preview_instances for cube in layout.cubes]
    assert all(item for item in instances)
    assert instances[0] != instances[1]


def test_a_wide_box_widens_the_canvas_instead_of_being_clipped() -> None:
    layout = pack_boxes([BoxSpec(id="wide", part_id="wide", width=32, height=4, depth=8)])
    assert layout.canvas[0] >= 2 * 8 + 2 * 32
    assert max(region.bbox[2] for region in layout.regions) <= layout.canvas[0]


def test_boxes_from_dict_normalises_and_validates() -> None:
    boxes = boxes_from_dict({"boxes": [{"id": "Head", "size": [8, 8, 8], "origin": [-4, -8, -4]}]})
    assert boxes[0].part_id == "head"
    assert boxes[0].origin == (-4, -8, -4)
    with pytest.raises(ValueError):
        boxes_from_dict({"boxes": []})
    with pytest.raises(ValueError):
        boxes_from_dict({"boxes": [{"id": "x", "size": [0, 8, 8]}]})
    with pytest.raises(ValueError):
        boxes_from_dict({"boxes": [{"id": "x", "size": [8, 8, 99]}]})
    with pytest.raises(ValueError):
        boxes_from_dict({"boxes": [{"id": "x", "size": [8, 8, 8]} for _ in range(40)]})
