"""Tests for UV layout canvas resolution and the shipped armour layouts."""

from __future__ import annotations

from pathlib import Path

from studio_next.contracts import AssetForm
from studio_next.quality import _layout_extent, _layout_matches_form
from studio_next.uv_layout import layout_canvas_extent

def test_declared_layout_size_wins_over_the_cube_bounding_box() -> None:
    """A leggings atlas is 64x32 even though its cubes stop at x=40."""
    from studio_next.uv_layout import layout_canvas_extent

    leggings = {
        "texture_width": 64,
        "texture_height": 32,
        "cubes": [
            {"id": "body", "part_id": "body", "u": 16, "v": 16, "width": 8, "height": 12, "depth": 4},
            {"id": "legs", "part_id": "legs", "u": 0, "v": 16, "width": 4, "height": 12, "depth": 4},
        ],
    }
    assert layout_canvas_extent(leggings) == (64, 32)


def test_a_region_larger_than_the_declaration_is_never_clipped() -> None:
    from studio_next.uv_layout import layout_canvas_extent

    oversized = {
        "texture_width": 16,
        "texture_height": 16,
        "regions": [{"id": "r", "part_id": "p", "bbox": [0, 0, 48, 20]}],
    }
    assert layout_canvas_extent(oversized) == (48, 20)


def test_the_shipped_armor_layouts_are_usable_entity_atlases() -> None:
    from studio_next.contracts import AssetForm
    from studio_next.quality import _layout_extent, _layout_matches_form

    for name in ("vanilla_1_12_armor_layer_1_64", "vanilla_1_12_armor_layer_2_64"):
        path = Path(__file__).resolve().parents[1] / "layouts" / (name + ".json")
        assert _layout_extent(path) == (64, 32)
        assert _layout_matches_form(path, AssetForm.ENTITY_UV)

