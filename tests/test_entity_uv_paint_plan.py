"""Tests for how an entity atlas keeps the model's own paint plan.

Normalization used to replace every model-authored primitive with a full-face
base coat, so a suit could never carry a face, neck or hand opening.
"""

from __future__ import annotations

from studio_next.contracts import (
    GeometrySpec,
    PartSpec,
    PrimitiveSpec,
    ShapeDescriptor,
    UvRegionSpec,
)
from studio_next.geometry import compile_geometry
from studio_next.plans import _normalize_entity_uv_geometry

REGIONS = [
    UvRegionSpec(id="head_top", part_id="head", bbox=[0, 0, 4, 4], face="top"),
    UvRegionSpec(id="head_front", part_id="head", bbox=[4, 0, 8, 4], face="front"),
]
PARTS = [PartSpec(id="head", meaning="head", style_role="entity_surface")]


def _descriptor() -> ShapeDescriptor:
    return ShapeDescriptor(target="helm", semantic="a helm", visual_identity=["shell"], parts=PARTS)


def _geometry(primitives: list[PrimitiveSpec]) -> GeometrySpec:
    return GeometrySpec(width=16, height=16, parts=PARTS, primitives=primitives, uv_regions=REGIONS)


def _painted(geometry: GeometrySpec) -> set[tuple[int, int]]:
    compiled = compile_geometry(geometry)
    return {
        (x, y)
        for y in range(16)
        for x in range(16)
        if compiled.mask.getpixel((x, y)) > 0
    }


def test_the_models_own_region_selection_survives() -> None:
    """A model may paint only some faces, which is how an opening is authored."""
    geometry = _geometry([
        PrimitiveSpec(
            id="head_uv_fill", part_id="head", primitive="uv_fill",
            params={"regions": ["head_top"]},
        )
    ])
    normalized = _normalize_entity_uv_geometry(geometry, _descriptor(), REGIONS)
    fills = [p for p in normalized.primitives if p.primitive == "uv_fill"]
    assert [p.params.get("regions") for p in fills] == [["head_top"]]
    assert _painted(normalized) == {(x, y) for y in range(4) for x in range(4)}


def test_a_part_the_model_left_unauthored_still_gets_a_base_coat() -> None:
    normalized = _normalize_entity_uv_geometry(_geometry([]), _descriptor(), REGIONS)
    assert _painted(normalized) == {
        (x, y) for y in range(4) for x in range(8)
    }


def test_a_model_cutout_is_kept_alongside_the_base_coat() -> None:
    """A cutout is how a model carves an opening; it must not be dropped."""
    geometry = _geometry([
        PrimitiveSpec(
            id="head_hole", part_id="head", primitive="cutout",
            params={"shape": "ellipse", "params": {"bbox": [0, 0, 2, 2]}},
            layer=5,
        )
    ])
    normalized = _normalize_entity_uv_geometry(geometry, _descriptor(), REGIONS)
    kinds = [p.primitive for p in normalized.primitives]
    assert "cutout" in kinds, "the model's cutout was dropped"
    assert "uv_fill" in kinds, "the unauthored part still needs a base coat"


def test_a_primitive_for_a_foreign_part_is_dropped() -> None:
    """Keeping the model's plan must not let it paint outside the atlas.

    The part is declared on the geometry (so the contract accepts it) but the
    layout has no region for it, so normalization must drop the primitive.
    """
    parts = PARTS + [PartSpec(id="ghost", meaning="ghost", style_role="entity_surface")]
    geometry = GeometrySpec(
        width=16,
        height=16,
        parts=parts,
        primitives=[PrimitiveSpec(id="ghost_fill", part_id="ghost", primitive="uv_fill", params={})],
        uv_regions=REGIONS,
    )
    normalized = _normalize_entity_uv_geometry(geometry, _descriptor(), REGIONS)
    assert all(p.part_id != "ghost" for p in normalized.primitives)
