"""Tests for the entity alpha authority selection.

Two armour layers are both 64x32, so size alone once handed layer 1's alpha to
a layer 2 target: the leggings came out with a helmet and arms stamped on them
and holes chewed out of the correct leg regions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from studio_next.contracts import (
    AppearanceSpec,
    AssetForm,
    AssetRequest,
    GeometrySpec,
    PartAppearance,
    PartSpec,
    PrimitiveSpec,
    ReferenceAsset,
    ReferenceRole,
    ShapeDescriptor,
    UvRegionSpec,
)
from studio_next.pipeline import GenerationPlan, _entity_reference_alpha_contract

REGIONS = [UvRegionSpec(id="body_front", part_id="body", bbox=[16, 16, 40, 32])]


def _plan(tmp_path: Path, reference_paths: list[Path]) -> GenerationPlan:
    parts = [PartSpec(id="body", meaning="body")]
    return GenerationPlan(
        request=AssetRequest(
            query="leggings",
            form=AssetForm.ENTITY_UV,
            width=64,
            height=32,
            shape_policy="target_model_uv",
        ),
        descriptor=ShapeDescriptor(target="leggings", semantic="armour leg texture", visual_identity=["plates"], parts=parts),
        geometry=GeometrySpec(
            width=64,
            height=32,
            parts=parts,
            # A full base coat over every declared region expresses no opinion,
            # which is precisely when a same-model reference may refine the
            # silhouette. Any cutout or omitted region disables it again.
            primitives=[
                PrimitiveSpec(
                    id="body_uv_fill",
                    part_id="body",
                    primitive="uv_fill",
                    params={"regions": [region.id for region in REGIONS]},
                )
            ],
            uv_regions=REGIONS,
        ),
        appearance=AppearanceSpec(palette={"wood": "#8A6A3F"}, parts={"body": PartAppearance(colors=["wood"])}),
        references=[
            ReferenceAsset(
                path=str(path),
                name=path.stem,
                roles=[ReferenceRole.SHAPE, ReferenceRole.PIXEL_STYLE],
                features={"width": 64, "height": 32},
            )
            for path in reference_paths
        ],
    )


def _alpha_image(path: Path, boxes: list[list[int]]) -> None:
    image = Image.new("RGBA", (64, 32), (90, 70, 45, 0))
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, fill=(90, 70, 45, 255))
    image.save(path)


def test_a_reference_that_stays_inside_the_regions_is_accepted(tmp_path: Path) -> None:
    inside = tmp_path / "inside.png"
    _alpha_image(inside, [[16, 16, 39, 31]])
    contract = _entity_reference_alpha_contract(_plan(tmp_path, [inside]))
    assert contract is not None
    assert contract.reference_path == str(inside)


def test_a_same_size_reference_that_paints_outside_the_regions_is_rejected(tmp_path: Path) -> None:
    """This is the layer_1-onto-layer_2 case: it must fall back to geometry."""
    outside = tmp_path / "outside.png"
    _alpha_image(outside, [[0, 0, 63, 31]])
    assert _entity_reference_alpha_contract(_plan(tmp_path, [outside])) is None


def test_a_model_authored_opening_disables_the_authority(tmp_path: Path) -> None:
    """An authority must never paint over an opening the model chose.

    A live run carved a face opening, the reference alpha put those cells back,
    and the renderer filled them from the reference's own pixels: a metal
    layer's grey appeared inside a wooden helmet.
    """
    inside = tmp_path / "inside.png"
    _alpha_image(inside, [[16, 16, 39, 31]])
    plan = _plan(tmp_path, [inside])
    cut = replace(
        plan,
        request=replace(plan.request, shape_policy="free"),
        geometry=replace(
            plan.geometry,
            primitives=[
                *plan.geometry.primitives,
                PrimitiveSpec(
                    id="face_opening",
                    part_id="body",
                    primitive="cutout",
                    params={"shape": "ellipse", "params": {"bbox": [20, 20, 24, 24]}},
                    layer=5,
                ),
            ],
        ),
    )
    assert _entity_reference_alpha_contract(cut) is None


def test_the_same_opening_keeps_the_authority_under_target_model_uv(tmp_path: Path) -> None:
    """A caller asking for the source model's contract still gets it."""
    inside = tmp_path / "inside.png"
    _alpha_image(inside, [[16, 16, 39, 31]])
    plan = _plan(tmp_path, [inside])  # shape_policy defaults to target_model_uv
    cut = replace(
        plan,
        geometry=replace(
            plan.geometry,
            primitives=[
                *plan.geometry.primitives,
                PrimitiveSpec(
                    id="face_opening",
                    part_id="body",
                    primitive="cutout",
                    params={"shape": "ellipse", "params": {"bbox": [20, 20, 24, 24]}},
                    layer=5,
                ),
            ],
        ),
    )
    assert _entity_reference_alpha_contract(cut) is not None


def test_a_dither_reference_is_rejected_even_inside_the_regions(tmp_path: Path) -> None:
    """A leather overlay is isolated pixels, not a model face.

    Its pixels all sit inside the UV regions, so containment alone accepted it
    and the solid armour plate came out punched full of holes.
    """
    dither = tmp_path / "overlay.png"
    image = Image.new("RGBA", (64, 32), (90, 70, 45, 0))
    draw = ImageDraw.Draw(image)
    for y in range(16, 32, 2):
        for x in range(16, 40, 2):
            draw.point((x, y), fill=(90, 70, 45, 255))
    image.save(dither)
    assert _entity_reference_alpha_contract(_plan(tmp_path, [dither])) is None


def test_a_solid_but_sparse_reference_is_rejected(tmp_path: Path) -> None:
    """A trim overlay is solid yet paints only part of the model.

    Vanilla armour measures 0.40-0.49 coverage for a full layer and 0.03-0.19
    for an overlay, so a solid 4x4 patch of a 384-pixel region is a trim.
    """
    trim = tmp_path / "trim.png"
    _alpha_image(trim, [[16, 16, 19, 19]])
    assert _entity_reference_alpha_contract(_plan(tmp_path, [trim])) is None


def test_the_best_matching_reference_wins(tmp_path: Path) -> None:
    outside = tmp_path / "layer_1.png"
    inside = tmp_path / "layer_2.png"
    _alpha_image(outside, [[0, 0, 63, 31]])
    _alpha_image(inside, [[16, 16, 39, 31]])
    contract = _entity_reference_alpha_contract(_plan(tmp_path, [outside, inside]))
    assert contract is not None
    assert contract.reference_path == str(inside)
