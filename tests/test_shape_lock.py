"""Tests for the vanilla-contour lock.

A live bow run asked for a locally broken limb (requires_silhouette_change was
true) but wrote "preserve the intact limb arc" in its strategy, and that single
word replaced the model's geometry with a partition of the vanilla bow alpha.

A second live run (a crystal bow) declared shape_edit_mode="appearance_only"
and described the reference as the "silhouette host", matched no preserve
keyword, and shipped a hand-drawn blob in place of the vanilla contour.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PIL import Image

from studio_next.contracts import (
    ArtDirection,
    GeometrySpec,
    PartSpec,
    PrimitiveSpec,
    ReferenceAsset,
    ReferenceRole,
    ShapeDescriptor,
)
from studio_next.geometry import compile_geometry
from studio_next.pipeline import _reference_shape_is_locked
from studio_next.reference_geometry import (
    _same_size_shape_reference,
    reference_silhouette_partition,
)

PARTS = [PartSpec(id="limb", meaning="bow limb")]

NEEDS_NEW_CONTOUR = ArtDirection(
    target="broken bow",
    primary_subject="a bow with a snapped upper limb",
    design_intent="replace the smooth taper with a frayed splintered edge",
    requires_silhouette_change=True,
)

KEEPS_CONTOUR = ArtDirection(
    target="recoloured bow",
    primary_subject="the same bow in a different wood",
    design_intent="recolour only",
    requires_silhouette_change=False,
)


def _descriptor(strategy: str, direction: ArtDirection | None = None) -> ShapeDescriptor:
    return ShapeDescriptor(
        target="bow",
        semantic="a bow",
        visual_identity=["arc", "string"],
        parts=PARTS,
        reference_strategy=strategy,
        art_direction=direction,
    )


def test_preserve_wording_locks_the_vanilla_contour() -> None:
    descriptor = _descriptor("preserve the vanilla bow silhouette, change the palette")
    assert _reference_shape_is_locked(descriptor) is True


def test_an_explicit_edit_cue_unlocks_it() -> None:
    descriptor = _descriptor("preserve the arc but cut a damaged notch into the limb")
    assert _reference_shape_is_locked(descriptor) is False


def test_the_art_direction_outranks_preserve_wording() -> None:
    """The immutable brief is decided before any vanilla image is seen."""
    descriptor = _descriptor(
        "preserve the intact limb arc, locally shorten the upper limb tip",
        NEEDS_NEW_CONTOUR,
    )
    assert _reference_shape_is_locked(descriptor) is False


def test_a_direction_that_keeps_the_contour_does_not_unlock() -> None:
    descriptor = _descriptor("preserve the vanilla bow silhouette", KEEPS_CONTOUR)
    assert _reference_shape_is_locked(descriptor) is True


def test_appearance_only_locks_even_when_the_prose_matches_no_cue() -> None:
    """The crystal-bow regression: structured intent, no preserve keyword."""
    descriptor = _descriptor(
        "use bow:bow_standby alpha as the silhouette host; recolor from the crystal palette",
        KEEPS_CONTOUR,
    )
    assert descriptor.shape_edit_mode == "model_decides"
    locked = replace(descriptor, shape_edit_mode="appearance_only")
    assert _reference_shape_is_locked(locked) is True


def test_a_structured_new_silhouette_outranks_preserve_wording() -> None:
    descriptor = _descriptor("preserve the vanilla bow silhouette", KEEPS_CONTOUR)
    rewritten = replace(descriptor, shape_edit_mode="new_silhouette")
    assert _reference_shape_is_locked(rewritten) is False


def _bow_like_reference(path: Path) -> ReferenceAsset:
    """A 16x16 anti-diagonal string plus a solid lower limb."""
    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for step in range(12):
        image.putpixel((13 - step, 1 + step), (255, 255, 255, 255))
    for y in range(4, 16):
        left = max(0, 13 - y - 2)
        right = max(0, 13 - y + 1)
        for x in range(left, max(left + 1, right)):
            image.putpixel((x, y), (200, 180, 120, 255))
    image.save(path)
    return ReferenceAsset(path=str(path), name="bow:bow_standby", roles=[ReferenceRole.SHAPE])


def _source_pixels(reference: ReferenceAsset) -> set[tuple[int, int]]:
    with Image.open(reference.path) as loaded:
        rgba = loaded.convert("RGBA")
    return {
        (x, y)
        for y in range(rgba.height)
        for x in range(rgba.width)
        if rgba.getpixel((x, y))[3] >= 8
    }


def _loose_draft() -> GeometrySpec:
    """Two overlapping rectangles: same rough diagonal, sloppy contour."""
    body = [
        "".join(
            "X" if 0 <= 13 - y - 3 <= x <= 13 - y + 3 and 0 <= y < 16 else "."
            for x in range(16)
        )
        for y in range(16)
    ]
    string = ["".join("X" if x == 14 - y else "." for x in range(16)) for y in range(16)]
    return GeometrySpec(
        width=16,
        height=16,
        parts=[PartSpec(id="bow_body", meaning="limb"), PartSpec(id="bow_string", meaning="cord")],
        primitives=[
            PrimitiveSpec(id="body", part_id="bow_body", primitive="custom_mask",
                          params={"offset": [0, 0], "marker": "X", "rows": body}, layer=0),
            PrimitiveSpec(id="string", part_id="bow_string", primitive="custom_mask",
                          params={"offset": [0, 0], "marker": "X", "rows": string}, layer=1),
        ],
    )


def _two_part_descriptor() -> ShapeDescriptor:
    return ShapeDescriptor(
        target="crystal bow",
        semantic="a crystal bow",
        visual_identity=["arc"],
        parts=[PartSpec(id="bow_body", meaning="limb"), PartSpec(id="bow_string", meaning="cord")],
        shape_edit_mode="appearance_only",
        art_direction=KEEPS_CONTOUR,
    )


def test_the_partition_reproduces_the_source_alpha_exactly(tmp_path: Path) -> None:
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    source = _source_pixels(reference)
    locked = reference_silhouette_partition(
        _two_part_descriptor(), _loose_draft(), [reference], 16, 16
    )
    assert locked is not None
    compiled = compile_geometry(locked)
    rendered = {(x, y) for y in range(16) for x in range(16) if compiled.mask.getpixel((x, y)) > 0}
    assert rendered == source, "the lock must reproduce the reference contour exactly"
    # The model partition survives: both declared supports still own pixels.
    for part_id in ("bow_body", "bow_string"):
        assert any(compiled.part_masks[part_id].getpixel((x, y)) > 0 for x, y in source), part_id


def test_the_partition_declines_when_a_required_part_owns_nothing(tmp_path: Path) -> None:
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    draft = _loose_draft()
    body_only = replace(draft, primitives=[draft.primitives[0]])
    assert reference_silhouette_partition(
        _two_part_descriptor(), body_only, [reference], 16, 16
    ) is None


def test_the_partition_needs_a_same_size_reference(tmp_path: Path) -> None:
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    assert reference_silhouette_partition(
        _two_part_descriptor(), _loose_draft(), [reference], 32, 32
    ) is None


def _frame(path: Path, name: str) -> ReferenceAsset:
    _bow_like_reference(path)
    return ReferenceAsset(
        path=str(path),
        name="bow:%s" % name,
        roles=[ReferenceRole.SHAPE],
        notes=["group=minecraft:item/bow", "family_member=%s" % name],
    )


def test_the_matching_family_frame_outranks_list_order(tmp_path: Path) -> None:
    """Every bow request carries all four frames with pulling_0 first.

    First-match selection conformed the idle frame to the half-drawn one and
    the animation the family was asked for never appeared.
    """
    drawn = _frame(tmp_path / "pulling_0.png", "bow_pulling_0")
    idle = _frame(tmp_path / "standby.png", "bow_standby")
    assert _same_size_shape_reference([drawn, idle], 16, 16, "bow_standby") is idle
    assert _same_size_shape_reference([drawn, idle], 16, 16, "bow_pulling_0") is drawn


def test_an_unknown_member_name_falls_back_to_list_order(tmp_path: Path) -> None:
    drawn = _frame(tmp_path / "pulling_0.png", "bow_pulling_0")
    idle = _frame(tmp_path / "standby.png", "bow_standby")
    assert _same_size_shape_reference([drawn, idle], 16, 16, "chestplate") is drawn
    assert _same_size_shape_reference([drawn, idle], 16, 16, None) is drawn


def test_a_different_sized_frame_is_never_the_shape_host(tmp_path: Path) -> None:
    drawn = _frame(tmp_path / "pulling_0.png", "bow_pulling_0")
    assert _same_size_shape_reference([drawn], 32, 32, "bow_pulling_0") is None


def test_a_set_prefixed_member_name_still_finds_its_frame(tmp_path: Path) -> None:
    """The planner names a member after its whole set: crystal_bow_standby."""
    drawn = _frame(tmp_path / "pulling_0.png", "bow_pulling_0")
    idle = _frame(tmp_path / "standby.png", "bow_standby")
    assert _same_size_shape_reference(
        [drawn, idle], 16, 16, "crystal_bow_standby"
    ) is idle
    assert _same_size_shape_reference(
        [drawn, idle], 16, 16, "crystal_bow_pulling_0"
    ) is drawn


def test_the_longest_frame_name_wins_an_ambiguous_suffix(tmp_path: Path) -> None:
    short = _frame(tmp_path / "short.png", "pulling_0")
    long = _frame(tmp_path / "long.png", "bow_pulling_0")
    assert _same_size_shape_reference(
        [short, long], 16, 16, "crystal_bow_pulling_0"
    ) is long


def test_a_uv_atlas_keeps_its_own_alpha_authority(tmp_path: Path) -> None:
    """entity_uv already scores its regions against the reference; do not clip."""
    from studio_next.contracts import UvRegionSpec

    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    draft = _loose_draft()
    regioned = replace(
        draft,
        uv_regions=[UvRegionSpec(id="body", part_id="bow_body", bbox=[0, 0, 16, 16])],
    )
    assert reference_silhouette_partition(
        _two_part_descriptor(), regioned, [reference], 16, 16
    ) is None


def test_an_off_by_one_draft_still_conforms_and_keeps_its_string(tmp_path: Path) -> None:
    """The live crystal-bow draft drew its cord one column right of the source.

    A strict overlap test found no string pixels at all and refused to conform
    a completely clear intent; nearest-mask ownership recovers it.
    """
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    draft = _loose_draft()
    shifted_body = [
        "".join(
            "X" if 0 <= 13 - y - 3 <= x <= 13 - y + 3 and 0 <= y < 16 else "."
            for x in range(16)
        )
        for y in range(16)
    ]
    shifted_string = ["".join("X" if x == 15 - y else "." for x in range(16)) for y in range(16)]
    off_by_one = replace(
        draft,
        primitives=[
            replace(draft.primitives[0], params={**draft.primitives[0].params, "rows": shifted_body}),
            replace(draft.primitives[1], params={**draft.primitives[1].params, "rows": shifted_string}),
        ],
    )
    locked = reference_silhouette_partition(
        _two_part_descriptor(), off_by_one, [reference], 16, 16
    )
    assert locked is not None, "an offset cord is still an unambiguous intent"
    compiled = compile_geometry(locked)
    source = _source_pixels(reference)
    rendered = {(x, y) for y in range(16) for x in range(16) if compiled.mask.getpixel((x, y)) > 0}
    assert rendered == source
    cord = compiled.part_masks["bow_string"]
    owned = sum(1 for x, y in source if cord.getpixel((x, y)) > 0)
    assert owned >= 6, "the thin part must not be swallowed by the body mask"


def test_a_specific_part_beats_the_body_it_crosses(tmp_path: Path) -> None:
    """Two masks are equally near a pixel surprisingly often."""
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    draft = _loose_draft()
    # A fat body slab covering everything, and a one-pixel cord beside it.
    slab = ["X" * 16 for _ in range(16)]
    cord = ["".join("X" if x == 14 - y else "." for x in range(16)) for y in range(16)]
    wide = replace(
        draft,
        primitives=[
            replace(draft.primitives[0], params={**draft.primitives[0].params, "rows": slab}),
            replace(draft.primitives[1], params={**draft.primitives[1].params, "rows": cord}),
        ],
    )
    locked = reference_silhouette_partition(
        _two_part_descriptor(), wide, [reference], 16, 16
    )
    assert locked is not None
    compiled = compile_geometry(locked)
    cord_mask = compiled.part_masks["bow_string"]
    assert any(
        cord_mask.getpixel((x, y)) > 0 for x, y in _source_pixels(reference)
    ), "the one-pixel cord must own something even across a full-canvas body"


def test_an_optional_part_with_no_mask_does_not_break_the_partition(tmp_path: Path) -> None:
    """A non-required support may simply have authored nothing."""
    reference = _bow_like_reference(tmp_path / "bow_standby.png")
    draft = _loose_draft()
    optional = PartSpec(id="tassel", meaning="optional cord tassel", required=False)
    descriptor = replace(
        _two_part_descriptor(),
        parts=[*_two_part_descriptor().parts, optional],
    )
    locked = reference_silhouette_partition(descriptor, draft, [reference], 16, 16)
    assert locked is not None
    assert {part.id for part in locked.parts} >= {"bow_body", "bow_string", "tassel"}


def test_a_local_silhouette_edit_with_no_named_edit_is_still_a_recolour() -> None:
    """The same recolour was labelled two ways by two runs of one request.

    One crystal-bow run wrote appearance_only; the next wrote
    local_silhouette_edit and described the reference only as the source for
    silhouette and proportions. A local edit with no named host is vacuous.
    """
    descriptor = _descriptor(
        "Use bow_standby for silhouette, proportions, and UV layout. Recolor with"
        " prismarine_crystals and end_crystal cyan-to-white palettes.",
        KEEPS_CONTOUR,
    )
    labelled = replace(descriptor, shape_edit_mode="local_silhouette_edit")
    assert _reference_shape_is_locked(labelled) is True


def test_a_local_silhouette_edit_that_names_the_edit_still_unlocks() -> None:
    descriptor = _descriptor(
        "Use the source contour but cut a chipped notch out of the upper limb.",
        KEEPS_CONTOUR,
    )
    labelled = replace(descriptor, shape_edit_mode="local_silhouette_edit")
    assert _reference_shape_is_locked(labelled) is False


def test_a_recolour_strategy_locks_without_an_ownership_map() -> None:
    """The map is corroboration the model rarely supplies, not the cue itself."""
    descriptor = _descriptor(
        "Use the reference for the body and recolor it into pale crystal.",
        KEEPS_CONTOUR,
    )
    assert descriptor.reference_part_map is None
    assert _reference_shape_is_locked(descriptor) is True


CONTRADICTORY_BRIEF = ArtDirection(
    target="crystal bow",
    primary_subject="a crystal bow in its standby state",
    design_intent="carve the bow from translucent crystal",
    # The boolean a live run wrote, next to prose that said the opposite.
    requires_silhouette_change=True,
    silhouette_actions=[
        "The outer bow contour stays continuous and readable as one crystal body.",
        "Transparent background remains exposed without opening the bow body contour.",
    ],
)


def test_the_descriptor_mode_outranks_a_contradictory_brief_boolean() -> None:
    """The anchor frame shipped the *next* frame's silhouette because of this.

    The art direction is a reference-free brief whose one boolean summarised
    several sentences and contradicted them; the descriptor is written after
    seeing the raster and names the mode explicitly.
    """
    descriptor = _descriptor(
        "Use bow_standby for bow silhouette and proportions; recolor into crystal.",
        CONTRADICTORY_BRIEF,
    )
    structured = replace(descriptor, shape_edit_mode="local_silhouette_edit")
    assert _reference_shape_is_locked(structured) is True
    assert _reference_shape_is_locked(
        replace(descriptor, shape_edit_mode="appearance_only")
    ) is True


def test_a_bare_run_still_lets_the_brief_veto_the_lock() -> None:
    """model_decides carries no structured answer, so the brief still rules."""
    descriptor = _descriptor(
        "Use bow_standby for bow silhouette and proportions; recolor into crystal.",
        CONTRADICTORY_BRIEF,
    )
    assert descriptor.shape_edit_mode == "model_decides"
    assert _reference_shape_is_locked(descriptor) is False


def test_a_structured_new_silhouette_survives_the_brief() -> None:
    descriptor = _descriptor("author a fresh contour", CONTRADICTORY_BRIEF)
    assert _reference_shape_is_locked(
        replace(descriptor, shape_edit_mode="new_silhouette")
    ) is False
