"""Tests for the dynamic family label and for family (set) generation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from asset_tree import write_png

from studio_next import quality
from studio_next.contracts import AppearanceSpec, PartAppearance, appearance_from_dict
from studio_next.reference_retrieval import (
    Candidate,
    build_router_manifest,
    family_token,
    manifest_label,
    parse_router_selection,
)


@dataclass(frozen=True)
class _FakeEntry:
    asset_id: str
    namespace: str = "minecraft"
    resource_path: str = ""
    category: str = "item"
    dimensions: dict = field(default_factory=dict)
    alpha: dict = field(default_factory=dict)
    palette: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)


def _candidate(name: str, category: str = "item", score: float = 0.0) -> Candidate:
    return Candidate(
        _FakeEntry(
            asset_id="minecraft:%s/%s" % (category, name),
            resource_path="%s/%s.png" % (category, name),
            category=category,
        ),
        score,
        [],
        [],
    )


# -- dynamic classification -------------------------------------------

def test_family_token_keeps_only_a_shared_head_noun() -> None:
    counts = Counter({"helmet": 3, "boots": 2, "wand": 1})
    assert family_token("iron_helmet", counts) == "helmet"
    assert family_token("golden_boots", counts) == "boots"
    assert family_token("weird_wand", counts) == ""   # only one asset owns it
    assert family_token("stick", counts) == ""        # no modifier, no head noun
    assert family_token("disc_11", counts) == ""      # numeric tail is not a class


def test_manifest_clusters_one_family_and_labels_the_line() -> None:
    manifest = build_router_manifest([
        _candidate("iron_helmet"),
        _candidate("stone_sword"),
        _candidate("diamond_helmet"),
        _candidate("iron_sword"),
    ])
    labels = [manifest_label(row) for row in manifest]
    assert labels[0] == "diamond_helmet item/helmet"
    assert labels[1] == "iron_helmet item/helmet"
    assert labels[2] == "iron_sword item/sword"
    assert labels[3] == "stone_sword item/sword"
    # the family is a label, never part of the name the model must return
    assert [row["name"] for row in manifest] == [
        "diamond_helmet", "iron_helmet", "iron_sword", "stone_sword",
    ]


def test_router_parser_accepts_the_printed_label() -> None:
    manifest = build_router_manifest([_candidate("iron_helmet"), _candidate("diamond_helmet")])
    label = manifest_label(manifest[0])
    selection = parse_router_selection(
        {"form": "item", "selections": [{"name": label, "roles": ["shape"]}]}, manifest
    )
    assert selection.selections[0].asset_id == manifest[0]["asset_id"]


# -- deterministic family consistency ---------------------------------

def test_family_consistency_separates_a_match_from_a_drifted_member(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.png"
    same = tmp_path / "same.png"
    drifted = tmp_path / "drifted.png"
    write_png(anchor, (200, 200, 200, 255))
    write_png(same, (200, 200, 200, 255))
    write_png(drifted, (10, 200, 30, 255))
    report = quality.family_consistency(anchor, [same, drifted], minimum_overlap=0.5)
    rows = {Path(row["sprite"]).name: row for row in report["members"]}
    assert rows["same.png"]["consistent"] is True
    assert rows["same.png"]["palette_overlap"] == 1.0
    assert rows["drifted.png"]["consistent"] is False
    assert rows["drifted.png"]["palette_overlap"] == 0.0


# -- family generation orchestration ----------------------------------

def _fake_member_report(out_dir: Path, colour) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sprite = out_dir / "sprite.png"
    write_png(sprite, colour)
    return {
        "selected_round": 0,
        "rounds_completed": 1,
        "selected_sprite": str(sprite),
        "rounds": [{
            "index": 0,
            "validation": {"passed": True},
            "blind_review": {"primary_object": "helmet"},
        }],
    }


def test_family_loop_anchors_every_member_and_repairs_a_drifted_one(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict] = []

    def fake_run(request, references, out_dir, **kwargs):
        calls.append({
            "query": request.query,
            "out_dir": Path(out_dir),
            "anchor": kwargs.get("style_anchor"),
            "anchor_palette": kwargs.get("anchor_palette"),
        })
        colour = (200, 200, 200, 255) if len(calls) == 1 else (10, 200, 30, 255)
        return _fake_member_report(Path(out_dir), colour)

    monkeypatch.setattr(quality, "run_quality_loop", fake_run)
    family = quality.run_family_loop(
        "iron armor set",
        tmp_path / "family",
        members=["helmet", "chestplate"],
        rounds=1,
    )
    # member 0, then member 1, then one drift repair for member 1
    assert len(calls) == 3
    assert calls[0]["anchor"] is None, "the first member defines the anchor"
    assert Path(calls[1]["anchor"]) == Path(family["anchor"])
    # The family ramp is a first-pass contract, not a drift repair: a set
    # that only matches after a gate trips has already shipped four palettes.
    assert calls[1]["anchor_palette"], "every sibling resolves onto the anchor ramp"
    assert calls[2]["anchor_palette"], "the repair pass carries it too"
    assert Path(calls[2]["anchor"]) == Path(family["anchor"])
    assert family["drift_repaired"] == 1
    assert family["members"][1]["drift_repaired"] is True
    assert [row["name"] for row in family["members"]] == ["helmet", "chestplate"]
    assert family["members"][0]["palette_overlap"] == 1.0
    written = json.loads((tmp_path / "family" / "family.json").read_text(encoding="utf-8"))
    assert written["set_name"] == "iron_armor_set"
    assert len(written["members"]) == 2


def test_family_loop_keeps_going_when_one_member_fails(tmp_path: Path, monkeypatch) -> None:
    calls: list[int] = []

    def fake_run(request, references, out_dir, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model returned an invalid plan")
        return _fake_member_report(Path(out_dir), (200, 200, 200, 255))

    monkeypatch.setattr(quality, "run_quality_loop", fake_run)
    family = quality.run_family_loop(
        "iron armor set",
        tmp_path / "family",
        members=["helmet", "chestplate"],
        rounds=1,
    )
    assert len(calls) == 2, "a failing member must not discard its siblings"
    assert family["members"][0]["error"] is not None
    assert family["members"][1]["error"] is None
    assert family["anchor"] == family["members"][1]["sprite"]


def test_family_loop_truncates_to_max_members(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        quality, "run_quality_loop",
        lambda request, references, out_dir, **kwargs: _fake_member_report(Path(out_dir), (200, 200, 200, 255)),
    )
    family = quality.run_family_loop(
        "tool set",
        tmp_path / "family",
        members=["sword", "pickaxe", "axe", "shovel"],
        max_members=2,
        rounds=1,
    )
    assert [row["name"] for row in family["members"]] == ["sword", "pickaxe"]


# -- model-shaped wire format (a real crash found by a live armour run) --

def test_nested_composite_object_is_normalized_instead_of_crashing() -> None:
    """A model that returns {mode, source} must not raise TypeError.

    The canonical wire format is a bare mode string, but a live run returned
    {"mode": "replace", "source": "layer:file"} and the membership test in
    AppearanceSpec raised 'unhashable type: dict' instead of a clean error.
    """
    spec = appearance_from_dict({
        "palette": {"iron": "#C8C8C8"},
        "parts": {"plate": {"colors": ["iron"]}},
        "part_reference_sampling": {"plate": "pattern"},
        "part_reference_composite": {"plate": {"mode": "replace", "source": "layer:file"}},
    })
    assert spec.part_reference_composite == {"plate": "replace"}
    assert spec.part_reference_sources == {"plate": "layer:file"}


def test_list_shaped_parts_are_keyed_by_their_id() -> None:
    """A parts list carrying appearance fields is the same data, keyed."""
    spec = appearance_from_dict({
        "palette": {"wood": "#8A6A3F"},
        "parts": [{"id": "body", "colors": ["wood"]}, {"id": "legs", "colors": ["wood"]}],
    })
    assert sorted(spec.parts) == ["body", "legs"]


def test_descriptor_shaped_parts_are_invalid_data_not_an_attribute_error() -> None:
    """A live leggings run returned descriptor parts inside an AppearanceSpec."""
    with pytest.raises(ValueError):
        appearance_from_dict({
            "palette": {"wood": "#8A6A3F"},
            "parts": [{"id": "body", "meaning": "leggings waist", "required": True}],
        })


def test_a_non_string_mode_is_invalid_data_not_a_type_error() -> None:
    with pytest.raises(ValueError):
        AppearanceSpec(
            palette={"iron": "#C8C8C8"},
            parts={"plate": PartAppearance(colors=["iron"])},
            part_reference_composite={"plate": ["overlay"]},
        )


# -- animation continuity ---------------------------------------------

def _framed(path: Path, pixels: dict[tuple[int, int], tuple[int, int, int, int]]) -> None:
    from PIL import Image

    image = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for point, value in pixels.items():
        image.putpixel(point, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def test_frame_continuity_reads_overlapping_frames_as_one_object(tmp_path: Path) -> None:
    body = {(x, y): (120, 90, 40, 255) for x in range(8) for y in range(8)}
    moved = dict(body)
    moved[(9, 9)] = (200, 200, 200, 255)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    far = tmp_path / "far.png"
    _framed(a, body)
    _framed(b, moved)
    _framed(far, {(x, y): (10, 200, 30, 255) for x in range(8, 16) for y in range(8, 16)})
    report = quality.frame_continuity([a, b])
    assert report["pair_count"] == 1
    assert report["minimum_agreement"] > 0.9, "one added pixel must barely move the body"
    # A helmet and a chestplate share no body, so they are not an animation.
    assert quality.frame_continuity([a, far])["pair_count"] == 0


def test_frame_continuity_is_relative_to_the_source_family(tmp_path: Path) -> None:
    """The vanilla bow only agrees on 29%-66% of its union pairwise."""
    body = {(x, y): (120, 90, 40, 255) for x in range(8) for y in range(8)}
    shifted = {(x + 1, y): (120, 90, 40, 255) for x in range(8) for y in range(8)}
    source_a = tmp_path / "source_a.png"
    source_b = tmp_path / "source_b.png"
    _framed(source_a, body)
    _framed(source_b, shifted)
    baseline = [source_a, source_b]
    same = quality.frame_continuity([source_a, source_b], baseline_sprites=baseline)
    assert same["consistent"] is True
    # An unrelated pair that happens to overlap offers no body agreement.
    odd = tmp_path / "odd.png"
    _framed(odd, {(x, y): (0, 0, 0, 255) for x in range(9) for y in range(8)})
    worse = quality.frame_continuity([source_a, odd], baseline_sprites=baseline)
    assert worse["consistent"] is False


def test_frame_continuity_claims_nothing_without_a_baseline(tmp_path: Path) -> None:
    body = {(x, y): (120, 90, 40, 255) for x in range(8) for y in range(8)}
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _framed(a, body)
    _framed(b, body)
    report = quality.frame_continuity([a, b])
    assert report["consistent"] is None
    assert report["minimum_agreement"] == 1.0


def test_family_loop_reports_frame_continuity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        quality, "run_quality_loop",
        lambda request, references, out_dir, **kwargs: _fake_member_report(
            Path(out_dir), (200, 200, 200, 255)
        ),
    )
    family = quality.run_family_loop(
        "crystal bow frames",
        tmp_path / "family",
        members=["bow_standby", "bow_pulling_0"],
        rounds=1,
    )
    continuity = family["continuity"]
    assert continuity is not None
    assert continuity["pair_count"] == 1
    assert continuity["minimum_agreement"] == 1.0
    assert continuity["consistent"] is None, "no source frames were recorded here"


def test_frame_continuity_separates_a_palette_shift_from_a_shading_shift(tmp_path: Path) -> None:
    """Exact and near agreement fail differently, and both are reported."""
    body = {(x, y): (100, 100, 100, 255) for x in range(8) for y in range(8)}
    nudged = {(x, y): (110, 104, 96, 255) for x, y in body}
    recoloured = {(x, y): (10, 200, 30, 255) for x, y in body}
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    c = tmp_path / "c.png"
    _framed(a, body)
    _framed(b, nudged)
    _framed(c, recoloured)
    shifted = quality.frame_continuity([a, b], colour_tolerance=24)
    assert shifted["minimum_agreement"] == 0.0, "the exact colours did move"
    assert shifted["minimum_near_agreement"] == 1.0, "but only within the tolerance"
    drifted = quality.frame_continuity([a, c], colour_tolerance=24)
    assert drifted["minimum_agreement"] == 0.0
    assert drifted["minimum_near_agreement"] == 0.0, "a real palette drift fails both"


def test_the_family_ramp_reaches_literal_hex_colours(tmp_path: Path) -> None:
    """A live bow family wrote hex straight into its parts.

    remap_palette_to_anchor only rewrites named swatches, so three of four
    frames never saw the anchor ramp and the set shipped four palettes.
    """
    ramp = [(0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 255, 255)]
    spec = appearance_from_dict({
        "palette": {"named": "#112233"},
        "parts": {"body": {"colors": ["#101010", "#A0A0A0", "#F0F0F0"], "material": "x"}},
    })
    remapped = quality.remap_appearance_to_anchor(spec, ramp)
    # The three literals keep their dark-to-light order, and the values spread
    # across the ramp: rank 0 -> #000000, rank 1 -> #808080, rank 2 -> #FFFFFF.
    assert remapped.parts["body"].colors == ["#000000", "#808080", "#FFFFFF"]
    # The named swatch keeps its name and its own single rank.
    assert remapped.palette["named"] == "#000000"


def test_the_family_ramp_reaches_the_outline_and_the_pixel_map(tmp_path: Path) -> None:
    """Every colour channel into the renderer is part of the contract."""
    ramp = [(0, 0, 0), (255, 255, 255)]
    spec = appearance_from_dict({
        "palette": {"body": "#808080"},
        "parts": {"body": {"colors": ["body"], "material": "x"}},
        "outline_color": "#101010",
        "pixel_map": {"legend": {"a": "#E0E0E0"}, "rows": ["a"]},
    })
    remapped = quality.remap_appearance_to_anchor(spec, ramp)
    assert remapped.outline_color == "#000000", "darkest literal takes the dark end"
    assert remapped.pixel_map["legend"]["a"] == "#FFFFFF", "lightest takes the light end"


# -- family paint inheritance ------------------------------------------

def _mask_spec(parts, rows_by_part, size=8):
    from studio_next.contracts import GeometrySpec, PartSpec, PrimitiveSpec

    specs = [PartSpec(id=pid, meaning=pid) for pid in parts]
    primitives = [
        PrimitiveSpec(
            id="p_%s" % pid,
            part_id=pid,
            primitive="custom_mask",
            params={"offset": [0, 0], "marker": "X", "rows": rows_by_part[pid]},
        )
        for pid in parts
    ]
    return GeometrySpec(width=size, height=size, parts=specs, primitives=primitives)


def _appearance(palette, parts, **spec_kwargs):
    """parts maps a part id to its PartAppearance kwargs."""
    from studio_next.contracts import AppearanceSpec, PartAppearance

    return AppearanceSpec(
        palette=palette,
        parts={pid: PartAppearance(**fields) for pid, fields in parts.items()},
        **spec_kwargs,
    )


def _block(columns):
    """A full-height mask occupying the given columns."""
    return ["".join("X" if x in columns else "." for x in range(8)) for _ in range(8)]


def test_a_part_on_the_same_pixels_takes_the_anchor_paint() -> None:
    anchor_geometry = _mask_spec(["body"], {"body": _block({0, 1, 2})})
    anchor_appearance = _appearance(
        {"dark": "#101010", "light": "#F0F0F0"},
        {"body": {"colors": ["dark", "light"], "shade_axis": "top"}},
    )
    member_geometry = _mask_spec(["torso"], {"torso": _block({0, 1, 2})})
    member_appearance = _appearance(
        {"a": "#000080", "b": "#800000"},
        {"torso": {"colors": ["a", "b"], "shade_axis": "bottom"}},
    )
    updated, audit = quality.inherit_anchor_paint(
        member_appearance, member_geometry, anchor_appearance, anchor_geometry
    )
    assert audit["parts"]["torso"]["anchor_part"] == "body"
    assert audit["parts"]["torso"]["overlap"] == 1.0
    assert updated.parts["torso"].colors == ["#101010", "#F0F0F0"]
    assert updated.parts["torso"].shade_axis == "top"
    assert audit["composition_from_anchor"] is True


def test_a_part_on_unrelated_pixels_keeps_its_own_paint() -> None:
    """A plain set (helmet, chestplate) must not inherit anything."""
    anchor_geometry = _mask_spec(["helmet"], {"helmet": _block({0, 1})})
    anchor_appearance = _appearance({"iron": "#C8C8C8"}, {"helmet": {"colors": ["iron"]}})
    member_geometry = _mask_spec(["chestplate"], {"chestplate": _block({5, 6, 7})})
    member_appearance = _appearance({"gold": "#FFD700"}, {"chestplate": {"colors": ["gold"]}})
    updated, audit = quality.inherit_anchor_paint(
        member_appearance, member_geometry, anchor_appearance, anchor_geometry
    )
    assert audit == {}
    assert updated.parts["chestplate"].colors == ["gold"], "the token survives untouched"


def test_a_paint_only_part_is_not_counted_as_a_physical_part() -> None:
    """compile_geometry makes an empty mask for every declared part."""
    anchor_geometry = _mask_spec(["body"], {"body": _block({0, 1, 2})})
    anchor_appearance = _appearance(
        {"dark": "#101010", "light": "#F0F0F0"},
        {"body": {"colors": ["dark", "light"]}},
        pixel_map={"legend": {"a": "#F0F0F0"}, "rows": ["a"]},
    )
    member_geometry = _mask_spec(["body"], {"body": _block({0, 1, 2})})
    member_appearance = _appearance(
        {"x": "#202020", "y": "#E0E0E0"},
        {"body": {"colors": ["x", "y"]}, "glint": {"colors": ["y"]}},
    )
    _updated, audit = quality.inherit_anchor_paint(
        member_appearance, member_geometry, anchor_appearance, anchor_geometry
    )
    # body matched, glint has no mask: the one match is enough for composition.
    assert audit["composition_from_anchor"] is True


def test_a_part_nested_inside_another_keeps_its_own_paint() -> None:
    """A nocked arrow sits inside the bow body's mask but is not the body.

    Coverage alone matched it to the body and handed the arrow the body's dark
    crystal ramp, which is the exact part the planner had just been asked to
    declare separately.
    """
    anchor_geometry = _mask_spec(["body"], {"body": _block({0, 1, 2, 3})})
    anchor_appearance = _appearance(
        {"dark": "#101010", "light": "#F0F0F0"},
        {"body": {"colors": ["dark", "light"]}},
    )
    member_geometry = _mask_spec(
        ["body", "arrow"],
        {"body": _block({0, 1, 2, 3}), "arrow": _block({1})},
    )
    member_appearance = _appearance(
        {"a": "#000080", "b": "#8080FF"},
        {"body": {"colors": ["a", "b"]}, "arrow": {"colors": ["b", "b"]}},
    )
    updated, audit = quality.inherit_anchor_paint(
        member_appearance, member_geometry, anchor_appearance, anchor_geometry
    )
    assert audit["parts"]["body"]["anchor_part"] == "body"
    assert "arrow" not in audit["parts"], "a nested part is not the same region"
    assert updated.parts["arrow"].colors == ["b", "b"], "the arrow keeps its own accent"
