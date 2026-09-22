"""Tests for the live catalogue to router to reference-expansion path."""

from __future__ import annotations

from pathlib import Path

from asset_tree import png_bytes

from studio_next.contracts import AssetForm, AssetRequest, ReferenceRole
from studio_next.group_index import GroupReferenceSource
from studio_next.quality import _resolve_auto_request
from studio_next.text_cache import TextFeatureCache


class _FakeClient:
    model = "fake-router"


class _FakePlanner:
    """Answers like the indexed router, but selects logical asset names."""

    def __init__(self, selections, form="item"):
        self.client = _FakeClient()
        self._selections = selections
        self._form = form
        self.captured = {}

    def route_indexed(self, query, candidates, layouts, cache_dir=None,
                      index_fingerprint="", model_name=None, art_direction=None):
        self.captured["manifest"] = candidates
        by_name = {item["name"]: item["asset_id"] for item in candidates}
        chosen = [
            {"asset_id": by_name[name], "roles": roles, "reason": "test", "confidence": 0.9}
            for name, roles in self._selections
        ]
        return {
            "form": self._form,
            "selections": chosen,
            "asset_ids": [item["asset_id"] for item in chosen],
            "uv_layout_index": None,
            "unmet_evidence": [],
            "reason": "test",
            "cache_hit": False,
        }


# -- content-addressed text cache --------------------------------------

def test_text_cache_hits_on_identical_bytes_and_misses_after_an_edit(tmp_path: Path) -> None:
    cache = TextFeatureCache(root=tmp_path / "text")
    original = png_bytes((10, 20, 30, 255))
    features, first_hit = cache.features(original)
    assert first_hit is False
    again, second_hit = cache.features(original)
    assert second_hit is True
    assert again == features
    edited = png_bytes((99, 20, 30, 255))
    _features, third_hit = cache.features(edited)
    assert third_hit is False
    assert cache.stats() == {"hits": 1, "misses": 2, "writes": 2}


def test_text_cache_key_includes_animation_annotation(tmp_path: Path) -> None:
    cache = TextFeatureCache(root=tmp_path / "text")
    data = png_bytes()
    cache.features(data, b'{"animation":{"frametime":1}}')
    _features, hit = cache.features(data, b'{"animation":{"frametime":2}}')
    assert hit is False
    _features, hit = cache.features(data, b'{"animation":{"frametime":1}}')
    assert hit is True


# -- group expansion ---------------------------------------------------

def test_selected_block_expands_into_all_of_its_faces(resource_root: Path, tmp_path: Path) -> None:
    source = GroupReferenceSource.from_sources([resource_root], tmp_path / "cache")
    try:
        entry = source.entry_for("demo:block/test_log")
        assets = source.planning_assets(
            entry, [ReferenceRole.SHAPE, ReferenceRole.MATERIAL], display_name="test_log"
        )
        assert sorted(asset.name for asset in assets) == ["test_log:log_side", "test_log:log_top"]
        assert all(asset.roles == [ReferenceRole.SHAPE, ReferenceRole.MATERIAL] for asset in assets)
        assert all(asset.features.get("width") == 16 for asset in assets)
        assert all(any("member=" in note for note in asset.notes) for asset in assets)
    finally:
        source.close()


def test_single_texture_group_keeps_a_plain_name(resource_root: Path, tmp_path: Path) -> None:
    source = GroupReferenceSource.from_sources([resource_root], tmp_path / "cache")
    try:
        group = source.entry_for("demo:item/disc_11").group
        assert len(group.textures) == 1
        assets = source.planning_assets(source.entry_for("demo:item/disc_11"), [ReferenceRole.MATERIAL])
        assert [asset.name for asset in assets] == ["disc_11"]
    finally:
        source.close()


def test_repeated_expansion_reuses_the_text_cache(resource_root: Path, tmp_path: Path) -> None:
    source = GroupReferenceSource.from_sources([resource_root], tmp_path / "cache")
    try:
        entry = source.entry_for("demo:block/test_log")
        source.planning_assets(entry, [ReferenceRole.MATERIAL])
        after_first = source.text_cache.stats()
        source.planning_assets(entry, [ReferenceRole.MATERIAL])
        after_second = source.text_cache.stats()
        assert after_first["misses"] == 2 and after_first["writes"] == 2
        assert after_second["hits"] == 2
        assert after_second["writes"] == after_first["writes"]
    finally:
        source.close()


# -- routing boundary --------------------------------------------------

def test_live_routing_expands_a_logical_name(resource_root: Path, tmp_path: Path) -> None:
    planner = _FakePlanner([("test_log", ["shape", "material"])])
    _request, references, _uv, routing, planning_assets = _resolve_auto_request(
        AssetRequest(query="blood gel", form=AssetForm.AUTO),
        [],
        planner,
        asset_sources=[str(resource_root)],
        cache_root=tmp_path / "cache",
    )
    assert planning_assets is not None
    assert len(references) == 2
    assert len(planning_assets) == 2
    assert sorted(asset.name for asset in planning_assets) == ["test_log:log_side", "test_log:log_top"]
    assert routing["mode"] == "model_auto"
    assert routing["selected"][0]["reference_count"] == 2
    assert routing["live_source"]["expanded_references"] == 2
    # The router prompt must stay name level: one line per logical asset.
    names = [row["name"] for row in planner.captured["manifest"]]
    assert "test_log" in names
    assert not [name for name in names if "log_top" in name or "log_side" in name]


def test_every_routed_selection_reaches_the_plan(resource_root: Path, tmp_path: Path) -> None:
    """A positional cap downstream once dropped the only material reference.

    The model answers structural candidates first and material or palette
    rasters last, so slicing the router's own (already bounded) list removed
    exactly the references the appearance stage needs.
    """
    planner = _FakePlanner([
        ("test_log", ["shape"]),
        ("watch", ["shape"]),
        ("disc_11", ["shape"]),
        ("disc_13", ["shape"]),
        ("lantern", ["material", "palette"]),
        ("dust", ["material"]),
    ])
    _request, _references, _uv, routing, planning_assets = _resolve_auto_request(
        AssetRequest(query="a wooden thing", form=AssetForm.AUTO),
        [],
        planner,
        asset_sources=[str(resource_root)],
        cache_root=tmp_path / "cache",
    )
    assert len(routing["selected"]) == 6
    assert planning_assets is not None
    names = [asset.name for asset in planning_assets]
    assert any("lantern" in name for name in names), names
    assert any("dust" in name for name in names), names


def test_live_routing_handles_a_frame_family_as_one_name(resource_root: Path, tmp_path: Path) -> None:
    planner = _FakePlanner([("watch", ["pixel_style", "material"])])
    _request, references, _uv, routing, planning_assets = _resolve_auto_request(
        AssetRequest(query="broken watch", form=AssetForm.AUTO),
        [],
        planner,
        asset_sources=[str(resource_root)],
        cache_root=tmp_path / "cache",
    )
    assert len(references) == 3
    assert planning_assets is not None
    assert [asset.name for asset in planning_assets] == [
        "watch:watch_00", "watch:watch_01", "watch:watch_02",
    ]
    assert routing["selected"][0]["reference_count"] == 3


# -- bounding a very large frame family --------------------------------

def test_frame_bounding_keeps_the_requested_frame_and_spreads_the_rest() -> None:
    """Vanilla clock owns 64 frames; attaching all of them wastes the budget."""
    from studio_next.group_index import _bounded_frames

    class _Texture:
        def __init__(self, name: str) -> None:
            self.name = name

    indexed = [(index, _Texture("clock_%02d" % index)) for index in range(64)]
    chosen = _bounded_frames(indexed, "clock_37", 8)
    names = [texture.name for _index, texture in chosen]
    assert len(names) == 8
    assert "clock_37" in names, "the frame this request is about must survive"
    assert names == sorted(names, key=lambda name: int(name.split("_")[1])), "order is preserved"
    assert names[-1] != "clock_07", "an even spread, not the first eight"


def test_frame_bounding_without_a_match_still_spreads_evenly() -> None:
    from studio_next.group_index import _bounded_frames

    class _Texture:
        def __init__(self, name: str) -> None:
            self.name = name

    indexed = [(index, _Texture("clock_%02d" % index)) for index in range(64)]
    chosen = _bounded_frames(indexed, "compass_00", 4)
    assert len(chosen) == 4
    assert [index for index, _texture in chosen] == sorted(index for index, _t in chosen)


def test_a_frame_family_is_bounded_through_the_real_expansion(resource_root: Path, tmp_path: Path) -> None:
    source = GroupReferenceSource.from_sources([str(resource_root)], tmp_path / "cache")
    entry = source.entry_for("demo:item/watch")
    full = source.planning_assets(entry, [ReferenceRole.PIXEL_STYLE])
    chosen = source.planning_assets(
        entry, [ReferenceRole.PIXEL_STYLE], preferred_member="watch_02", max_frames=2
    )
    assert len(full) == 3
    assert [asset.name for asset in chosen] == ["watch:watch_00", "watch:watch_02"]
    # The family index stays the source index, not the position in the slice.
    assert "member=3/3" in chosen[-1].notes


def test_only_the_matching_frame_keeps_the_shape_role(resource_root: Path, tmp_path: Path) -> None:
    """The standby bow came back with the half-drawn frame's silhouette.

    Every state of a code-driven family was attached with roles=shape, and the
    planner copied whichever one the list happened to start with. The member
    that answers this request is knowable, so say it.
    """
    planner = _FakePlanner([("watch", ["shape", "scale"])])
    _request, _references, _uv, _routing, planning_assets = _resolve_auto_request(
        AssetRequest(query="a brass watch at the second state", form=AssetForm.AUTO, name="watch_01"),
        [],
        planner,
        asset_sources=[str(resource_root)],
        cache_root=tmp_path / "cache",
    )
    assert planning_assets is not None
    by_name = {asset.name: asset for asset in planning_assets}
    authority = by_name["watch:watch_01"]
    assert [role.value for role in authority.roles] == ["shape", "scale"]
    assert any("shape_authority=this request" in note for note in authority.notes)
    for sibling in ("watch:watch_00", "watch:watch_02"):
        roles = [role.value for role in by_name[sibling].roles]
        assert "shape" not in roles, sibling
        assert any("shape_authority=context only" in note for note in by_name[sibling].notes), sibling


def test_a_family_with_no_matching_member_is_left_alone(resource_root: Path, tmp_path: Path) -> None:
    planner = _FakePlanner([("watch", ["shape", "scale"])])
    _request, _references, _uv, _routing, planning_assets = _resolve_auto_request(
        AssetRequest(query="a brass watch", form=AssetForm.AUTO, name="clock_frame_midnight"),
        [],
        planner,
        asset_sources=[str(resource_root)],
        cache_root=tmp_path / "cache",
    )
    assert planning_assets is not None
    for asset in planning_assets:
        assert "shape" in [role.value for role in asset.roles], asset.name
