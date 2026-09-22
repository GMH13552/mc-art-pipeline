"""Behaviour tests for the image-free logical asset catalogue."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from asset_tree import png_bytes, write_json, write_png

from studio_next import asset_groups as module
from studio_next.asset_groups import build_catalogue


def test_block_faces_collapse_into_one_name(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        group = catalogue.group("demo:block/test_log")
        assert group.texture_names == ("log_side", "log_top")


def test_frame_models_fold_into_their_base_name(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        watch = catalogue.group("demo:item/watch")
        assert watch.texture_names == ("watch_00", "watch_01", "watch_02")
        assert "demo:item/watch_01" not in catalogue.groups


def test_unrelated_numeric_siblings_stay_separate(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        assert catalogue.group("demo:item/disc_11").texture_names == ("disc_11",)
        assert catalogue.group("demo:item/disc_13").texture_names == ("disc_13",)


def test_entity_textures_group_by_leading_name(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        group = catalogue.group("demo:entity/pig")
        assert group.texture_names == ("pig", "pig_saddle")


def test_unreferenced_family_becomes_one_orphan_name(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        group = catalogue.group("demo:texture/dust")
        assert group.texture_names == ("dust_0", "dust_1")


def test_animation_metadata_travels_with_its_texture(resource_root: Path) -> None:
    with build_catalogue([resource_root]) as catalogue:
        group = catalogue.group("demo:item/lantern")
        assert group.textures[0].animated
        assert group.textures[0].animation == {"frametime": 4, "interpolate": False, "frames": None}


def test_catalogue_reads_json_and_mcmeta_but_never_an_image(resource_root: Path, monkeypatch) -> None:
    seen: list[str] = []
    original = module.AssetRoot.read

    def spy(self, resource_path):
        seen.append(resource_path)
        return original(self, resource_path)

    monkeypatch.setattr(module.AssetRoot, "read", spy)
    with build_catalogue([resource_root]):
        pass
    assert seen, "the catalogue should have read something"
    assert not [path for path in seen if path.endswith(".png")], seen


def test_extract_is_the_only_step_that_reads_images(resource_root: Path, tmp_path: Path, monkeypatch) -> None:
    with build_catalogue([resource_root]) as catalogue:
        seen: list[str] = []
        original = module.AssetRoot.read

        def spy(self, resource_path):
            seen.append(resource_path)
            return original(self, resource_path)

        monkeypatch.setattr(module.AssetRoot, "read", spy)
        written = catalogue.extract("demo:block/test_log", tmp_path / "out")
    assert sorted(path.name for path in written) == ["log_side.png", "log_top.png"]
    assert seen and all(path.endswith(".png") for path in seen)


def test_later_source_wins_like_a_resource_pack(tmp_path: Path) -> None:
    jar = tmp_path / "pack.jar"
    with zipfile.ZipFile(jar, "w") as bundle:
        bundle.writestr("assets/demo/models/item/thing.json", json.dumps({
            "parent": "item/generated",
            "textures": {"layer0": "items/thing_jar"},
        }))
        bundle.writestr("assets/demo/textures/items/thing_jar.png", png_bytes())

    overlay = tmp_path / "overlay" / "assets" / "demo"
    write_json(overlay / "models" / "item" / "thing.json", {
        "parent": "item/generated",
        "textures": {"layer0": "items/thing_overlay"},
    })
    write_png(overlay / "textures" / "items" / "thing_overlay.png")

    with build_catalogue([jar, tmp_path / "overlay"]) as catalogue:
        assert catalogue.group("demo:item/thing").texture_names == ("thing_overlay",)
