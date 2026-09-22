"""Synthetic Minecraft-style resource trees shared by the tests."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image


def png_bytes(colour=(200, 40, 40, 255), size=(16, 16)) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


def write_png(path: Path, colour=(200, 40, 40, 255), size=(16, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(colour, size))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_resource_root(base: Path) -> Path:
    """A miniature resource root: one block, one frame family, two discs.

    Every leaf texture gets distinct bytes. Sharing pixels would make two
    different assets collide in the content-addressed text cache, which is
    correct behaviour but hides what the tests are actually asserting.
    """
    root = base / "resources" / "assets" / "demo"

    # one block, two face textures
    write_json(root / "blockstates" / "test_log.json", {
        "variants": {
            "axis=y": {"model": "test_log"},
            "axis=z": {"model": "test_log", "x": 90},
        },
    })
    write_json(root / "models" / "block" / "test_log.json", {
        "parent": "block/cube_column",
        "textures": {"end": "blocks/log_top", "side": "blocks/log_side"},
    })
    write_png(root / "textures" / "blocks" / "log_top.png", (150, 110, 70, 255))
    write_png(root / "textures" / "blocks" / "log_side.png", (110, 80, 50, 255))

    # one item id whose frames live in sibling models, as clock/compass do
    write_json(root / "models" / "item" / "watch.json", {
        "parent": "item/generated",
        "textures": {"layer0": "items/watch_00"},
    })
    for frame in range(3):
        write_json(root / "models" / "item" / ("watch_0%d.json" % frame), {
            "parent": "item/generated",
            "textures": {"layer0": "items/watch_0%d" % frame},
        })
        write_png(root / "textures" / "items" / ("watch_0%d.png" % frame), (180, 160, 30 + frame, 255))

    # two genuinely separate items that merely share a numeric prefix
    for disc in (11, 13):
        write_json(root / "models" / "item" / ("disc_%d.json" % disc), {
            "parent": "item/generated",
            "textures": {"layer0": "items/disc_%d" % disc},
        })
        write_png(root / "textures" / "items" / ("disc_%d.png" % disc), (40 + disc, 90, 160, 255))

    # entity textures, which have no data-driven mapping
    write_png(root / "textures" / "entity" / "pig.png", (230, 150, 160, 255))
    write_png(root / "textures" / "entity" / "pig_saddle.png", (150, 90, 60, 255))

    # an unreferenced numeric family
    write_png(root / "textures" / "blocks" / "dust_0.png", (90, 80, 70, 255))
    write_png(root / "textures" / "blocks" / "dust_1.png", (120, 110, 100, 255))

    # an animated texture plus its metadata
    write_png(root / "textures" / "blocks" / "glow.png", (250, 220, 90, 255))
    write_json(root / "textures" / "blocks" / "glow.png.mcmeta", {"animation": {"frametime": 4}})
    write_json(root / "models" / "item" / "lantern.json", {
        "parent": "item/generated",
        "textures": {"layer0": "blocks/glow"},
    })
    return root.parents[1]
