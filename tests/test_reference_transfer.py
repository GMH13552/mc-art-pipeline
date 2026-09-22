"""Tests for reference value transfer into the authored palette.

A live dark oak armour run bound every part to a 16x16 plank texture with
sampling "pattern" and composite "replace", asking for plank grain "instead of
a flat brown fill". The renderer produced exactly the flat fill: 43 sampled
source colours collapsed onto 4 authored swatches.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image

from studio_next.appearance import render_appearance
from studio_next.contracts import (
    AppearanceSpec,
    GeometrySpec,
    PartAppearance,
    PartSpec,
    PrimitiveSpec,
    ReferenceAsset,
    ReferenceRole,
)
from studio_next.geometry import compile_geometry

PALETTE = {"deep": "#1E1309", "mid": "#3E2912", "light": "#492F17"}


def _grainy(path: Path, size: int = 16, shades: int = 24) -> None:
    """A plank-like tile whose value rhythm is finer than the palette."""
    rng = random.Random(7)
    image = Image.new("RGBA", (size, size))
    for y in range(size):
        for x in range(size):
            base = 0x3A + (y % 4) * 6
            value = max(0, min(255, base + rng.randint(-6, 6)))
            image.putpixel((x, y), (value, int(value * 0.72), int(value * 0.45), 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    assert shades <= 256


def _plan(reference_path: Path, composite: str) -> tuple[GeometrySpec, AppearanceSpec, list[ReferenceAsset]]:
    parts = [PartSpec(id="plate", meaning="armour plate")]
    geometry = GeometrySpec(
        width=16,
        height=16,
        parts=parts,
        primitives=[
            PrimitiveSpec(id="plate_shape", part_id="plate", primitive="ellipse", params={"bbox": [0, 0, 16, 16]})
        ],
    )
    appearance = AppearanceSpec(
        palette=PALETTE,
        parts={"plate": PartAppearance(colors=["deep", "mid", "light"], material="wood")},
        reference_sampling="pattern",
        part_reference_sampling={"plate": "pattern"},
        part_reference_sources={"plate": "grain"},
        part_reference_composite={"plate": composite},
    )
    references = [
        ReferenceAsset(
            path=str(reference_path),
            name="grain",
            roles=[ReferenceRole.MATERIAL, ReferenceRole.PALETTE],
            features={},
        )
    ]
    return geometry, appearance, references


def _render(reference_path: Path, composite: str):
    geometry, appearance, references = _plan(reference_path, composite)
    compiled = compile_geometry(geometry)
    return render_appearance(geometry, compiled, appearance, seed=0, references=references)


def test_a_replace_reference_keeps_its_own_value_rhythm(tmp_path: Path) -> None:
    reference = tmp_path / "grain.png"
    _grainy(reference)
    result = _render(reference, "replace")
    colours = {
        result.getpixel((x, y))[:3]
        for y in range(result.height)
        for x in range(result.width)
        if result.getpixel((x, y))[3] >= 8
    }
    # The authored palette alone can only ever produce its own three entries.
    assert len(colours) > len(PALETTE), len(colours)


def test_a_hairline_part_keeps_its_ramp_instead_of_going_flat(tmp_path: Path) -> None:
    """A five-pixel arrow is all edge, so the inner-edge rule erased it.

    The rule gives a small part a readable contour, but on a part one or two
    pixels thick every pixel qualifies and the whole part was painted in its
    darkest stop -- a nocked arrow came out dark instead of the icy accent the
    planner had authored for it.
    """
    parts = [
        PartSpec(id="body", meaning="bow body"),
        PartSpec(id="arrow", meaning="nocked arrow"),
    ]
    body_rows = ["".join("X" if 0 <= 10 - y - 3 <= x <= 10 - y + 3 else "." for x in range(16)) for y in range(16)]
    arrow_rows = ["".join("X" if x == 5 and y in (2, 3) else "." for x in range(16)) for y in range(16)]
    geometry = GeometrySpec(
        width=16,
        height=16,
        parts=parts,
        primitives=[
            PrimitiveSpec(id="b", part_id="body", primitive="custom_mask",
                          params={"offset": [0, 0], "marker": "X", "rows": body_rows}),
            PrimitiveSpec(id="a", part_id="arrow", primitive="custom_mask",
                          params={"offset": [0, 0], "marker": "X", "rows": arrow_rows}),
        ],
    )
    appearance = AppearanceSpec(
        palette={"dark": "#0C3730", "mid": "#2CCDB1", "ice": "#E6F8FF"},
        parts={
            "body": PartAppearance(colors=["dark", "mid"], material="crystal"),
            "arrow": PartAppearance(colors=["dark", "mid", "ice"], material="crystal", shade_axis="top"),
        },
        reference_sampling="none",
    )
    compiled = compile_geometry(geometry)
    image = render_appearance(geometry, compiled, appearance)
    arrow = {
        image.getpixel((x, y))[:3]
        for y in range(16)
        for x in range(16)
        if compiled.part_masks["arrow"].getpixel((x, y)) > 0
    }
    assert arrow, "the arrow must be painted"
    assert arrow != {(12, 55, 48)}, "the hairline part must not be flattened to its darkest stop"
