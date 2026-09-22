"""Magnify 16x16 sprites into a labelled contact sheet for eyeball review."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

SCALE = 12


def load(p):
    return Image.open(p).convert("RGBA")


def sheet(entries, out, scale=SCALE, cols=None):
    """entries: list of (label, path)."""
    cols = cols or len(entries)
    tiles = []
    for label, path in entries:
        im = load(path)
        big = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        tiles.append((label, big))
    pad = 8
    label_h = 20
    cell_w = max(t.width for _, t in tiles) + pad * 2
    cell_h = max(t.height for _, t in tiles) + pad * 2 + label_h
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGBA", (cell_w * cols, cell_h * rows), (24, 24, 30, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (label, tile) in enumerate(tiles):
        cx = (i % cols) * cell_w
        cy = (i // cols) * cell_h
        draw.text((cx + pad, cy + 4), label, fill=(230, 230, 235, 255))
        canvas.alpha_composite(tile, (cx + pad, cy + label_h + pad))
    canvas.save(out)
    print(out, canvas.size)


if __name__ == "__main__":
    import json
    spec = json.loads(sys.argv[1])
    sheet([(e[0], e[1]) for e in spec["entries"]], spec["out"], cols=spec.get("cols"))
