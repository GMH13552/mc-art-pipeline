"""Render a labelled contact sheet for one generated family and its sources."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

SCALE = 12


def tile(path, scale=SCALE):
    with Image.open(path) as loaded:
        image = loaded.convert("RGBA")
    return image.resize((image.width * scale, image.height * scale), Image.NEAREST)


def sheet(rows, out, cols=4):
    pad, label_h = 10, 22
    tiles = [(label, tile(path)) for label, path in rows]
    cell_w = max(t.width for _, t in tiles) + pad * 2
    cell_h = max(t.height for _, t in tiles) + pad * 2 + label_h
    grid = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGBA", (cell_w * min(cols, len(tiles)), cell_h * grid), (26, 26, 32, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(tiles):
        cx = (index % cols) * cell_w
        cy = (index // cols) * cell_h
        draw.text((cx + pad, cy + 5), label, fill=(235, 235, 240, 255))
        canvas.alpha_composite(image, (cx + pad, cy + label_h + pad))
    canvas.save(out)
    print(out, canvas.size)


if __name__ == "__main__":
    family = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = sys.argv[2]
    rows = [(row["name"], row["sprite"]) for row in family["members"] if row.get("sprite")]
    sheet(rows, out, cols=len(rows))
