"""Render one family's frames next to their source frames, with the metrics."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from studio_next.quality import frame_continuity, _member_shape_host

SCALE = 12


def tile(path):
    with Image.open(path) as loaded:
        image = loaded.convert("RGBA")
    return image.resize((image.width * SCALE, image.height * SCALE), Image.NEAREST)


def sheet(rows, out, cols):
    pad, label_h = 10, 22
    tiles = [(label, tile(path)) for label, path in rows]
    cell_w = max(t.width for _, t in tiles) + pad * 2
    cell_h = max(t.height for _, t in tiles) + pad * 2 + label_h
    grid = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGBA", (cell_w * cols, cell_h * grid), (26, 26, 32, 255))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(tiles):
        cx = (index % cols) * cell_w
        cy = (index // cols) * cell_h
        draw.text((cx + pad, cy + 5), label, fill=(235, 235, 240, 255))
        canvas.alpha_composite(image, (cx + pad, cy + label_h + pad))
    canvas.save(out)
    print("sheet ->", out, canvas.size)


def main(family_path):
    family = json.loads(Path(family_path).read_text(encoding="utf-8"))
    members = [row for row in family["members"] if row.get("sprite")]
    generated = [Path(row["sprite"]) for row in members]
    sources = []
    for row in members:
        host = _member_shape_host(Path(row["out_dir"]), row["name"], row.get("selected_round"))
        sources.append(host)

    print("members:", len(members))
    for row, host in zip(members, sources):
        sprite = Path(row["sprite"])
        if host is None:
            print("  %-24s validation=%-5s (no shape host recorded)" % (
                row["name"], row["validation_passed"]))
            continue
        a = Image.open(sprite).convert("RGBA")
        b = Image.open(host).convert("RGBA")
        A = {(x, y) for y in range(a.height) for x in range(a.width) if a.getpixel((x, y))[3] >= 8}
        B = {(x, y) for y in range(b.height) for x in range(b.width) if b.getpixel((x, y))[3] >= 8}
        iou = len(A & B) / float(max(len(A | B), 1))
        print("  %-24s validation=%-5s contour IoU vs source=%.4f identical=%s" % (
            row["name"], row["validation_passed"], iou, A == B))

    generated = [p for p in generated if p.exists()]
    sources = [p for p in sources if p is not None and Path(p).exists()]
    if len(generated) > 1:
        report = frame_continuity(generated, baseline_sprites=sources or None)
        print()
        print("generated continuity : exact min=%s mean=%s | near min=%s mean=%s" % (
            report["minimum_agreement"], report["mean_agreement"],
            report["minimum_near_agreement"], report["mean_near_agreement"]))
        if report["baseline"]:
            base = report["baseline"]
            print("source family        : exact min=%s mean=%s | near min=%s mean=%s" % (
                base["minimum_agreement"], base["mean_agreement"],
                base["minimum_near_agreement"], base["mean_near_agreement"]))
    if sources and len(sources) == len(generated):
        sheet(
            [("SOURCE %d" % i, p) for i, p in enumerate(sources)]
            + [("GENERATED %d" % i, p) for i, p in enumerate(generated)],
            Path(family_path).parent / "frames_compare.png",
            cols=len(generated),
        )
    else:
        sheet([(p.parent.name, p) for p in generated],
              Path(family_path).parent / "frames_compare.png", cols=len(generated))


if __name__ == "__main__":
    main(sys.argv[1])
