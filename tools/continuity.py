"""Pairwise continuity metrics for a set of same-object frames."""
import sys
from pathlib import Path
from PIL import Image


def load(path):
    return Image.open(path).convert("RGBA")


def pair(a, b):
    pa = {(x, y): a.getpixel((x, y)) for y in range(a.height) for x in range(a.width)}
    pb = {(x, y): b.getpixel((x, y)) for y in range(b.height) for x in range(b.width)}
    opa = {p for p, v in pa.items() if v[3] >= 8}
    opb = {p for p, v in pb.items() if v[3] >= 8}
    union = opa | opb
    shared = opa & opb
    iou = len(shared) / max(len(union), 1)
    agree = sum(1 for p in union if pa[p] == pb[p])
    return {
        "iou": round(iou, 4),
        "union": len(union),
        "colour_agreement": round(agree / max(len(union), 1), 4),
    }


def report(paths):
    imgs = [(Path(p).stem, load(p)) for p in paths]
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            na, a = imgs[i]
            nb, b = imgs[j]
            print("  %-14s vs %-14s %s" % (na, nb, pair(a, b)))


if __name__ == "__main__":
    report(sys.argv[1:])
