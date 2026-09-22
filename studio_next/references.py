"""Reference-image feature extraction and explicit role assignment.

Images are evidence for structure, material and pixel style. They are never
silently treated as the final alpha mask by this module.
"""

from __future__ import annotations

from collections import Counter, deque
from io import BytesIO
from math import atan2, degrees, sqrt
import json
from pathlib import Path

from PIL import Image

from .contracts import ReferenceAsset, ReferenceRole


def _opaque_points(image: Image.Image, threshold: int = 8) -> set[tuple[int, int]]:
    rgba = image.convert("RGBA")
    return {
        (x, y)
        for y in range(rgba.height)
        for x in range(rgba.width)
        if rgba.getpixel((x, y))[3] >= threshold
    }


def _components(points: set[tuple[int, int]]) -> int:
    unseen = set(points)
    count = 0
    while unseen:
        count += 1
        queue = deque([unseen.pop()])
        while queue:
            x, y = queue.popleft()
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return count


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#%02X%02X%02X" % rgb


def _principal_axis(points: set[tuple[int, int]]) -> tuple[float, float]:
    """Return major-axis angle and anisotropy for an opaque silhouette."""
    if len(points) < 2:
        return 0.0, 0.0
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    xx = yy = xy = 0.0
    for x, y in points:
        dx, dy = x - mean_x, y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    xx /= len(points)
    yy /= len(points)
    xy /= len(points)
    angle = 0.5 * atan2(2.0 * xy, xx - yy)
    trace = xx + yy
    determinant = max(xx * yy - xy * xy, 0.0)
    root = sqrt(max(trace * trace - 4.0 * determinant, 0.0))
    major, minor = (trace + root) / 2.0, (trace - root) / 2.0
    anisotropy = (major - minor) / max(major + minor, 1e-9)
    return degrees(angle), anisotropy


def _pixel_text(image: Image.Image, palette: list[tuple[tuple[int, int, int], int]]) -> str:
    """Encode a pixel texture as a compact, lossless-for-small-images text map.

    `.` is transparent. Opaque pixels use a stable palette index, with a legend
    on the first line. Larger images are nearest-neighbour reduced only for this
    textual side channel; the original image is still attached to Vision.
    """
    rgba = image.convert("RGBA")
    if max(rgba.size) > 64:
        scale = 64 / float(max(rgba.size))
        rgba = rgba.resize((max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale))), Image.Resampling.NEAREST)
    colors = [color for color, _ in palette]
    symbols = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    legend = " ".join("%s=%s" % (symbols[index], _hex(color)) for index, color in enumerate(colors))
    rows = ["pixel_map (.=transparent; %s)" % legend]
    for y in range(rgba.height):
        row: list[str] = []
        for x in range(rgba.width):
            red, green, blue, alpha = rgba.getpixel((x, y))
            if alpha < 8:
                row.append(".")
                continue
            if not colors:
                row.append("0")
                continue
            nearest = min(
                range(len(colors)),
                key=lambda index: sum((channel - color_channel) ** 2 for channel, color_channel in zip((red, green, blue), colors[index])),
            )
            row.append(symbols[nearest])
        rows.append("".join(row))
    return "\n".join(rows)


def _run_lengths(values: list[int]) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _opaque_runs(values: list[int]) -> list[list[int]]:
    """Return inclusive-exclusive opaque runs for one pixel row/column."""
    runs: list[list[int]] = []
    start: int | None = None
    for index, value in enumerate(values + [0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index])
            start = None
    return runs


def _silhouette_profile(image: Image.Image) -> dict[str, object]:
    """Expose an alpha-only contour without making it a reusable template.

    Counts and run spans are easier for a text/vision planner to follow than a
    colour-index map alone. They preserve staircase width and gaps while still
    leaving the target geometry open for a new object.
    """
    alpha = image.convert("RGBA").getchannel("A")
    row_spans: list[list[list[int]]] = []
    column_spans: list[list[list[int]]] = []
    rows: list[str] = []
    for y in range(alpha.height):
        opaque = [1 if alpha.getpixel((x, y)) >= 8 else 0 for x in range(alpha.width)]
        row_spans.append(_opaque_runs(opaque))
        rows.append("".join("#" if value else "." for value in opaque))
    for x in range(alpha.width):
        opaque = [1 if alpha.getpixel((x, y)) >= 8 else 0 for y in range(alpha.height)]
        column_spans.append(_opaque_runs(opaque))
    return {
        "row_spans": row_spans,
        "column_spans": column_spans,
        "silhouette_map": "\n".join(rows),
    }


def _stroke_profile(image: Image.Image) -> dict[str, float | int | list[int]]:
    """Expose pixel-scale stroke evidence without turning it into a template.

    Row/column occupancy and contiguous run widths tell a planner whether a
    reference uses mostly one-pixel shafts, chunky blocks or a mixture. They
    are descriptive statistics only; no target shape is copied from them.
    """
    alpha = image.convert("RGBA").getchannel("A")
    row_counts: list[int] = []
    col_counts: list[int] = []
    runs: list[int] = []
    for y in range(alpha.height):
        row = [1 if alpha.getpixel((x, y)) >= 8 else 0 for x in range(alpha.width)]
        row_counts.append(sum(row))
        runs.extend(_run_lengths(row))
    for x in range(alpha.width):
        col_counts.append(sum(1 if alpha.getpixel((x, y)) >= 8 else 0 for y in range(alpha.height)))
    nonzero_rows = [value for value in row_counts if value]
    nonzero_cols = [value for value in col_counts if value]
    def median(values: list[int]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return {
        "row_opaque_median": median(nonzero_rows),
        "row_opaque_max": max(nonzero_rows, default=0),
        "column_opaque_median": median(nonzero_cols),
        "column_opaque_max": max(nonzero_cols, default=0),
        "contiguous_run_median": median(runs),
        "contiguous_run_max": max(runs, default=0),
        "row_opaque_counts": row_counts,
    }


def _symmetry_profile(image: Image.Image) -> dict[str, float]:
    """Measure repeated pixels across useful axes of a small reference.

    Item sprites are often authored around a diagonal, so anti-diagonal and
    diagonal matches are reported alongside ordinary horizontal/vertical ones.
    This is evidence for a texture planner, never a command to mirror a new
    object's silhouette.
    """
    rgba = image.convert("RGBA")
    width, height = rgba.size
    transforms = {
        "horizontal": lambda x, y: (width - 1 - x, y),
        "vertical": lambda x, y: (x, height - 1 - y),
        "diagonal": lambda x, y: (y, x),
        "anti_diagonal": lambda x, y: (width - 1 - y, height - 1 - x),
    }
    result: dict[str, float] = {}
    for name, transform in transforms.items():
        same = total = 0
        for y in range(height):
            for x in range(width):
                other_x, other_y = transform(x, y)
                if 0 <= other_x < width and 0 <= other_y < height:
                    same += rgba.getpixel((x, y)) == rgba.getpixel((other_x, other_y))
                    total += 1
        result[name] = same / float(max(total, 1))
    return result


def analyze_png(path: str | Path, palette_limit: int = 16) -> dict[str, object]:
    source = Path(path)
    with Image.open(source) as loaded:
        image = loaded.convert("RGBA")
    return _analyze_image(image, palette_limit)


def analyze_png_bytes(data: bytes, palette_limit: int = 16) -> dict[str, object]:
    """Same text description as analyze_png, for bytes already in memory.

    The live asset catalogue reads a texture once and hands the same bytes to
    both the content-addressed text cache and this analyzer, so a reference is
    never decoded twice.
    """
    with Image.open(BytesIO(data)) as loaded:
        image = loaded.convert("RGBA")
    return _analyze_image(image, palette_limit)


def _analyze_image(image: "Image.Image", palette_limit: int = 16) -> dict[str, object]:
    points = _opaque_points(image)
    colors = Counter(
        (r, g, b)
        for r, g, b, alpha in image.get_flattened_data()
        if alpha >= 8
    )
    if points:
        xs, ys = zip(*points)
        left, top, right, bottom = min(xs), min(ys), max(xs) + 1, max(ys) + 1
        bbox = [left, top, right, bottom]
        bbox_width, bbox_height = right - left, bottom - top
        orientation_degrees, axis_anisotropy = _principal_axis(points)
    else:
        bbox = None
        bbox_width = bbox_height = 0
        orientation_degrees = axis_anisotropy = 0.0
    palette = colors.most_common(palette_limit)
    stroke_profile = _stroke_profile(image)
    symmetry_profile = _symmetry_profile(image)
    silhouette_profile = _silhouette_profile(image)
    return {
        "width": image.width,
        "height": image.height,
        "opaque_pixels": len(points),
        "occupancy_ratio": len(points) / float(image.width * image.height),
        "bbox": bbox,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "aspect_ratio": bbox_width / float(max(bbox_height, 1)),
        "orientation_degrees": orientation_degrees,
        "axis_anisotropy": axis_anisotropy,
        "components_4_connected": _components(points),
        "palette": [_hex(color) for color, _ in palette],
        "stroke_profile": stroke_profile,
        "symmetry_profile": symmetry_profile,
        "silhouette_profile": silhouette_profile,
        "pixel_text": _pixel_text(image, palette),
    }


def reference_from_png(path: str | Path, roles: list[ReferenceRole] | None = None,
                       notes: list[str] | None = None) -> ReferenceAsset:
    source = Path(path).resolve()
    if not source.exists():
        raise FileNotFoundError("reference image not found: %s" % source)
    return ReferenceAsset(
        path=str(source),
        name=source.stem,
        roles=roles or [ReferenceRole.PIXEL_STYLE, ReferenceRole.MATERIAL],
        notes=notes or [],
        features=analyze_png(source),
    )


def summarize_reference(reference: ReferenceAsset, include_pixel_map: bool = False) -> str:
    features = reference.features
    summary = (
        "%s roles=%s size=%sx%s bbox=%s components=%s axis=%.1fdeg anisotropy=%.2f palette=%s"
        % (
            reference.name,
            ",".join(role.value for role in reference.roles),
            features.get("width", "?"),
            features.get("height", "?"),
            features.get("bbox", "?"),
            features.get("components_4_connected", "?"),
            float(features.get("orientation_degrees", 0.0)),
            float(features.get("axis_anisotropy", 0.0)),
            ",".join(features.get("palette", [])[:5]),
        )
    )
    if reference.notes:
        summary += " notes=" + "; ".join(str(note).replace("\n", " ").strip() for note in reference.notes if str(note).strip())
    stroke = features.get("stroke_profile", {})
    if isinstance(stroke, dict):
        summary += " stroke_rows=median:%s,max:%s run=median:%s,max:%s" % (
            stroke.get("row_opaque_median", "?"),
            stroke.get("row_opaque_max", "?"),
            stroke.get("contiguous_run_median", "?"),
            stroke.get("contiguous_run_max", "?"),
        )
    symmetry = features.get("symmetry_profile", {})
    if isinstance(symmetry, dict):
        summary += " symmetry=diag:%.2f anti_diag:%.2f h:%.2f v:%.2f" % (
            float(symmetry.get("diagonal", 0.0)),
            float(symmetry.get("anti_diagonal", 0.0)),
            float(symmetry.get("horizontal", 0.0)),
            float(symmetry.get("vertical", 0.0)),
        )
    if include_pixel_map:
        summary += "\n" + str(features.get("pixel_text", "pixel_map unavailable"))
        silhouette = features.get("silhouette_profile", {})
        if isinstance(silhouette, dict):
            summary += "\nsilhouette_map (#=opaque; .=transparent)\n" + str(
                silhouette.get("silhouette_map", "unavailable")
            )
            summary += "\nrow_spans (x0 inclusive, x1 exclusive)=" + json.dumps(
                silhouette.get("row_spans", []), separators=(",", ":")
            )
    return summary
