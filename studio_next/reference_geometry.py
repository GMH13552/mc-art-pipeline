"""Safe shape-only fallback from an explicitly shape-labelled reference."""

from __future__ import annotations

from PIL import Image

from .contracts import GeometrySpec, PrimitiveSpec, ReferenceAsset, ReferenceRole, ShapeDescriptor, is_overlay_part
from .geometry import compile_geometry


def _reference_member_name(reference: ReferenceAsset) -> str | None:
    """The source family member a reference stands for, when it is one.

    A code-driven family (bow, clock, compass) expands one logical name into
    every frame, and the router attaches all of them to every request. The
    frame that matches *this* request is only visible through the family
    identity recorded when the group was expanded.
    """
    for note in reference.notes:
        if note.startswith("family_member="):
            return note.split("=", 1)[1].strip() or None
    name = str(reference.name or "")
    if ":" in name:
        return name.rsplit(":", 1)[1].strip() or None
    return None


def family_member_match(member: str, wanted: str) -> int:
    """How well a source family member name answers a request name; 0 = no.

    The planner names a family member after its whole set, so the request says
    crystal_bow_standby where the source frame is bow_standby. Accept either
    direction and score by length, which is what separates bow_pulling_1 from
    pulling_1.
    """
    member = str(member or "").strip().lower()
    wanted = str(wanted or "").strip().lower()
    if not member or not wanted:
        return 0
    if member == wanted:
        return len(member) + 1
    if wanted.endswith(member) or member.endswith(wanted):
        return len(member)
    return 0


def _same_size_shape_reference(
    references: list[ReferenceAsset],
    width: int,
    height: int,
    preferred_name: str | None = None,
) -> ReferenceAsset | None:
    """Pick the shape reference that shares the canvas pixel grid.

    When the request is one member of a code-driven family, the matching
    frame outranks list order. Every bow request carries all four vanilla bow
    frames with bow_pulling_0 first, so the first-match rule would conform the
    idle frame to the half-drawn one and silently destroy the animation the
    family was asked to produce.
    """
    candidates: list[ReferenceAsset] = []
    for reference in references:
        if ReferenceRole.SHAPE not in reference.roles:
            continue
        try:
            with Image.open(reference.path) as loaded:
                if loaded.size == (width, height):
                    candidates.append(reference)
        except (OSError, ValueError):
            continue
    if not candidates:
        return None
    if preferred_name:
        best: tuple[int, ReferenceAsset] | None = None
        for reference in candidates:
            score = family_member_match(_reference_member_name(reference), preferred_name)
            if score and (best is None or score > best[0]):
                best = (score, reference)
        if best is not None:
            return best[1]
    return candidates[0]


def reference_silhouette_partition(
    descriptor: ShapeDescriptor,
    geometry: GeometrySpec,
    references: list[ReferenceAsset],
    width: int,
    height: int,
    preferred_name: str | None = None,
) -> GeometrySpec | None:
    """Conform model-authored part masks to a same-size shape reference.

    A descriptor that chose appearance_only has already answered the shape
    question: the source contour *is* the requested contour and the design
    lives in the paint. The least reliable way to reach that answer is to ask
    the same model to re-draw the source raster cell by cell in ASCII. A live
    crystal-bow run wrote 'use bow_standby alpha as the silhouette host' and
    then returned a hand-drawn blob, which every later stage faithfully
    painted.

    The model's own part masks are the ownership evidence here. They already
    say which part sits where -- they are simply rasterised too loosely. Every
    source pixel is handed to the declared part whose mask lies nearest, which
    reproduces the source contour exactly while keeping the model's partition
    of it, so the downstream appearance plan still addresses the parts it was
    authored against.

    Nearest-mask rather than overlap, because a loose draft is often offset
    by a pixel: a live crystal-bow draft drew its string on column 13 where
    the source string sits on column 12, so a strict overlap test found no
    string at all and refused to conform a perfectly clear intent.

    This is reference conformance, not a category template: it is only reached
    when the descriptor itself asked to preserve the silhouette, and it
    declines when a required support has no mask to be nearest to, so a draft
    that genuinely describes a different object is left to the repair loop.
    """
    # A UV atlas already has a validated alpha authority of its own (the
    # model's declared regions scored against same-size references), so this
    # contour recovery must not second-guess it.
    if geometry.uv_regions:
        return None
    shape_reference = _same_size_shape_reference(references, width, height, preferred_name)
    if shape_reference is None:
        return None
    try:
        with Image.open(shape_reference.path) as loaded:
            alpha = loaded.convert('RGBA').getchannel('A')
    except (OSError, ValueError):
        return None
    try:
        compiled = compile_geometry(geometry)
    except (KeyError, TypeError, ValueError):
        return None
    if (compiled.width, compiled.height) != (width, height):
        return None

    source: list[tuple[int, int]] = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if alpha.getpixel((x, y)) >= 8
    ]
    if not source:
        return None

    claimants: list[tuple[str, list[tuple[int, int]]]] = []
    for part in descriptor.parts:
        if part.paint_only:
            continue
        mask = compiled.part_masks.get(part.id)
        points = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if mask is not None and mask.getpixel((x, y)) > 0
        ]
        if not points:
            # A declared support with no mask at all is not evidence about this
            # silhouette; the draft is describing something else.
            if part.required and not is_overlay_part(part):
                return None
            continue
        claimants.append((part.id, points))
    if not claimants:
        return None

    points_by_part: dict[str, set[tuple[int, int]]] = {part_id: set() for part_id, _ in claimants}
    for x, y in source:
        # Distance first, then the smaller mask. Two masks are equally near a
        # pixel surprisingly often, and the smaller one is the more specific
        # reading of it: a thin cord beats the body it is drawn across.
        part_id, _points = min(
            claimants,
            key=lambda item: (
                min(max(abs(px - x), abs(py - y)) for px, py in item[1]),
                len(item[1]),
                item[0],
            ),
        )
        points_by_part[part_id].add((x, y))

    primitives: list[PrimitiveSpec] = []
    for part in descriptor.parts:
        if part.paint_only:
            continue
        # A non-required part may have had no mask to claim anything with; it
        # simply keeps no geometry here instead of raising.
        points = points_by_part.get(part.id, set())
        if not points:
            continue
        mask_rows = [
            ''.join('X' if (x, y) in points else '.' for x in range(width))
            for y in range(height)
        ]
        primitives.append(
            PrimitiveSpec(
                id='reference_%s' % part.id,
                part_id=part.id,
                primitive='custom_mask',
                params={'offset': [0, 0], 'marker': 'X', 'rows': mask_rows},
                layer=part.layer,
            )
        )
    if not primitives:
        return None
    return GeometrySpec(
        width=width,
        height=height,
        parts=list(descriptor.parts),
        primitives=primitives,
        connections=[],
        constraints=[],
        background_transparent=True,
    )


def reference_mask_primitive(reference: ReferenceAsset, part_id: str, width: int, height: int) -> PrimitiveSpec:
    """Convert a shape reference's alpha silhouette into a centered custom mask."""
    with Image.open(reference.path) as loaded:
        alpha = loaded.convert("RGBA").getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("shape reference has an empty alpha mask: %s" % reference.path)
    alpha = alpha.crop(bbox)
    scale = min((width - 2) / float(max(alpha.width, 1)), (height - 2) / float(max(alpha.height, 1)))
    resized = alpha.resize(
        (max(1, round(alpha.width * scale)), max(1, round(alpha.height * scale))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("L", (width, height), 0)
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    rows = ["".join("X" if canvas.getpixel((x, y)) >= 8 else "." for x in range(width)) for y in range(height)]
    return PrimitiveSpec(
        id="reference_shape",
        part_id=part_id,
        primitive="custom_mask",
        params={"offset": [0, 0], "rows": rows, "marker": "X"},
        layer=0,
    )

def reference_fallback_geometry(
    descriptor: ShapeDescriptor,
    references: list[ReferenceAsset],
    width: int,
    height: int,
    preferred_name: str | None = None,
) -> GeometrySpec | None:
    """Use a shape reference only for a one-part auto fallback."""
    required = [part for part in descriptor.parts if part.required and not part.paint_only]
    if len(required) != 1:
        return None
    chosen = _same_size_shape_reference(references, width, height, preferred_name)
    if chosen is None:
        candidates = [reference for reference in references if ReferenceRole.SHAPE in reference.roles]
        chosen = candidates[0] if candidates else None
    if chosen is None:
        return None
    return GeometrySpec(
        width=width,
        height=height,
        parts=list(descriptor.parts),
        primitives=[reference_mask_primitive(chosen, required[0].id, width, height)],
        connections=[],
        constraints=[],
    )


def reference_partition_geometry(
    descriptor: ShapeDescriptor,
    references: list[ReferenceAsset],
    width: int,
    height: int,
    preferred_name: str | None = None,
) -> GeometrySpec | None:
    """Recover a multi-part mask from model-authored ownership evidence.

    A vision planner may describe a same-size variant correctly and still return
    an unusable geometry draft (for example, an empty handle).  When it has also
    supplied ``reference_part_map`` we can recover the exact source silhouette
    without inventing a category template: the map assigns source pixels to the
    descriptor's own part IDs, and any unlabelled opaque source pixel is assigned
    to the nearest labelled part.  This is a last-resort repair path only; normal
    requests continue to use the model's geometry directly.
    """
    evidence = descriptor.reference_part_map
    if not isinstance(evidence, dict):
        return None
    legend = evidence.get("legend")
    rows = evidence.get("rows")
    if not isinstance(legend, dict) or not isinstance(rows, list):
        return None
    if len(rows) != height or any(not isinstance(row, str) for row in rows):
        return None
    part_ids = {part.id for part in descriptor.parts}
    decoded: dict[str, str] = {
        str(marker): str(part_id)
        for marker, part_id in legend.items()
        if str(part_id) in part_ids
    }
    if not decoded:
        return None
    shape_reference = _same_size_shape_reference(references, width, height, preferred_name)
    if shape_reference is None:
        return None
    try:
        with Image.open(shape_reference.path) as loaded:
            alpha = loaded.convert("RGBA").getchannel("A")
    except (OSError, ValueError):
        return None

    # Vision text maps occasionally carry one trailing dot after a row. Treat
    # that as a serialization slip when it does not alter the source-sized
    # coordinate grid; reject larger or ambiguous maps instead of silently
    # shifting a physical part.
    normalized_rows: list[str] = []
    for raw_row in rows:
        row = raw_row
        while len(row) > width and row.endswith("."):
            row = row[:-1]
        if len(row) < width:
            row = row + "." * (width - len(row))
        if len(row) != width:
            return None
        normalized_rows.append(row)

    points_by_part: dict[str, list[tuple[int, int]]] = {part_id: [] for part_id in part_ids}
    unassigned: list[tuple[int, int]] = []
    for y, row in enumerate(normalized_rows):
        for x, marker in enumerate(row):
            if alpha.getpixel((x, y)) < 8:
                continue
            part_id = decoded.get(marker)
            if part_id is None:
                unassigned.append((x, y))
            else:
                points_by_part[part_id].append((x, y))

    # A target ownership map can add a newly authored local motif on pixels
    # that were transparent in the source.  It is overlay evidence: source
    # support points above remain intact, while target-only part IDs contribute
    # their model-chosen cells to a second labelled mask.
    target = descriptor.target_part_map
    if isinstance(target, dict) and isinstance(target.get("legend"), dict) and isinstance(target.get("rows"), list):
        target_rows = list(target.get("rows", []))
        if len(target_rows) == height:
            target_decoded = {
                str(marker): str(part_id)
                for marker, part_id in target["legend"].items()
                if str(part_id) in part_ids
            }
            target_normalized: list[str] = []
            for raw_row in target_rows:
                row = str(raw_row)
                while len(row) > width and row.endswith("."):
                    row = row[:-1]
                if len(row) < width:
                    row += "." * (width - len(row))
                target_normalized.append(row if len(row) == width else "")
            if all(target_normalized):
                source_owned = {part_id for part_id, points in points_by_part.items() if points}
                target_only = {
                    part_id for part_id in part_ids
                    if part_id not in source_owned
                }
                for y, row in enumerate(target_normalized):
                    for x, marker in enumerate(row):
                        part_id = target_decoded.get(marker)
                        if part_id in target_only:
                            points_by_part[part_id].append((x, y))

    # A required part with no ownership evidence cannot be reconstructed safely
    # when it is an unchanged support.  A newly authored local motif is absent
    # from the source by definition, so its empty source mask is valid; the
    # geometry model or target ownership map will author it separately.
    if any(
        part.required and not part.paint_only
        and not points_by_part[part.id]
        and not is_overlay_part(part)
        for part in descriptor.parts
    ):
        return None
    labelled = [(part_id, point) for part_id, points in points_by_part.items() for point in points]
    if not labelled and unassigned:
        return None
    for x, y in unassigned:
        part_id, _point = min(
            labelled,
            key=lambda item: (max(abs(item[1][0] - x), abs(item[1][1] - y)), item[0]),
        )
        points_by_part[part_id].append((x, y))

    primitives: list[PrimitiveSpec] = []
    for part in descriptor.parts:
        if part.paint_only:
            continue
        points = set(points_by_part[part.id])
        if not points:
            continue
        mask_rows = [
            "".join("X" if (x, y) in points else "." for x in range(width))
            for y in range(height)
        ]
        primitives.append(
            PrimitiveSpec(
                id="reference_%s" % part.id,
                part_id=part.id,
                primitive="custom_mask",
                params={"offset": [0, 0], "marker": "X", "rows": mask_rows},
                layer=part.layer,
            )
        )
    if not primitives:
        return None
    return GeometrySpec(
        width=width,
        height=height,
        parts=list(descriptor.parts),
        primitives=primitives,
        connections=[],
        constraints=[],
        background_transparent=True,
    )
