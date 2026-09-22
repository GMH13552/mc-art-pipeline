"""Load persisted plans and create fully dynamic plans with an optional model."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .contracts import (
    ArtDirection,
    AssetRequest,
    AssetForm,
    AppearanceSpec,
    ConnectionSpec,
    GeometrySpec,
    PartAppearance,
    PartSpec,
    PrimitiveSpec,
    ReferenceAsset,
    ReferenceRole,
    UvRegionSpec,
    appearance_from_dict,
    descriptor_from_dict,
    geometry_from_dict,
    request_from_dict,
    is_overlay_part,
    to_jsonable,
)
from .llm import ModelPlanner, OpenAICompatibleClient, _target_map_covers_declared_parts
from .pipeline import GenerationPlan
from .reference_geometry import reference_partition_geometry
from .references import reference_from_png
from .uv_layout import layout_regions, layout_summary
from .geometry import compile_geometry
from .validation import validate_geometry


def _empty_target_overlay_ids(descriptor: Any) -> set[str]:
    """Find physical overlay markers declared but never placed by the model."""
    target = descriptor.target_part_map
    if not isinstance(target, dict):
        return set()
    legend = target.get("legend")
    rows = target.get("rows")
    if not isinstance(legend, dict) or not isinstance(rows, list):
        return set()
    marker_by_part = {str(part_id): str(marker) for marker, part_id in legend.items()}
    return {
        part.id
        for part in descriptor.parts
        if (
            part.required
            and not part.paint_only
            and is_overlay_part(part)
            and (marker := marker_by_part.get(part.id)) is not None
            and not any(isinstance(row, str) and marker in row for row in rows)
        )
    }


def _normalize_paint_only_appearance(
    appearance: AppearanceSpec,
    descriptor: Any,
) -> AppearanceSpec:
    """Keep a surface-only feature from projecting a whole raster on its host.

    A paint-only part has deliberately no independent mask or bounds. Direct
    source sampling for it therefore has no stable local coordinate system and
    can turn a secondary material/motif reference into a full-host texture.
    The model can still use that evidence through palette, marks, region rules
    or pixel_map, but must author its actual coverage. This is a contract
    consequence of ``paint_only``, not an object-specific rule.
    """
    paint_only_ids = {part.id for part in descriptor.parts if part.paint_only}
    if not paint_only_ids:
        return appearance
    return replace(
        appearance,
        part_reference_sampling={
            part_id: mode
            for part_id, mode in appearance.part_reference_sampling.items()
            if part_id not in paint_only_ids
        },
        part_reference_sources={
            part_id: source
            for part_id, source in appearance.part_reference_sources.items()
            if part_id not in paint_only_ids
        },
        part_reference_composite={
            part_id: composite
            for part_id, composite in appearance.part_reference_composite.items()
            if part_id not in paint_only_ids
        },
    )


def _augment_descriptor_for_uv(descriptor: Any, regions: list[UvRegionSpec]) -> Any:
    """Make the supplied model layout authoritative about physical UV parts."""
    known = {part.id for part in descriptor.parts}
    additions: list[PartSpec] = []
    layer = max((part.layer for part in descriptor.parts), default=0) + 1
    by_part: dict[str, list[UvRegionSpec]] = {}
    for region in regions:
        by_part.setdefault(region.part_id, []).append(region)
    for part_id, part_regions in by_part.items():
        if part_id in known:
            continue
        notes = next((region.notes for region in part_regions if region.notes), "")
        additions.append(PartSpec(
            id=part_id,
            meaning=notes or ("physical model part from supplied UV layout: " + part_id),
            required=any(region.required for region in part_regions),
            layer=layer,
            style_role="entity_surface",
        ))
        layer += 1
    if not additions:
        return descriptor
    from dataclasses import replace

    return replace(descriptor, parts=[*descriptor.parts, *additions])


def _normalize_entity_uv_geometry(
    geometry: Any,
    descriptor: Any,
    regions: list[UvRegionSpec],
) -> Any:
    """Bind an entity atlas to its declared UV faces, never to a 2-D icon mask.

    A model may describe eyes, tails or other visible details that live inside
    an existing cube's texture region.  Those details are paint-only in the
    atlas contract; allowing the geometry response to invent a second alpha
    canvas part would either paint outside the model UV or make compilation
    fail.  Each declared UV part therefore gets one deterministic ``uv_fill``
    base mask, while appearance marks carry within-face detail.
    """
    from dataclasses import replace

    allowed_ids = list(dict.fromkeys(region.part_id for region in regions))
    descriptor_by_id = {part.id: part for part in descriptor.parts}
    parts = [part for part in geometry.parts if part.id in allowed_ids]
    geometry_by_id = {part.id: part for part in parts}
    for part_id in allowed_ids:
        if part_id in geometry_by_id:
            continue
        part = descriptor_by_id.get(part_id)
        if part is not None:
            parts.append(part)
            geometry_by_id[part_id] = part
    # The descriptor is authoritative for UV parts, so a missing synthetic
    # part is only possible with a malformed model response. Keep a valid
    # contract rather than manufacturing semantic names here.
    allowed_ids = [part_id for part_id in allowed_ids if part_id in geometry_by_id]
    # The model owns the atlas paint plan, and the geometry prompt already
    # invites it to use "narrower face treatment" where a part needs it. This
    # step used to discard every model-authored primitive and substitute a
    # full-face base coat, so no suit could ever have a face, neck or hand
    # opening -- and a request for a deliberately different silhouette had no
    # way to say so. Keep the model's own atlas primitives for these parts and
    # synthesise a base coat only where it authored none.
    primitives: list[PrimitiveSpec] = []
    authored_parts: set[str] = set()
    for primitive in geometry.primitives:
        if primitive.part_id not in allowed_ids:
            continue
        if primitive.primitive not in {"uv_fill", "cutout"}:
            continue
        primitives.append(primitive)
        if primitive.primitive == "uv_fill":
            authored_parts.add(primitive.part_id)
    for part_id in allowed_ids:
        if part_id in authored_parts:
            continue
        primitives.append(PrimitiveSpec(
            id="%s_uv_fill" % part_id,
            part_id=part_id,
            primitive="uv_fill",
            params={"regions": [region.id for region in regions if region.part_id == part_id]},
            layer=geometry_by_id[part_id].layer,
        ))
    return replace(
        geometry,
        parts=parts,
        primitives=primitives,
        # Atlas regions are spatially disjoint texture rectangles; model-world
        # touch/overlap constraints do not apply to this 2-D representation.
        connections=[],
        constraints=[],
    )


def _normalize_entity_appearance(
    appearance: AppearanceSpec,
    descriptor: Any,
    geometry: Any,
) -> AppearanceSpec:
    """Normalize atlas paint declarations without inferring an object taxonomy.

    Geometry owns the set of UV regions and the model owns the material intent.
    This pass only binds marks/rules to those exact regions and, for pattern
    sampling, gives an under-specified ramp a few value bands. It deliberately
    does not inspect names such as ``muzzle`` or ``blade`` and never invents
    feature pixels. That keeps the same path usable for arbitrary entities,
    blocks, clothing, tools and modded layouts.
    """
    from dataclasses import replace

    del descriptor  # Appearance normalization is intentionally descriptor-free.
    regions_by_part: dict[str, list[UvRegionSpec]] = {}
    for region in geometry.uv_regions:
        regions_by_part.setdefault(region.part_id, []).append(region)

    normalized_palette = dict(appearance.palette)
    normalized_parts: dict[str, PartAppearance] = {}

    def rgb_for_token(token: str) -> tuple[int, int, int]:
        raw = str(normalized_palette.get(token, token)).strip().lstrip("#")
        if len(raw) == 3:
            raw = "".join(char * 2 for char in raw)
        try:
            if len(raw) != 6:
                raise ValueError
            return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
        except (TypeError, ValueError):
            return (96, 96, 96)

    def token_luma(token: str) -> float:
        red, green, blue = rgb_for_token(token)
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    def add_palette_token(prefix: str, color: tuple[int, int, int]) -> str:
        token = prefix
        suffix = 1
        while token in normalized_palette and rgb_for_token(token) != color:
            suffix += 1
            token = "%s_%d" % (prefix, suffix)
        normalized_palette.setdefault(
            token,
            "#%02X%02X%02X" % tuple(max(0, min(255, value)) for value in color),
        )
        return token

    def pattern_ramp(part_id: str, style: PartAppearance, regions: list[UvRegionSpec]) -> list[str]:
        colors: list[str] = []
        seen: set[tuple[int, int, int]] = set()
        for token in style.colors:
            color = rgb_for_token(token)
            if color in seen:
                continue
            seen.add(color)
            colors.append(token)
        # A single swatch can be intentional for a tiny accent. For a larger
        # UV surface, however, pattern sampling needs at least dark/mid/light
        # control points or all source texture bands collapse to one colour.
        area = sum(
            max(0, region.bbox[2] - region.bbox[0]) * max(0, region.bbox[3] - region.bbox[1])
            for region in regions
        )
        material = style.material.strip().lower()
        accent_materials = {"accent", "glow", "eye", "symbol"}
        if appearance.reference_sampling != "pattern" or area < 16 or material in accent_materials:
            return colors or list(style.colors)
        if len(colors) == 1:
            base = rgb_for_token(colors[0])
            dark = tuple(round(channel * 0.68) for channel in base)
            light = tuple(min(255, round(channel * 1.24 + 4)) for channel in base)
            colors = [
                add_palette_token("%s_pattern_dark" % part_id, dark),
                colors[0],
                add_palette_token("%s_pattern_light" % part_id, light),
            ]
        elif len(colors) == 2:
            first, second = (rgb_for_token(token) for token in colors)
            midpoint = tuple(round((first[channel] + second[channel]) / 2) for channel in range(3))
            ordered = sorted(colors, key=token_luma)
            colors = [ordered[0], add_palette_token("%s_pattern_mid" % part_id, midpoint), ordered[1]]
        # Keep a small set of evenly distributed control points. Midpoints are
        # derived from the authored palette, so the requested hue remains the
        # model's choice and no source-specific swatch is imported here.
        target_count = min(8, max(3, len(colors)))
        while len(colors) < target_count:
            ordered = sorted(colors, key=token_luma)
            gaps = [
                (token_luma(ordered[index + 1]) - token_luma(ordered[index]), index)
                for index in range(len(ordered) - 1)
            ]
            if not gaps or max(gap for gap, _index in gaps) <= 0:
                break
            _gap, gap_index = max(gaps)
            left, right = (rgb_for_token(token) for token in ordered[gap_index:gap_index + 2])
            midpoint = tuple(round((left[channel] + right[channel]) / 2) for channel in range(3))
            if midpoint in {rgb_for_token(token) for token in ordered}:
                break
            mid_token = add_palette_token("%s_pattern_%d" % (part_id, len(ordered)), midpoint)
            colors = ordered[:gap_index + 1] + [mid_token] + ordered[gap_index + 1:]
        return colors

    for part_id, style in appearance.parts.items():
        regions = regions_by_part.get(part_id, [])
        # A part without a physical UV region cannot be painted on an atlas.
        if not regions:
            continue
        region_ids = {region.id for region in regions}
        front_regions = [
            region for region in regions
            if region.face.lower() in {"front", "north", "south"}
        ]
        marks: list[dict[str, Any]] = []
        for raw_mark in style.marks:
            if not isinstance(raw_mark, dict):
                continue
            mark = dict(raw_mark)
            raw_regions = mark.get("regions", [])
            if not raw_regions and mark.get("region") is not None:
                raw_regions = [mark.get("region")]
            if isinstance(raw_regions, str):
                raw_regions = [raw_regions]
            if not isinstance(raw_regions, list):
                raw_regions = []
            resolved_ids = [str(region_id) for region_id in raw_regions if str(region_id) in region_ids]
            # Shorthand is only a binding convenience; it does not inspect the
            # meaning of the part and never stamps a mark onto every face.
            if not resolved_ids and front_regions:
                shorthand = any(str(item).lower() in {part_id.lower(), "front", "face"} for item in raw_regions)
                if shorthand or not raw_regions:
                    resolved_ids = [front_regions[0].id]
            if not resolved_ids:
                continue
            mark["regions"] = resolved_ids
            rows = mark.get("rows")
            if rows is None and isinstance(mark.get("grid"), list):
                # Normalize a compact numeric mask emitted by some vision
                # models into the canonical textual row contract.  Non-zero
                # cells remain authored pixels; positions and colour still
                # come entirely from the model response.
                rows = [
                    "".join("." if str(cell).strip() in {"", "0", "."} else "X" for cell in row)
                    if isinstance(row, list)
                    else str(row)
                    for row in mark.get("grid", [])
                ]
            if isinstance(rows, list) and rows:
                target = next((region for region in regions if region.id == resolved_ids[0]), None)
                if target is None:
                    continue
                target_width = max(0, target.bbox[2] - target.bbox[0])
                target_height = max(0, target.bbox[3] - target.bbox[1])
                cleaned: list[str] = []
                for row in rows[:target_height]:
                    # The canonical mark raster uses only ``X`` and ``.``.
                    # Vision models sometimes return A/B/C (or other labels)
                    # to describe value bands; treating every non-dot as an
                    # opaque pixel turns those labels into invented motifs.
                    # Keep the declared contract strict and discard labels
                    # while retaining a lowercase-x compatibility path.
                    text = "".join("X" if char in {"X", "x"} else "." for char in str(row))
                    cleaned.append(text[:target_width].ljust(target_width, "."))
                mark["rows"] = cleaned
                if not any(char not in {".", " "} for row in cleaned for char in row):
                    continue
            marks.append(mark)
        colors = pattern_ramp(part_id, style, regions)
        normalized_parts[part_id] = replace(style, colors=colors, marks=marks)

    valid_region_ids = {region.id for region in geometry.uv_regions}
    normalized_rules: list[dict[str, Any]] = []
    for raw_rule in getattr(appearance, "region_rules", []) or []:
        if not isinstance(raw_rule, dict):
            continue
        raw_regions = raw_rule.get("regions", [])
        if not raw_regions and raw_rule.get("region_id") is not None:
            raw_regions = [raw_rule.get("region_id")]
        if isinstance(raw_regions, str):
            raw_regions = [raw_regions]
        if not isinstance(raw_regions, list):
            continue
        regions = [str(region_id) for region_id in raw_regions if str(region_id) in valid_region_ids]
        if not regions:
            continue
        mode = str(raw_rule.get("mode", "")).strip().lower()
        # Accept the compact field form some vision models use when they
        # return a rule (`retint: "purple"`) instead of the canonical mode /
        # target_color pair.  The renderer still receives one normalized
        # contract and never infers region meaning from names.
        if not mode:
            if raw_rule.get("retint") is not None:
                mode = "retint"
            elif raw_rule.get("source_exact") is not None:
                mode = "source_exact"
            else:
                mode = "palette"
        if mode not in {"palette", "retint", "recolor", "tint", "source_exact", "exact"}:
            continue
        # A whole-face palette rewrite discards the directional grain, rings,
        # seams or other pixel rhythm that the model explicitly chose to keep
        # through reference pattern/value sampling.  Let the reference-owning
        # face retain its structure; model-authored rules may still retint or
        # add local marks without replacing that evidence.
        region_part_ids = {
            str(region.part_id) for region in geometry.uv_regions if str(region.id) in regions
        }
        if mode == "palette" and any(
            appearance.part_reference_sampling.get(str(part_id), appearance.reference_sampling)
            in {"pattern", "value"}
            for part_id in region_part_ids
        ):
            continue
        cleaned: dict[str, Any] = {"regions": regions, "mode": mode}
        target_color = raw_rule.get("target_color")
        if target_color is None and mode == "retint":
            target_color = raw_rule.get("retint")
        if target_color is not None:
            cleaned["target_color"] = str(target_color)
        if "cluster_only" in raw_rule:
            cleaned["cluster_only"] = bool(raw_rule["cluster_only"])
        if "strength" in raw_rule:
            try:
                cleaned["strength"] = max(0.0, min(1.0, float(raw_rule["strength"])))
            except (TypeError, ValueError):
                cleaned["strength"] = 1.0
        normalized_rules.append(cleaned)

    physical_part_ids = set(normalized_parts)
    return replace(
        appearance,
        palette=normalized_palette,
        parts=normalized_parts,
        region_rules=normalized_rules,
        # Keep reference bindings on the same concrete UV parts as the paint
        # declarations. A descriptor may contain semantic surface notes whose
        # ids have no UV rectangle; retaining their bindings silently loses
        # the face/material relationship downstream.
        part_reference_sampling={
            part_id: mode
            for part_id, mode in appearance.part_reference_sampling.items()
            if part_id in physical_part_ids
        },
        part_reference_sources={
            part_id: source
            for part_id, source in appearance.part_reference_sources.items()
            if part_id in physical_part_ids
        },
        part_reference_composite={
            part_id: mode
            for part_id, mode in appearance.part_reference_composite.items()
            if part_id in physical_part_ids
        },
    )

def reference_from_dict(data: dict[str, Any]) -> ReferenceAsset:
    normalized = dict(data)
    normalized["roles"] = [ReferenceRole(role) for role in normalized.get("roles", [])]
    if not normalized.get("features"):
        source = Path(str(normalized.get("path", "")))
        if source.exists() and source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            normalized["features"] = reference_from_png(source).features
    return ReferenceAsset(**normalized)


def plan_from_dict(data: dict[str, Any]) -> GenerationPlan:
    return GenerationPlan(
        request=request_from_dict(data["request"]),
        descriptor=descriptor_from_dict(data["descriptor"]),
        geometry=geometry_from_dict(data["geometry"]),
        appearance=appearance_from_dict(data["appearance"]),
        references=[reference_from_dict(item) for item in data.get("references", [])],
    )


def plan_from_file(path: str | Path) -> GenerationPlan:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if data.get("references"):
        data = dict(data)
        resolved_references = []
        for reference in data["references"]:
            item = dict(reference)
            reference_path = Path(str(item["path"]))
            if not reference_path.is_absolute():
                item["path"] = str((source.parent / reference_path).resolve())
            resolved_references.append(item)
        data["references"] = resolved_references
    layout_file = data.get("uv_layout_file")
    if layout_file:
        data = dict(data)
        geometry = dict(data["geometry"])
        geometry["uv_regions"] = to_jsonable(
            uv_layout_from_file((source.parent / str(layout_file)).resolve())
        )
        data["geometry"] = geometry
    return plan_from_dict(data)


def reference_from_spec(spec: str) -> ReferenceAsset:
    """Parse `path::role,role` without making roles a positional CLI option.

    The delimiter is chosen so a normal Windows drive prefix (`C:`) remains
    untouched. Omitting it records the source as material and pixel-style
    evidence, never as an implicit shape lock.
    """
    path, separator, roles_text = spec.rpartition("::")
    if not separator:
        return reference_from_png(spec)
    if not path:
        raise ValueError("reference path cannot be blank")
    roles = [ReferenceRole(item.strip()) for item in roles_text.split(",") if item.strip()]
    if not roles:
        raise ValueError("reference roles after :: cannot be blank")
    return reference_from_png(path, roles=roles)


def uv_layout_from_file(path: str | Path) -> list[UvRegionSpec]:
    """Load a caller-supplied generic UV layout (`[{...}]` or `{regions:[...]}`)."""
    source = Path(path)
    return layout_regions(json.loads(source.read_text(encoding="utf-8")))


def vanilla_layout_catalog(root: str | Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Return local model UV layouts with summaries for automatic routing."""
    project_root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    layout_root = project_root / "layouts"
    result: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(layout_root.glob("*.json")):
        try:
            regions = uv_layout_from_file(path)
        except (OSError, ValueError, TypeError):
            continue
        result.append((str(path.resolve()), layout_summary(regions)))
    return result


def create_model_plan(request: AssetRequest, reference_paths: list[str] | None = None,
                      client: OpenAICompatibleClient | None = None,
                      planner: ModelPlanner | None = None,
                      uv_layout_path: str | Path | None = None,
                      reference_assets: list[ReferenceAsset] | None = None,
                      art_direction: ArtDirection | None = None) -> GenerationPlan:
    # Automatic routing already has semantic display names and role decisions
    # for each selected image.  Preserve those objects when supplied instead
    # of reparsing content-addressed blob paths, which would turn ``cow`` and
    # ``magma`` into opaque hashes and reset their roles to defaults.  The
    # path-based form remains the public compatibility path for CLI callers.
    references = (
        list(reference_assets)
        if reference_assets is not None
        else [reference_from_spec(path) for path in reference_paths or []]
    )
    # Shape evidence is assigned by the automatic router (or explicitly by a
    # caller using ``PATH::shape``). Do not infer it from a transparent PNG:
    # an unrelated material swatch such as leather can also be a compact
    # raster, and treating every selected image as a silhouette source makes
    # the geometry model mix two unrelated contours.
    planner = planner or ModelPlanner(client or OpenAICompatibleClient.from_env())
    descriptor = planner.describe(
        query=request.query,
        form=request.form,
        width=request.width,
        height=request.height,
        references=references,
        art_direction=art_direction,
    )
    # A target ownership map is geometry evidence only.  For an explicitly
    # silhouette-preserving request it carries no new geometry and commonly
    # degenerates into a labelled copy of the source.  Discard that duplicate
    # instead of allowing it to masquerade as a required paint layout.
    direction = descriptor.art_direction or art_direction
    if direction is not None and not direction.requires_silhouette_change:
        descriptor = replace(descriptor, target_part_map=None)
    # Preserve the model's open edit decision at the contract boundary.  When
    # it explicitly chose a silhouette-preserving mode but failed to provide a
    # usable target ownership map for a newly named local overlay, that overlay
    # is surface-authored by definition: promote it to paint-only instead of
    # allowing an empty alpha part to poison geometry compilation.  This is a
    # generic consistency repair, not an object-specific shape rule.
    empty_overlay_ids = _empty_target_overlay_ids(descriptor)
    preserve_without_map = (
        descriptor.shape_edit_mode in {"appearance_only", "preserve_silhouette"}
        and not _target_map_covers_declared_parts(descriptor, request.width, request.height)
    )
    if request.form in {AssetForm.ITEM, AssetForm.CROSS} and (preserve_without_map or empty_overlay_ids):
        descriptor = replace(
            descriptor,
            parts=[
                replace(part, paint_only=True)
                if is_overlay_part(part) and (preserve_without_map or part.id in empty_overlay_ids)
                else part
                for part in descriptor.parts
            ],
        )
    uv_regions = uv_layout_from_file(uv_layout_path) if uv_layout_path else []
    if request.form.value == "entity_uv" and not uv_regions:
        raise ValueError("entity_uv planning requires --uv-layout from the target model")
    if uv_regions:
        descriptor = _augment_descriptor_for_uv(descriptor, uv_regions)
        if request.form.value in {"entity_uv", "block_multi"}:
            uv_part_ids = {region.part_id for region in uv_regions}
            descriptor = replace(
                descriptor,
                parts=[
                    replace(
                        part,
                        # Paint-only parts have no physical UV owner, but an
                        # explicitly requested one is still required output.
                        required=(part.required and (part.paint_only or part.id in uv_part_ids)),
                        meaning=(
                            part.meaning
                            if part.id in uv_part_ids
                            else part.meaning + " (paint-only detail inside the supplied model UV)"
                        ),
                    )
                    for part in descriptor.parts
                ],
            )
    try:
        geometry = planner.geometry(
            descriptor,
            request.width,
            request.height,
            uv_regions=uv_regions,
            form=request.form,
            references=references,
        )
    except (KeyError, TypeError, ValueError, RuntimeError):
        # A provider can return a truncated JSON response after a long vision
        # correction pass. Keep the unattended pipeline reviewable by falling
        # back to the model-authored source/target ownership evidence when it
        # can be rasterized; this is not an object template and the outer
        # quality loop still reports the missing local motif for repair.
        geometry = reference_partition_geometry(
            descriptor, references, request.width, request.height, request.name
        )
        if geometry is None:
            raise
    if uv_regions:
        # UV regions are the model's authoritative atlas contract. If a
        # vision response forgot a declared region part, keep the stage flow
        # valid by adding a conservative base coat for that part; this does
        # not invent a silhouette and leaves the chosen atlas pixels intact.
        geometry_part_ids = {part.id for part in geometry.parts}
        descriptor_by_id = {part.id: part for part in descriptor.parts}
        missing_part_ids = [part_id for part_id in layout_summary(uv_regions)["parts"] if part_id not in geometry_part_ids]
        extra_parts = [descriptor_by_id[part_id] for part_id in missing_part_ids if part_id in descriptor_by_id]
        extra_primitives = [
            PrimitiveSpec(
                id="%s_uv_fill" % part_id,
                part_id=part_id,
                primitive="uv_fill",
                params={},
                layer=descriptor_by_id[part_id].layer,
            )
            for part_id in missing_part_ids
            if part_id in descriptor_by_id
        ]
        geometry = replace(
            geometry,
            parts=[*geometry.parts, *extra_parts],
            primitives=[*geometry.primitives, *extra_primitives],
            uv_regions=uv_regions,
        )
    if request.form.value in {"item", "cross"} and not uv_regions and max(request.width, request.height) <= 32:
        # Validate the model's raster before appearance planning. If its
        # required masks are empty/duplicated or its own constraints contradict
        # the pixels, keep the descriptor and references but use the open,
        # role-driven safety net instead of sending a broken silhouette into
        # the rest of the pipeline.
        try:
            candidate_compiled = compile_geometry(geometry)
            candidate_validation = validate_geometry(geometry, candidate_compiled, request.form)
        except (KeyError, TypeError, ValueError):
            candidate_validation = None
        if candidate_validation is not None and not candidate_validation.passed:
            # Keep every compilable model-authored draft, including one with a
            # missing/empty required part, so the pipeline's model repair and
            # blind-review stages can correct the actual response. Replacing a
            # draft here with a synthetic role mask hides the model's contour
            # and makes later repair learn from pixels the model never chose.
            pass
    if request.form.value in {"entity_uv", "block_multi"}:
        geometry = _normalize_entity_uv_geometry(geometry, descriptor, uv_regions)
    appearance = _normalize_paint_only_appearance(
        planner.appearance(descriptor, geometry, references=references), descriptor
    )
    if request.form.value in {"entity_uv", "block_multi"}:
        # A same-size vanilla atlas is a pixel-level model/texture reference,
        # so preserve its local value bands and markings while recolouring.
        # This keeps a cow recognisably cow-like without hard-coding cow paint
        # into the generic renderer.
        try:
            from PIL import Image

            same_size_reference = False
            for reference in references:
                with Image.open(reference.path) as image:
                    if image.size == (request.width, request.height):
                        same_size_reference = True
                        break
        except (OSError, ValueError):
            same_size_reference = False
        if same_size_reference and appearance.reference_sampling == "none":
            appearance = replace(appearance, reference_sampling="pattern")
        appearance = _normalize_entity_appearance(appearance, descriptor, geometry)
    return GenerationPlan(
        request=request,
        descriptor=descriptor,
        geometry=geometry,
        appearance=appearance,
        references=references,
    )
