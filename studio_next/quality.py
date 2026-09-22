"""Unattended model-driven generation and visual quality loop."""

from __future__ import annotations

import colorsys
import json
import math
import os
import re
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image

from .appearance import _resolve_colors
from .contracts import (
    ArtDirection,
    AssetForm,
    AssetRequest,
    ReferenceAsset,
    ReferenceRole,
    appearance_from_dict,
    geometry_from_dict,
    is_overlay_part,
    to_jsonable,
)
from .geometry import compile_geometry
from .group_index import GroupReferenceSource
from .llm import BlindReviewer, ModelPlanner, ModelSemanticCritic, OpenAICompatibleClient, TargetVisualReviewer
from .pipeline import (
    GenerationPipeline,
    GenerationPlan,
    _geometry_can_be_rendered,
    _reference_shape_is_locked,
    _target_overlay_descriptor,
    is_material_only_descriptor,
)
from .plans import (
    _normalize_entity_appearance,
    create_model_plan,
    reference_from_spec,
    uv_layout_from_file,
    vanilla_layout_catalog,
)
from .reference_geometry import (
    _reference_member_name,
    family_member_match,
)
from .reference_index import ReferenceIndex
from .reference_retrieval import build_router_manifest, retrieve_candidates
from .references import reference_from_png
from .validation import validate_geometry


# The one cap on routed evidence lives at the parse boundary (see
# parse_router_selection). A second positional cap downstream is what silently
# dropped the only wood textures on a live run, so the fallback recall window
# below uses this same number and nothing else truncates the router's answer.
_MAX_ROUTED_REFERENCES = 6


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _blind_alignment(plan: GenerationPlan, blind: dict[str, Any]) -> bool:
    """Use a soft noun overlap only as a stopping hint, never as a hard gate."""
    target_text = " ".join(
        [plan.request.query, plan.descriptor.target, plan.descriptor.semantic]
        + [part.meaning for part in plan.descriptor.parts]
    ).lower()
    tokens = {token for token in re.findall(r"[a-z][a-z0-9_-]{2,}", target_text)}
    # The planner is asked for an English target, but a model may still return
    # a Chinese noun in semantic fields. Add a small language bridge for common
    # Minecraft nouns so the blind gate can compare an English review without
    # exposing the target to that reviewer. This is alignment metadata only;
    # it never changes geometry or chooses a shape template.
    chinese_aliases = {
        "牛": {"cow", "bull", "animal", "mob"},
        "动物": {"animal", "mob", "creature"},
        "皮革": {"leather", "hide", "patch", "material"},
        "皮": {"leather", "hide", "patch"},
        "方块": {"block", "cube"},
        "村民": {"villager", "character", "humanoid", "person", "mob"},
        "蘑菇": {"mushroom", "plant"},
        "剑": {"sword", "blade", "weapon"},
        "刀": {"knife", "dagger", "blade"},
        "镐": {"pickaxe", "tool"},
        "斧": {"axe", "tool"},
        "铲": {"shovel", "tool"},
    }
    for source, aliases in chinese_aliases.items():
        if source in target_text:
            tokens.update(aliases)
    # The blind reviewer describes the render contract when the image is an
    # atlas or a neutral swatch rather than naming the user's language. Keep
    # this as form metadata only; it does not inspect the noun or select a
    # visual template.
    form_aliases = {
        AssetForm.BLOCK_MULTI: {"block", "cube"},
        AssetForm.ENTITY_UV: {"entity", "atlas", "model"},
        AssetForm.CROSS: {"cross", "billboard", "plant"},
        AssetForm.ITEM: {"item", "icon"},
    }
    tokens.update(form_aliases.get(plan.request.form, set()))
    primary = str(blind.get("primary_object", "")).lower().strip()
    if primary in {"", "unknown", "unrendered", "ambiguous"}:
        return False
    # Compare meaningful noun tokens instead of accepting any substring. A
    # phrase such as ``flint and steel`` otherwise aligns with a sword merely
    # because the descriptor contains the stopword ``and``. Form aliases are
    # already in ``tokens`` and still allow a reviewer to answer ``block`` or
    # ``entity`` for an atlas without exposing the user query.
    primary_tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", primary))
    stopwords = {
        "and", "the", "with", "from", "for", "like", "this", "that",
        "item", "object", "asset", "pixel", "art", "texture", "model",
    }
    meaningful_primary = primary_tokens - stopwords
    return bool(meaningful_primary & tokens)


def _target_review_can_accept_preserved_variant(
    plan: GenerationPlan,
    blind: dict[str, Any],
    target_visual_ok: bool,
) -> bool:
    """Accept a verified recolour/finish even when blind naming uses a vanilla noun.

    A target-free reviewer cannot be expected to coin a mod's new material
    name.  For example, a well-rendered blood gel may be described as a
    redstone-like or nether-wart-like blob.  That disagreement is useful
    diagnostic evidence, but it must not override a target-aware review when
    the planner explicitly retained a source silhouette.  New silhouettes and
    embedded features remain subject to the ordinary blind/topology checks.
    """
    if not target_visual_ok:
        return False
    if plan.descriptor.shape_edit_mode not in {"appearance_only", "preserve_silhouette"}:
        return False
    primary = str(blind.get("primary_object", "")).strip().lower()
    return primary not in {"", "unknown", "unrendered", "ambiguous"}


def _reference_variant_shape_gate(
    plan: GenerationPlan, comparison: dict[str, Any] | None
) -> bool:
    """Check a model-declared reference variant without naming an object class."""
    strategy = str(getattr(plan.descriptor, "reference_strategy", "")).lower()
    variant_words = (
        "variant", "preserve", "keep", "unchanged", "recolor", "recolour",
        "damage", "broken", "chipped",
    )
    same_size_shape = any(
        ReferenceRole.SHAPE in reference.roles
        and int(reference.features.get("width") or 0) == plan.request.width
        and int(reference.features.get("height") or 0) == plan.request.height
        for reference in plan.references
    )
    if not same_size_shape or not any(word in strategy for word in variant_words):
        return True
    if not isinstance(comparison, dict):
        return False
    shape = comparison.get("shape")
    if not isinstance(shape, dict):
        return False
    try:
        iou = float(shape.get("iou"))
    except (TypeError, ValueError):
        return False
    # A local variant may alter its named feature, but a broad contour rewrite
    # is evidence that unnamed supports were not preserved.
    return iou >= 0.60


def _embedded_part_ids(plan: GenerationPlan) -> set[str]:
    """Find model-declared local motifs for repair non-regression checks.

    This reads only the descriptor's own language. It does not classify an
    object or prescribe a minimum shape; it protects a part that the model
    itself described as embedded/inlaid/detail while a later repair is trying
    to make that part clearer.
    """
    descriptor_text = " ".join(
        [plan.descriptor.semantic, plan.descriptor.reference_strategy, *plan.descriptor.visual_identity]
    ).lower()
    if not any(word in descriptor_text for word in (
        "embedded", "inlaid", "inlay", "inserted", "attached", "mounted",
        "encased", "grafted", "set into", "镶嵌", "嵌入",
    )):
        return set()
    result: set[str] = set()
    for part in plan.descriptor.parts:
        if is_overlay_part(part):
            result.add(part.id)
    return result


def _embedded_feature_visible(plan: GenerationPlan, blind: dict[str, Any]) -> bool:
    """Require a target-free review to notice a named embedded feature."""
    motif_ids = _embedded_part_ids(plan)
    if not motif_ids:
        return True
    observed = " ".join(
        [
            str(blind.get("primary_object", "")),
            " ".join(str(item) for item in blind.get("alternatives", []) if item),
            " ".join(str(item) for item in blind.get("evidence", []) if item),
            " ".join(str(item) for item in blind.get("visible_parts", []) if item),
        ]
    ).lower()
    generic = {
        "embedded", "inlaid", "inlay", "inserted", "attached", "mounted",
        "encased", "grafted", "set", "into", "part", "motif", "detail",
        "accent", "ornament", "the", "and", "with",
    }
    for part in plan.descriptor.parts:
        if part.id not in motif_ids:
            continue
        tokens = [
            token
            for term in part.recognition_terms
            for token in re.findall(r"[a-z][a-z0-9_-]{2,}", str(term).lower())
        ]
        if not tokens:
            # Old saved descriptors did not contain inspection vocabulary.
            # Their identifier is less ambiguous than prose that also names
            # the host support (for example an inset "in the blade").
            tokens = re.findall(r"[a-z][a-z0-9_-]{2,}", part.id.lower())
        meaningful = [token for token in tokens if token not in generic]
        aliases = set(meaningful)
        if "eyeball" in aliases:
            aliases.add("eye")
        if "crossguard" in aliases:
            aliases.add("guard")
        if meaningful and any(token in observed for token in aliases):
            continue
        return False
    return True


def _motif_area_non_regression(
    before: Any, candidate: Any, motif_part_ids: set[str], minimum_ratio: float = 0.90
) -> tuple[bool, dict[str, tuple[int, int]]]:
    """Reject a repair that shrinks a named embedded part while fixing it."""
    if not motif_part_ids:
        return True, {}
    try:
        before_compiled = compile_geometry(before)
        candidate_compiled = compile_geometry(candidate)
    except (KeyError, TypeError, ValueError):
        return True, {}
    changes: dict[str, tuple[int, int]] = {}
    for part_id in motif_part_ids:
        before_mask = before_compiled.part_masks.get(part_id)
        candidate_mask = candidate_compiled.part_masks.get(part_id)
        if before_mask is None or candidate_mask is None:
            continue
        before_area = sum(1 for pixel in before_mask.getdata() if pixel >= 8)
        candidate_area = sum(1 for pixel in candidate_mask.getdata() if pixel >= 8)
        changes[part_id] = (before_area, candidate_area)
        if before_area > 0 and candidate_area < max(1, math.ceil(before_area * minimum_ratio)):
            return False, changes
    return True, changes


def _round_quality_score(entry: dict[str, Any]) -> float:
    """Rank retained artifacts without inventing a target-specific success rule."""
    validation = entry.get("validation") if isinstance(entry.get("validation"), dict) else {}
    blind = entry.get("blind_review") if isinstance(entry.get("blind_review"), dict) else {}
    score = 0.0
    if entry.get("sprite"):
        score += 10.0
    if bool(validation.get("passed")):
        score += 6.0
    else:
        # A rendered but semantically rejected round is useful feedback for
        # the next model pass, but it must not outrank a contract-valid round
        # merely because its alpha happened to overlap the reference more.
        # Keep the failed artifact in the report while making selection prefer
        # a round that actually passed its full validation.
        score -= 6.0
    if bool(entry.get("blind_alignment_hint")):
        score += 4.0
    else:
        score -= 2.0
    if bool(entry.get("reference_shape_gate")):
        score += 2.0
    try:
        score += min(1.0, max(0.0, float(blind.get("confidence", 0.0))))
    except (TypeError, ValueError):
        pass
    comparison = entry.get("reference_comparison")
    if isinstance(comparison, dict) and isinstance(comparison.get("shape"), dict):
        try:
            score += 2.0 * min(1.0, max(0.0, float(comparison["shape"].get("iou", 0.0))))
        except (TypeError, ValueError):
            pass
    return score


def _normalize_routed_form(value: object) -> AssetForm:
    """Accept a few ordinary model aliases while keeping one internal enum."""
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "entity": AssetForm.ENTITY_UV,
        "entity_texture": AssetForm.ENTITY_UV,
        "mob": AssetForm.ENTITY_UV,
        "creature": AssetForm.ENTITY_UV,
        "block": AssetForm.BLOCK_MULTI,
        "cube": AssetForm.BLOCK_MULTI,
        "billboard": AssetForm.CROSS,
        "plant": AssetForm.CROSS,
    }
    if text in aliases:
        return aliases[text]
    try:
        form = AssetForm(text)
    except ValueError:
        return AssetForm.ITEM
    return AssetForm.ITEM if form == AssetForm.AUTO else form


def _layout_extent(path: str | Path) -> tuple[int, int]:
    """Find the required canvas extent of a compact or explicit UV layout."""
    from .uv_layout import layout_canvas_extent

    return layout_canvas_extent(json.loads(Path(path).read_text(encoding="utf-8")))


def _layout_matches_form(path: str | Path, form: AssetForm) -> bool:
    """Check a model-selected layout against the requested generic contract."""
    try:
        regions = uv_layout_from_file(path)
    except (OSError, ValueError, TypeError):
        return False
    if form == AssetForm.BLOCK_MULTI:
        # A block atlas needs three visible cube planes. The names are face
        # roles, not object templates, so custom layouts may use ``right`` or
        # ``east`` for the third plane.
        faces = {region.face.strip().lower() for region in regions}
        has_top = "top" in faces or "up" in faces
        has_front = bool(faces & {"front", "north", "south"})
        has_side = bool(faces & {"right", "east", "left", "west", "side"})
        return has_top and has_front and has_side and len(regions) <= 12
    if form == AssetForm.ENTITY_UV:
        # Entity/model layouts carry preview placements for at least one cube;
        # a flat three-face block strip must never be selected for this form.
        return len(regions) >= 6 and any(
            region.preview_instances is not None or region.preview_origin is not None
            for region in regions
        )
    return True


def _safe_routed_target_path(value: object) -> str | None:
    """Accept a model-proposed resource path only when it is pack-relative."""
    if not isinstance(value, str):
        return None
    candidate = value.strip().replace("\\", "/")
    if not candidate or candidate.startswith("/") or ":" in candidate:
        return None
    pieces = [piece for piece in candidate.split("/") if piece]
    if not pieces or any(piece in {".", ".."} for piece in pieces):
        return None
    if pieces[-1].lower().endswith(".png") is False:
        return None
    return "/".join(pieces)


def resolve_reference_index_path(explicit: str | Path | None = None) -> Path | None:
    """Resolve an explicitly configured or project-local active index."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None
    configured = os.environ.get("MC_ART_REFERENCE_INDEX", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.exists() else None
    default = Path(__file__).resolve().parents[1] / "references" / "index.json"
    return default if default.exists() else None


def load_active_index(path: str | Path | None = None) -> ReferenceIndex | None:
    """Load an index only when its source still has the indexed fingerprint."""
    index_path = resolve_reference_index_path(path)
    if index_path is None:
        return None
    try:
        index = ReferenceIndex.load(index_path)
        from .reference_index import fingerprint_source
        if fingerprint_source(index.source.path).fingerprint != index.source.fingerprint:
            return None
        return index
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _resolve_auto_request(
    request: AssetRequest,
    explicit_references: list[str],
    planner: ModelPlanner,
    supplied_uv_layout: str | Path | None = None,
    reference_index_path: str | Path | None = None,
    router_model: str | None = None,
    art_direction: ArtDirection | None = None,
    asset_sources: list[str] | None = None,
    cache_root: str | Path | None = None,
    recall_limit: int | None = None,
) -> tuple[AssetRequest, list[str], str | Path | None, dict[str, Any], list[ReferenceAsset] | None]:
    """Let the model route a bare query to a concrete form and evidence set.

    Two evidence backends are supported. The historical one reads a persisted
    reference index. The live one resolves logical asset names from the given
    roots on every run, so a mod's own art is always the style anchor and the
    committed JSON index can be retired.
    """
    if request.form != AssetForm.AUTO:
        resolved = request
        if supplied_uv_layout is not None and request.form in {AssetForm.ENTITY_UV, AssetForm.BLOCK_MULTI}:
            # An explicit form plus an explicit layout must still get a canvas
            # that contains that layout, whether the layout is a shipped vanilla
            # model or one derived from the object's own box decomposition.
            layout_width, layout_height = _layout_extent(supplied_uv_layout)
            resolved = replace(
                request,
                width=max(request.width, layout_width),
                height=max(request.height, layout_height),
            )
        return resolved, explicit_references, supplied_uv_layout, {
            "mode": "explicit",
            "form": resolved.form.value,
            "resolved_dimensions": [resolved.width, resolved.height],
            "uv_layout": str(supplied_uv_layout) if supplied_uv_layout else None,
            "reference_indices": [],
            "uv_layout_index": None,
        }, None
    layouts = vanilla_layout_catalog()
    live_source: GroupReferenceSource | None = None
    if asset_sources and not explicit_references:
        live_source = GroupReferenceSource.from_sources(
            list(asset_sources),
            cache_root or (Path("references") / ".cache" / "live"),
        )
        active_index = live_source
    else:
        active_index = None if explicit_references else load_active_index(reference_index_path)
    if active_index is None and not explicit_references:
        raise RuntimeError(
            "no reference evidence: pass --source <jar or resources directory> or run index-vanilla first"
        )
    # The index is already a compact name/category catalogue, so let the
    # router see every entry from the current source.  Recall still sorts
    # lexical matches first; the model can therefore use the whole 1.12 pack
    # without confusing a top-N window for the complete catalogue.
    indexed_candidates = (
        retrieve_candidates(request.query, active_index, limit=recall_limit)
        if active_index else []
    )
    indexed_manifest = build_router_manifest(indexed_candidates)
    indexed_route: dict[str, Any] | None = None
    if active_index:
        try:
            indexed_route = planner.route_indexed(
                request.query,
                indexed_manifest,
                layouts,
                cache_dir=active_index.source_cache_dir,
                index_fingerprint=active_index.source.fingerprint,
                model_name=router_model,
                art_direction=art_direction,
            )
        except AssertionError:
            # A few third-party callers still provide a legacy test/client
            # double whose ``complete`` method insists on attached images.
            # Keep that old route contract usable while the real indexed
            # client remains image-free.
            indexed_route = None
        # The old route schema used numeric ``reference_indices``.  Treat it
        # as an explicit compatibility signal instead of interpreting those
        # positions against the new global index.
        if indexed_route is not None and indexed_route.get("_legacy_numeric_route"):
            indexed_route = None
    if indexed_route is not None:
        route = indexed_route
        selected_ids = [str(item) for item in route.get("asset_ids", []) if isinstance(item, str)]
        selected_entries = [entry for asset_id in selected_ids for entry in active_index.entries if entry.asset_id == asset_id]
        if not selected_entries:
            selected_entries = [candidate.entry for candidate in indexed_candidates[:_MAX_ROUTED_REFERENCES]]
        selected_roles: list[list[str]] = []
        raw_selections = route.get("selections", [])
        role_map = {
            str(item.get("asset_id")): [str(role) for role in item.get("roles", [])]
            for item in raw_selections if isinstance(item, dict) and item.get("asset_id")
        }
        selected_specs = []
        display_name_by_id = {
            str(item.get("asset_id")): str(item.get("name", ""))
            for item in indexed_manifest
            if item.get("asset_id") and item.get("name")
        }
        catalog: list[ReferenceAsset] = []
        entry_rows: list[dict[str, Any]] = []
        # No second cap here. The router's own response is already bounded by
        # parse_router_selection, so slicing again only ever removed evidence:
        # structural candidates come first in the model's answer and material or
        # palette rasters come last, so a positional cut systematically dropped
        # exactly the references the appearance stage needs.
        for entry in selected_entries:
            roles = role_map.get(entry.asset_id) or [ReferenceRole.PIXEL_STYLE.value, ReferenceRole.MATERIAL.value]
            display = display_name_by_id.get(entry.asset_id, "")
            if live_source is not None:
                # One logical name can own several textures. Expand the whole
                # family so the planners see every face or frame together,
                # instead of one arbitrary member standing in for the rest.
                expanded = live_source.planning_assets(
                    entry,
                    [ReferenceRole(role) for role in roles],
                    display_name=display or None,
                    preferred_member=getattr(request, "name", None),
                    max_frames=_MAX_GROUP_FRAMES,
                )
            else:
                parsed = reference_from_spec(
                    str(active_index.materialize(entry)) + "::" + ",".join(dict.fromkeys(roles))
                )
                # Keep the human/model-facing display name after materialization;
                # the blob filename is content-addressed and must never become
                # the semantic name in prompts or persisted plans.
                expanded = [replace(parsed, name=display) if display else parsed]
            # A code-driven family arrives as several states of one object, and
            # the router marks every one of them roles=shape. The planner then
            # copied whichever frame was listed first: a standby bow came back
            # with the half-drawn frame's silhouette. The pipeline already knows
            # which member answers this request, so say it instead of leaving
            # four equal authorities in the prompt.
            authority = _shape_authority(expanded, getattr(request, "name", None))
            for asset in expanded:
                asset_roles = list(roles)
                if authority is not None:
                    notes = list(asset.notes)
                    if asset is authority:
                        notes.append("shape_authority=this request; copy this silhouette")
                    else:
                        asset_roles = [role for role in asset_roles if role != ReferenceRole.SHAPE.value]
                        asset_roles = asset_roles or [ReferenceRole.PIXEL_STYLE.value]
                        notes.append(
                            "shape_authority=context only; another state of the same "
                            "object, not this request's silhouette"
                        )
                    asset = replace(asset, roles=[ReferenceRole(role) for role in asset_roles], notes=notes)
                selected_roles.append(asset_roles)
                catalog.append(asset)
            entry_rows.append({
                "asset_id": entry.asset_id,
                "roles": roles,
                "reference_count": len(expanded),
                "references": [asset.name for asset in expanded],
            })
        selected_indices = list(range(len(catalog)))
        routing_index = {
            "path": "" if live_source is not None else str(resolve_reference_index_path(reference_index_path) or ""),
            "source_fingerprint": active_index.source.fingerprint,
            "entry_count": len(active_index.entries),
            "cache_hit": active_index.cache_stats.get("entries_reused", 0) > 0,
        }
    else:
        # Explicit references remain available to callers embedding the
        # pipeline, but the public CLI always uses the full local index.
        active_index = None
        catalog = [reference_from_spec(item) for item in explicit_references]
        route = planner.route(request.query, catalog, layouts, art_direction=art_direction)
        selected_roles = []
        selected_indices = []
        routing_index = {"status": "missing" if not explicit_references else "explicit"}
    form = _normalize_routed_form(route.get("form"))

    def _indices(key: str) -> list[int]:
        value = route.get(key, [])
        if isinstance(value, int):
            value = [value]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, int) and 0 <= item < len(catalog)]

    if not active_index:
        selected_indices = _indices("reference_indices")[:_MAX_ROUTED_REFERENCES]
    selected = [catalog[index] for index in selected_indices]
    if form in {AssetForm.ITEM, AssetForm.CROSS} and selected and selected[0].features.get("width") != 16:
        item_candidates = [
            (index, item) for index, item in enumerate(catalog)
            if item.features.get("width") == 16 and item.features.get("height") == 16
        ]
        if item_candidates and not any(index in selected_indices for index, _ in item_candidates):
            selected_indices = [item_candidates[0][0]] + selected_indices[:3]
            selected = [catalog[index] for index in selected_indices]

    layout_index = route.get("uv_layout_index")
    if not isinstance(layout_index, int) or not 0 <= layout_index < len(layouts):
        layout_index = None
    layout_correction: dict[str, Any] | None = None
    if form in {AssetForm.ENTITY_UV, AssetForm.BLOCK_MULTI}:
        if layout_index is not None and not _layout_matches_form(layouts[layout_index][0], form):
            layout_correction = {
                "model_index": layout_index,
                "model_path": layouts[layout_index][0],
                "reason": "selected layout does not satisfy the requested form contract",
            }
            layout_index = None
        if layout_index is None:
            candidates = [
                (index, path) for index, (path, _summary) in enumerate(layouts)
                if _layout_matches_form(path, form)
            ]
            if candidates:
                # Keep the model's reference choice, but use the first
                # contract-compatible local layout when its index was invalid.
                layout_index, _path = candidates[0]
                if layout_correction is not None:
                    layout_correction["replacement_index"] = layout_index
    uv_layout: str | Path | None = supplied_uv_layout
    if form in {AssetForm.ENTITY_UV, AssetForm.BLOCK_MULTI} and uv_layout is None:
        if layout_index is not None:
            uv_layout = layouts[layout_index][0]
        else:
            # A generic filename match covers any local model/layout pair and
            # does not encode a query-specific shape template in the router.
            names = " ".join(item.name.lower() for item in selected)
            for path, _summary in layouts:
                stem = Path(path).stem.lower()
                tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]
                if any(token in names for token in tokens if len(token) >= 4):
                    uv_layout = path
                    break
        if uv_layout is None:
            raise ValueError(
                "automatic entity routing found no matching UV layout; add a model layout under layouts/"
            )

    width = route.get("width")
    height = route.get("height")
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        width = request.width
        height = request.height
    width = max(1, min(width, 512))
    height = max(1, min(height, 512))
    if uv_layout is not None:
        layout_width, layout_height = _layout_extent(uv_layout)
        width = max(width, layout_width)
        height = max(height, layout_height)
    elif selected:
        # Keep dimensions tied to the selected source for atlas recolors; the
        # model can still choose a different canvas when no source is relevant.
        if form == AssetForm.ENTITY_UV:
            width = int(selected[0].features.get("width") or width)
            height = int(selected[0].features.get("height") or height)
    # An atlas must fit its model, so entity_uv defaults to the source model's
    # own contract. "free" is the caller's way to say this is a new design for
    # that model, in which case the model's paint plan outranks the reference
    # silhouette instead of being refined by it.
    shape_policy = request.shape_policy
    if form == AssetForm.ENTITY_UV and shape_policy != "free":
        shape_policy = "target_model_uv"
    target_path = request.target_path or _safe_routed_target_path(route.get("target_path"))
    resolved = AssetRequest(
        query=request.query,
        form=form,
        name=request.name,
        namespace=request.namespace,
        width=width,
        height=height,
        novelty=request.novelty,
        target_path=target_path,
        shape_policy=shape_policy,
        seed=request.seed,
        pack_format=request.pack_format,
    )
    routing = {
        "mode": "model_auto",
        "query": request.query,
        "decision": route,
        "resolved_form": form.value,
        "resolved_dimensions": [width, height],
        "selected_references": [catalog[index].name for index in selected_indices],
        "selected_reference_paths": [catalog[index].path for index in selected_indices],
        "selected_reference_roles": [
            [role.value for role in catalog[index].roles] for index in selected_indices
        ],
        "selected_uv_layout": str(uv_layout) if uv_layout else None,
    }
    if active_index:
        routing["index"] = routing_index
        routing["local_recall"] = indexed_manifest
        routing["router"] = {
            "method": "llm",
            "model": router_model or getattr(planner.client, "model", None),
            "cache_hit": bool(route.get("cache_hit")),
            "selected_ids": [entry.asset_id for entry in selected_entries],
        }
        routing["selected"] = entry_rows
        routing["attached_images"] = [str(catalog[index].path) for index in selected_indices]
        if live_source is not None:
            routing["live_source"] = {
                "roots": [str(root.path) for root in live_source.catalogue.roots],
                "fingerprint": live_source.fingerprint,
                "groups": live_source.catalogue.stats.get("groups", {}),
                "expanded_references": len(catalog),
                "text_cache": live_source.text_cache.stats(),
            }
    if layout_correction is not None:
        routing["layout_correction"] = layout_correction
    # A live source already produced cache-backed reference objects. Returning
    # them lets the quality loop skip the path round trip that would re-decode
    # every image and defeat the content-addressed text cache.
    planning_assets = catalog if live_source is not None else None
    return resolved, [item.path for item in selected], uv_layout, routing, planning_assets


def _planning_reference_assets(
    reference_specs: list[str], routing: dict[str, Any] | None = None,
) -> list[ReferenceAsset]:
    """Materialize planning references without losing router semantics.

    The indexed router works with friendly manifest names and explicit roles,
    while the renderer needs local paths.  Reconstructing each object from a
    blob path alone would replace both with a content hash and the default
    ``pixel_style/material`` roles.  Keep the path parser as the compatibility
    boundary, then overlay only the model-selected name/role metadata.
    """
    assets = [reference_from_spec(spec) for spec in reference_specs]
    routing = routing or {}
    names = routing.get("selected_references")
    role_rows = routing.get("selected_reference_roles")
    decision = routing.get("decision")
    decision_selections = decision.get("selections") if isinstance(decision, dict) else []
    if not isinstance(names, list):
        names = []
    if not isinstance(role_rows, list):
        role_rows = []
    normalized: list[ReferenceAsset] = []
    for index, asset in enumerate(assets):
        replacement: dict[str, Any] = {}
        if index < len(names) and str(names[index]).strip():
            replacement["name"] = str(names[index])
        if index < len(role_rows) and isinstance(role_rows[index], list):
            roles: list[ReferenceRole] = []
            for raw_role in role_rows[index]:
                try:
                    role = ReferenceRole(str(raw_role))
                except ValueError:
                    continue
                if role not in roles:
                    roles.append(role)
            if roles:
                replacement["roles"] = roles
        if index < len(decision_selections) and isinstance(decision_selections[index], dict):
            reason = str(decision_selections[index].get("reason", "")).strip()
            if reason:
                replacement["notes"] = [reason]
        normalized.append(replace(asset, **replacement) if replacement else asset)
    return normalized


def _reference_texture_gate(comparison: dict[str, Any] | None) -> bool:
    """Return whether a descriptive texture comparison is good enough to stop.

    This is a soft quality hint rather than a pixel-equality requirement:
    requested recolours and motifs are allowed to change colour.  It only
    reopens the appearance stage when a pattern-mode atlas is clearly losing
    the reference's local value/edge rhythm, so the model can inspect the
    current render and decide how to recover it.
    """
    if not isinstance(comparison, dict):
        return True
    assessment = comparison.get("assessment")
    if isinstance(assessment, dict) and str(assessment.get("texture", "")).lower() == "diverged":
        return False
    texture = comparison.get("texture")
    if not isinstance(texture, dict):
        return True
    try:
        correlation = float(texture.get("luma_correlation"))
    except (TypeError, ValueError):
        correlation = None
    try:
        edge_agreement = float(texture.get("edge_agreement"))
    except (TypeError, ValueError):
        edge_agreement = None
    if correlation is not None and edge_agreement is not None:
        return correlation >= 0.88 or edge_agreement >= 0.92
    return True


def run_quality_loop(
    request: AssetRequest | str,
    references: list[str] | None = None,
    out_dir: str | Path | None = None,
    rounds: int = 3,
    max_repairs: int = 1,
    package: bool = False,
    uv_layout_path: str | Path | None = None,
    reference_index_path: str | Path | None = None,
    router_model: str | None = None,
    asset_sources: list[str] | None = None,
    cache_root: str | Path | None = None,
    recall_limit: int | None = None,
    style_anchor: str | Path | None = None,
    anchor_palette: list[tuple[int, int, int]] | None = None,
    shape_mode_by_group: dict[str, str] | None = None,
    anchor_paint: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Run planning, deterministic checks, blind review and geometry revision.

    Every round is retained under ``round_00``, ``round_01`` ... so a caller
    can inspect exactly which stage changed. The loop is intentionally generic:
    it feeds the blind noun and evidence back into the open geometry planner,
    never into a target-specific template or a color post-process.
    """
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if out_dir is None:
        raise ValueError("out_dir is required")
    if isinstance(request, str):
        request = AssetRequest(query=request, form=AssetForm.AUTO)
    input_request = request
    references = references or []
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient.from_env(trace_dir=root / "llm_raw")
    planner = ModelPlanner(client)
    blind_reviewer = BlindReviewer(client)
    target_visual_reviewer = TargetVisualReviewer(client)
    # This is the deliberate first planning boundary. It receives only the
    # user's request, then remains immutable while retrieval contributes
    # vanilla evidence for form, material and pixel language.
    art_direction = planner.art_direct(request.query)
    request, references, uv_layout_path, routing, cached_planning_assets = _resolve_auto_request(
        request,
        references,
        planner,
        supplied_uv_layout=uv_layout_path,
        reference_index_path=reference_index_path,
        router_model=router_model,
        art_direction=art_direction,
        asset_sources=asset_sources,
        cache_root=cache_root,
        recall_limit=recall_limit,
    )
    routing["art_direction"] = to_jsonable(art_direction)
    _write_json(root / "routing.json", routing)
    _write_json(root / "art_direction.json", art_direction)
    planning_references = references
    if (
        routing.get("mode") == "model_auto"
        and references
    ):
        # Keep the model-selected roles for every routed form.  For a small
        # icon, only the first source receives the compatibility shape role;
        # later selections remain material/style evidence so an unrelated
        # swatch cannot contribute a second alpha contour to geometry.
        route_roles = routing.get("selected_reference_roles")
        if not isinstance(route_roles, list):
            route_roles = []
        routed_specs: list[str] = []
        for index, path in enumerate(references):
            raw_roles = route_roles[index] if index < len(route_roles) else []
            if not isinstance(raw_roles, list):
                raw_roles = []
            roles = [str(role) for role in raw_roles if str(role)]
            if request.form in {AssetForm.ITEM, AssetForm.CROSS} and index == 0 and "shape" not in roles:
                roles.append("shape")
            if not roles:
                roles = ["pixel_style", "material", "palette"]
            routed_specs.append(str(path) + "::" + ",".join(dict.fromkeys(roles)))
        planning_references = routed_specs
    # Keep the router's semantic names and role assignments attached to the
    # images that enter every planning stage.  Re-parsing a materialized blob
    # path would otherwise expose only its hash filename (and default roles),
    # making it easy for the vision model to confuse a cow atlas with a magma
    # palette reference even though both images are attached.
    # A live source already handed back cache-backed reference objects, so the
    # path round trip (which would re-decode every selected image) is skipped.
    planning_assets = (
        cached_planning_assets
        if cached_planning_assets is not None
        else _planning_reference_assets(planning_references, routing)
    )
    if style_anchor is not None:
        # A sibling generated earlier in the same family run. It is pinned as
        # palette and pixel-style evidence only: it must not redefine this
        # member's subject or silhouette, so it never receives a shape role.
        anchor_path = Path(style_anchor)
        if anchor_path.exists():
            planning_assets = list(planning_assets) + [
                reference_from_png(
                    anchor_path,
                    roles=[ReferenceRole.PIXEL_STYLE, ReferenceRole.PALETTE],
                    notes=["family style anchor: keep this palette and pixel language"],
                )
            ]
            routing["style_anchor"] = str(anchor_path)
    plan = create_model_plan(
        request,
        planning_references,
        planner=planner,
        uv_layout_path=uv_layout_path,
        reference_assets=planning_assets,
        art_direction=art_direction,
    )
    if shape_mode_by_group:
        # Frames of one object must share one shape relation: the planner
        # freely wrote appearance_only, local_silhouette_edit and
        # preserve_silhouette across four frames of the same bow, and the two
        # loose readings shipped a hand-drawn contour while their siblings
        # kept the source one. The anchor's reading of the object governs.
        for reference in plan.references:
            group = _reference_group(reference)
            inherited = shape_mode_by_group.get(group) if group else None
            if inherited:
                descriptor = replace(plan.descriptor, shape_edit_mode=inherited)
                direction = descriptor.art_direction
                if direction is not None and direction.requires_silhouette_change:
                    # Within one object's frame set, "does the contour change?"
                    # is an object-level question. A draw-state frame says yes
                    # because the limbs bend and the string moves, but its own
                    # source frame already shows that bend: the reference-free
                    # brief was written against the rest state, while this
                    # member is conforming to the matching source frame.
                    descriptor = replace(
                        descriptor,
                        art_direction=replace(direction, requires_silhouette_change=False),
                    )
                    routing["family_contour_change_cleared"] = True
                plan = replace(plan, descriptor=descriptor)
                routing["inherited_shape_edit_mode"] = inherited
                routing["inherited_shape_group"] = group
                break
    # The effective contour decision, not the label the model happened to
    # choose: a family propagates this, and two runs of the same recolour
    # wrote appearance_only and local_silhouette_edit for identical intent.
    routing["shape_contour_locked"] = _reference_shape_is_locked(plan.descriptor)
    if anchor_palette:
        # The named swatches stay the model's; their values become the family's.
        plan = replace(
            plan,
            appearance=remap_appearance_to_anchor(plan.appearance, anchor_palette),
        )
        routing["anchor_palette"] = ["#%02X%02X%02X" % colour for colour in anchor_palette]
    if anchor_paint is not None:
        # Colour values alone did not hold the set together: each frame still
        # authored its own shading, so one bow resolved into four materials.
        # A part that sits on the same pixels as an anchor part takes the
        # anchor's whole paint specification instead.
        inherited_appearance, inherited = inherit_anchor_paint(
            plan.appearance,
            plan.geometry,
            anchor_paint[0],
            anchor_paint[1],
        )
        if inherited:
            plan = replace(plan, appearance=inherited_appearance)
            routing["inherited_anchor_paint"] = inherited
    report: dict[str, Any] = {
        "input_request": to_jsonable(input_request),
        "request": to_jsonable(request),
        "routing": routing,
        "art_direction": to_jsonable(art_direction),
        "rounds_requested": rounds,
        "max_geometry_repairs": max_repairs,
        "rounds": [],
    }
    material_only = is_material_only_descriptor(plan.descriptor)
    for index in range(rounds):
        round_dir = root / ("round_%02d" % index)
        _write_json(round_dir / "plan.json", plan)
        pipeline = GenerationPipeline(
            critic=ModelSemanticCritic(client),
            repairer=planner,
            max_geometry_repairs=max_repairs,
        )
        result = pipeline.run(plan, round_dir / "generated", package=package)
        blind: dict[str, Any] | None = None
        blind_image = result.sprite_path
        if request.form == AssetForm.BLOCK_MULTI:
            # A block texture is a flat atlas by design, so a blind model
            # looking at the 16x16 source only sees a texture swatch. Review
            # the generated isometric block preview instead; keep the atlas
            # as the actual artifact and validation source.
            block_preview = round_dir / "generated" / "isometric_preview.png"
            if block_preview.exists():
                blind_image = block_preview
        elif request.form == AssetForm.ENTITY_UV:
            # Review the transparent side projection itself. A checkerboard is
            # useful for a human inspecting dark pixels, but its alternating
            # high-contrast tiles become extra rectangular masses to a blind
            # vision model; the same cow that is recognised on the transparent
            # preview can be misread as an axe on that background. Keep the
            # checker and isometric views as artifacts for manual inspection.
            entity_preview = round_dir / "generated" / "front_preview.png"
            if not entity_preview.exists():
                entity_preview = round_dir / "generated" / "front_preview_checker.png"
            if not entity_preview.exists():
                entity_preview = round_dir / "generated" / "entity_preview.png"
            if entity_preview.exists():
                blind_image = entity_preview
        if blind_image is not None:
            blind = blind_reviewer.review(str(blind_image))
            _write_json(round_dir / "blind_review.json", blind)
        else:
            blind = {
                "primary_object": "unrendered",
                "alternatives": [],
                "confidence": 0.0,
                "evidence": list(result.validation.errors),
                "orientation": "unknown",
                "visible_parts": [],
            }
            _write_json(round_dir / "blind_review.json", blind)
        target_visual_review: dict[str, Any] | None = None
        target_visual_ok = True
        if blind_image is not None:
            try:
                # The blind reviewer needs an isometric block preview to name
                # the object. The target-aware reviewer instead judges the
                # texture contract itself, so it must see the native atlas:
                # perspective is a diagnostic presentation, not a requested
                # property of a block's 2-D texture.
                target_image = result.sprite_path if request.form == AssetForm.BLOCK_MULTI else blind_image
                target_visual_review = target_visual_reviewer.review(plan.descriptor, str(target_image))
                target_visual_ok = bool(target_visual_review.get("passed", False))
                _write_json(round_dir / "target_visual_review.json", target_visual_review)
            except Exception as exc:  # visual audit is additive; retain a rendered round on API failure
                target_visual_review = {"error": str(exc)}
                _write_json(round_dir / "target_visual_review.json", target_visual_review)
        # Keep the noun review independent: only after it is written do we
        # load the actual vanilla evidence and create a pixel-level diagnosis.
        # This lets a human (or a later planner pass) see whether a miss came
        # from contour pixels, colour changes, or lost local texture without
        # contaminating the blind reviewer's judgement.
        reference_comparison: dict[str, object] | None = None
        if result.sprite_path is not None:
            try:
                reference_comparison = GenerationPipeline.reference_comparison(
                    round_dir / "generated",
                    result.sprite_path,
                    plan.references,
                )
            except (OSError, ValueError, TypeError) as exc:
                # A malformed optional evidence file must not discard a valid
                # render. Keep the failure visible in the round report.
                reference_comparison = {"error": str(exc)}
        aligned = _blind_alignment(plan, blind)
        blind_alignment_reason = "noun_overlap" if aligned else "noun_mismatch"
        if not target_visual_ok:
            aligned = False
            blind_alignment_reason = "target_visual_contract_failed"
        elif _target_review_can_accept_preserved_variant(plan, blind, target_visual_ok):
            # Keep the blind noun in the report as a useful independent
            # description, but let the target-aware visual contract decide
            # whether a deliberately recoloured/finished variant succeeded.
            aligned = True
            blind_alignment_reason = "target_verified_preserved_variant"
        if aligned and not _embedded_feature_visible(plan, blind):
            aligned = False
            blind_alignment_reason = "named_embedded_feature_not_visible_to_target_free_review"
        if material_only and result.validation.passed and target_visual_ok:
            # A blind noun such as "chestplate" or "turtle" is not a useful
            # identity test for a material swatch; even the vanilla leather
            # reference receives those labels. Keep the blind result in the
            # report, but stop on the deterministic contour/material gate and
            # avoid feeding the noun back into geometry repairs.
            aligned = True
        reference_shape_ok = _reference_variant_shape_gate(plan, reference_comparison)
        reference_texture_ok = _reference_texture_gate(reference_comparison)
        uv_layout_diagnostic: dict[str, object] | None = None
        if request.form == AssetForm.ENTITY_UV:
            diagnostic_path = round_dir / "generated" / "alpha" / "uv_coverage_vs_reference_alpha.json"
            if diagnostic_path.exists():
                try:
                    loaded_diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_diagnostic, dict):
                        uv_layout_diagnostic = loaded_diagnostic
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    # The rendered atlas remains reviewable if an optional
                    # diagnostic was interrupted; surface no invented score.
                    uv_layout_diagnostic = {"error": "could not read UV coverage diagnostic"}
        round_entry = {
            "index": index,
            "directory": str(round_dir),
            "validation": to_jsonable(result.validation),
            "sprite": str(result.sprite_path) if result.sprite_path else None,
            "blind_image": str(blind_image) if blind_image else None,
            "blind_review": blind,
            "blind_alignment_hint": aligned,
            "target_visual_review": target_visual_review,
            "target_visual_gate": target_visual_ok,
            "reference_shape_gate": reference_shape_ok,
            "reference_texture_gate": reference_texture_ok,
        }
        if reference_comparison is not None:
            round_entry["reference_comparison"] = reference_comparison
        if uv_layout_diagnostic is not None:
            # This reaches the coordinator report after blind review, never
            # the blind reviewer. It lets the next planning/routing pass tell
            # a bad model/layout from a bad material plan without making an
            # object-specific pixel repair in Python.
            round_entry["uv_layout_diagnostic"] = uv_layout_diagnostic
        if material_only:
            round_entry["blind_alignment_reason"] = "material_only_visual_identity_is_undefined"
        else:
            round_entry["blind_alignment_reason"] = blind_alignment_reason
        report["rounds"].append(round_entry)
        if index >= rounds - 1 or (
            result.validation.passed
            and aligned
            and reference_shape_ok
            and reference_texture_ok
        ):
            break
        if (
            not result.validation.passed
            and request.form in {AssetForm.ITEM, AssetForm.CROSS}
            and not material_only
        ):
            # A semantic failure can be caused by a valid motif mask whose
            # colours collapse into the supporting part. Give the appearance
            # planner first chance when the critic explicitly calls out an
            # embedded/inlaid feature; geometry repair is allowed afterwards
            # only if the repaint does not resolve the target-free review.
            embedded_errors = [
                str(error).lower() for error in result.validation.errors
                if "embedded motif" in str(error).lower()
            ]
            repair_scope = {"geometry": True, "appearance": False, "reason": "semantic validation requires geometry review"}
            if embedded_errors:
                try:
                    repair_scope = planner.recommend_repair_scope(
                        plan.descriptor,
                        result.validation,
                        blind,
                        current_image=str(blind_image) if blind_image is not None else None,
                    )
                    # A model director may be uncertain, but a semantic failure
                    # must still reach a model-authored geometry review rather
                    # than silently ending the quality loop.
                    if not repair_scope["geometry"] and not repair_scope["appearance"]:
                        repair_scope["geometry"] = True
                        repair_scope["reason"] = (
                            repair_scope["reason"] or "semantic failure needs geometry review"
                        )
                except Exception as exc:
                    round_entry["model_repair_scope_error"] = str(exc)
            round_entry["model_repair_scope"] = repair_scope
            appearance_changed = False
            if embedded_errors and repair_scope["appearance"]:
                try:
                    revised_appearance = planner.revise_appearance_from_review(
                        request,
                        plan.descriptor,
                        plan.geometry,
                        plan.appearance,
                        blind,
                        references=plan.references,
                        current_image=blind_image,
                    )
                    if to_jsonable(revised_appearance) != to_jsonable(plan.appearance):
                        round_entry["appearance_revision_from_embedded_failure"] = True
                        plan = replace(plan, appearance=revised_appearance)
                        appearance_changed = True
                    round_entry["appearance_revision_from_embedded_failure"] = False
                except Exception as exc:
                    round_entry["appearance_revision_from_embedded_failure_error"] = str(exc)
            if not repair_scope["geometry"]:
                if appearance_changed:
                    continue
                # Keep a semantic failure live even when the director sees a
                # paint issue but returns no altered AppearanceSpec.
                repair_scope["geometry"] = True
            # Give the open planner one more chance with the complete contract
            # errors. The inner repair loop may have spent its response budget
            # on schema/compile repair; this pass receives the semantic
            # critic's concrete missing-part evidence and can preserve a better
            # reference-scale support.
            try:
                active_geometry_path = round_dir / "generated" / "geometry.json"
                active_geometry = geometry_from_dict(
                    json.loads(active_geometry_path.read_text(encoding="utf-8"))
                ) if active_geometry_path.exists() else plan.geometry
                candidate = planner.revise_geometry_from_review(
                    request,
                    plan.descriptor,
                    active_geometry,
                    result.validation,
                    blind,
                    compiled=result.compiled,
                    references=plan.references,
                    reference_comparison=(
                        reference_comparison
                        if isinstance(reference_comparison, dict)
                        else None
                    ),
                )
                candidate_compiled = compile_geometry(candidate)
                candidate_validation = validate_geometry(candidate, candidate_compiled, request.form)
                motif_ok, motif_changes = _motif_area_non_regression(
                    active_geometry, candidate, _embedded_part_ids(plan)
                )
                if not motif_ok:
                    round_entry["model_geometry_repair_rejected"] = list(candidate_validation.errors)
                    round_entry["model_geometry_repair_motif_area"] = {
                        part_id: {"before": values[0], "after": values[1]}
                        for part_id, values in motif_changes.items()
                    }
                    candidate_validation = replace(
                        candidate_validation,
                        errors=[
                            *candidate_validation.errors,
                            "embedded motif area regressed during geometry repair",
                        ],
                    )
                    motif_ok = False
                if not motif_ok:
                    candidate = active_geometry
                if candidate_validation.passed and to_jsonable(candidate) != to_jsonable(active_geometry):
                    plan = replace(plan, geometry=candidate)
                    round_entry["model_geometry_repair_from_validation"] = True
                    continue
                if (
                    to_jsonable(candidate) != to_jsonable(active_geometry)
                    and _geometry_can_be_rendered(candidate, candidate_compiled)
                ):
                    plan = replace(plan, geometry=candidate)
                    round_entry["model_geometry_repair_unvalidated"] = True
                    round_entry["model_geometry_repair_soft_errors"] = list(candidate_validation.errors)
                    continue
                round_entry["model_geometry_repair_rejected"] = list(candidate_validation.errors)
                # Keep a model-authored candidate that measurably reduces hard
                # geometry errors for the next quality round. Replacing it
                # with a role-driven mask makes repeated requests converge on
                # the same generic diagonal contour and hides the actual model
                # decision from the next repair prompt.
                try:
                    active_validation = validate_geometry(
                        active_geometry, compile_geometry(active_geometry), request.form
                    )
                except (KeyError, TypeError, ValueError):
                    active_validation = result.validation
                if (
                    to_jsonable(candidate) != to_jsonable(active_geometry)
                    and len(candidate_validation.errors) < len(active_validation.errors)
                ):
                    plan = replace(plan, geometry=candidate)
                    round_entry["model_geometry_repair_unvalidated"] = True
                    round_entry["model_geometry_repair_error_reduction"] = {
                        "before": len(active_validation.errors),
                        "after": len(candidate_validation.errors),
                    }
                    continue
            except Exception as exc:  # noqa: BLE001 - safety net remains auditable
                round_entry["model_geometry_repair_error"] = str(exc)
            if appearance_changed:
                continue
        if (
            result.validation.passed
            and not reference_shape_ok
            and request.form in {AssetForm.ITEM, AssetForm.CROSS}
            and not material_only
            and not _target_overlay_descriptor(plan.descriptor)
            and index < rounds - 1
        ):
            # A candidate can satisfy the open geometry/semantic contract yet
            # still diverge too far from a same-size variant's reference. Keep
            # that rendered candidate for ranking, but ask for a fresh
            # model-authored draft rather than feeding the valid raster through
            # a repair prompt that may redesign its supports. This is a generic
            # preserve-vs-retry decision; no object-specific mask is supplied.
            try:
                fresh_geometry = planner.geometry(
                    plan.descriptor,
                    request.width,
                    request.height,
                    uv_regions=plan.geometry.uv_regions,
                    form=request.form,
                    references=plan.references,
                )
                if to_jsonable(fresh_geometry) != to_jsonable(plan.geometry):
                    plan = replace(plan, geometry=fresh_geometry)
                    round_entry["fresh_geometry_for_reference_shape"] = True
                    continue
            except Exception as exc:  # keep the valid candidate auditable
                round_entry["fresh_geometry_for_reference_shape_error"] = str(exc)
        # A valid entity atlas can fail blind recognition because of paint
        # hierarchy (for example a bright lower patch reading as a sword).
        # Give the appearance stage one explicit correction pass before asking
        # the geometry stage to touch a model UV contract.
        if result.validation.passed and (not aligned or not reference_texture_ok):
            try:
                appearance_feedback = dict(blind)
                if isinstance(target_visual_review, dict):
                    appearance_feedback["target_visual_review"] = target_visual_review
                if isinstance(reference_comparison, dict):
                    appearance_feedback["reference_comparison"] = reference_comparison
                revised_appearance = planner.revise_appearance_from_review(
                    request,
                    plan.descriptor,
                    plan.geometry,
                    plan.appearance,
                    appearance_feedback,
                    references=plan.references,
                    current_image=blind_image,
                )
                if request.form == AssetForm.ENTITY_UV:
                    revised_appearance = _normalize_entity_appearance(
                        revised_appearance,
                        plan.descriptor,
                        plan.geometry,
                    )
                if to_jsonable(revised_appearance) != to_jsonable(plan.appearance):
                    round_entry["appearance_revision_changed"] = True
                    round_entry["appearance_revision_reason"] = (
                        "blind_mismatch" if not aligned else "reference_texture_diverged"
                    )
                    plan = replace(plan, appearance=revised_appearance)
                    continue
                round_entry["appearance_revision_unchanged"] = True
            except Exception as exc:  # retain the valid visual artifact and continue to geometry feedback
                round_entry["appearance_revision_error"] = str(exc)
            if request.form == AssetForm.ENTITY_UV:
                # Atlas geometry is already mechanically locked to the vanilla
                # model. A geometry repair here would invent a second
                # silhouette and cannot improve the blind preview.
                break
        try:
            # GenerationPipeline may have accepted a repaired geometry. Feed
            # that actual artifact forward; revising the original failed plan
            # would silently discard the previous repair and break stage flow.
            active_geometry_path = round_dir / "generated" / "geometry.json"
            active_geometry = geometry_from_dict(
                json.loads(active_geometry_path.read_text(encoding="utf-8"))
            ) if active_geometry_path.exists() else plan.geometry
            revised = None
            best_invalid: tuple[int, Any] | None = None
            try:
                active_geometry_validation = validate_geometry(
                    active_geometry, compile_geometry(active_geometry), request.form
                )
                active_geometry_error_count = len(active_geometry_validation.errors)
            except (KeyError, TypeError, ValueError):
                active_geometry_error_count = len(result.validation.errors)
            revision_feedback = dict(blind)
            if isinstance(reference_comparison, dict):
                revision_feedback["reference_comparison"] = reference_comparison
            round_entry["revision_attempts"] = 0
            # Vision models occasionally echo a valid but rejected geometry.
            # Give that same round one explicit rebuild attempt before stopping;
            # this keeps `rounds` meaningful without introducing a target
            # specific fallback shape; an unresolved candidate remains visible
            # to the next model revision round instead.
            for revision_attempt in range(2):
                round_entry["revision_attempts"] = revision_attempt + 1
                if revision_attempt:
                    revision_feedback["previous_revision_unchanged"] = True
                    revision_feedback["revision_instruction"] = (
                        "Rebuild the primitive list with visibly different proportions and contour; "
                        "do not repeat the current geometry or only restate the errors."
                    )
                candidate = planner.revise_geometry_from_review(
                    request,
                    plan.descriptor,
                    active_geometry,
                    result.validation,
                    revision_feedback,
                    compiled=result.compiled,
                    references=plan.references,
                    reference_comparison=reference_comparison if isinstance(reference_comparison, dict) else None,
                )
                if to_jsonable(candidate) == to_jsonable(active_geometry):
                    revision_feedback["candidate_contract_errors"] = [
                        "revision returned unchanged geometry"
                    ]
                    continue
                candidate_compiled = None
                try:
                    candidate_compiled = compile_geometry(candidate)
                    candidate_contract = validate_geometry(
                        candidate,
                        candidate_compiled,
                        request.form,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    candidate_contract = None
                    revision_feedback["candidate_contract_errors"] = [str(exc)]
                motif_ok, motif_changes = _motif_area_non_regression(
                    active_geometry, candidate, _embedded_part_ids(plan)
                )
                if not motif_ok:
                    revision_feedback["candidate_contract_errors"] = [
                        *(revision_feedback.get("candidate_contract_errors", [])),
                        "embedded motif area regressed during geometry repair: %s" % json.dumps(
                            {
                                part_id: {"before": values[0], "after": values[1]}
                                for part_id, values in motif_changes.items()
                            },
                            ensure_ascii=False,
                        ),
                    ]
                    candidate_contract = None
                if candidate_contract is not None and not candidate_contract.passed:
                    revision_feedback["candidate_contract_errors"] = list(candidate_contract.errors)
                    if candidate_compiled is not None and _geometry_can_be_rendered(candidate, candidate_compiled):
                        revised = candidate
                        round_entry["revision_changed"] = True
                        round_entry["revision_unvalidated"] = True
                        break
                    candidate_error_count = len(candidate_contract.errors)
                    if (
                        candidate_error_count < active_geometry_error_count
                        and (best_invalid is None or candidate_error_count < best_invalid[0])
                    ):
                        best_invalid = (candidate_error_count, candidate)
                    continue
                revised = candidate
                round_entry["revision_changed"] = True
                break
            if revised is None:
                if best_invalid is not None:
                    plan = replace(plan, geometry=best_invalid[1])
                    round_entry["revision_changed"] = True
                    round_entry["revision_unvalidated"] = True
                    round_entry["revision_error_reduction"] = {
                        "before": active_geometry_error_count,
                        "after": best_invalid[0],
                    }
                    continue
                if index < rounds - 1:
                    # A schema-valid response can still be unusable (for
                    # example every support mask is empty). Ask the same
                    # model for a fresh open geometry draft before spending
                    # another round on a blind revision of an empty canvas.
                    try:
                        fresh_geometry = planner.geometry(
                            plan.descriptor,
                            request.width,
                            request.height,
                            uv_regions=plan.geometry.uv_regions,
                            form=request.form,
                            references=plan.references,
                        )
                        if to_jsonable(fresh_geometry) != to_jsonable(active_geometry):
                            plan = replace(plan, geometry=fresh_geometry)
                            round_entry["fresh_geometry_retry"] = True
                            continue
                    except Exception as exc:  # keep the failed artifact auditable
                        round_entry["fresh_geometry_retry_error"] = str(exc)
                round_entry["revision_unchanged"] = True
                round_entry["revision_changed"] = False
                break
        except Exception as exc:  # retain the failed round and stop cleanly
            round_entry["revision_error"] = str(exc)
            break
        plan = replace(plan, geometry=revised)
    report["rounds_completed"] = len(report["rounds"])
    if report["rounds"]:
        selected = max(report["rounds"], key=_round_quality_score)
        report["selected_round"] = selected["index"]
        report["selected_sprite"] = selected.get("sprite")
        report["selected_quality_score"] = _round_quality_score(selected)
    else:
        report["selected_round"] = None
        report["selected_sprite"] = None
    _write_json(root / "quality_report.json", report)
    return report


def run_query_pipeline(
    query: str,
    out_dir: str | Path,
    rounds: int = 3,
    max_repairs: int = 1,
    package: bool = False,
) -> dict[str, Any]:
    """Public one-argument entry point for unattended generation.

    ``query`` is the only asset-specific input.  Routing, vanilla references,
    model UV selection, rendering and blind review all remain inside the
    quality pipeline.
    """
    return run_quality_loop(
        AssetRequest(query=query, form=AssetForm.AUTO),
        [],
        out_dir,
        rounds=rounds,
        max_repairs=max_repairs,
        package=package,
    )


def author_box_layout(
    query: str,
    out_dir: str | Path,
    *,
    max_boxes: int = 16,
    canvas_width: int | None = None,
    planner: ModelPlanner | None = None,
) -> Path:
    """Derive a UV layout from a model-authored box decomposition.

    This is the path for a model that has no vanilla equivalent: the object
    declares its own boxes and the atlas follows from them, instead of the run
    being forced onto an unrelated shipped layout such as the cow's.
    """
    from .box_model import pack_boxes

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    active = planner or ModelPlanner(
        OpenAICompatibleClient.from_env(trace_dir=root / "llm_raw")
    )
    boxes = active.author_boxes(query, max_boxes=max_boxes)
    layout = pack_boxes(boxes, canvas_width=canvas_width)
    _write_json(root / "box_decomposition.json", {
        "query": query,
        "boxes": [
            {
                "id": box.id,
                "part_id": box.part_id,
                "size": [box.width, box.height, box.depth],
                "origin": list(box.origin) if box.origin else None,
                "notes": box.notes,
            }
            for box in boxes
        ],
        "canvas": list(layout.canvas),
        "cube_count": len(layout.cubes),
        "region_count": len(layout.regions),
    })
    return layout.write(root / "authored_layout.json")


def _set_slug(value: str, fallback: str = "asset_set") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug[:48] or fallback


def sprite_palette_profile(sprite_path: str | Path, limit: int = 12) -> dict[str, Any]:
    """A comparison-stable description of a rendered sprite's colours.

    Exact RGB counting is deliberately avoided. Pixel art carries many
    near-identical shades, so two textures that plainly belong to one family
    can share almost no exact colour: a live run scored 0.09 exact-RGB overlap
    on a pair whose mean nearest-neighbour distance was 7 out of 441. Colours
    are therefore quantised to 16 levels per channel, and the saturation and
    brightness split is measured separately, because "the sibling drifted to
    grey" is a real failure that a palette intersection alone can miss.
    """
    with Image.open(sprite_path) as loaded:
        image = loaded.convert("RGBA")
    pixels = [
        (red, green, blue)
        for red, green, blue, alpha in image.get_flattened_data()
        if alpha >= 8
    ]
    if not pixels:
        return {"opaque": 0, "quantised": [], "grey_ratio": 0.0, "dark_ratio": 0.0, "mean_saturation": 0.0}
    quantised = Counter((red >> 4, green >> 4, blue >> 4) for red, green, blue in pixels)
    saturations: list[float] = []
    values: list[float] = []
    for red, green, blue in pixels:
        _hue, saturation, value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
        saturations.append(saturation)
        values.append(value)
    total = float(len(pixels))
    return {
        "opaque": len(pixels),
        "quantised": [colour for colour, _count in quantised.most_common(limit)],
        "grey_ratio": sum(1 for item in saturations if item < 0.22) / total,
        "dark_ratio": sum(1 for item in values if item < 0.22) / total,
        "mean_saturation": sum(saturations) / total,
    }


# A drifted member costs one extra generation pass. Capping it keeps a bad set
# from turning into an unbounded retry loop.
_MAX_FAMILY_DRIFT_REPAIRS = 2


def sprite_palette_ramp(sprite_path: str | Path, count: int = 8) -> list[tuple[int, int, int]]:
    """An ordered dark-to-light ramp taken from a rendered anchor sprite."""
    with Image.open(sprite_path) as loaded:
        image = loaded.convert("RGBA")
    counts = Counter(
        (red, green, blue)
        for red, green, blue, alpha in image.get_flattened_data()
        if alpha >= 8
    )
    if not counts:
        return []
    ordered = sorted((colour for colour, _n in counts.most_common(24)), key=lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
    if len(ordered) <= count:
        return ordered
    return [
        ordered[round(index * (len(ordered) - 1) / float(max(count - 1, 1)))]
        for index in range(count)
    ]


def remap_palette_to_anchor(
    palette: dict[str, str],
    ramp: list[tuple[int, int, int]],
) -> dict[str, str]:
    """Move an authored palette onto a family anchor's ramp, keeping its order.

    The model names its swatches and decides their dark/mid/light order; the
    family owns the actual values. Entries are re-assigned by luma rank, so the
    authored structure survives while every member of the set resolves to the
    same colours. This is a family-level contract the caller asked for, not a
    per-asset colour post-process.
    """
    if not palette or not ramp:
        return dict(palette)

    def luma(value: str) -> float:
        text = str(value).strip().lstrip("#")
        if len(text) != 6:
            return 128.0
        try:
            red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
        except ValueError:
            return 128.0
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    ordered = sorted(palette.items(), key=lambda item: luma(item[1]))
    ranked = sorted(ramp, key=lambda colour: 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2])
    result: dict[str, str] = {}
    for index, (name, _value) in enumerate(ordered):
        position = round(index * (len(ranked) - 1) / float(max(len(ordered) - 1, 1)))
        red, green, blue = ranked[position]
        result[name] = "#%02X%02X%02X" % (red, green, blue)
    return result


_HEX_COLOUR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def remap_appearance_to_anchor(
    appearance: Any,
    ramp: list[tuple[int, int, int]],
) -> Any:
    """Force every colour a member named onto the family's ramp.

    ``remap_palette_to_anchor`` only rewrites *named* swatches, and a live bow
    family showed why that is not enough: three of the four frames wrote
    literal hex colours straight into their parts, so the anchor's ramp never
    reached a single pixel and the four frames of one bow shipped in four
    unrelated palettes. Named swatches keep their names; every literal the
    member used is ranked by luma against the ramp, so the member's own light
    to dark structure survives while the values become the family's.
    """
    if not ramp:
        return appearance
    palette = remap_palette_to_anchor(appearance.palette, ramp)
    ranked = sorted(
        ramp, key=lambda colour: 0.2126 * colour[0] + 0.7152 * colour[1] + 0.0722 * colour[2]
    )

    def luma(value: str) -> float:
        text = str(value).strip().lstrip("#")
        try:
            red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
        except (ValueError, IndexError):
            return 128.0
        return 0.2126 * red + 0.7152 * green + 0.0722 * blue

    literals: set[str] = set()
    for part in appearance.parts.values():
        literals.update(colour for colour in part.colors if _HEX_COLOUR.match(str(colour)))
    if appearance.outline_color and _HEX_COLOUR.match(str(appearance.outline_color)):
        literals.add(appearance.outline_color)
    pixel_map = appearance.pixel_map
    if isinstance(pixel_map, dict) and isinstance(pixel_map.get("legend"), dict):
        literals.update(
            value for value in pixel_map["legend"].values() if _HEX_COLOUR.match(str(value))
        )
    ordered = sorted(literals, key=luma)
    literal_map: dict[str, str] = {}
    for index, colour in enumerate(ordered):
        position = round(index * (len(ranked) - 1) / float(max(len(ordered) - 1, 1)))
        red, green, blue = ranked[position]
        literal_map[colour] = "#%02X%02X%02X" % (red, green, blue)

    def resolved(colour: str) -> str:
        return literal_map.get(colour, colour)

    parts = {
        part_id: replace(part, colors=[resolved(colour) for colour in part.colors])
        for part_id, part in appearance.parts.items()
    }
    updated_map = pixel_map
    if isinstance(pixel_map, dict) and isinstance(pixel_map.get("legend"), dict):
        updated_map = dict(pixel_map)
        updated_map["legend"] = {
            key: resolved(str(value)) for key, value in pixel_map["legend"].items()
        }
    return replace(
        appearance,
        palette=palette,
        parts=parts,
        outline_color=(resolved(appearance.outline_color) if appearance.outline_color else None),
        pixel_map=updated_map,
    )


def inherit_anchor_paint(
    appearance: Any,
    geometry: Any,
    anchor_appearance: Any,
    anchor_geometry: Any,
    minimum_overlap: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    """Give each member part the anchor part's whole paint specification.

    The family contract already forces the colour *values* onto one ramp, and a
    live crystal bow showed why that is not enough: each frame still authored
    its own shading vocabulary, so the body resolved into six value bands on
    one frame and four dark ones on the next. The four frames of one bow read
    as four materials.

    Matching is by mask overlap, never by part name: a member calls the same
    region bow_body, crystal_bow_body or crystal_bow_body_upper, and only the
    pixels agree. Overlap is also the guard -- members of a plain set
    (helmet, chestplate) sit on different atlases, so nothing matches and each
    keeps its own paint.

    Returns the revised appearance and a per-part audit of what was matched.
    """
    try:
        member = compile_geometry(geometry)
        anchor = compile_geometry(anchor_geometry)
    except (KeyError, TypeError, ValueError):
        return appearance, {}
    anchor_ramps: dict[str, list[str]] = {}
    for part_id, spec in anchor_appearance.parts.items():
        anchor_ramps[part_id] = [
            "#%02X%02X%02X" % colour
            for colour in _resolve_colors(spec, anchor_appearance.palette)
        ]

    matched: dict[str, Any] = {}
    parts: dict[str, Any] = {}
    for part_id, spec in appearance.parts.items():
        parts[part_id] = spec
        mask = member.part_masks.get(part_id)
        if mask is None:
            continue
        own = [
            (x, y)
            for y in range(member.height)
            for x in range(member.width)
            if mask.getpixel((x, y)) > 0
        ]
        if not own:
            continue
        best_id, best_score = None, 0.0
        for candidate_id, candidate_spec in anchor_appearance.parts.items():
            candidate = anchor.part_masks.get(candidate_id)
            if candidate is None:
                continue
            shared = sum(1 for x, y in own if candidate.getpixel((x, y)) > 0)
            if not shared:
                continue
            # Intersection over union, not coverage of this part alone. A
            # nocked arrow sits *inside* the bow body's mask, so coverage
            # alone says "same region" and hands the arrow the body's dark
            # crystal ramp -- which is exactly the arrow the prompt asked to
            # declare. Two parts are the same region only if they mostly
            # contain each other.
            candidate_points = sum(
                1
                for y in range(anchor.height)
                for x in range(anchor.width)
                if candidate.getpixel((x, y)) > 0
            )
            union = len(own) + candidate_points - shared
            if union <= 0:
                continue
            score = shared / float(union)
            if score > best_score:
                best_id, best_score = candidate_id, score
        if best_id is None or best_score < minimum_overlap:
            continue
        source = anchor_appearance.parts[best_id]
        parts[part_id] = replace(
            spec,
            colors=list(anchor_ramps[best_id]),
            shade_axis=source.shade_axis,
            noise=source.noise,
            highlight_ratio=source.highlight_ratio,
            marks=list(source.marks),
        )
        matched[part_id] = {"anchor_part": best_id, "overlap": round(best_score, 3)}

    if not matched:
        return appearance, {}
    # The anchor's own hand-authored composition is the pattern the request
    # asked to keep, so it carries across the frames that locked onto the same
    # object. A member that matched almost nothing keeps its own composition.
    # A declared part owns geometry here only when its mask actually holds
    # pixels; compile_geometry creates an empty mask for every declared part,
    # so presence alone would count paint-only parts as physical.
    physical = [
        part_id
        for part_id in appearance.parts
        if part_id in member.part_masks
        and any(
            member.part_masks[part_id].getpixel((x, y)) > 0
            for y in range(member.height)
            for x in range(member.width)
        )
    ]
    carries_composition = len(matched) >= max(1, len(physical) // 2)
    updated = replace(
        appearance,
        parts=parts,
        pixel_map=anchor_appearance.pixel_map if carries_composition else appearance.pixel_map,
        outline_color=(anchor_appearance.outline_color if carries_composition else appearance.outline_color),
        outline_width=(anchor_appearance.outline_width if carries_composition else appearance.outline_width),
    )
    return updated, {
        "parts": matched,
        "composition_from_anchor": carries_composition,
    }


def family_consistency(
    anchor_sprite: str | Path,
    member_sprites: list[str | Path],
    limit: int = 12,
    minimum_overlap: float = 0.35,
    maximum_grey_delta: float = 0.25,
) -> dict[str, Any]:
    """Measure how closely each sibling actually matches the anchor's palette.

    This is measured evidence rather than a model opinion, and it reports both
    numbers it used so a low verdict can be audited: a quantised palette
    intersection, and how far the desaturated share drifted. A sibling that
    quietly turned from wood into grey iron trips the second one even when its
    colour histogram still overlaps.
    """
    anchor = sprite_palette_profile(anchor_sprite, limit)
    anchor_colours = set(anchor["quantised"])
    rows: list[dict[str, Any]] = []
    for path in member_sprites:
        profile = sprite_palette_profile(path, limit)
        colours = set(profile["quantised"])
        union = anchor_colours | colours
        overlap = len(anchor_colours & colours) / float(len(union)) if union else 0.0
        grey_delta = abs(anchor["grey_ratio"] - profile["grey_ratio"])
        rows.append({
            "sprite": str(path),
            "palette_overlap": round(overlap, 4),
            "grey_ratio": round(profile["grey_ratio"], 4),
            "grey_ratio_delta": round(grey_delta, 4),
            "consistent": overlap >= minimum_overlap and grey_delta <= maximum_grey_delta,
        })
    return {
        "anchor": str(anchor_sprite),
        "anchor_grey_ratio": round(anchor["grey_ratio"], 4),
        "members": rows,
        "minimum_overlap": minimum_overlap,
        "maximum_grey_delta": maximum_grey_delta,
    }


def _frame_pixels(path: str | Path) -> dict[tuple[int, int], tuple[int, int, int, int]] | None:
    try:
        with Image.open(path) as loaded:
            image = loaded.convert("RGBA")
    except (OSError, ValueError):
        return None
    return {
        (x, y): image.getpixel((x, y))
        for y in range(image.height)
        for x in range(image.width)
    }



# One logical name can own a great many near-identical frames -- vanilla
# clock has 64 and compass 32. Attaching every one of them spends the whole
# vision budget on one object, so a selection expands to at most this many
# states: the frame this request is about, plus an even spread of the rest.
_MAX_GROUP_FRAMES = 8


def _shape_authority(
    expanded: list[ReferenceAsset], preferred_name: str | None
) -> ReferenceAsset | None:
    """The one member of a family whose silhouette answers this request.

    Only meaningful when a logical name expanded into several states; a single
    texture, or a set of distinct faces, has no frame to single out.
    """
    if not preferred_name or len(expanded) < 2:
        return None
    best: tuple[int, ReferenceAsset] | None = None
    for asset in expanded:
        score = family_member_match(_reference_member_name(asset), preferred_name)
        if score and (best is None or score > best[0]):
            best = (score, asset)
    return best[1] if best is not None else None


def _reference_group(reference: ReferenceAsset) -> str | None:
    """The logical asset a reference came from, when the live source recorded it."""
    for note in reference.notes:
        if note.startswith("group="):
            value = note.split("=", 1)[1].strip()
            return value or None
    return None


def _anchor_paint(member_dir: Path, selected_round: Any) -> tuple[Any, Any] | None:
    """The appearance and geometry one finished member actually rendered with.

    Read back from the member's own report so the family inherits a plan the
    model produced, not a plan the family invented.
    """
    try:
        index = int(selected_round)
    except (TypeError, ValueError):
        index = 0
    generated = member_dir / ("round_%02d" % index) / "generated"
    try:
        appearance = appearance_from_dict(
            json.loads((generated / "appearance.json").read_text(encoding="utf-8"))
        )
        geometry = geometry_from_dict(
            json.loads((generated / "geometry.json").read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return appearance, geometry


def _member_shape_group(references: list[dict[str, Any]]) -> str | None:
    """The logical asset a finished member's shape reference came from.

    Read back from the member's own report rather than assumed, so a family
    propagates a reading only between frames of the same source object.
    """
    group: str | None = None
    for entry in references:
        if not isinstance(entry, dict):
            continue
        roles = entry.get("roles") or []
        if "shape" not in roles:
            continue
        for note in entry.get("notes") or []:
            if isinstance(note, str) and note.startswith("group="):
                group = note.split("=", 1)[1].strip() or None
                break
        if group:
            break
    return group


def _member_references(member_dir: Path, selected_round: Any) -> list[dict[str, Any]]:
    """The reference list one finished member recorded for its selected round."""
    try:
        index = int(selected_round)
    except (TypeError, ValueError):
        index = 0
    references_path = member_dir / ("round_%02d" % index) / "generated" / "references.json"
    try:
        entries = json.loads(references_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def _member_shape_host(member_dir: Path, member_name: str, selected_round: Any) -> Path | None:
    """The reference one member was actually conformed to.

    Read back from the member's own report so the continuity baseline is the
    source family the run really used, not a second guess at it. The match is
    the same suffix rule the geometry stage uses, or the baseline would
    compare a frame against a sibling frame it was never conformed to.
    """
    entries = _member_references(member_dir, selected_round)
    candidates: list[tuple[str, Path]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        roles = entry.get("roles") or []
        if "shape" not in roles:
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not Path(path).exists():
            continue
        name = str(entry.get("name") or "")
        member = (name.rsplit(":", 1)[-1] if ":" in name else name).strip().lower()
        candidates.append((member, Path(path)))
    if not candidates:
        return None
    wanted = str(member_name or "").strip().lower()
    best: tuple[int, Path] | None = None
    for member, path in candidates:
        if not member or not wanted:
            continue
        if member == wanted:
            score = len(member) + 1
        elif wanted.endswith(member) or member.endswith(wanted):
            score = len(member)
        else:
            continue
        if best is None or score > best[0]:
            best = (score, path)
    return best[1] if best is not None else candidates[0][1]


def frame_continuity(
    sprites: list[str | Path],
    *,
    baseline_sprites: list[str | Path] | None = None,
    minimum_shared: float = 0.35,
    tolerance: float = 0.15,
    colour_tolerance: int = 24,
) -> dict[str, Any]:
    """Measure whether several frames read as one object in different states.

    A palette intersection cannot answer this: two frames of the same draw
    animation share a palette even when the drawing underneath is unrelated.
    What the eye checks is pixel agreement on the body the frames have in
    common, so that is what is measured here -- restricted to pairs whose
    silhouettes genuinely overlap, because the members of a plain set
    (helmet, chestplate, leggings) share no body at all.

    The verdict is relative on purpose. Even the vanilla bow family only
    agrees on 29%-66% of its union pairwise: the string moves, the limb bends
    a little, and thin shapes make every moved pixel visible. Comparing a
    generated family against that same-family baseline is the only honest
    question; an absolute threshold would fail the source art.

    Two numbers are reported because they fail differently. Exact agreement
    asks whether the same pixel was painted the same way. Near agreement
    (within colour_tolerance per channel) asks whether the frames at least
    speak one colour language. A family that shares a palette but moves its
    highlights scores low on the first and high on the second; a family
    whose members drifted into different palettes scores low on both, which
    is the difference a single number would hide.
    """
    loaded = [(Path(path), _frame_pixels(path)) for path in sprites]
    loaded = [(path, pixels) for path, pixels in loaded if pixels is not None]
    rows: list[dict[str, Any]] = []
    for index in range(len(loaded)):
        for other in range(index + 1, len(loaded)):
            path_a, pixels_a = loaded[index]
            path_b, pixels_b = loaded[other]
            if len(pixels_a) != len(pixels_b):
                continue
            opaque_a = {point for point, value in pixels_a.items() if value[3] >= 8}
            opaque_b = {point for point, value in pixels_b.items() if value[3] >= 8}
            union = opaque_a | opaque_b
            if not union:
                continue
            iou = len(opaque_a & opaque_b) / float(len(union))
            if iou < minimum_shared:
                continue
            exact = 0
            near = 0
            delta_total = 0
            for point in union:
                left, right = pixels_a[point], pixels_b[point]
                if left == right:
                    exact += 1
                delta = max(abs(left[index] - right[index]) for index in range(4))
                delta_total += delta
                if delta <= colour_tolerance:
                    near += 1
            rows.append({
                "a": str(path_a),
                "b": str(path_b),
                "silhouette_iou": round(iou, 4),
                "agreement": round(exact / float(len(union)), 4),
                "near_agreement": round(near / float(len(union)), 4),
                "mean_colour_delta": round(delta_total / float(len(union)), 2),
            })
    measured = [row["agreement"] for row in rows]
    baseline: dict[str, Any] | None = None
    if baseline_sprites:
        baseline = frame_continuity(
            list(baseline_sprites),
            minimum_shared=minimum_shared,
            tolerance=tolerance,
            colour_tolerance=colour_tolerance,
        )
        baseline = baseline if baseline["pairs"] else None
    if not measured:
        verdict: bool | None = None
    elif baseline is None:
        # Nothing to compare against: report the number, claim nothing.
        verdict = None
    else:
        verdict = min(measured) >= baseline["minimum_agreement"] - tolerance
    near_measured = [row["near_agreement"] for row in rows]
    return {
        "pairs": rows,
        "pair_count": len(rows),
        "mean_agreement": round(sum(measured) / len(measured), 4) if measured else None,
        "minimum_agreement": round(min(measured), 4) if measured else None,
        "mean_near_agreement": (
            round(sum(near_measured) / len(near_measured), 4) if near_measured else None
        ),
        "minimum_near_agreement": round(min(near_measured), 4) if near_measured else None,
        "colour_tolerance": colour_tolerance,
        "baseline": (
            None
            if baseline is None
            else {
                "minimum_agreement": baseline["minimum_agreement"],
                "mean_agreement": baseline["mean_agreement"],
                "minimum_near_agreement": baseline["minimum_near_agreement"],
                "mean_near_agreement": baseline["mean_near_agreement"],
                "pair_count": baseline["pair_count"],
            }
        ),
        "tolerance": tolerance,
        "consistent": verdict,
    }


def run_family_loop(
    query: str,
    out_dir: str | Path,
    *,
    members: list[str] | None = None,
    max_members: int = 6,
    form: AssetForm | None = None,
    shape_policy: str = "planned",
    family_plan: dict[str, Any] | None = None,
    rounds: int = 1,
    max_repairs: int = 1,
    package: bool = True,
    asset_sources: list[str] | None = None,
    cache_root: str | Path | None = None,
    recall_limit: int | None = None,
    router_model: str | None = None,
    reference_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a whole family of sibling textures from one request.

    The first member runs through the normal pipeline and becomes the style
    anchor. Every later member runs the same pipeline but with that anchor
    attached as palette and pixel-style evidence, so an armour set or a tool
    set shares one visual language instead of drifting member by member.

    Each member keeps its own output directory and quality report; 'family.json'
    records the plan, the per-member outcome and a deterministic palette
    comparison against the anchor.
    """
    if out_dir is None:
        raise ValueError("out_dir is required")
    root = Path(out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if family_plan is not None:
        # The caller already asked the model whether this request is one asset
        # or a set; planning again would both cost a call and risk disagreeing
        # with the split it just decided on.
        plan = dict(family_plan)
        plan.setdefault("source", "model")
    elif members:
        plan: dict[str, Any] = {
            "set_name": _set_slug(query),
            "shared_style": "",
            "members": [
                {"name": _set_slug(name, "member_%d" % (index + 1)), "target": "%s: %s" % (query, name)}
                for index, name in enumerate(members[:max_members])
            ],
        }
        plan["source"] = "explicit"
    else:
        planner = ModelPlanner(OpenAICompatibleClient.from_env(trace_dir=root / "llm_raw"))
        plan = planner.plan_family(query, max_members=max_members)
        plan["source"] = "model"
    _write_json(root / "family_plan.json", plan)

    anchor: Path | None = None
    anchor_dir: Path | None = None
    anchor_round: Any = 0
    rows: list[dict[str, Any]] = []
    member_requests: list[tuple[int, AssetRequest, Path]] = []
    shared_style = str(plan.get("shared_style") or "").strip()
    # Frames of one object must agree on whether the source silhouette is the
    # answer. The first member decides it for its reference group and every
    # later member drawn from that same group inherits the reading; a set
    # whose members come from different groups is untouched.
    shape_mode_by_group: dict[str, str] = {}
    # The anchor's own ramp, captured once it exists. Every later member is
    # resolved onto it on the first pass, not only when a drift gate trips:
    # a family that only repairs colour after shipping four different
    # palettes has already failed the request.
    anchor_ramp: list[tuple[int, int, int]] | None = None
    anchor_paint: tuple[Any, Any] | None = None
    for index, member in enumerate(plan["members"]):
        member_dir = root / ("%02d_%s" % (index, member["name"]))
        # The shared style is part of every member's immutable brief, not just
        # a note in the report. The first member has no rendered anchor yet, so
        # this sentence is the only thing keeping it in the same family.
        member_query = member["target"]
        if shared_style:
            member_query = "%s\nShared style that every piece of this set must keep: %s" % (
                member_query, shared_style
            )
        request = AssetRequest(
            query=member_query,
            form=form or AssetForm.AUTO,
            name=member["name"],
            # The family path builds its own requests, so anything the caller
            # set on the run has to be carried across explicitly: a flag like
            # free-silhouette was silently dropped here.
            shape_policy=shape_policy,
        )
        member_requests.append((index, request, member_dir))
        if anchor is not None and anchor_ramp is None:
            anchor_ramp = sprite_palette_ramp(anchor)
            anchor_paint = _anchor_paint(anchor_dir, anchor_round)
        report: dict[str, Any] = {}
        error: str | None = None
        try:
            report = run_quality_loop(
                request,
                [],
                member_dir,
                rounds=rounds,
                max_repairs=max_repairs,
                package=package,
                asset_sources=asset_sources,
                cache_root=cache_root,
                recall_limit=recall_limit,
                router_model=router_model,
                reference_index_path=reference_index_path,
                style_anchor=anchor,
                anchor_palette=anchor_ramp,
                shape_mode_by_group=shape_mode_by_group or None,
                anchor_paint=anchor_paint,
            )
        except Exception as exc:  # one bad member must not discard its siblings
            error = "%s: %s" % (type(exc).__name__, exc)
            _write_json(member_dir / "family_member_error.json", {
                "member": member["name"],
                "error": error,
                "traceback": traceback.format_exc(),
            })
        sprite = report.get("selected_sprite")
        if sprite and not Path(sprite).exists():
            sprite = None
        selected_index = report.get("selected_round")
        if anchor is None and sprite:
            anchor = Path(sprite)
            anchor_dir = member_dir
            anchor_round = selected_index
        if error is None and sprite and not shape_mode_by_group:
            group = _member_shape_group(_member_references(member_dir, selected_index))
            if group and (report.get("routing") or {}).get("shape_contour_locked"):
                # Whether the first frame locked the source contour is a
                # property of the object, so every later frame drawn from the
                # same source keeps its own source contour too.
                shape_mode_by_group[group] = "preserve_silhouette"
        selected = next(
            (entry for entry in report.get("rounds", []) if entry.get("index") == selected_index),
            {},
        )
        rows.append({
            "index": index,
            "name": member["name"],
            "target": member["target"],
            "out_dir": str(member_dir),
            "sprite": sprite,
            "selected_round": selected_index,
            "rounds_completed": report.get("rounds_completed", 0),
            "validation_passed": bool(selected.get("validation", {}).get("passed")),
            "blind_object": (selected.get("blind_review") or {}).get("primary_object"),
            "error": error,
        })

    consistency: dict[str, Any] | None = None
    if anchor is not None:
        siblings = [
            Path(row["sprite"])
            for row in rows
            if row["sprite"] and Path(row["sprite"]) != anchor
        ]
        consistency = family_consistency(anchor, siblings)
        measured = {item["sprite"]: item for item in consistency["members"]}
        for row in rows:
            if not row["sprite"]:
                continue
            if Path(row["sprite"]) == anchor:
                row["palette_overlap"] = 1.0
                row["consistent"] = True
            else:
                item = measured.get(str(row["sprite"])) or {}
                row["palette_overlap"] = item.get("palette_overlap")
                row["consistent"] = item.get("consistent")

    # The anchor is attached to every member, but a member can still drift past
    # it. The gate already measured that, so spend one extra pass on the drifted
    # members rather than shipping a set that no longer matches: the anchor's own
    # ramp becomes the retry's palette while the model keeps its own swatch names
    # and its own target.
    repaired = 0
    if anchor is not None and consistency is not None:
        ramp = sprite_palette_ramp(anchor)
        drifted = [
            row for row in rows
            if row["sprite"] and Path(row["sprite"]) != anchor and not row.get("consistent", True)
        ]
        for row in drifted[:_MAX_FAMILY_DRIFT_REPAIRS]:
            _index, request, member_dir = member_requests[row["index"]]
            try:
                retry = run_quality_loop(
                    request,
                    [],
                    member_dir,
                    rounds=rounds,
                    max_repairs=max_repairs,
                    package=package,
                    asset_sources=asset_sources,
                    cache_root=cache_root,
                    recall_limit=recall_limit,
                    router_model=router_model,
                    reference_index_path=reference_index_path,
                    style_anchor=anchor,
                    anchor_palette=ramp,
                    shape_mode_by_group=shape_mode_by_group or None,
                    anchor_paint=anchor_paint,
                )
            except Exception as exc:  # a failed retry keeps the first result
                row["drift_repair_error"] = "%s: %s" % (type(exc).__name__, exc)
                continue
            sprite = retry.get("selected_sprite")
            if not (sprite and Path(sprite).exists()):
                row["drift_repair_error"] = "drift repair produced no sprite"
                continue
            row["sprite_before_drift_repair"] = row["sprite"]
            row["sprite"] = sprite
            selected_index = retry.get("selected_round")
            row["selected_round"] = selected_index
            selected = next(
                (entry for entry in retry.get("rounds", []) if entry.get("index") == selected_index),
                {},
            )
            row["validation_passed"] = bool(selected.get("validation", {}).get("passed"))
            row["blind_object"] = (selected.get("blind_review") or {}).get("primary_object")
            row["drift_repaired"] = True
            repaired += 1
        if repaired:
            siblings = [
                Path(row["sprite"])
                for row in rows
                if row["sprite"] and Path(row["sprite"]) != anchor
            ]
            consistency = family_consistency(anchor, siblings)
            remeasured = {item["sprite"]: item for item in consistency["members"]}
            for row in rows:
                if not row["sprite"] or Path(row["sprite"]) == anchor:
                    continue
                item = remeasured.get(str(row["sprite"])) or {}
                row["palette_overlap"] = item.get("palette_overlap")
                row["consistent"] = item.get("consistent")

    continuity: dict[str, Any] | None = None
    generated_sprites = [Path(row["sprite"]) for row in rows if row["sprite"]]
    if len(generated_sprites) > 1:
        # The baseline is the source family this run was actually conformed
        # to, so the comparison answers "is this set as coherent as the vanilla
        # frames it came from" rather than an invented absolute threshold.
        hosts: list[Path] = []
        for row in rows:
            host = _member_shape_host(
                Path(row["out_dir"]), row["name"], row.get("selected_round")
            )
            if host is not None and host not in hosts:
                hosts.append(host)
        continuity = frame_continuity(
            generated_sprites,
            baseline_sprites=hosts if len(hosts) > 1 else None,
        )

    family = {
        "query": query,
        "set_name": plan["set_name"],
        "plan_source": plan.get("source"),
        "shared_style": plan.get("shared_style", ""),
        "rounds": rounds,
        "anchor": str(anchor) if anchor else None,
        "drift_repaired": repaired,
        "members": rows,
        "consistency": consistency,
        "continuity": continuity,
        "inherited_shape_modes": shape_mode_by_group or None,
    }
    _write_json(root / "family.json", family)
    return family

