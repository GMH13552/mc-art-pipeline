"""Generic local recall and strict parsing for indexed reference selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .contracts import ReferenceRole
from .reference_index import IndexEntry, ReferenceIndex


# This is a language vocabulary, not a catalogue of shape families.  It maps
# common natural-language material/resource words to tokens already present in
# Minecraft paths, leaving the final semantic choice to the router model.
TOKEN_ALIASES: dict[str, tuple[str, ...]] = {
    "矿石": ("ore",),
    "矿": ("ore",),
    "铁": ("iron",),
    "金": ("gold", "golden"),
    "钻石": ("diamond",),
    "绿宝石": ("emerald",),
    "青金石": ("lapis",),
    "红石": ("redstone",),
    "紫晶": ("amethyst",),
    "石英": ("quartz",),
    "绿": ("green",),
    "红": ("red",),
    "蓝": ("blue",),
    "紫": ("purple",),
    "黑": ("black",),
    "白": ("white",),
    "方块": ("block", "blocks"),
    "块": ("block", "blocks"),
    "剑": ("sword",),
    "刀": ("sword", "knife"),
    "斧": ("axe",),
    "镐": ("pickaxe",),
    "铲": ("shovel",),
    "锄": ("hoe",),
    "工具": ("tool",),
    "武器": ("weapon",),
    "牛": ("cow",),
    "村民": ("villager",),
    "生物": ("entity",),
    "实体": ("entity",),
    "皮肤": ("entity",),
    "皮革": ("leather",),
    "木": ("wood", "plank", "log"),
    "石": ("stone",),
    "玻璃": ("glass",),
}

_CATEGORY_TERMS = {
    "entity": ("实体", "生物", "皮肤", "怪物", "动物", "entity", "mob"),
    "block": ("方块", "块", "矿", "矿石", "砖", "石", "木", "玻璃", "block", "ore"),
    "item": ("物品", "武器", "工具", "剑", "刀", "斧", "镐", "铲", "锄", "item", "sword", "tool"),
}

# The source index is complete; this is only the broad local-recall window
# handed to the semantic router.  Keeping the window separate from the index
# size lets a large pack remain searchable without forcing every one of its
# entries into a model request.
DEFAULT_RECALL_LIMIT = 64


@dataclass(frozen=True)
class Candidate:
    entry: IndexEntry
    score: float
    evidence: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RouterChoice:
    asset_id: str
    roles: list[str]
    reason: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class RouterSelection:
    form: str | None
    width: int | None
    height: int | None
    selections: list[RouterChoice]
    uv_layout_index: int | None = None
    target_path: str | None = None
    unmet_evidence: list[str] = field(default_factory=list)
    reason: str = ""


def _path_tokens(entry: IndexEntry) -> set[str]:
    text = "%s %s %s %s" % (entry.namespace, entry.resource_path, entry.category, entry.asset_id)
    return {token for token in re.split(r"[^a-zA-Z0-9]+", text.lower()) if token}


def _query_terms(query: str) -> set[str]:
    terms = {token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if token}
    for phrase, aliases in sorted(TOKEN_ALIASES.items(), key=lambda pair: len(pair[0]), reverse=True):
        if phrase in query:
            terms.update(aliases)
    return terms


def _category_intent(query: str) -> str | None:
    lowered = query.lower()
    for category, terms in _CATEGORY_TERMS.items():
        if any(term in lowered for term in terms):
            return category
    return None


def _score(entry: IndexEntry, terms: set[str], category: str | None) -> tuple[float, list[str], list[str]]:
    tokens = _path_tokens(entry)
    path = PurePosixPath(entry.resource_path).stem.lower()
    score = 0.0
    evidence: list[str] = []
    matched: list[str] = []
    for term in sorted(terms):
        if term in tokens:
            score += 4.0
            matched.append(term)
            evidence.append("path:%s" % term)
        elif len(term) >= 4 and term in path:
            score += 1.5
            matched.append(term)
            evidence.append("substring:%s" % term)
    if category:
        if entry.category == category:
            score += 2.5
            evidence.append("category:%s" % category)
        else:
            score -= 0.5
    # Generic dimensional evidence: block textures are usually opaque tiles;
    # entity atlases are larger and item sprites commonly have transparency.
    if category == "block" and entry.alpha.get("opaque_ratio", 0.0) >= 0.95:
        score += 0.5
        evidence.append("opaque_tile")
    if category == "entity" and entry.dimensions.get("width", 0) >= 32:
        score += 0.5
        evidence.append("atlas_size")
    return score, evidence, matched


def retrieve_candidates(
    query: str,
    index: ReferenceIndex,
    limit: int | None = DEFAULT_RECALL_LIMIT,
) -> list[Candidate]:
    if not query.strip():
        raise ValueError("query cannot be blank")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    terms = _query_terms(query)
    category = _category_intent(query)
    scored = []
    for entry in index.entries:
        score, evidence, matched = _score(entry, terms, category)
        scored.append(Candidate(entry, score, evidence, matched))
    scored.sort(key=lambda candidate: (-candidate.score, candidate.entry.resource_path, candidate.entry.asset_id))
    if not scored:
        return []
    positive = [candidate for candidate in scored if candidate.score > 0]
    if limit is None:
        # Explicit ``None`` means the caller requested the complete source
        # catalogue.  Positive matches remain first because ``scored`` is
        # already sorted, followed by the rest of the index.
        return scored
    return (positive or scored)[:limit]


def family_token(name: str, counts: Mapping[str, int], minimum: int = 2) -> str:
    """The head noun a logical asset shares with its siblings.

    The catalogue decides the class, not a hardcoded list: 'leather_helmet',
    'iron_helmet' and 'gold_helmet' all end in a token several assets share, so
    they are labelled 'helmet'. A token only one asset owns yields no family,
    so the label never invents a class of one.
    """
    tokens = [token for token in re.split(r"[^a-z0-9]+", name.lower()) if token]
    if len(tokens) < 2:
        return ""
    last = tokens[-1]
    if last.isdigit():
        return ""
    return last if counts.get(last, 0) >= minimum else ""


def manifest_label(row: Mapping[str, Any]) -> str:
    """The one line the router prompt prints for a candidate.

    The prompt builder and the response parser share this function so a model
    that copies the whole line back still parses.
    """
    label = "%s %s" % (str(row.get("name", "")).strip(), str(row.get("category", "misc")).strip())
    family = str(row.get("family", "") or "").strip()
    return "%s/%s" % (label, family) if family else label


def build_router_manifest(candidates: Sequence[Candidate]) -> list[dict[str, Any]]:
    entries = [candidate.entry for candidate in candidates]
    base_names = [entry.resource_path.rsplit("/", 1)[-1].removesuffix(".png") for entry in entries]
    base_counts = Counter(base_names)
    namespace_category_counts = Counter(
        (entry.namespace, entry.category, base_name)
        for entry, base_name in zip(entries, base_names)
    )
    display_names: list[str] = []
    for entry, base_name in zip(entries, base_names):
        if base_counts[base_name] == 1:
            display_name = base_name
        elif namespace_category_counts[(entry.namespace, entry.category, base_name)] == 1:
            # Namespace/category disambiguation is still compact and keeps
            # the model-facing name independent of the full resource path.
            display_name = "%s:%s:%s" % (entry.namespace, entry.category, base_name)
        else:
            parent = PurePosixPath(entry.resource_path).parent.name or "root"
            display_name = "%s:%s:%s:%s" % (entry.namespace, entry.category, parent, base_name)
        display_names.append(display_name)

    token_lists = [[token for token in re.split(r"[^a-z0-9]+", name.lower()) if token] for name in base_names]
    last_token_counts = Counter(tokens[-1] for tokens in token_lists if len(tokens) > 1)
    families = [family_token(name, last_token_counts) for name in base_names]

    # Show one family at a time. Similar assets sitting next to each other is
    # what makes the list readable and the choice between them explicit.
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            families[index] or "~",
            entries[index].category,
            -candidates[index].score,
            display_names[index],
            entries[index].asset_id,
        ),
    )
    manifest: list[dict[str, Any]] = []
    for rank, index in enumerate(order):
        candidate = candidates[index]
        entry = entries[index]
        base_name = base_names[index]
        display_name = display_names[index]
        manifest.append({
            "rank": rank,
            "asset_id": entry.asset_id,
            "namespace": entry.namespace,
            "resource_path": entry.resource_path,
            "name": display_name,
            "base_name": base_name,
            "family": families[index],
            "category": entry.category,
            "dimensions": entry.dimensions,
            "alpha": {
                "opaque_ratio": entry.alpha.get("opaque_ratio"),
                "bbox": entry.alpha.get("bbox"),
                "components": entry.alpha.get("components"),
            },
            "palette": {
                "count": entry.palette.get("count"),
                "top": list(entry.palette.get("top", []))[:6],
                "luma_range": entry.palette.get("luma_range"),
            },
            "structure": {
                "axis_angle": entry.structure.get("axis_angle"),
                "axis_anisotropy": entry.structure.get("axis_anisotropy"),
                "pattern_density": entry.structure.get("pattern_density"),
                "stroke": entry.structure.get("stroke", {}),
            },
            "local_score": round(candidate.score, 4),
            "evidence": list(candidate.evidence),
            "matched_terms": list(candidate.matched_terms),
        })
    return manifest


def _roles(value: Any) -> list[str]:
    if value is None:
        return [ReferenceRole.PIXEL_STYLE.value, ReferenceRole.MATERIAL.value]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ValueError("roles must be a non-empty list")
    allowed = {role.value for role in ReferenceRole}
    result: list[str] = []
    for item in value:
        role = str(item).strip().lower()
        if role not in allowed:
            raise ValueError("unknown reference role: %s" % role)
        if role not in result:
            result.append(role)
    return result


def parse_router_selection(
    data: dict[str, Any],
    manifest: Sequence[dict[str, Any]],
    max_selected: int = 6,
) -> RouterSelection:
    if not isinstance(data, dict):
        raise ValueError("router response must be an object")
    by_id = {str(item.get("asset_id")): item for item in manifest if item.get("asset_id")}
    by_rank = {int(item.get("rank", index)): str(item["asset_id"]) for index, item in enumerate(manifest) if item.get("asset_id")}
    by_name = {str(item.get("name")): str(item["asset_id"]) for item in manifest if item.get("name") and item.get("asset_id")}
    # The prompt prints compact `name category` lines.  Accepting that whole
    # line in a model response is harmless and avoids rejecting a response
    # merely because the model copied the category token along with the name.
    by_label = {
        manifest_label(item): str(item["asset_id"])
        for item in manifest
        if item.get("name") and item.get("asset_id")
    }
    raw = data.get("selections")
    if raw is None:
        raw = data.get("reference_indices", [])
    if isinstance(raw, (str, int)):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("selections must be a list")
    if len(raw) > max_selected:
        raise ValueError("too many selected references")
    selections: list[RouterChoice] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, int):
            asset_id = by_rank.get(item)
            if asset_id is None:
                raise ValueError("unknown candidate index: %s" % item)
            item = {"asset_id": asset_id}
        elif isinstance(item, str):
            item = {"asset_id": item}
        if not isinstance(item, dict):
            raise ValueError("selection must be an object")
        if not item.get("asset_id") and item.get("name") is not None:
            name = str(item.get("name")).strip()
            asset_id = by_name.get(name)
            if asset_id is None:
                asset_id = by_label.get(name)
            if asset_id is None:
                raise ValueError("unknown candidate name: %s" % name)
            item = dict(item)
            item["asset_id"] = asset_id
        if not item.get("asset_id") and item.get("candidate") is not None:
            try:
                candidate_rank = int(item.get("candidate"))
            except (TypeError, ValueError):
                raise ValueError("candidate must be an integer") from None
            asset_id = by_rank.get(candidate_rank)
            if asset_id is None:
                raise ValueError("unknown candidate index: %s" % candidate_rank)
            item = dict(item)
            item["asset_id"] = asset_id
        asset_id = str(item.get("asset_id", "")).strip()
        if asset_id not in by_id:
            raise ValueError("unknown asset_id: %s" % asset_id)
        if asset_id in seen:
            raise ValueError("duplicate asset_id: %s" % asset_id)
        seen.add(asset_id)
        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                raise ValueError("confidence must be numeric") from None
        selections.append(RouterChoice(asset_id, _roles(item.get("roles")), str(item.get("reason", "")), confidence))
    def _optional_int(name: str) -> int | None:
        value = data.get(name)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError("%s must be an integer" % name) from None
        return parsed if parsed > 0 else None
    layout = data.get("uv_layout_index")
    if layout is not None:
        try:
            layout = int(layout)
        except (TypeError, ValueError):
            raise ValueError("uv_layout_index must be an integer or null") from None
    return RouterSelection(
        form=str(data.get("form")).strip() if data.get("form") is not None else None,
        width=_optional_int("width"),
        height=_optional_int("height"),
        selections=selections,
        uv_layout_index=layout,
        target_path=str(data["target_path"]) if data.get("target_path") else None,
        unmet_evidence=[str(item) for item in data.get("unmet_evidence", [])] if isinstance(data.get("unmet_evidence", []), list) else [],
        reason=str(data.get("reason", "")),
    )


def route_cache_key(
    query: str,
    form: str | None,
    manifest: Sequence[dict[str, Any]],
    index_fingerprint: str,
    model: str | None,
    schema_version: int = 1,
    art_direction: dict[str, Any] | None = None,
) -> str:
    payload = {
        "query": query,
        "form": form,
        "manifest": list(manifest),
        "index_fingerprint": index_fingerprint,
        "model": model,
        "schema_version": schema_version,
        "art_direction": art_direction,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()
