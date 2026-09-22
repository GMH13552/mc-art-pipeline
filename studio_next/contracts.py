"""Stable JSON contracts shared by every stage of the pipeline.

The contracts deliberately encode relationships and measurable constraints instead
of a closed set of item categories. A planner can describe a brand-new object as
parts plus geometry, while the compiler and validators stay deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
import re
from typing import Any


class AssetForm(StrEnum):
    # AUTO is used by the unattended query entry point.  It is resolved to a
    # concrete render contract (item/cross/block_multi/entity_uv/custom) by
    # the planner before geometry compilation; persisted plans always carry a
    # concrete form.
    AUTO = "auto"
    ITEM = "item"
    CROSS = "cross"
    BLOCK_MULTI = "block_multi"
    ENTITY_UV = "entity_uv"
    CUSTOM = "custom"


class ReferenceRole(StrEnum):
    SHAPE = "shape"
    SCALE = "scale"
    MATERIAL = "material"
    PALETTE = "palette"
    PIXEL_STYLE = "pixel_style"
    UV_LAYOUT = "uv_layout"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class AssetRequest:
    query: str
    form: AssetForm = AssetForm.ITEM
    name: str = "generated_asset"
    namespace: str = "demo"
    width: int = 16
    height: int = 16
    novelty: float = 0.6
    target_path: str | None = None
    shape_policy: str = "planned"
    seed: int = 0
    pack_format: int = 15

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be blank")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 <= self.novelty <= 1.0:
            raise ValueError("novelty must be within 0..1")
        if self.shape_policy not in {"planned", "reference", "free", "auto", "target_model_uv"}:
            raise ValueError("shape_policy must be planned/reference/free/auto/target_model_uv")
        if self.pack_format <= 0:
            raise ValueError("pack_format must be positive")


@dataclass(frozen=True)
class ReferenceAsset:
    path: str
    name: str
    roles: list[ReferenceRole] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PartSpec:
    id: str
    meaning: str
    required: bool = True
    # A paint-only part is a model-declared visual feature with no independent
    # alpha geometry.  It must be authored by AppearanceSpec marks/pixel_map.
    # This keeps the contract open: the model decides whether a local feature
    # changes the silhouette or only the surface appearance.
    paint_only: bool = False
    layer: int = 0
    style_role: str = "body"
    recognition_terms: list[str] = field(default_factory=list)
    contour_intent: str = "free"

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("part id must be a simple identifier")
        if self.contour_intent not in {"free", "compact", "elongated", "surface"}:
            raise ValueError("part contour_intent must be free/compact/elongated/surface")


@dataclass(frozen=True)
class ArtDirection:
    """Reference-free, immutable intent established before retrieval.

    This is deliberately a relationship brief rather than a category template.
    It tells every downstream stage what must remain the visual subject, what
    may change, and how any new feature relates to its host.
    """
    target: str
    primary_subject: str
    design_intent: str
    visual_hierarchy: list[str] = field(default_factory=list)
    # Model-authored bridge between the high-level brief and geometry. It is
    # true only when the requested result needs a different alpha silhouette;
    # surface-only motifs and material variants keep it false.
    requires_silhouette_change: bool = False
    # These are a concrete, reference-free change brief. They explain what
    # should visibly change or remain, while geometry and appearance choose
    # the actual native-pixel coordinates after seeing their evidence.
    composition: str = ""
    silhouette_actions: list[str] = field(default_factory=list)
    feature_actions: list[str] = field(default_factory=list)
    surface_actions: list[str] = field(default_factory=list)
    preservation_rules: list[str] = field(default_factory=list)
    feature_relationships: list[str] = field(default_factory=list)
    negative_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.target.strip() or not self.primary_subject.strip() or not self.design_intent.strip():
            raise ValueError("art direction needs target, primary_subject and design_intent")


_OVERLAY_PART_WORDS = {
    "embedded", "inlaid", "inlay", "inserted", "attached", "mounted",
    "encased", "grafted", "motif", "emblem", "boss", "socket", "gem",
    "eye", "eyeball", "detail", "accent", "ornament", "镶嵌", "嵌入",
}


def is_overlay_part(part: PartSpec) -> bool:
    """Return whether a part is a newly authored local overlay.

    The check is intentionally role-driven.  A host described as
    ``host support for the eye`` remains a support; the word ``eye`` in that
    sentence must not turn the host into the overlay itself.
    """
    role = str(part.style_role or "").strip().lower()
    # The part's declared role outranks incidental feature words in its
    # description. A blade that "hosts an eye" is still the silhouette
    # support; reading the word eye from its prose must not turn the whole
    # blade into an overlay/paint-only part.
    if any(cue in role for cue in (
        "host", "support", "silhouette", "structural", "junction", "continuation",
    )):
        return False
    if role in {"support", "primary support", "body", "handle", "grip", "shaft", "stem"}:
        return False
    text = " ".join((str(part.id), str(part.meaning), role)).lower()
    words = set(re.findall(r"[a-z]+", text))
    return bool(words & _OVERLAY_PART_WORDS) or any(token in text for token in ("镶嵌", "嵌入"))


@dataclass(frozen=True)
class ShapeDescriptor:
    target: str
    semantic: str
    visual_identity: list[str]
    parts: list[PartSpec]
    negative_identities: list[str] = field(default_factory=list)
    orientation: str = "auto"
    reference_strategy: str = "learn_structure_and_style"
    # The planner decides whether this request is a recolour/detail pass, a
    # local reference edit, or a new contour. The renderer only executes it.
    shape_edit_mode: str = "model_decides"
    # Optional model-authored ownership evidence for a same-size reference.
    # The map is deliberately opaque to the renderer: it helps the geometry
    # planner partition a union alpha silhouette into physical parts, while
    # leaving the final contour and any local variant edit to the model.
    reference_part_map: dict[str, Any] | None = None
    # Optional model-authored target ownership evidence.  Unlike
    # ``reference_part_map`` this map may place a newly requested local motif
    # on pixels that are transparent in the source.  It is still only
    # planning evidence: the geometry model owns the final contour.
    target_part_map: dict[str, Any] | None = None
    # Attached by the orchestration layer after the reference-free director
    # pass. Geometry, appearance and review see the same immutable brief.
    art_direction: ArtDirection | None = None

    def __post_init__(self) -> None:
        ids = [part.id for part in self.parts]
        if not self.target.strip() or not self.semantic.strip():
            raise ValueError("shape descriptor needs target and semantic")
        if not self.parts:
            raise ValueError("shape descriptor needs at least one part")
        if len(ids) != len(set(ids)):
            raise ValueError("shape descriptor contains duplicate part ids")
        if self.shape_edit_mode not in {
            "model_decides", "appearance_only", "preserve_silhouette",
            "local_silhouette_edit", "new_silhouette",
        }:
            raise ValueError("shape_edit_mode is not a supported open edit mode")


@dataclass(frozen=True)
class PrimitiveSpec:
    id: str
    part_id: str
    primitive: str
    params: dict[str, Any]
    layer: int = 0


@dataclass(frozen=True)
class ConnectionSpec:
    a: str
    b: str
    required: bool = True


@dataclass(frozen=True)
class ConstraintSpec:
    """A generic, measurable requirement emitted at runtime by a planner."""

    metric: str
    minimum: float | None = None
    maximum: float | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("constraint needs minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("constraint minimum cannot exceed maximum")


@dataclass(frozen=True)
class UvRegionSpec:
    """A labelled target rectangle in a supplied entity texture layout.

    This is deliberately a layout description, not a hard-coded player or
    villager template. A caller can bring any model's UV map and bind its
    semantic parts to the regions actually used by that model.
    """

    id: str
    part_id: str
    bbox: list[int]
    face: str = "custom"
    required: bool = True
    notes: str = ""
    preview_origin: list[int] | None = None
    preview_size: list[int] | None = None
    preview_layer: int = 0
    preview_instances: list[list[int]] | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.part_id:
            raise ValueError("UV region needs id and part_id")
        if len(self.bbox) != 4:
            raise ValueError("UV region bbox must be [left, top, right, bottom]")
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("UV region bbox must have positive area")
        if self.preview_origin is not None and len(self.preview_origin) != 2:
            raise ValueError("UV preview_origin must be [x, y]")
        if self.preview_size is not None and (len(self.preview_size) != 2 or min(self.preview_size) <= 0):
            raise ValueError("UV preview_size must contain positive [width, height]")
        if self.preview_instances is not None:
            if not self.preview_instances or any(len(item) != 4 or min(item[2:]) <= 0 for item in self.preview_instances):
                raise ValueError("UV preview_instances must contain [x, y, width, height] boxes")


@dataclass(frozen=True)
class GeometrySpec:
    width: int
    height: int
    parts: list[PartSpec]
    primitives: list[PrimitiveSpec]
    connections: list[ConnectionSpec] = field(default_factory=list)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    uv_regions: list[UvRegionSpec] = field(default_factory=list)
    background_transparent: bool = True

    def __post_init__(self) -> None:
        known = {part.id for part in self.parts}
        if len(known) != len(self.parts):
            raise ValueError("geometry contains duplicate part ids")
        unknown = {primitive.part_id for primitive in self.primitives} - known
        if unknown:
            raise ValueError("primitive refers to unknown part(s): %s" % ", ".join(sorted(unknown)))
        connection_parts = {endpoint for connection in self.connections for endpoint in (connection.a, connection.b)}
        unknown_connections = connection_parts - known
        if unknown_connections:
            raise ValueError(
                "connection refers to unknown part(s): %s" % ", ".join(sorted(unknown_connections))
            )
        primitive_ids = [primitive.id for primitive in self.primitives]
        if len(primitive_ids) != len(set(primitive_ids)):
            raise ValueError("geometry contains duplicate primitive ids")
        uv_ids = [region.id for region in self.uv_regions]
        if len(uv_ids) != len(set(uv_ids)):
            raise ValueError("geometry contains duplicate UV region ids")
        unknown_uv_parts = {region.part_id for region in self.uv_regions} - known
        if unknown_uv_parts:
            raise ValueError("UV region refers to unknown part(s): %s" % ", ".join(sorted(unknown_uv_parts)))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("geometry dimensions must be positive")
        for region in self.uv_regions:
            left, top, right, bottom = region.bbox
            if left < 0 or top < 0 or right > self.width or bottom > self.height:
                raise ValueError("UV region %s lies outside geometry canvas" % region.id)


@dataclass(frozen=True)
class PartAppearance:
    colors: list[str]
    material: str = "generic"
    shade_axis: str = "auto"
    noise: float = 0.0
    highlight_ratio: float = 0.18
    marks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.colors:
            raise ValueError("part appearance needs at least one color")
        if not 0.0 <= self.noise <= 1.0:
            raise ValueError("appearance noise must be within 0..1")
        if not 0.0 <= self.highlight_ratio <= 1.0:
            raise ValueError("appearance highlight_ratio must be within 0..1")


@dataclass(frozen=True)
class AppearanceSpec:
    palette: dict[str, str]
    parts: dict[str, PartAppearance]
    outline_color: str | None = None
    outline_width: int = 0
    transparent_background: bool = True
    # Optional value/palette-band transfer from a pixel-style/material
    # reference. ``pattern`` preserves discrete source bands; together with
    # ``motif_policy=reference_locked`` it also keeps unruled opaque source
    # pixels exact. The geometry and alpha remain fully locked by the mask.
    reference_sampling: str = "none"
    # Model-authored local material instructions. Python validates region IDs
    # and applies these rules, but does not infer that a region is a nose, ear,
    # gem, seam or any other semantic feature.
    region_rules: list[dict[str, Any]] = field(default_factory=list)
    # Optional per-part override for reference transfer. A composite asset
    # can keep a source support in ``pattern`` mode while letting a newly
    # authored inset, emblem or other motif use its own palette in ``none``
    # mode. An absent entry inherits the global reference_sampling value.
    part_reference_sampling: dict[str, str] = field(default_factory=dict)
    # Optional model-authored binding from a part to one of the named
    # references supplied for this request.  This lets a composite asset use
    # a structural source for its supports and a separate material/palette
    # source for a newly authored surface without any object-specific Python
    # routing.  Unknown names simply fall back to the global source.
    part_reference_sources: dict[str, str] = field(default_factory=dict)
    # How a named source participates in the part: ``overlay`` keeps the
    # structural/global raster as a base and lays down the source's salient
    # pixel clusters; ``replace`` lets the named raster provide the full ramp.
    part_reference_composite: dict[str, str] = field(default_factory=dict)
    # The model chooses whether local marks are copied from the reference or
    # deliberately authored. This keeps motif ownership explicit for any
    # asset family instead of inferring it from object names in Python.
    motif_policy: str = "free"
    # Optional model-authored whole-canvas colour composition. ``.`` delegates
    # to normal rendering; another symbol resolves through ``legend``. It is
    # RGB-only, so this open raster contract cannot alter alpha or silhouette.
    pixel_map: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outline_width not in {0, 1}:
            raise ValueError("outline_width currently supports 0 or 1 pixels")
        if self.reference_sampling not in {"none", "value", "pattern"}:
            raise ValueError("reference_sampling must be none, value or pattern")
        if self.motif_policy not in {"free", "reference_locked", "model_authored", "none"}:
            raise ValueError("motif_policy must be free, reference_locked, model_authored or none")
        # A non-string mode must be reported as invalid data, never as a
        # TypeError from the membership test below.
        invalid_sampling = {
            part_id: mode
            for part_id, mode in self.part_reference_sampling.items()
            if not isinstance(mode, str) or mode not in {"none", "value", "pattern"}
        }
        if invalid_sampling:
            raise ValueError("part_reference_sampling values must be none, value or pattern")
        invalid_composite = {
            part_id: mode
            for part_id, mode in self.part_reference_composite.items()
            if not isinstance(mode, str) or mode not in {"overlay", "replace"}
        }
        if invalid_composite:
            raise ValueError("part_reference_composite values must be overlay or replace")
        if self.pixel_map is not None:
            if not isinstance(self.pixel_map, dict):
                raise ValueError("pixel_map must be an object when present")
            legend = self.pixel_map.get("legend")
            rows = self.pixel_map.get("rows")
            if not isinstance(legend, dict) or not isinstance(rows, list):
                raise ValueError("pixel_map needs legend and rows")
            if any(len(str(symbol)) != 1 or str(symbol) == "." for symbol in legend):
                raise ValueError("pixel_map legend keys must be one non-dot character")
            if any(not isinstance(row, str) for row in rows):
                raise ValueError("pixel_map rows must be strings")


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    stage: str
    metrics: dict[str, float | int | str | bool]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    """Convert contracts, enums and nested values into JSON-safe primitives."""
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def part_from_dict(data: dict[str, Any]) -> PartSpec:
    normalized = dict(data)
    # Vision models sometimes reuse the primitive spelling ``part_id`` (or a
    # display ``name``) when returning the part table.  These are unambiguous
    # aliases for the canonical identifier, so normalize them at the contract
    # boundary instead of failing an otherwise usable plan.
    if "id" not in normalized:
        for alias in ("part_id", "name"):
            if str(normalized.get(alias, "")).strip():
                normalized["id"] = normalized[alias]
                break
    # ``role`` is a common compact spelling for the appearance/style role in
    # model-authored JSON. It is unambiguous at this boundary and keeps the
    # small final-authoring prompt free of a large field glossary.
    if "style_role" not in normalized and str(normalized.get("role", "")).strip():
        normalized["style_role"] = normalized["role"]
    normalized.pop("role", None)
    if "meaning" not in normalized and str(normalized.get("description", "")).strip():
        normalized["meaning"] = normalized["description"]
    normalized.pop("description", None)
    if "meaning" not in normalized and str(normalized.get("kind", "")).strip():
        normalized["meaning"] = normalized["kind"]
    normalized.pop("kind", None)
    # Keep appearance-only annotations out of the geometry contract. Vision
    # models often place ``color``/``material``/``notes`` beside a part even
    # when the schema asks for them in AppearanceSpec; ignoring those extras is
    # safer than rejecting an otherwise usable raster plan.
    normalized = {
        key: value
        for key, value in normalized.items()
        if key in {"id", "meaning", "required", "paint_only", "layer", "style_role", "recognition_terms", "contour_intent"}
    }
    normalized.pop("part_id", None)
    normalized.pop("name", None)
    # Keep malformed vision responses recoverable without inventing geometry:
    # the identifier is a useful concise meaning until the descriptor/UV
    # binding stage supplies the authoritative model-part description.
    if not str(normalized.get("meaning", "")).strip():
        normalized["meaning"] = str(normalized.get("id", "part"))
    return PartSpec(**normalized)


def art_direction_from_dict(data: dict[str, Any]) -> ArtDirection:
    return ArtDirection(
        target=str(data.get("target", "")),
        primary_subject=str(data.get("primary_subject", "")),
        design_intent=str(data.get("design_intent", "")),
        visual_hierarchy=[str(item) for item in data.get("visual_hierarchy", []) if str(item).strip()],
        requires_silhouette_change=bool(data.get("requires_silhouette_change", False)),
        composition=str(data.get("composition", "")),
        silhouette_actions=[str(item) for item in data.get("silhouette_actions", []) if str(item).strip()],
        feature_actions=[str(item) for item in data.get("feature_actions", []) if str(item).strip()],
        surface_actions=[str(item) for item in data.get("surface_actions", []) if str(item).strip()],
        preservation_rules=[str(item) for item in data.get("preservation_rules", []) if str(item).strip()],
        feature_relationships=[str(item) for item in data.get("feature_relationships", []) if str(item).strip()],
        negative_constraints=[str(item) for item in data.get("negative_constraints", []) if str(item).strip()],
    )


def request_from_dict(data: dict[str, Any]) -> AssetRequest:
    normalized = dict(data)
    normalized["form"] = AssetForm(normalized.get("form", AssetForm.ITEM))
    return AssetRequest(**normalized)


def descriptor_from_dict(data: dict[str, Any]) -> ShapeDescriptor:
    normalized = dict(data)
    normalized["parts"] = [part_from_dict(item) for item in data["parts"]]
    if isinstance(normalized.get("art_direction"), dict):
        normalized["art_direction"] = art_direction_from_dict(normalized["art_direction"])
    edit_mode = str(normalized.get("shape_edit_mode", "model_decides")).strip().lower()
    aliases = {
        "appearance": "appearance_only",
        "recolor": "appearance_only",
        "recolour": "appearance_only",
        "preserve": "preserve_silhouette",
        "local": "local_silhouette_edit",
        "local_edit": "local_silhouette_edit",
        "new": "new_silhouette",
    }
    normalized["shape_edit_mode"] = aliases.get(edit_mode, edit_mode)
    if normalized["shape_edit_mode"] not in {
        "model_decides", "appearance_only", "preserve_silhouette",
        "local_silhouette_edit", "new_silhouette",
    }:
        normalized["shape_edit_mode"] = "model_decides"
    return ShapeDescriptor(**normalized)


def geometry_from_dict(data: dict[str, Any]) -> GeometrySpec:
    normalized = dict(data)
    normalized["parts"] = [part_from_dict(item) for item in data["parts"]]
    known_part_ids = {part.id for part in normalized["parts"]}
    primitives: list[PrimitiveSpec] = []
    for index, raw in enumerate(data["primitives"], start=1):
        item = dict(raw)
        # `type` and `part` are conventional names in drawing DSLs. A model
        # sometimes emits either alongside our canonical names; accept the
        # spelling without creating a second, target-specific format.
        if "primitive" not in item and "type" in item:
            item["primitive"] = item["type"]
        item.pop("type", None)
        if "part_id" not in item and "part" in item:
            item["part_id"] = item["part"]
        item.pop("part", None)
        if "part_id" not in item:
            # A one-part plan has an unambiguous owner even when a model
            # forgets to repeat it on every primitive.  For multi-part plans,
            # accept the same inference only when the primitive id carries a
            # unique declared-part prefix; ambiguous responses still fail
            # loudly instead of silently painting the wrong component.
            primitive_id = str(item.get("id", ""))
            if len(known_part_ids) == 1:
                item["part_id"] = next(iter(known_part_ids))
            else:
                matching_parts = [
                    part_id for part_id in known_part_ids
                    if primitive_id == part_id or primitive_id.startswith(part_id + "_")
                ]
                if len(matching_parts) == 1:
                    item["part_id"] = matching_parts[0]
        if "params" not in item and "parameters" in item:
            item["params"] = item.pop("parameters")
        reserved = {"id", "part_id", "primitive", "layer", "params"}
        extras = {key: value for key, value in item.items() if key not in reserved}
        params = item.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("primitive params must be an object")
        item["params"] = {**extras, **params}
        item = {key: value for key, value in item.items() if key in reserved}
        if "id" not in item:
            item["id"] = "%s_%s_%d" % (
                str(item.get("part_id", "part")),
                str(item.get("primitive", "primitive")),
                index,
            )
        primitives.append(PrimitiveSpec(**item))
    normalized["primitives"] = primitives
    primitive_part_by_id = {
        str(item.get("id")): str(item.get("part_id", item.get("part", "")))
        for item in data.get("primitives", [])
        if isinstance(item, dict) and item.get("id")
    }

    def normalize_connection_endpoint(raw: Any) -> Any:
        endpoint = str(raw)
        if endpoint in known_part_ids:
            return endpoint
        # Vision planners occasionally copy a primitive id (`head_base`) into
        # a connection. Map it only when the primitive itself identifies one
        # declared part; ambiguous or unknown names still fail the contract.
        mapped = primitive_part_by_id.get(endpoint)
        return mapped if mapped in known_part_ids else raw

    normalized["connections"] = [
        ConnectionSpec(
            **{
                **dict(item),
                "a": normalize_connection_endpoint(item.get("a")),
                "b": normalize_connection_endpoint(item.get("b")),
            }
        )
        for item in data.get("connections", [])
    ]
    normalized["constraints"] = [ConstraintSpec(**item) for item in data.get("constraints", [])]
    normalized["uv_regions"] = [UvRegionSpec(**item) for item in data.get("uv_regions", [])]
    return GeometrySpec(**normalized)


def appearance_from_dict(data: dict[str, Any]) -> AppearanceSpec:
    normalized = dict(data)
    # The canonical wire format is a named colour mapping, but vision models
    # quite reasonably sometimes return a compact ordered palette after an
    # audit pass.  That representation contains the same colour evidence and
    # is unambiguous, so normalize it at the JSON boundary instead of asking a
    # later renderer to interpret a list as a mapping (or abandoning the whole
    # quality run).  Generated names are intentionally neutral: styles and
    # pixel-map legends may continue to use literal hex values.
    raw_palette = data.get("palette", {})
    if isinstance(raw_palette, (list, tuple)):
        normalized["palette"] = {
            "color_%d" % (index + 1): str(color)
            for index, color in enumerate(raw_palette)
            if str(color).strip()
        }
    elif isinstance(raw_palette, dict):
        normalized["palette"] = {
            str(name): str(color)
            for name, color in raw_palette.items()
            if str(name).strip() and str(color).strip()
        }
    else:
        raise ValueError("palette must be an object or an ordered colour list")
    # `model_authored` is a valid motif policy but not a sampling mode.  Models
    # commonly use it to mean that a new feature must not sample a reference;
    # translate that schema alias to the equivalent renderer mode rather than
    # discarding an otherwise valid appearance plan.
    def _normalize_mode_map(
        field: str, model_authored: str, allowed: frozenset[str]
    ) -> dict[str, Any] | None:
        """Accept a bare mode string or the richer {mode, source} object.

        The canonical wire format is a bare mode string, but a model commonly
        returns {"mode": "replace", "source": "layer:file"}. Both express the
        same decision, so keep the mode and fold the source into the source map
        instead of failing the whole run with a type error.
        """
        raw = data.get(field)
        if not isinstance(raw, dict):
            return None
        modes: dict[str, Any] = {}
        sources = dict(normalized.get("part_reference_sources") or {})
        for part_id, value in raw.items():
            key = str(part_id)
            if isinstance(value, dict):
                mode = value.get("mode") or value.get("type") or ""
                source = value.get("source")
            else:
                mode = value
                source = None
            if source is not None and key not in sources:
                sources[key] = str(source)
            resolved = model_authored if mode == "model_authored" else mode
            if not isinstance(resolved, str) or resolved not in allowed:
                # An unrecognised mode is a model typo, not a reason to discard
                # the whole asset: leave that part unset and keep rendering.
                continue
            modes[key] = resolved
        if sources:
            normalized["part_reference_sources"] = sources
        return modes

    sampling = _normalize_mode_map(
        "part_reference_sampling", "none", frozenset({"none", "value", "pattern"})
    )
    if sampling is not None:
        normalized["part_reference_sampling"] = sampling
    composite = _normalize_mode_map(
        "part_reference_composite", "overlay", frozenset({"overlay", "replace"})
    )
    if composite is not None:
        normalized["part_reference_composite"] = composite
    raw_parts = data.get("parts")
    if isinstance(raw_parts, list):
        # A model sometimes returns parts as an ordered list of objects with an
        # explicit id instead of a mapping. That carries the same information,
        # so key it by id instead of failing the whole quality run.
        keyed_parts: dict[str, Any] = {}
        for index, item in enumerate(raw_parts):
            if not isinstance(item, dict):
                raise ValueError("parts list entries must be objects")
            key = item.get("id") or item.get("part_id") or item.get("name")
            if not key:
                raise ValueError("parts list entry %d needs an id" % index)
            keyed_parts[str(key)] = {
                name: value for name, value in item.items()
                if name not in {"id", "part_id", "name"}
            }
        raw_parts = keyed_parts
    if not isinstance(raw_parts, dict):
        raise ValueError("parts must be an object keyed by part id")
    try:
        normalized["parts"] = {
            str(key): PartAppearance(**item) for key, item in raw_parts.items()
        }
    except TypeError as exc:
        # A descriptor-shaped entry (meaning/required/paint_only) is model data
        # in the wrong schema; report it as invalid data so the caller's retry
        # and fallback logic can handle it instead of dying on an AttributeError.
        raise ValueError("invalid part appearance entry: %s" % exc) from exc
    return AppearanceSpec(**normalized)
