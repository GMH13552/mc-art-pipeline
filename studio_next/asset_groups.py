"""Logical asset grouping: one name, all of its textures.

A block is not a file. 'oak_log' is a blockstate that resolves through a model
parent chain to two textures, and a large mod such as AoA3 ships 1403
blockstates for 3701 textures. A router that has to choose a reference cannot
reason about 'log_oak_top'; it can reason about 'oak_log'.

This module builds that name-level catalogue from one or more asset roots
(vanilla JAR, mod JAR, resource pack, or a mod's src/main/resources tree). The
catalogue is deliberately image-free: it reads blockstate and model JSON only.
Image bytes are fetched later, and only for the handful of names a planner
actually selects.

Nothing here is persisted. Callers rebuild the catalogue from the current roots
on demand, which is what keeps generated art aligned with a mod's existing art
instead of drifting every time a new run starts.

Two known limits, both deliberately handled rather than hidden:

* Entity textures have no data-driven mapping in 1.12.2 (the mapping lives in
  the entity renderer's Java code), so entity groups fall back to a documented
  prefix heuristic.
* A few texture families are code-driven: 'clock' has 64 frames but its item
  model references only 'clock_00'. Numeric-suffix families are expanded so
  those siblings are not lost.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile


CATEGORY_BLOCK = "block"
CATEGORY_ITEM = "item"
CATEGORY_ENTITY = "entity"
CATEGORY_TEXTURE = "texture"

CATEGORIES = (CATEGORY_BLOCK, CATEGORY_ITEM, CATEGORY_ENTITY, CATEGORY_TEXTURE)

# Model paths are namespace-relative and rooted at one of these two folders.
_MODEL_ROOTS = frozenset({"block", "item"})
_NUMERIC_FAMILY = re.compile(r"^(?P<base>.+?)_(?P<number>\d{1,3})$")
_MAX_PARENT_DEPTH = 16


def _split_ref(reference: str, default_namespace: str) -> tuple[str, str]:
    """Split a namespace:path reference and drop any blockstate/model state."""
    value = str(reference).split("[", 1)[0].strip()
    if ":" in value:
        namespace, _, rest = value.partition(":")
        return namespace, rest
    return default_namespace, value


def _entity_group_name(relative: str) -> str:
    """Group an entity texture by folder, else by its leading name token.

    'entity/cow.png' -> 'cow'; 'entity/horse_black.png' -> 'horse';
    'entity/banner/base.png' -> 'banner'. This is a heuristic because the real
    entity to texture mapping is compiled into the renderer, not shipped as
    data.
    """
    parts = relative.split("/")
    if len(parts) > 1:
        return parts[0]
    stem = parts[0][:-4] if parts[0].endswith(".png") else parts[0]
    return stem.split("_", 1)[0]


class AssetRoot:
    """A read-only view over one asset root: a JAR or a directory tree.

    The underlying ZipFile stays open for the lifetime of the root. That
    matters: re-opening a JAR re-parses its entire central directory (45 ms for
    the vanilla 1.12.2 JAR, 93 ms for AoA3), which is the difference between a
    one-second scan and a five-minute one.
    """

    def __init__(self, path: str | Path) -> None:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError("asset root does not exist: %s" % source)
        self.path = source
        self._bundle: ZipFile | None = None
        self._keys: set[str] = set()
        if source.is_dir():
            for item in source.rglob("*"):
                if item.is_file():
                    self._keys.add(item.relative_to(source).as_posix())
        elif source.suffix.lower() == ".jar":
            bundle = ZipFile(source)
            self._bundle = bundle
            self._keys = {name for name in bundle.namelist() if not name.endswith("/")}
        else:
            raise ValueError("asset root must be a .jar or a directory: %s" % source)

    @property
    def kind(self) -> str:
        return "directory" if self._bundle is None else "jar"

    def keys(self) -> set[str]:
        return self._keys

    def read(self, resource_path: str) -> bytes | None:
        if resource_path not in self._keys:
            return None
        if self._bundle is not None:
            return self._bundle.read(resource_path)
        return (self.path / resource_path).read_bytes()

    def close(self) -> None:
        if self._bundle is not None:
            self._bundle.close()
            self._bundle = None

    def __enter__(self) -> "AssetRoot":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "AssetRoot(%s, %s, %d keys)" % (self.path, self.kind, len(self._keys))


def open_roots(sources: Iterable[str | Path]) -> list[AssetRoot]:
    """Open every source, expanding a version or resource folder into its JARs."""
    roots: list[AssetRoot] = []
    for source in sources:
        candidate = Path(source).expanduser()
        if candidate.is_dir() and not (candidate / "assets").is_dir():
            jars = sorted(candidate.glob("*.jar"))
            if jars:
                roots.extend(AssetRoot(jar) for jar in jars)
                continue
        roots.append(AssetRoot(candidate))
    return roots


@dataclass(frozen=True)
class TextureRef:
    """One texture owned by a group, with its optional animation metadata."""

    namespace: str
    resource_path: str
    animation: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return PurePosixPath(self.resource_path).name[: -len(".png")]

    @property
    def animated(self) -> bool:
        return bool(self.animation and "frametime" in self.animation)


@dataclass(frozen=True)
class AssetGroup:
    """One selectable logical asset and every texture that belongs to it."""

    name: str
    namespace: str
    category: str
    textures: tuple[TextureRef, ...] = ()
    models: tuple[str, ...] = ()
    origins: tuple[str, ...] = ()
    note: str = ""

    @property
    def asset_id(self) -> str:
        return "%s:%s/%s" % (self.namespace, self.category, self.name)

    @property
    def texture_names(self) -> tuple[str, ...]:
        return tuple(texture.name for texture in self.textures)

    def to_manifest_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "asset_id": self.asset_id,
            "name": self.name,
            "namespace": self.namespace,
            "category": self.category,
            "texture_count": len(self.textures),
            "textures": list(self.texture_names),
        }
        if self.note:
            row["note"] = self.note
        animated = [texture.name for texture in self.textures if texture.animated]
        if animated:
            row["animated"] = animated
        return row

    def describe(self) -> str:
        return "%s [%s] %d tex: %s" % (
            self.asset_id,
            self.category,
            len(self.textures),
            ", ".join(self.texture_names),
        )


@dataclass
class GroupCatalogue:
    """A name-level, image-free view of every asset root that was scanned."""

    groups: dict[str, AssetGroup] = field(default_factory=dict)
    roots: list[AssetRoot] = field(default_factory=list)
    owner: dict[str, AssetRoot] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    def group(self, asset_id: str) -> AssetGroup:
        if asset_id in self.groups:
            return self.groups[asset_id]
        matches = [item for item in self.groups.values() if item.name == asset_id]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError("unknown asset group: %s" % asset_id)
        raise KeyError(
            "ambiguous asset name %s: %s" % (asset_id, [item.asset_id for item in matches])
        )

    def select(
        self,
        category: str | None = None,
        namespace: str | None = None,
        contains: str | None = None,
        minimum_textures: int = 0,
    ) -> list[AssetGroup]:
        needle = contains.lower() if contains else None
        chosen = [
            group
            for group in self.groups.values()
            if (category is None or group.category == category)
            and (namespace is None or group.namespace == namespace)
            and (needle is None or needle in group.name.lower() or needle in group.asset_id.lower())
            and len(group.textures) >= minimum_textures
        ]
        return sorted(chosen, key=lambda group: group.asset_id)

    def to_manifest(self) -> list[dict[str, Any]]:
        """Router-facing catalogue: names and texture names, never pixels."""
        return [group.to_manifest_row() for group in self.select()]

    def read(self, resource_path: str) -> bytes | None:
        root = self.owner.get(resource_path)
        return root.read(resource_path) if root is not None else None

    def extract(self, asset_id: str, destination: str | Path) -> list[Path]:
        """Write every texture of one group to a destination directory.

        This is the only place image bytes are touched, and it happens after a
        name has already been chosen.
        """
        group = self.group(asset_id)
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        used: dict[str, str] = {}
        for texture in group.textures:
            data = self.read(texture.resource_path)
            if data is None:
                continue
            stem = texture.name
            if stem in used and used[stem] != texture.resource_path:
                index = 2
                while "%s_%d" % (stem, index) in used:
                    index += 1
                stem = "%s_%d" % (stem, index)
            used[stem] = texture.resource_path
            path = target / (stem + ".png")
            path.write_bytes(data)
            written.append(path)
        return written

    def close(self) -> None:
        for root in self.roots:
            root.close()

    def __enter__(self) -> "GroupCatalogue":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _register(
    groups: dict[str, AssetGroup],
    claimed: set[str],
    namespace: str,
    category: str,
    name: str,
    texture_paths: Iterable[str],
    models: Iterable[str],
    origin: str,
    animations: Any,
    note: str = "",
) -> None:
    asset_id = "%s:%s/%s" % (namespace, category, name)
    wanted = sorted(set(texture_paths))
    existing = groups.get(asset_id)
    if existing is not None:
        wanted = sorted(set(wanted) | {texture.resource_path for texture in existing.textures})
        models = list(models) + list(existing.models)
        origin_list = list(existing.origins)
        if origin not in origin_list:
            origin_list.append(origin)
        note = note or existing.note
    else:
        origin_list = [origin]
    textures = tuple(
        TextureRef(
            namespace=path.split("/")[1],
            resource_path=path,
            animation=animations(path),
        )
        for path in wanted
    )
    groups[asset_id] = AssetGroup(
        name=name,
        namespace=namespace,
        category=category,
        textures=textures,
        models=tuple(dict.fromkeys(models)),
        origins=tuple(origin_list),
        note=note,
    )
    claimed.update(wanted)


def build_catalogue(
    sources: Iterable[str | Path],
    *,
    include_entities: bool = True,
    include_orphans: bool = True,
) -> GroupCatalogue:
    """Resolve every logical asset name reachable from the given sources.

    Later sources win, exactly like a resource pack stack, so a mod's own
    namespace can override a vanilla name without any special casing.
    """
    roots = open_roots(sources)
    catalogue = GroupCatalogue(roots=roots)
    try:
        owner: dict[str, AssetRoot] = {}
        for root in roots:
            for key in root.keys():
                owner[key] = root
        catalogue.owner = owner

        def load_json(resource_path: str) -> Any:
            data = catalogue.read(resource_path)
            if data is None:
                return None
            try:
                return json.loads(data)
            except ValueError:
                return None

        # texture path -> sibling .mcmeta, without reading either one yet
        animation_index = {
            key[: -len(".mcmeta")]: key for key in owner if key.endswith(".mcmeta")
        }
        animation_cache: dict[str, dict[str, Any] | None] = {}

        def animation_for(texture_path: str) -> dict[str, Any] | None:
            meta = animation_index.get(texture_path)
            if meta is None:
                return None
            if meta not in animation_cache:
                document = load_json(meta) or {}
                info = document.get("animation")
                if isinstance(info, dict):
                    frames = info.get("frames")
                    animation_cache[meta] = {
                        "frametime": info.get("frametime", 1),
                        "interpolate": bool(info.get("interpolate")),
                        "frames": len(frames) if isinstance(frames, list) else None,
                    }
                elif isinstance(document.get("texture"), dict):
                    animation_cache[meta] = {"texture_properties": sorted(document["texture"])}
                else:
                    animation_cache[meta] = {}
            return animation_cache[meta] or None

        model_cache: dict[tuple[str, str], Any] = {}

        def model_resolve(namespace: str, relative: str) -> tuple[list[str], list[str]]:
            """Walk a model's parent chain; return texture paths and model ids."""
            texture_paths: list[str] = []
            visited: list[str] = []
            seen: set[str] = set()
            node, current_namespace, depth = relative, namespace, 0
            while node and node not in seen and depth < _MAX_PARENT_DEPTH:
                seen.add(node)
                depth += 1
                key = (current_namespace, node)
                if key not in model_cache:
                    model_cache[key] = load_json(
                        "assets/%s/models/%s.json" % (current_namespace, node)
                    )
                document = model_cache[key]
                if document is None:
                    break
                visited.append("%s/%s" % (current_namespace, node))
                for value in (document.get("textures") or {}).values():
                    if isinstance(value, str) and "#" not in value:
                        texture_namespace, texture_relative = _split_ref(value, current_namespace)
                        texture_paths.append(
                            "assets/%s/textures/%s.png" % (texture_namespace, texture_relative)
                        )
                parent = document.get("parent") or ""
                parent_namespace, parent_relative = _split_ref(parent, current_namespace)
                node = (
                    parent_relative
                    if parent_relative.split("/", 1)[0] in _MODEL_ROOTS
                    else None
                )
                current_namespace = parent_namespace
            return texture_paths, visited

        groups = catalogue.groups
        claimed: set[str] = set()
        animations = animation_for

        # ---- blocks: blockstate -> model chain -> textures ----
        blockstate_keys = sorted(
            key
            for key in owner
            if key.startswith("assets/") and "/blockstates/" in key and key.endswith(".json")
        )
        for path in blockstate_keys:
            namespace = path.split("/")[1]
            name = path.split("/blockstates/", 1)[1][: -len(".json")]
            document = load_json(path)
            if not isinstance(document, dict):
                continue
            references: list[str] = []

            def walk(node: Any) -> None:
                if isinstance(node, list):
                    for item in node:
                        walk(item)
                elif isinstance(node, dict):
                    if "model" in node:
                        references.append(node["model"])
                    if "apply" in node:
                        walk(node["apply"])

            for variant in (document.get("variants") or {}).values():
                walk(variant)
            walk(document.get("multipart") or [])

            texture_paths: list[str] = []
            models: list[str] = []
            for reference in references:
                model_namespace, model_relative = _split_ref(reference, namespace)
                if model_relative.split("/", 1)[0] not in _MODEL_ROOTS:
                    model_relative = "block/" + model_relative
                found_textures, found_models = model_resolve(model_namespace, model_relative)
                texture_paths.extend(found_textures)
                models.extend(found_models)
            _register(
                groups, claimed, namespace, CATEGORY_BLOCK, name,
                texture_paths, models, path, animations,
            )

        # ---- items: models/item -> parent chain -> layer0..N ----
        item_keys = sorted(
            key
            for key in owner
            if key.startswith("assets/") and "/models/item/" in key and key.endswith(".json")
        )
        for path in item_keys:
            namespace = path.split("/")[1]
            name = path.split("/models/item/", 1)[1][: -len(".json")]
            texture_paths, models = model_resolve(namespace, "item/" + name)
            _register(
                groups, claimed, namespace, CATEGORY_ITEM, name,
                texture_paths, models, path, animations,
            )

        # ---- entities: prefix heuristic, documented as such ----
        if include_entities:
            entity_paths: dict[tuple[str, str], list[str]] = {}
            for key in owner:
                if not key.startswith("assets/") or "/textures/entity/" not in key:
                    continue
                if not key.endswith(".png"):
                    continue
                namespace = key.split("/")[1]
                relative = key.split("/textures/entity/", 1)[1]
                entity_paths.setdefault((namespace, _entity_group_name(relative)), []).append(key)
            for (namespace, name), paths in entity_paths.items():
                _register(
                    groups, claimed, namespace, CATEGORY_ENTITY, name,
                    paths, [], "textures/entity", animations,
                    note="entity grouping is a name heuristic; 1.12.2 has no data-driven entity map",
                )

        # ---- numeric families (clock_00..63, destroy_stage_0..9, ...) ----
        families: dict[str, list[str]] = {}
        for key in owner:
            if not key.endswith(".png"):
                continue
            directory, _, filename = key.rpartition("/")
            match = _NUMERIC_FAMILY.match(filename[: -len(".png")])
            if match:
                families.setdefault(directory + "/" + match.group("base"), []).append(key)
        families = {key: sorted(value) for key, value in families.items() if len(value) > 1}
        family_of = {member: key for key, members in families.items() for member in members}

        def family_owner(base: str) -> AssetGroup | None:
            """Find the single logical name a numeric family belongs to.

            'clock' owns clock_00..63 outright. 'bow' owns bow_pulling_0..2
            through the pulling_ prefix. When nothing owns the family (the
            music discs 'record_11' and 'record_13' are separate real items),
            the members are left alone rather than merged into a fake 'record'.
            """
            selectable = (CATEGORY_ITEM, CATEGORY_BLOCK)
            exact = [g for g in groups.values() if g.category in selectable and g.name == base]
            if len(exact) == 1:
                return exact[0]
            prefixed = [
                g
                for g in groups.values()
                if g.category in selectable and base.startswith(g.name + "_")
            ]
            return prefixed[0] if len(prefixed) == 1 else None

        pulled_in = 0
        folded = 0
        for family, members in sorted(families.items()):
            base = family.rpartition("/")[2]
            target = family_owner(base)
            if target is None:
                continue
            before = {texture.resource_path for texture in target.textures}
            merged = sorted(before | set(members))
            pulled_in += len(set(merged) - before)
            _register(
                groups, claimed, target.namespace, target.category, target.name,
                merged, target.models, target.origins[0], animations, target.note,
            )
            # A frame model such as item/clock_01 is an implementation detail of
            # the clock, not a second selectable asset.
            for member in members:
                stem = PurePosixPath(member).name[: -len(".png")]
                sibling_id = "%s:%s/%s" % (target.namespace, target.category, stem)
                if sibling_id != target.asset_id and sibling_id in groups:
                    del groups[sibling_id]
                    folded += 1

        # ---- anything still unclaimed becomes its own group ----
        orphan_groups = 0
        orphan_textures = 0
        if include_orphans:
            leftovers: dict[tuple[str, str], list[str]] = {}
            for key in sorted(owner):
                if not key.startswith("assets/") or "/textures/" not in key:
                    continue
                if not key.endswith(".png") or key in claimed:
                    continue
                namespace = key.split("/")[1]
                relative = key.split("/textures/", 1)[1]
                stem = PurePosixPath(relative).name[: -len(".png")]
                family = family_of.get(key)
                name = family.rpartition("/")[2] if family else stem
                leftovers.setdefault((namespace, name), []).append(key)
            for (namespace, name), paths in leftovers.items():
                _register(
                    groups, claimed, namespace, CATEGORY_TEXTURE, name,
                    paths, [], "textures", animations,
                    note="unreferenced by any blockstate or item model",
                )
                orphan_groups += 1
                orphan_textures += len(paths)

        counts: dict[str, int] = {}
        texture_counts = {category: 0 for category in CATEGORIES}
        for group in groups.values():
            counts[group.category] = counts.get(group.category, 0) + 1
            texture_counts[group.category] = texture_counts.get(group.category, 0) + len(group.textures)
        catalogue.stats = {
            "roots": ["%s (%s)" % (root.path, root.kind) for root in roots],
            "resource_paths": len(owner),
            "models_resolved": len(model_cache),
            "groups": counts,
            "grouped_textures": texture_counts,
            "numeric_families": len(families),
            "family_textures_pulled_in": pulled_in,
            "family_member_groups_folded": folded,
            "orphan_groups": orphan_groups,
            "orphan_textures": orphan_textures,
            "animated_textures": sum(
                1 for group in groups.values() for t in group.textures if t.animated
            ),
        }
        return catalogue
    except BaseException:
        catalogue.close()
        raise
