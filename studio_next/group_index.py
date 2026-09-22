"""Present a live GroupCatalogue through the reference-source interface.

The router was written against the static ReferenceIndex. Instead of forking
it, this adapter exposes a logical-asset catalogue in the same shape, so local
recall, manifest building, route caching and strict response parsing are all
reused unchanged.

The one real difference: a single selected logical name can expand into several
reference images -- every face texture of a block, or every frame of a
code-driven family such as 'clock' or 'bow'. That expansion is what lets the
downstream planners see a coherent family instead of one arbitrary frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .asset_groups import AssetGroup, GroupCatalogue, build_catalogue
from .contracts import ReferenceAsset, ReferenceRole
from .text_cache import TextFeatureCache


def _bounded_frames(
    indexed: list[tuple[int, Any]],
    preferred_member: str | None,
    max_frames: int,
) -> list[tuple[int, Any]]:
    """Keep the frame this request is about plus an even spread of the rest."""
    wanted = str(preferred_member or "").strip().lower()
    chosen: list[tuple[int, Any]] = []
    rest: list[tuple[int, Any]] = []
    for item in indexed:
        name = str(getattr(item[1], "name", "")).strip().lower()
        if wanted and not chosen and name == wanted:
            chosen.append(item)
        else:
            rest.append(item)
    remaining = max_frames - len(chosen)
    if remaining > 0 and rest:
        step = len(rest) / float(remaining)
        seen = {item[0] for item in chosen}
        for position in range(remaining):
            item = rest[min(int(position * step), len(rest) - 1)]
            if item[0] not in seen:
                chosen.append(item)
                seen.add(item[0])
    return sorted(chosen, key=lambda item: item[0])


@dataclass(frozen=True)
class GroupEntry:
    """One logical asset, shaped like an IndexEntry so the router accepts it."""

    group: AssetGroup
    asset_id: str
    namespace: str
    category: str
    resource_path: str
    dimensions: dict[str, Any] = field(default_factory=dict)
    alpha: dict[str, Any] = field(default_factory=dict)
    palette: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceStamp:
    kind: str
    path: str
    minecraft_version: str | None
    fingerprint: str


class GroupReferenceSource:
    """A GroupCatalogue viewed as a ReferenceIndex-compatible source."""

    def __init__(
        self,
        catalogue: GroupCatalogue,
        cache_root: str | Path,
        *,
        text_root: str | Path | None = None,
        blob_root: str | Path | None = None,
    ) -> None:
        self.catalogue = catalogue
        # Absolute on purpose: these paths are written into a plan and replayed
        # later, and plan_from_file resolves a relative reference against the
        # plan's own directory. A relative cache root silently produced
        # references that could not be reopened.
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.fingerprint = _catalogue_fingerprint(catalogue)
        self.source = SourceStamp(
            kind="live",
            path=";".join(str(root.path) for root in catalogue.roots),
            minecraft_version=None,
            fingerprint=self.fingerprint,
        )
        self.entries: list[GroupEntry] = [
            GroupEntry(
                group=group,
                asset_id=group.asset_id,
                namespace=group.namespace,
                category=group.category,
                # The router only needs a stable, human-meaningful path; using
                # category/name keeps every group distinguishable even when two
                # namespaces reuse a name.
                resource_path="%s/%s.png" % (group.category, group.name),
            )
            for group in catalogue.select()
        ]
        self.cache_stats = {"entries_reused": 0, "entries_built": len(self.entries)}
        self.text_cache = TextFeatureCache(
            root=Path(text_root).expanduser().resolve() if text_root is not None else self.cache_root / "text"
        )
        self.blob_root = (
            Path(blob_root).expanduser().resolve() if blob_root is not None else self.cache_root / "blobs"
        )
        self.extracted_textures = 0

    # -- construction ---------------------------------------------------
    @classmethod
    def from_sources(
        cls,
        sources: list[str | Path],
        cache_root: str | Path,
        **kwargs: Any,
    ) -> "GroupReferenceSource":
        return cls(build_catalogue(sources), cache_root, **kwargs)

    # -- ReferenceIndex-compatible surface ------------------------------
    @property
    def source_cache_dir(self) -> Path:
        label = self.fingerprint.removeprefix("sha256:")
        return self.cache_root / "routes" / label

    def entry_for(self, asset_id: str) -> GroupEntry:
        for entry in self.entries:
            if entry.asset_id == asset_id:
                return entry
        raise KeyError("unknown asset id: %s" % asset_id)

    def _annotation(self, resource_path: str) -> bytes:
        return self.catalogue.read(resource_path + ".mcmeta") or b""

    def _materialize_texture(self, resource_path: str) -> tuple[Path | None, dict[str, Any], bool]:
        data = self.catalogue.read(resource_path)
        if data is None:
            return None, {}, False
        annotation = self._annotation(resource_path)
        features, cache_hit = self.text_cache.features(data, annotation)
        digest = sha256(data).hexdigest()
        blob = self.blob_root / digest[:2] / (digest + ".png")
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(data)
        return blob, features, cache_hit

    def materialize(self, entry: GroupEntry) -> Path:
        """First texture only, for callers that expect a single path."""
        paths = self.materialize_all(entry)
        if not paths:
            raise FileNotFoundError("asset group has no readable texture: %s" % entry.asset_id)
        return paths[0]

    def materialize_all(self, entry: GroupEntry) -> list[Path]:
        paths: list[Path] = []
        for texture in entry.group.textures:
            blob, _features, _hit = self._materialize_texture(texture.resource_path)
            if blob is not None:
                paths.append(blob)
        self.extracted_textures += len(paths)
        return paths

    def planning_assets(
        self,
        entry: GroupEntry,
        roles: list[ReferenceRole],
        display_name: str | None = None,
        preferred_member: str | None = None,
        max_frames: int | None = None,
    ) -> list[ReferenceAsset]:
        """Expand one logical name into a family of reference images.

        Every member carries the router's roles, plus notes that keep the
        family identity and the member index visible to the downstream model.

        A code-driven family can be very large -- vanilla clock owns 64 frames
        and compass 32 -- and attaching all of them spends the whole vision
        budget on near-duplicates of one object. Above ``max_frames`` the
        expansion keeps the frame this request is about plus an even spread of
        the remaining states, so the planners still see the range of the
        animation without paying for every degree of it.
        """
        group = entry.group
        base = (display_name or group.name).strip() or group.name
        total = len(group.textures)
        indexed = list(enumerate(group.textures))
        if max_frames is not None and total > max_frames >= 1:
            indexed = _bounded_frames(indexed, preferred_member, max_frames)
        assets: list[ReferenceAsset] = []
        for index, texture in indexed:
            blob, features, _hit = self._materialize_texture(texture.resource_path)
            if blob is None:
                continue
            notes = [
                "group=%s" % group.asset_id,
                "member=%d/%d" % (index + 1, total),
            ]
            if total > 1:
                notes.append("family_member=%s" % texture.name)
            if texture.animation:
                notes.append("animation=%s" % json.dumps(texture.animation, ensure_ascii=False, sort_keys=True))
            if group.note:
                notes.append("group_note=%s" % group.note)
            name = base if total == 1 else "%s:%s" % (base, texture.name)
            assets.append(
                ReferenceAsset(
                    path=str(blob),
                    name=name,
                    roles=list(roles),
                    notes=notes,
                    features=features,
                )
            )
        self.extracted_textures += len(assets)
        return assets

    def close(self) -> None:
        self.catalogue.close()

    def __enter__(self) -> "GroupReferenceSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _catalogue_fingerprint(catalogue: GroupCatalogue) -> str:
    """Identity of the catalogue itself, independent of any query."""
    digest = sha256()
    for group in catalogue.select():
        digest.update(group.asset_id.encode("utf-8"))
        digest.update(b"\0")
        for texture in group.textures:
            digest.update(texture.resource_path.encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()
