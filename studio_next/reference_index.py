"""Source-aware Minecraft texture index with lazy, content-addressed caching.

The index is deliberately descriptive.  It records resource paths and measured
pixel features, but never turns a resource name into a closed shape class.  A
caller can scan a 1.12.2 JAR once and materialize only the references selected
for a later pipeline run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Iterable
from zipfile import ZipFile

from PIL import Image

from .minecraft_assets import game_archives, list_assets
from .references import analyze_png


INDEX_SCHEMA_VERSION = 2
FEATURE_SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_label(value: str) -> str:
    return value.removeprefix("sha256:").replace("/", "_").replace("\\", "_")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as tmp:
        tmp.write(data)
        temp_name = Path(tmp.name)
    os.replace(temp_name, path)


def _atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _directory_fingerprint(root: Path) -> str:
    assets_root = root / "assets" if (root / "assets").is_dir() else root
    rows: list[str] = []
    for path in sorted(p for p in assets_root.rglob("*") if p.is_file()):
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            rows.append("%s|%d|%d|%s" % (relative, stat.st_size, stat.st_mtime_ns, _sha256_file(path)))
        except OSError:
            continue
    return "sha256:" + _sha256_bytes("\n".join(rows).encode("utf-8"))


def _jar_fingerprint(paths: Iterable[Path]) -> str:
    rows: list[str] = []
    for path in sorted(paths):
        stat = path.stat()
        rows.append("%s|%d|%d|%s" % (path.name, stat.st_size, stat.st_mtime_ns, _sha256_file(path)))
    return "sha256:" + _sha256_bytes("\n".join(rows).encode("utf-8"))


@dataclass(frozen=True)
class IndexSource:
    kind: str
    path: str
    minecraft_version: str | None
    fingerprint: str


def fingerprint_source(source: str | Path) -> IndexSource:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError("asset source does not exist: %s" % path)
    if path.is_file():
        if path.suffix.lower() != ".jar":
            raise ValueError("asset source file must be a .jar: %s" % path)
        return IndexSource("jar", str(path), None, _jar_fingerprint([path]))
    if path.name.lower() == "assets" and path.is_dir():
        return IndexSource("directory", str(path), None, _directory_fingerprint(path))
    if (path / "assets").is_dir():
        kind = "resource_pack" if (path / "pack.mcmeta").exists() else "directory"
        return IndexSource(kind, str(path), None, _directory_fingerprint(path))
    jars = list(path.glob("*.jar"))
    if jars:
        version = path.name.split("-")[0] if path.name else None
        return IndexSource("jar", str(path), version, _jar_fingerprint(jars))
    raise ValueError("asset source must be a JAR, version directory, or directory containing assets/: %s" % path)


def asset_id_for(namespace: str, resource_path: str, fingerprint: str) -> str:
    normalized = resource_path.replace("\\", "/").lstrip("/")
    return "%s:%s@%s" % (namespace, normalized, fingerprint)


def _classify_resource(resource_path: str) -> str:
    parts = PurePosixPath(resource_path).parts
    try:
        texture_index = parts.index("textures")
    except ValueError:
        return "misc"
    section = parts[texture_index + 1] if texture_index + 1 < len(parts) else ""
    return {
        "item": "item",
        "items": "item",
        "block": "block",
        "blocks": "block",
        "entity": "entity",
        "entities": "entity",
        "particle": "particle",
        "particles": "particle",
        "gui": "gui",
    }.get(section, "misc")


def _namespace_resource(path: str) -> tuple[str, str] | None:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if len(parts) < 4 or parts[0] != "assets":
        return None
    return parts[1], PurePosixPath(*parts[2:]).as_posix()


def _feature_record(image_bytes: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(image_bytes)) as loaded:
        image = loaded.convert("RGBA")
        features = analyze_png_from_image(image)
    palette = list(features.get("palette", []))
    lumas: list[int] = []
    for color in palette:
        try:
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        except (TypeError, ValueError):
            continue
        lumas.append(round(0.2126 * r + 0.7152 * g + 0.0722 * b))
    silhouette = features.get("silhouette_profile") or {}
    stroke = features.get("stroke_profile") or {}
    alpha = {
        "opaque_ratio": float(features.get("occupancy_ratio", 0.0)),
        "opaque_pixels": int(features.get("opaque_pixels", 0)),
        "bbox": features.get("bbox"),
        "components": int(features.get("components_4_connected", 0)),
    }
    palette_record = {
        "count": len(palette),
        "top": palette,
        "luma_range": [min(lumas), max(lumas)] if lumas else [0, 0],
    }
    # The manifest must stay small even when a resource pack contains large
    # GUI/entity atlases.  Preserve exact spans for normal pixel-art sheets;
    # large sources are re-analyzed after lazy materialization instead of
    # embedding thousands of rows in the searchable manifest.
    keep_spans = max(image.size) <= 128
    compact_stroke = {
        key: value
        for key, value in stroke.items()
        if key != "row_opaque_counts"
    }
    structure = {
        "row_spans": silhouette.get("row_spans", []) if keep_spans else [],
        "column_spans": silhouette.get("column_spans", []) if keep_spans else [],
        "axis_angle": float(features.get("orientation_degrees", 0.0)),
        "axis_anisotropy": float(features.get("axis_anisotropy", 0.0)),
        "symmetry": features.get("symmetry_profile", {}),
        "pattern_density": _pattern_density(image),
        "stroke": compact_stroke,
    }
    return {
        "dimensions": {"width": int(features.get("width", image.width)), "height": int(features.get("height", image.height))},
        "alpha": alpha,
        "palette": palette_record,
        "structure": structure,
        "feature_text": str(features.get("pixel_text", "")),
        "silhouette_text": str(silhouette.get("silhouette_map", "")) if keep_spans else "",
    }


def analyze_png_from_image(image: Image.Image) -> dict[str, Any]:
    """Run the shared analyzer on an in-memory image without a temp file."""
    # `analyze_png` intentionally accepts paths for the existing public API;
    # write a small temporary PNG only for the analyzer's stable feature code.
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        image.save(temporary, "PNG")
        return analyze_png(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def _pattern_density(image: Image.Image) -> float:
    rgba = image.convert("RGBA")
    pixels = [rgba.getpixel((x, y))[:3] for y in range(rgba.height) for x in range(rgba.width) if rgba.getpixel((x, y))[3] >= 8]
    if not pixels:
        return 0.0
    unique = len(set(pixels))
    return unique / float(len(pixels))


@dataclass
class IndexEntry:
    asset_id: str
    namespace: str
    resource_path: str
    category: str
    source_locator: dict[str, str]
    dimensions: dict[str, int]
    alpha: dict[str, Any]
    palette: dict[str, Any]
    structure: dict[str, Any]
    related: dict[str, list[str]] = field(default_factory=lambda: {"model_ids": [], "blockstate_ids": [], "faces": []})
    content_sha256: str = ""
    feature_text: str = ""
    silhouette_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # The active manifest is intentionally a name/path catalogue.  Full
        # pixel maps are optional and remain in feature blobs only when the
        # caller explicitly asks for them.
        for key in ("alpha", "palette", "structure", "feature_text", "silhouette_text"):
            if not data.get(key):
                data.pop(key, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexEntry":
        values = dict(data)
        values.setdefault("source_locator", {})
        values.setdefault("dimensions", {})
        values.setdefault("alpha", {})
        values.setdefault("palette", {})
        values.setdefault("structure", {})
        values.setdefault("related", {"model_ids": [], "blockstate_ids": [], "faces": []})
        values.setdefault("feature_text", "")
        values.setdefault("silhouette_text", "")
        return cls(**values)


@dataclass
class ReferenceIndex:
    source: IndexSource
    entries: list[IndexEntry]
    cache_root: Path
    cache_stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    schema_version: int = INDEX_SCHEMA_VERSION

    @property
    def source_cache_dir(self) -> Path:
        return self.cache_root / "sources" / _fingerprint_label(self.source.fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source": asdict(self.source),
            "cache_root": str(self.cache_root),
            "cache_stats": self.cache_stats,
            "errors": self.errors,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def save(self, path: str | Path) -> None:
        _atomic_write_json(Path(path).expanduser().resolve(), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ReferenceIndex":
        index_path = Path(path).expanduser().resolve()
        data = json.loads(index_path.read_text(encoding="utf-8"))
        source = IndexSource(**data["source"])
        cache_root = Path(data.get("cache_root") or index_path.parent).expanduser().resolve()
        return cls(
            source=source,
            entries=[IndexEntry.from_dict(row) for row in data.get("entries", [])],
            cache_root=cache_root,
            cache_stats=dict(data.get("cache_stats") or {}),
            errors=list(data.get("errors") or []),
            schema_version=int(data.get("schema_version", INDEX_SCHEMA_VERSION)),
        )

    def _entry(self, entry_or_id: IndexEntry | str) -> IndexEntry:
        if isinstance(entry_or_id, IndexEntry):
            return entry_or_id
        for entry in self.entries:
            if entry.asset_id == entry_or_id:
                return entry
        raise KeyError("unknown asset_id: %s" % entry_or_id)

    def feature_text(self, entry_or_id: IndexEntry | str) -> str:
        entry = self._entry(entry_or_id)
        if entry.feature_text:
            return entry.feature_text
        try:
            image_path = self.materialize(entry)
            with Image.open(image_path) as loaded:
                return str(_feature_record(loaded.convert("RGBA")).get("feature_text", ""))
        except (OSError, ValueError, KeyError):
            return ""

    def materialize(self, entry_or_id: IndexEntry | str) -> Path:
        entry = self._entry(entry_or_id)
        destination = self.source_cache_dir / "blobs" / (entry.content_sha256 + ".png")
        if destination.exists() and _sha256_file(destination) == entry.content_sha256:
            return destination
        locator = entry.source_locator
        kind = locator.get("kind")
        if kind == "directory":
            data = Path(locator["path"]).read_bytes()
        elif kind == "jar":
            with ZipFile(Path(locator["archive"])) as bundle:
                data = bundle.read(locator["resource_path"])
        else:
            raise ValueError("unsupported source locator kind: %s" % kind)
        if _sha256_bytes(data) != entry.content_sha256:
            raise ValueError("source changed since index was built: %s" % entry.asset_id)
        _atomic_write_bytes(destination, data)
        return destination


def _directory_entries(source: IndexSource) -> list[tuple[str, str, dict[str, str], bytes]]:
    root = Path(source.path)
    assets_root = root / "assets" if (root / "assets").is_dir() else root
    rows: list[tuple[str, str, dict[str, str], bytes]] = []
    for path in sorted(assets_root.rglob("*.png")):
        relative = path.relative_to(assets_root).as_posix()
        pieces = PurePosixPath(relative).parts
        if len(pieces) < 3 or pieces[1] != "textures":
            continue
        namespace, resource = pieces[0], PurePosixPath(*pieces[1:]).as_posix()
        rows.append((namespace, resource, {"kind": "directory", "path": str(path.resolve())}, path.read_bytes()))
    return rows


def _jar_entries(source: IndexSource) -> list[tuple[str, str, dict[str, str], bytes]]:
    rows: list[tuple[str, str, dict[str, str], bytes]] = []
    # Opening a ZipFile re-parses the whole central directory (45 ms for the
    # vanilla 1.12.2 JAR, 93 ms for a large mod).  Opening it once per texture
    # turned a one-second scan into a 76-second one, so every archive stays
    # open for the whole walk and is closed exactly once at the end.
    bundles: dict[str, ZipFile] = {}
    try:
        for asset in list_assets(source.path, prefix="assets/"):
            parsed = _namespace_resource(asset.resource_path)
            if parsed is None:
                continue
            namespace, resource = parsed
            if not resource.startswith("textures/") or not resource.lower().endswith(".png"):
                continue
            archive = str(asset.archive.resolve())
            bundle = bundles.get(archive)
            if bundle is None:
                bundle = ZipFile(asset.archive)
                bundles[archive] = bundle
            data = bundle.read(asset.resource_path)
            rows.append((namespace, resource, {"kind": "jar", "archive": archive, "resource_path": asset.resource_path}, data))
    finally:
        for bundle in bundles.values():
            bundle.close()
    return rows


def build_index(
    source: str | Path,
    cache_root: str | Path,
    rebuild: bool = False,
    with_pixel_text: bool = False,
) -> ReferenceIndex:
    source_info = fingerprint_source(source)
    cache = Path(cache_root).expanduser().resolve()
    source_cache = cache / "sources" / _fingerprint_label(source_info.fingerprint)
    manifest_path = source_cache / "manifest.json"
    if manifest_path.exists() and not rebuild:
        try:
            cached = ReferenceIndex.load(manifest_path)
            if cached.source.fingerprint == source_info.fingerprint and cached.schema_version == INDEX_SCHEMA_VERSION:
                cached.cache_root = cache
                cached.cache_stats = {"entries_reused": len(cached.entries), "entries_built": 0}
                return cached
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    rows = _directory_entries(source_info) if source_info.kind in {"directory", "resource_pack"} else _jar_entries(source_info)
    entries: list[IndexEntry] = []
    errors: list[str] = []
    for namespace, resource, locator, data in rows:
        try:
            with Image.open(BytesIO(data)) as loaded:
                dimensions = {"width": loaded.width, "height": loaded.height}
            feature = {
                "dimensions": dimensions,
                "alpha": {},
                "palette": {},
                "structure": {},
                "feature_text": "",
                "silhouette_text": "",
            }
            if with_pixel_text:
                feature = _feature_record(data)
        except Exception as exc:  # pragma: no cover - corrupt third-party PNG
            errors.append("%s: %s" % (resource, exc))
            continue
        entry = IndexEntry(
            asset_id=asset_id_for(namespace, resource, source_info.fingerprint),
            namespace=namespace,
            resource_path=resource,
            category=_classify_resource(resource),
            source_locator=locator,
            dimensions=feature["dimensions"],
            alpha=feature["alpha"],
            palette=feature["palette"],
            structure=feature["structure"],
            content_sha256=_sha256_bytes(data),
            feature_text=feature["feature_text"] if with_pixel_text else "",
            silhouette_text=feature["silhouette_text"] if with_pixel_text else "",
        )
        entries.append(entry)
        if with_pixel_text:
            feature_path = source_cache / "features" / (entry.content_sha256 + ".json")
            if not feature_path.exists() or rebuild:
                _atomic_write_json(feature_path, {"schema_version": FEATURE_SCHEMA_VERSION, "asset_id": entry.asset_id, **feature})

    index = ReferenceIndex(
        source=source_info,
        entries=entries,
        cache_root=cache,
        cache_stats={"entries_reused": 0, "entries_built": len(entries), "errors": len(errors)},
        errors=errors,
    )
    _atomic_write_json(manifest_path, index.to_dict())
    return index
