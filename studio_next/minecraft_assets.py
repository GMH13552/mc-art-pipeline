"""Read vanilla-style assets directly from a Minecraft version JAR.

This is deliberately an archive reader, not a version-specific lookup table.
The caller supplies the version directory (or a single JAR) and the resource
path it wants to use as reference evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile


@dataclass(frozen=True)
class MinecraftAsset:
    archive: Path
    resource_path: str


def _normal_resource_path(resource_path: str) -> str:
    value = resource_path.replace("\\", "/").lstrip("/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError("resource path must be a relative path inside the JAR")
    return path.as_posix()


def game_archives(version_path: str | Path) -> list[Path]:
    """Find readable JARs in a version folder, or accept one JAR explicitly."""
    source = Path(version_path).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".jar":
            raise ValueError("version path must be a .jar file or a directory containing JARs")
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(source.glob("*.jar"))
    else:
        raise FileNotFoundError("Minecraft version path does not exist: %s" % source)
    readable: list[Path] = []
    for archive in candidates:
        try:
            with ZipFile(archive) as bundle:
                bundle.infolist()
            readable.append(archive)
        except BadZipFile:
            continue
    if not readable:
        raise FileNotFoundError("no readable JAR found under: %s" % source)
    return readable


def list_assets(version_path: str | Path, prefix: str = "assets/") -> list[MinecraftAsset]:
    """List unique resource paths from each archive in deterministic precedence order."""
    normalized_prefix = _normal_resource_path(prefix)
    if prefix.endswith("/"):
        normalized_prefix += "/"
    found: dict[str, MinecraftAsset] = {}
    for archive in game_archives(version_path):
        with ZipFile(archive) as bundle:
            for name in bundle.namelist():
                normalized = name.replace("\\", "/")
                if normalized.endswith("/") or not normalized.startswith(normalized_prefix):
                    continue
                found.setdefault(normalized, MinecraftAsset(archive=archive, resource_path=normalized))
    return [found[name] for name in sorted(found)]


def extract_asset(version_path: str | Path, resource_path: str, destination: str | Path) -> MinecraftAsset:
    """Copy one exact resource out of a version JAR without unpacking the whole archive."""
    wanted = _normal_resource_path(resource_path)
    for archive in game_archives(version_path):
        with ZipFile(archive) as bundle:
            try:
                content = bundle.read(wanted)
            except KeyError:
                continue
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return MinecraftAsset(archive=archive, resource_path=wanted)
    raise FileNotFoundError("resource not found in Minecraft version JARs: %s" % wanted)
