"""Content-addressed cache for the text description of a reference image.

The catalogue in 'asset_groups' is deliberately image-free. This module is the
next step: once a planner has chosen a logical name, its textures are read and
turned into the text features the downstream planners actually read.

The cache key is the SHA-256 of the image bytes plus any sibling .mcmeta bytes.
That choice matters:

* editing or replacing a texture changes the key, so the text is regenerated
  automatically and a stale description can never be reused;
* moving, renaming or re-indexing identical bytes stays a hit, so the cache
  survives the Windows to WSL move that broke the old path-bound index;
* animation metadata participates in the key, because frametime/frames change
  how a texture should be described.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .references import analyze_png_bytes


TEXT_FEATURE_SCHEMA = 1


@dataclass
class TextFeatureCache:
    """Stores analyzer output by content hash, never by path."""

    root: Path
    hits: int = 0
    misses: int = 0
    writes: int = 0

    @staticmethod
    def key(data: bytes, annotation: bytes = b"") -> str:
        digest = sha256()
        digest.update(("mc-art-text-v%d" % TEXT_FEATURE_SCHEMA).encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        digest.update(annotation)
        return digest.hexdigest()

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / (key + ".json")

    def get(self, data: bytes, annotation: bytes = b"") -> dict[str, Any] | None:
        target = self.path_for(self.key(data, annotation))
        if not target.exists():
            self.misses += 1
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            target.unlink(missing_ok=True)
            self.misses += 1
            return None
        if payload.get("schema_version") != TEXT_FEATURE_SCHEMA:
            self.misses += 1
            return None
        self.hits += 1
        return dict(payload.get("features") or {})

    def put(self, data: bytes, annotation: bytes, features: dict[str, Any]) -> Path:
        key = self.key(data, annotation)
        target = self.path_for(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": TEXT_FEATURE_SCHEMA,
                    "key": key,
                    "features": features,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        self.writes += 1
        return target

    def features(self, data: bytes, annotation: bytes = b"") -> tuple[dict[str, Any], bool]:
        """Return (features, cache_hit), computing and storing on a miss."""
        cached = self.get(data, annotation)
        if cached is not None:
            return cached, True
        features = analyze_png_bytes(data)
        self.put(data, annotation, features)
        return features, False

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}
