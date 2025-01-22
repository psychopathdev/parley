"""Tiny content-addressed cache for episode results.

We hash the cache key (already a deterministic string) and store the
serialized result under ``<cache_dir>/<hash>.json``. Hits skip the
expensive parts (audio perturbation + ASR + env rollout); misses run
the pipeline and write the result back.

Cache values are kept as plain JSON-friendly dicts; the engine inflates
them back into :class:`EpisodeResult` instances on read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContentCache:
    """File-backed key→value cache. Single-process, no locking.

    Set ``cache_dir=None`` to disable the cache (every operation is a
    no-op) — the runner uses this in CI where each invocation is fresh.
    """

    def __init__(self, cache_dir: str | Path | None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    def _path_for(self, key: str) -> Path:
        assert self.cache_dir is not None
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if self.cache_dir is None:
            return None
        p = self._path_for(key)
        if not p.exists():
            return None
        try:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.cache_dir is None:
            return
        p = self._path_for(key)
        # Atomic write: stage to a tmp file then rename.
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(value, f)
        tmp.replace(p)

    def clear(self) -> None:
        if self.cache_dir is None:
            return
        for f in self.cache_dir.glob("*.json"):
            f.unlink(missing_ok=True)
