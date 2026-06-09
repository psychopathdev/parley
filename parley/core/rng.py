"""Seeded, reproducible RNG management.

Two design choices worth flagging:

1. We use ``numpy.random.Generator`` (the modern PCG64 stream) rather than
   the legacy global state. That means seeding is local and predictable
   regardless of what the host program has done with ``np.random.seed``.

2. Sub-streams are derived deterministically from a parent seed + a name
   string via :func:`derive_seed`. The name is hashed with BLAKE2b so the
   same (seed, name) always yields the same sub-seed across processes and
   Python versions — important for cache hits on a redo.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def derive_seed(parent_seed: int, name: str) -> int:
    """Deterministically derive a 64-bit sub-seed from a parent + name.

    BLAKE2b is used (rather than ``hash(name)``) because the built-in
    ``hash`` of ``str`` is randomized per-process and is therefore unsafe
    for any reproducibility guarantee.

    The result is a non-negative ``int`` in ``[0, 2**63)``, which fits the
    ``seed`` argument of every numpy/python RNG without surprises.
    """
    if parent_seed < 0:
        raise ValueError(f"parent_seed must be non-negative, got {parent_seed}")
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8, salt=b"parley\0\0")
    h.update(parent_seed.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False) & ((1 << 63) - 1)


@dataclass
class RngManager:
    """A tiny façade over numpy ``Generator`` with named sub-streams.

    Usage::

        rng = RngManager(seed=42)
        audio_rng = rng.stream("audio.noise")          # deterministic
        scene_rng = rng.stream("env.scene.episode-7")

    Calling :meth:`stream` twice with the same name returns the *same*
    generator object so consumers that hold the reference keep advancing
    its state instead of accidentally rewinding it.
    """

    seed: int
    _streams: dict[str, np.random.Generator] = field(default_factory=dict, repr=False)

    def stream(self, name: str) -> np.random.Generator:
        gen = self._streams.get(name)
        if gen is None:
            gen = np.random.default_rng(derive_seed(self.seed, name))
            self._streams[name] = gen
        return gen

    def fresh(self, name: str) -> np.random.Generator:
        """Force-create a new generator for ``name``, discarding any cached state.

        Use this between episodes when you want each episode to see the
        identical starting stream regardless of evaluation order.
        """
        gen = np.random.default_rng(derive_seed(self.seed, name))
        self._streams[name] = gen
        return gen
