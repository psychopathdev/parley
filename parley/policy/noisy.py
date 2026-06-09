"""Wrap another policy and add Gaussian noise to its action vector.

Useful for empirically probing how much success-rate the env will tolerate
before metrics collapse — and as a controlled "imperfect VLA" baseline.
"""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Action, Observation
from parley.policy.base import VLAPolicy


@registry.policy.register("noisy")
class NoisyPolicy:
    name = "noisy"

    def __init__(self, base: VLAPolicy, sigma: float = 0.05) -> None:
        if sigma < 0:
            raise ValueError("NoisyPolicy.sigma must be non-negative")
        self._base = base
        self._sigma = float(sigma)
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._base.reset(rng)

    def act(self, obs: Observation) -> Action:
        a = self._base.act(obs)
        rng = self._rng
        assert rng is not None
        noise = rng.normal(0.0, self._sigma, size=a.vec.shape).astype(np.float32)
        return Action(vec=(a.vec + noise).astype(np.float32), space=a.space, label=a.label)
