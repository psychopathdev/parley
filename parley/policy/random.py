"""Uniform-random policy. The floor baseline."""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Action, Observation


@registry.policy.register("random")
class RandomPolicy:
    """Sample an action uniformly from ``[-1, 1]^2`` plus a coin-flip pick/place."""

    name = "random"
    _PICK_PROB = 0.05
    _PLACE_PROB = 0.05

    def __init__(self) -> None:
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def act(self, obs: Observation) -> Action:
        rng = self._rng
        assert rng is not None, "RandomPolicy.act called before reset"
        u = float(rng.random())
        vec = rng.uniform(0.0, 1.0, size=2).astype(np.float32)
        if u < self._PICK_PROB:
            return Action(vec=vec, space="xy_pick_place", label="pick")
        if u < self._PICK_PROB + self._PLACE_PROB:
            return Action(vec=vec, space="xy_pick_place", label="place")
        delta = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        return Action(vec=delta, space="xy_pick_place", label="move")
