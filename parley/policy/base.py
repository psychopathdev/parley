"""Policy protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from parley.core.types import Action, Observation


@runtime_checkable
class VLAPolicy(Protocol):
    """Map an observation (frame + transcript + grounding) to an action.

    Policies are stateful within an episode: ``reset`` is called at the
    start, ``act`` at every step. Cross-episode state should be cleared
    in ``reset``. ``rng`` is per-episode and seedable.
    """

    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def act(self, obs: Observation) -> Action: ...
