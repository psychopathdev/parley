"""Environment protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from parley.core.types import Action, Frame, GoalSpec, SceneSpec


@runtime_checkable
class Environment(Protocol):
    """Episodic environment with a Gym-style step interface.

    ``reset`` is given the scene and goal up-front (Parley is goal-
    conditioned: the env knows what the policy is supposed to do so it
    can score success). ``step`` returns ``(frame, reward, done, info)``.
    """

    name: str
    action_space: str
    seed: int

    def reset(
        self,
        scene: SceneSpec,
        goal: GoalSpec,
        rng: np.random.Generator,
    ) -> Frame: ...

    def step(self, action: Action) -> tuple[Frame, float, bool, dict[str, object]]: ...
