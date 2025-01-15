"""Scripted oracle-ish policy.

Uses the parsed :class:`Grounding` and the scene description embedded in
the frame to locate the target object and emit a small program:

  state 0 ("approach"): move toward target's (x, y)
  state 1 ("pick"):     when within proximity, emit a pick action
  state 2 ("transit"):  if a destination is set, move toward its zone
  state 3 ("place"):    emit a place action and stop

If grounding is missing or no object matches, the policy emits no-op
moves until the episode ends. This is the success-rate *ceiling* for
the synthetic environment under a given grounder.
"""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Action, Observation
from parley.env.tabletop import DIRECTION_ZONES


@registry.policy.register("scripted")
class ScriptedPolicy:
    name = "scripted"

    def __init__(self) -> None:
        self._stage = 0
        self._holding = False

    def reset(self, rng: np.random.Generator) -> None:
        self._stage = 0
        self._holding = False

    def act(self, obs: Observation) -> Action:
        grounding = obs.grounding
        frame = obs.frame
        scene = frame.scene
        if grounding is None or scene is None or grounding.verb == "<unknown>":
            return self._noop()

        target_pos = self._find_target(scene.objects, grounding)
        if target_pos is None:
            return self._noop()

        eef = frame.state[:2]

        # Stage machine
        if self._stage == 0:  # approach
            d = float(np.hypot(eef[0] - target_pos[0], eef[1] - target_pos[1]))
            if d < 0.08:
                self._stage = 1
            else:
                direction = np.asarray(target_pos, dtype=np.float32) - eef
                norm = float(np.linalg.norm(direction)) or 1.0
                return Action(
                    vec=(direction / norm).astype(np.float32),
                    space="xy_pick_place",
                    label="move",
                )

        if self._stage == 1:  # pick
            self._stage = 2
            self._holding = True
            return Action(
                vec=np.asarray(target_pos, dtype=np.float32),
                space="xy_pick_place",
                label="pick",
            )

        if self._stage == 2:  # transit / place
            if grounding.verb == "pick":
                return self._noop()
            dest = DIRECTION_ZONES.get(grounding.destination or "center")
            if dest is None:
                return self._noop()
            d = float(np.hypot(eef[0] - dest[0], eef[1] - dest[1]))
            if d < 0.06:
                self._stage = 3
                return Action(
                    vec=np.asarray(dest, dtype=np.float32),
                    space="xy_pick_place",
                    label="place",
                )
            direction = np.asarray(dest, dtype=np.float32) - eef
            norm = float(np.linalg.norm(direction)) or 1.0
            return Action(
                vec=(direction / norm).astype(np.float32),
                space="xy_pick_place",
                label="move",
            )

        return self._noop()

    # ------------------------------------------------------------------
    @staticmethod
    def _find_target(objects, grounding) -> tuple[float, float] | None:  # type: ignore[no-untyped-def]
        if grounding.target is None:
            return None
        parts = grounding.target.lower().split()
        for o in objects:
            color = str(o.get("color", "")).lower()
            shape = str(o.get("shape", "")).lower()
            if all(p in (color, shape) for p in parts):
                pos = o.get("pos", (0.5, 0.5))
                return float(pos[0]), float(pos[1])
        return None

    @staticmethod
    def _noop() -> Action:
        return Action(vec=np.zeros(2, dtype=np.float32), space="xy_pick_place", label="move")
