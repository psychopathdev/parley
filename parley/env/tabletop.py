"""A self-contained 2-D tabletop environment.

The world is a unit square populated with colored objects. The agent
holds a 2-D end-effector position; continuous actions move it; the
``pick`` and ``place`` discrete actions interact with whichever object
the effector currently overlaps. Success is judged by comparing the
final state against a :class:`GoalSpec`.

This env is *not* a robotics simulator — it's a deterministic abstract
testbed designed to be small enough that the entire pipeline (audio →
ASR → grounding → policy → action → success) runs in milliseconds. Real
simulators (LIBERO, ManiSkill) plug in via the same :class:`Environment`
protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from parley.core.errors import ValidationError
from parley.core.registry import registry
from parley.core.types import Action, Frame, GoalSpec, SceneSpec

# State vector layout, all in workspace coordinates:
#  [eef_x, eef_y, holding_idx, holding_color_id, holding_shape_id, n_objects]
STATE_DIM = 6
PROXIMITY = 0.08  # how close eef must be to grab an object
PLACE_TOLERANCE = 0.15  # how close a destination zone must be to call placement done

# Direction zones for "to the left/right/center"
DIRECTION_ZONES: dict[str, tuple[float, float]] = {
    "left": (0.15, 0.5),
    "right": (0.85, 0.5),
    "center": (0.5, 0.5),
    "front": (0.5, 0.85),
    "back": (0.5, 0.15),
}


@registry.env.register("tabletop")
class TabletopEnv:
    """Deterministic 2-D tabletop with pickable colored objects."""

    name = "tabletop"
    action_space = "xy_pick_place"

    def __init__(self, max_steps: int = 64, seed: int = 0) -> None:
        if max_steps <= 0:
            raise ValueError("TabletopEnv.max_steps must be positive")
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self._step = 0
        self._eef = np.array([0.5, 0.5], dtype=np.float32)
        self._holding: int | None = None
        self._objects: list[dict[str, Any]] = []
        self._goal: GoalSpec | None = None
        self._scene: SceneSpec | None = None
        self._done = False

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        scene: SceneSpec,
        goal: GoalSpec,
        rng: np.random.Generator,
    ) -> Frame:
        # We mutate object dicts (e.g. set positions during ``place``) so
        # copy them to avoid leaking modifications back into the scene.
        self._objects = [dict(o) for o in scene.objects]
        self._scene = scene
        self._goal = goal
        self._step = 0
        self._holding = None
        self._done = False
        # Spawn the effector at a small random offset from center so episodes
        # with the same scene/goal still differ at the action level.
        self._eef = np.array(
            [0.5 + rng.uniform(-0.05, 0.05), 0.5 + rng.uniform(-0.05, 0.05)],
            dtype=np.float32,
        )
        return self._make_frame()

    def step(self, action: Action) -> tuple[Frame, float, bool, dict[str, Any]]:
        if self._done:
            raise ValidationError("step called after episode terminated")
        self._step += 1
        info: dict[str, Any] = {"action_kind": action.label or "move"}
        reward = 0.0

        if action.space == "xy_pick_place":
            if action.label == "pick":
                # Vec is interpreted as the (x, y) location to pick at.
                self._eef = np.asarray(action.vec[:2], dtype=np.float32)
                target = self._object_at(self._eef)
                if target is not None and self._holding is None:
                    self._holding = target
                    info["picked"] = self._objects[target].get("id")
                    reward = 0.5
            elif action.label == "place":
                self._eef = np.asarray(action.vec[:2], dtype=np.float32)
                if self._holding is not None:
                    self._objects[self._holding]["pos"] = tuple(self._eef.tolist())
                    info["placed"] = self._objects[self._holding].get("id")
                    self._holding = None
                    reward = 0.5
            else:
                # plain delta move
                self._eef = np.clip(
                    self._eef + np.asarray(action.vec[:2], dtype=np.float32) * 0.05,
                    0.0,
                    1.0,
                )
        elif action.space == "xy_delta":
            self._eef = np.clip(
                self._eef + np.asarray(action.vec[:2], dtype=np.float32) * 0.05,
                0.0,
                1.0,
            )
        else:
            raise ValidationError(f"TabletopEnv does not support action space {action.space!r}")

        success = self._evaluate_success()
        if success:
            reward += 1.0
        self._done = success or self._step >= self.max_steps
        return self._make_frame(), float(reward), self._done, {**info, "success": success}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_frame(self) -> Frame:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0:2] = self._eef
        state[2] = -1.0 if self._holding is None else float(self._holding)
        state[5] = float(len(self._objects))
        return Frame(state=state, scene=self._scene, step=self._step)

    def _object_at(self, pos: np.ndarray) -> int | None:
        """Return the index of the closest object within PROXIMITY, or None."""
        best: int | None = None
        best_d = PROXIMITY
        for i, obj in enumerate(self._objects):
            ox, oy = obj.get("pos", (0.0, 0.0))
            d = float(np.hypot(pos[0] - ox, pos[1] - oy))
            if d < best_d:
                best_d = d
                best = i
        return best

    def _evaluate_success(self) -> bool:
        goal = self._goal
        if goal is None:
            return False
        verb = goal.verb
        target = goal.target
        target_idx = self._match_target_index(target)
        if target_idx is None:
            return False
        if verb == "pick":
            return self._holding == target_idx
        if verb == "place":
            if self._holding is not None:
                return False  # not yet placed
            zone = DIRECTION_ZONES.get(goal.destination or "")
            if zone is None:
                return False
            ox, oy = self._objects[target_idx].get("pos", (0.0, 0.0))
            return float(np.hypot(ox - zone[0], oy - zone[1])) <= PLACE_TOLERANCE
        if verb == "push":
            zone = DIRECTION_ZONES.get(goal.destination or "")
            if zone is None:
                return False
            ox, oy = self._objects[target_idx].get("pos", (0.0, 0.0))
            return float(np.hypot(ox - zone[0], oy - zone[1])) <= PLACE_TOLERANCE
        return False

    def _match_target_index(self, target: str | None) -> int | None:
        if not target:
            return None
        parts = target.lower().split()
        for i, obj in enumerate(self._objects):
            color = str(obj.get("color", "")).lower()
            shape = str(obj.get("shape", "")).lower()
            if all(p in (color, shape) for p in parts):
                return i
        return None
