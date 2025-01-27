"""Tests for the tabletop environment + scripted/random policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from parley.core.errors import ValidationError
from parley.core.types import (
    Action,
    GoalSpec,
    Grounding,
    Observation,
    SceneSpec,
    Transcript,
)
from parley.env import TabletopEnv
from parley.policy import NoisyPolicy, RandomPolicy, ScriptedPolicy


def _scene() -> SceneSpec:
    return SceneSpec(
        objects=(
            {"id": "o1", "color": "red", "shape": "cube", "pos": (0.2, 0.7)},
            {"id": "o2", "color": "blue", "shape": "sphere", "pos": (0.8, 0.3)},
        )
    )


def _roll(
    env: TabletopEnv,
    goal: GoalSpec,
    policy: Any,
    grounding: Grounding,
    text: str = "",
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    frame = env.reset(_scene(), goal, rng)
    policy.reset(rng)
    done = False
    info: dict[str, Any] = {}
    while not done:
        obs = Observation(frame=frame, transcript=Transcript(text=text), grounding=grounding)
        action = policy.act(obs)
        frame, _, done, info = env.step(action)
    return info


def test_env_rejects_zero_max_steps() -> None:
    with pytest.raises(ValueError, match="positive"):
        TabletopEnv(max_steps=0)


def test_env_step_after_termination_raises() -> None:
    env = TabletopEnv(max_steps=2)
    env.reset(_scene(), GoalSpec(verb="pick", target="red cube"), np.random.default_rng(0))
    move = Action(vec=np.zeros(2, dtype=np.float32), space="xy_pick_place", label="move")
    done = False
    while not done:
        _, _, done, _ = env.step(move)
    with pytest.raises(ValidationError, match="terminated"):
        env.step(move)


def test_env_rejects_unknown_action_space() -> None:
    env = TabletopEnv(max_steps=4)
    env.reset(_scene(), GoalSpec(verb="pick"), np.random.default_rng(0))
    with pytest.raises(ValidationError, match="action space"):
        env.step(Action(vec=np.zeros(2, dtype=np.float32), space="discrete"))


def test_scripted_solves_pick() -> None:
    info = _roll(
        TabletopEnv(max_steps=30),
        GoalSpec(verb="pick", target="red cube"),
        ScriptedPolicy(),
        Grounding(
            verb="pick", target="red cube", modifier="red", slots={"color": "red", "shape": "cube"}
        ),
    )
    assert info["success"] is True


def test_scripted_solves_place_left() -> None:
    info = _roll(
        TabletopEnv(max_steps=40),
        GoalSpec(verb="place", target="blue sphere", destination="left"),
        ScriptedPolicy(),
        Grounding(
            verb="place",
            target="blue sphere",
            destination="left",
            slots={"color": "blue", "shape": "sphere", "direction": "left"},
        ),
    )
    assert info["success"] is True


def test_scripted_fails_on_unknown_grounding() -> None:
    info = _roll(
        TabletopEnv(max_steps=10),
        GoalSpec(verb="pick", target="red cube"),
        ScriptedPolicy(),
        Grounding(verb="<unknown>"),
    )
    assert info["success"] is False


def test_random_policy_runs() -> None:
    info = _roll(
        TabletopEnv(max_steps=15),
        GoalSpec(verb="pick", target="red cube"),
        RandomPolicy(),
        Grounding(verb="pick", target="red cube"),
    )
    assert "success" in info


def test_noisy_policy_wraps_scripted() -> None:
    info = _roll(
        TabletopEnv(max_steps=40),
        GoalSpec(verb="pick", target="red cube"),
        NoisyPolicy(base=ScriptedPolicy(), sigma=0.02),
        Grounding(
            verb="pick", target="red cube", modifier="red", slots={"color": "red", "shape": "cube"}
        ),
    )
    assert info["success"] is True


def test_noisy_policy_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NoisyPolicy(base=ScriptedPolicy(), sigma=-0.1)
