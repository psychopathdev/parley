"""Tests for the shared dataclasses in :mod:`parley.core.types`."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.types import Action, Audio, Frame, GoalSpec, Grounding, SceneSpec, Transcript


def test_audio_rejects_stereo() -> None:
    stereo = np.zeros((2, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="must be 1-D"):
        Audio(samples=stereo, sample_rate=16_000)


def test_audio_rejects_nonpositive_sr() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        Audio(samples=np.zeros(16, dtype=np.float32), sample_rate=0)


def test_audio_duration_matches_samples() -> None:
    samples = np.zeros(8_000, dtype=np.float32)
    audio = Audio(samples=samples, sample_rate=16_000)
    assert audio.duration == pytest.approx(0.5)


def test_transcript_has_sane_defaults() -> None:
    t = Transcript(text="pick up the red cube")
    assert t.tokens == ()
    assert t.confidence is None
    assert t.metadata == {}


def test_grounding_optional_slots() -> None:
    g = Grounding(verb="pick", target="cube", slots={"color": "red"})
    assert g.modifier is None
    assert g.slots["color"] == "red"


def test_action_default_space() -> None:
    a = Action(vec=np.zeros(2, dtype=np.float32))
    assert a.space == "xy_delta"
    assert a.label is None


def test_frame_with_optional_image() -> None:
    f = Frame(state=np.zeros(4, dtype=np.float32))
    assert f.image is None
    assert f.step == 0


def test_scene_and_goal_are_plain_records() -> None:
    scene = SceneSpec(objects=({"id": "o1", "color": "red", "pos": (0.2, 0.3)},))
    goal = GoalSpec(verb="pick", target="o1")
    assert scene.workspace == (0.0, 0.0, 1.0, 1.0)
    assert goal.destination is None
