"""Tests for the synthetic dataset generator and on-disk format."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parley.core.types import Audio
from parley.data import SynthConfig, generate_dataset, load_episodes, save_episodes
from parley.data.synth import vocab_for
from parley.speech import CodecSpeechFrontend


def test_synth_is_deterministic() -> None:
    a = generate_dataset(SynthConfig(n_episodes=4, seed=0))
    b = generate_dataset(SynthConfig(n_episodes=4, seed=0))
    for x, y in zip(a, b, strict=True):
        assert x.episode_id == y.episode_id
        assert x.instruction.text == y.instruction.text
        assert np.array_equal(x.audio, y.audio)
        assert x.scene.objects == y.scene.objects


def test_synth_clean_round_trip() -> None:
    """The codec frontend must decode synth audio to the original text."""
    cfg = SynthConfig(n_episodes=8, seed=0)
    eps = generate_dataset(cfg)
    fe = CodecSpeechFrontend(vocab=vocab_for(cfg), sample_rate=cfg.sample_rate)
    matched = 0
    for ep in eps:
        t = fe.transcribe(Audio(samples=ep.audio, sample_rate=ep.sample_rate))
        assert t.text == ep.instruction.text, (ep.instruction.text, t.text)
        matched += 1
    assert matched == len(eps)


def test_synth_target_appears_in_scene() -> None:
    cfg = SynthConfig(n_episodes=12, seed=1)
    for ep in generate_dataset(cfg):
        assert ep.goal.target is not None
        color, shape = ep.goal.target.split()
        assert any(o["color"] == color and o["shape"] == shape for o in ep.scene.objects)


def test_round_trip_save_load(tmp_path: Path) -> None:
    eps = generate_dataset(SynthConfig(n_episodes=4, seed=2))
    out = tmp_path / "ds.jsonl"
    save_episodes(eps, out)
    loaded = load_episodes(out)
    assert len(loaded) == len(eps)
    for a, b in zip(eps, loaded, strict=True):
        assert a.episode_id == b.episode_id
        assert a.instruction.text == b.instruction.text
        assert np.array_equal(a.audio, b.audio)
        assert a.scene.objects == b.scene.objects
        assert a.goal.target == b.goal.target


def test_load_missing_index_raises(tmp_path: Path) -> None:
    try:
        load_episodes(tmp_path / "nope.jsonl")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")
