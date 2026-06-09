"""Regression tests for issues surfaced by the v0.1 review pass.

Each test pins a behavior that was previously wrong:

- linguistic perturbations were silent at the speech layer (the codec
  audio wasn't re-encoded after the text changed);
- the cache key ignored resolved params, so two suites sharing
  (pipeline, perturbation, episode, seed) names but differing in params
  collided;
- _resample_linear crashed on empty input.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from parley.core.config import (
    BenchmarkConfig,
    EnvConfig,
    PerturbationGroup,
    PipelineConfig,
    PluginSpec,
    RunnerConfig,
)
from parley.data import SynthConfig, generate_dataset
from parley.perturb.audio import _resample_linear
from parley.runner import BenchmarkEngine
from parley.runner.pipeline import cache_key


def _cfg(perturbations: list[PerturbationGroup], out: str, workers: int = 1) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="reg",
        seed=1,
        env=EnvConfig(name="tabletop"),
        pipelines=[
            PipelineConfig(
                name="codec+scripted",
                speech=PluginSpec(name="codec"),
                policy=PluginSpec(name="scripted"),
            )
        ],
        perturbations=perturbations,
        metrics=["wer", "success_rate"],
        runner=RunnerConfig(max_steps=40, workers=workers, cache_dir=None),
        output_dir=out,
    )


def test_linguistic_perturbation_moves_wer() -> None:
    """A filler perturbation must produce non-zero WER (it was silent before)."""
    eps = generate_dataset(SynthConfig(n_episodes=8, seed=1))
    with tempfile.TemporaryDirectory() as td:
        cfg = _cfg(
            [
                PerturbationGroup(
                    name="filler", steps=[PluginSpec(name="filler", params={"rate": 1.0})]
                )
            ],
            td,
        )
        results = BenchmarkEngine(cfg).run(eps)
        filler = [r for r in results if r.perturbation == "filler"]
        clean = [r for r in results if r.perturbation == "clean"]
        assert all(r.metrics["wer"] == 0.0 for r in clean)
        assert sum(r.metrics["wer"] for r in filler) > 0.0


def test_cache_key_includes_param_fingerprint() -> None:
    """Same names, different fingerprint => different key (no collision)."""
    a = cache_key("p", "noise", "ep0", 0, config_fingerprint="aaaa")
    b = cache_key("p", "noise", "ep0", 0, config_fingerprint="bbbb")
    assert a != b
    # Empty fingerprint is backward-compatible (no suffix).
    assert cache_key("p", "noise", "ep0", 0) == "p::noise::ep0::seed=0"


def test_cache_does_not_collide_across_param_change(tmp_path: Path) -> None:
    """Two suites with the same perturbation NAME but different snr must not
    reuse each other's cached results."""
    eps = generate_dataset(SynthConfig(n_episodes=4, seed=1))
    cache_dir = str(tmp_path / "cache")

    def run_with_snr(snr: float) -> list[float]:
        cfg = BenchmarkConfig(
            name="reg",
            seed=1,
            env=EnvConfig(name="tabletop"),
            pipelines=[
                PipelineConfig(
                    name="codec+scripted",
                    speech=PluginSpec(name="codec"),
                    policy=PluginSpec(name="scripted"),
                )
            ],
            perturbations=[
                PerturbationGroup(
                    name="noise",
                    steps=[PluginSpec(name="additive_noise", params={"snr_db": snr})],
                )
            ],
            metrics=["wer", "success_rate"],
            runner=RunnerConfig(max_steps=40, workers=1, cache_dir=cache_dir),
            output_dir=str(tmp_path / "runs"),
        )
        res = BenchmarkEngine(cfg).run(eps)
        return [r.metrics["wer"] for r in res if r.perturbation == "noise"]

    clean_snr = run_with_snr(40.0)  # essentially clean
    harsh_snr = run_with_snr(-40.0)  # essentially destroyed
    # If the cache collided, harsh would reuse clean's (zero) WER.
    assert sum(harsh_snr) > sum(clean_snr)


def test_resample_linear_empty_input() -> None:
    out = _resample_linear(np.array([], dtype=np.float64), 1.1)
    assert out.shape[0] == 0


def test_threaded_engine_matches_serial(tmp_path: Path) -> None:
    """workers>1 must produce the same per-episode results as workers=1.

    Guards the per-thread pipeline fix: a shared, stateful policy under a
    thread pool would otherwise race and diverge from the serial run.
    """
    eps = generate_dataset(SynthConfig(n_episodes=8, seed=3))
    perts = [PerturbationGroup(name="mu_law", steps=[PluginSpec(name="mu_law")])]

    serial = BenchmarkEngine(_cfg(perts, str(tmp_path / "s"), workers=1)).run(eps)
    threaded = BenchmarkEngine(_cfg(perts, str(tmp_path / "t"), workers=4)).run(eps)

    def key(r: object) -> tuple[str, str, str]:
        return (r.pipeline, r.perturbation, r.episode_id)  # type: ignore[attr-defined]

    s = {key(r): r.success for r in serial}
    t = {key(r): r.success for r in threaded}
    assert s == t
