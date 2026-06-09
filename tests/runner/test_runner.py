"""Engine + cache + suite + pipeline integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parley.core.config import (
    BenchmarkConfig,
    DatasetConfig,
    EnvConfig,
    PerturbationGroup,
    PipelineConfig,
    PluginSpec,
    RunnerConfig,
)
from parley.core.errors import ConfigError
from parley.data import SynthConfig, generate_dataset
from parley.runner import BenchmarkEngine, ContentCache, expand_suite
from parley.runner.pipeline import build_pipeline, cache_key

# ---- ContentCache --------------------------------------------------------


def test_cache_disabled_when_dir_none() -> None:
    cache = ContentCache(None)
    assert cache.enabled is False
    cache.put("k", {"a": 1})  # no-op
    assert cache.get("k") is None


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path / "cache")
    assert cache.enabled
    cache.put("alpha", {"x": 1})
    assert cache.get("alpha") == {"x": 1}


def test_cache_clear(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path / "cache")
    cache.put("alpha", {"x": 1})
    cache.clear()
    assert cache.get("alpha") is None


def test_cache_bad_json_falls_back_to_miss(tmp_path: Path) -> None:
    cache = ContentCache(tmp_path / "cache")
    cache.put("alpha", {"x": 1})
    # Corrupt the on-disk file.
    f = next((tmp_path / "cache").glob("*.json"))
    f.write_text("not json", encoding="utf-8")
    assert cache.get("alpha") is None  # graceful


def test_cache_key_is_stable() -> None:
    a = cache_key("p1", "clean", "ep0", 0)
    b = cache_key("p1", "clean", "ep0", 0)
    assert a == b
    assert cache_key("p1", "clean", "ep0", 1) != a


# ---- expand_suite --------------------------------------------------------


def _cfg(perturbations: list[PerturbationGroup] | None = None) -> BenchmarkConfig:
    return BenchmarkConfig(
        name="t",
        seed=0,
        env=EnvConfig(name="tabletop"),
        pipelines=[
            PipelineConfig(
                name="P1", speech=PluginSpec(name="codec"), policy=PluginSpec(name="scripted")
            ),
            PipelineConfig(
                name="P2", speech=PluginSpec(name="codec"), policy=PluginSpec(name="random")
            ),
        ],
        perturbations=perturbations or [],
        metrics=["success_rate"],
        runner=RunnerConfig(max_steps=40, workers=1, cache_dir=None),
        output_dir="runs",
    )


def test_expand_suite_cartesian_product() -> None:
    eps = generate_dataset(SynthConfig(n_episodes=3, seed=0))
    cfg = _cfg(
        perturbations=[
            PerturbationGroup(
                name="noise", steps=[PluginSpec(name="additive_noise", params={"snr_db": 0.0})]
            ),
        ]
    )
    runs = expand_suite(cfg, eps)
    # 2 pipelines * (clean + noise) * 3 episodes = 12
    assert len(runs) == 12
    names = {r.perturbation_name for r in runs}
    assert names == {"clean", "noise"}


def test_expand_suite_clean_only_when_no_perturbations() -> None:
    eps = generate_dataset(SynthConfig(n_episodes=2, seed=0))
    cfg = _cfg()
    runs = expand_suite(cfg, eps)
    assert len(runs) == 2 * 1 * 2
    assert all(r.perturbation_name == "clean" for r in runs)


# ---- build_pipeline -----------------------------------------------------


def test_build_pipeline_missing_name_errors() -> None:
    with pytest.raises(ConfigError, match="name"):
        build_pipeline(
            name="x",
            speech={},
            grounding={"name": "rule_based"},
            policy={"name": "scripted"},
        )


# ---- Engine end-to-end --------------------------------------------------


def test_engine_runs_full_suite_and_writes_traces(tmp_path: Path) -> None:
    eps = generate_dataset(SynthConfig(n_episodes=2, seed=0))
    cfg = _cfg(
        perturbations=[
            PerturbationGroup(name="mu_law", steps=[PluginSpec(name="mu_law")]),
        ]
    )
    cfg = cfg.model_copy(update={"output_dir": str(tmp_path / "runs")})
    engine = BenchmarkEngine(cfg)
    results = engine.run(eps)
    assert len(results) == 2 * 2 * 2  # pipelines * (clean+mu_law) * episodes

    # Scripted under clean must succeed every episode (success-rate ceiling).
    scripted_clean = [r for r in results if r.pipeline == "P1" and r.perturbation == "clean"]
    assert all(r.success for r in scripted_clean)
    # Random under clean has near-zero success on a 20-step horizon.
    random_clean = [r for r in results if r.pipeline == "P2" and r.perturbation == "clean"]
    assert sum(1 for r in random_clean if r.success) <= 1

    # Trace JSON files exist and parse.
    trace_files = list((tmp_path / "runs" / "traces").glob("*.json"))
    assert len(trace_files) == len(results)
    sample = json.loads(trace_files[0].read_text())
    assert {"episode_id", "pipeline", "perturbation", "success", "n_steps"} <= sample.keys()


def test_engine_uses_cache_on_second_run(tmp_path: Path) -> None:
    eps = generate_dataset(SynthConfig(n_episodes=2, seed=0))
    cfg = _cfg().model_copy(
        update={
            "output_dir": str(tmp_path / "runs"),
            "runner": RunnerConfig(max_steps=20, workers=1, cache_dir=str(tmp_path / "cache")),
        }
    )
    BenchmarkEngine(cfg).run(eps)
    second = BenchmarkEngine(cfg).run(eps)
    # All results must arrive populated (cached values still carry success+metrics).
    assert all(r.metrics for r in second)


def test_engine_threadpool_runs(tmp_path: Path) -> None:
    eps = generate_dataset(SynthConfig(n_episodes=4, seed=0))
    cfg = _cfg().model_copy(
        update={
            "output_dir": str(tmp_path / "runs"),
            "runner": RunnerConfig(max_steps=20, workers=2, cache_dir=None),
        }
    )
    results = BenchmarkEngine(cfg).run(eps)
    # 2 pipelines * 1 (clean) * 4 = 8
    assert len(results) == 8


def test_engine_unknown_pipeline_in_suite() -> None:
    """A RunSpec referencing an unregistered pipeline name must raise ConfigError."""
    eps = generate_dataset(SynthConfig(n_episodes=1, seed=0))
    cfg = _cfg()
    cfg.pipelines[0].speech.params  # touch to keep mypy happy  # noqa: B018
    engine = BenchmarkEngine(cfg)
    # Manually mutate cfg.pipelines to remove names — reproduce the error path.
    engine.cfg = cfg.model_copy(update={"pipelines": []})
    with pytest.raises(ConfigError, match="unknown pipeline"):
        engine._get_or_build_pipeline("P1", eps)


def test_engine_runs_against_loaded_dataset(tmp_path: Path) -> None:
    """The engine should accept episodes loaded from disk too."""
    from parley.data import load_episodes, save_episodes

    eps = generate_dataset(SynthConfig(n_episodes=2, seed=0))
    save_episodes(eps, tmp_path / "ds.jsonl")
    loaded = load_episodes(tmp_path / "ds.jsonl")
    cfg = _cfg().model_copy(update={"output_dir": str(tmp_path / "runs")})
    cfg = cfg.model_copy(
        update={
            "dataset": DatasetConfig(source="file", path=str(tmp_path / "ds.jsonl"), episodes=2),
        }
    )
    results = BenchmarkEngine(cfg).run(loaded)
    assert len(results) == 4
