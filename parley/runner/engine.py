"""The orchestrator. Iterates the expanded suite, runs each unit, collects results."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from parley.core.config import BenchmarkConfig
from parley.core.errors import ConfigError
from parley.core.registry import registry
from parley.core.rng import RngManager
from parley.core.types import EpisodeResult, Trace
from parley.data.schema import Episode
from parley.data.synth import SynthConfig, vocab_for
from parley.env.base import Environment
from parley.metrics.base import Metric
from parley.runner.cache import ContentCache
from parley.runner.pipeline import Pipeline, build_pipeline, cache_key, run_episode
from parley.runner.suite import RunSpec, expand_suite


@dataclass
class EngineConfig:
    """Knobs shared across all RunSpecs in a single ``run`` invocation."""

    seed: int = 0
    max_steps: int = 64
    workers: int = 1
    cache_dir: str | None = None
    output_dir: str = "runs"


@dataclass
class _Artifacts:
    results: list[EpisodeResult] = field(default_factory=list)
    traces_dir: Path | None = None


class BenchmarkEngine:
    """Run an expanded suite of :class:`RunSpec` items.

    Pipelines are built lazily: the first time a ``RunSpec`` references
    a pipeline name, we materialize it and cache. Real-world pipelines
    that load weights into memory only do so once per run.
    """

    def __init__(
        self,
        cfg: BenchmarkConfig,
        engine_cfg: EngineConfig | None = None,
    ) -> None:
        self.cfg = cfg
        self.engine_cfg = engine_cfg or EngineConfig(
            seed=cfg.seed,
            max_steps=cfg.runner.max_steps,
            workers=cfg.runner.workers,
            cache_dir=cfg.runner.cache_dir,
            output_dir=cfg.output_dir,
        )
        self._cache = ContentCache(self.engine_cfg.cache_dir)
        self._pipelines: dict[str, Pipeline] = {}
        self._metrics: list[Metric] = self._build_metrics(cfg.metrics)

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def run(self, episodes: Sequence[Episode]) -> list[EpisodeResult]:
        """Run the full suite and return all per-episode results."""
        if not episodes:
            return []
        suite = expand_suite(self.cfg, episodes)
        env_factory = self._env_factory()

        out_dir = Path(self.engine_cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = out_dir / "traces"
        traces_dir.mkdir(exist_ok=True)

        if self.engine_cfg.workers <= 1:
            return [self._run_one(spec, env_factory, traces_dir, episodes) for spec in suite]

        with ThreadPoolExecutor(max_workers=self.engine_cfg.workers) as ex:
            futures = [
                ex.submit(self._run_one, spec, env_factory, traces_dir, episodes) for spec in suite
            ]
            return [f.result() for f in futures]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_one(
        self,
        spec: RunSpec,
        env_factory: Callable[[], Environment],
        traces_dir: Path,
        all_episodes: Sequence[Episode],
    ) -> EpisodeResult:
        key = cache_key(
            spec.pipeline.name,
            spec.perturbation_name,
            spec.episode.episode_id,
            self.engine_cfg.seed,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return EpisodeResult(
                pipeline=cached["pipeline"],
                perturbation=cached["perturbation"],
                episode_id=cached["episode_id"],
                success=cached["success"],
                metrics=dict(cached.get("metrics", {})),
                trace_path=cached.get("trace_path"),
            )

        pipeline = self._get_or_build_pipeline(spec.pipeline.name, all_episodes)
        env = env_factory()
        rng_mgr = RngManager(seed=self.engine_cfg.seed)
        trace = run_episode(
            pipeline=pipeline,
            env=env,
            episode=spec.episode,
            perturbation=spec.perturbation,
            rng_mgr=rng_mgr,
            max_steps=self.engine_cfg.max_steps,
            perturbation_name=spec.perturbation_name,
        )
        metrics = self._compute_metrics(trace)
        trace_path = self._dump_trace(trace, traces_dir, spec)
        result = EpisodeResult(
            pipeline=spec.pipeline.name,
            perturbation=spec.perturbation_name,
            episode_id=trace.episode_id,
            success=trace.success,
            metrics=metrics,
            trace_path=str(trace_path) if trace_path else None,
        )
        self._cache.put(
            key,
            {
                "pipeline": result.pipeline,
                "perturbation": result.perturbation,
                "episode_id": result.episode_id,
                "success": result.success,
                "metrics": result.metrics,
                "trace_path": result.trace_path,
            },
        )
        return result

    def _get_or_build_pipeline(self, name: str, episodes: Sequence[Episode]) -> Pipeline:
        cached = self._pipelines.get(name)
        if cached is not None:
            return cached
        spec = next((p for p in self.cfg.pipelines if p.name == name), None)
        if spec is None:
            raise ConfigError(f"unknown pipeline name in suite: {name!r}")
        # The codec frontend needs the dataset vocab. We provide it
        # transparently when the spec asks for the codec by name.
        extra: dict[str, dict[str, object]] = {}
        if spec.speech.name == "codec":
            sample_rate = int(episodes[0].sample_rate) if episodes else 16_000
            extra["speech"] = {"vocab": vocab_for(SynthConfig(sample_rate=sample_rate))}
        pipeline = build_pipeline(
            name=name,
            speech=dict(spec.speech.model_dump()),
            grounding=dict(spec.grounding.model_dump()),
            policy=dict(spec.policy.model_dump()),
            extra_kwargs=extra,
        )
        self._pipelines[name] = pipeline
        return pipeline

    def _build_metrics(self, names: Sequence[str]) -> list[Metric]:
        metrics: list[Metric] = []
        for n in names:
            cls = registry.metric.get(n)
            metrics.append(cast(Metric, cls()))
        return metrics

    def _compute_metrics(self, trace: Trace) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self._metrics:
            out.update(m.compute(trace))
        return out

    def _env_factory(self) -> Callable[[], Environment]:
        env_name = self.cfg.env.name
        cls = registry.env.get(env_name)
        params: dict[str, Any] = dict(self.cfg.env.params or {})
        params.setdefault("max_steps", self.engine_cfg.max_steps)

        def factory() -> Environment:
            return cast(Environment, cls(**params))

        return factory

    def _dump_trace(self, trace: Trace, traces_dir: Path, spec: RunSpec) -> Path | None:
        """Persist a JSON-serializable summary of the trace.

        We deliberately drop the audio waveform — it's recoverable from
        the dataset + perturbation seed, and including it would balloon
        run directories.
        """
        path = (
            traces_dir / f"{spec.pipeline.name}--{spec.perturbation_name}--{trace.episode_id}.json"
        )
        body = {
            "episode_id": trace.episode_id,
            "pipeline": spec.pipeline.name,
            "perturbation": spec.perturbation_name,
            "instruction": {
                "text": trace.instruction.text,
                "reference": trace.instruction.reference,
            },
            "transcript": {"text": trace.transcript.text},
            "grounding": (
                None
                if trace.grounding is None
                else {
                    "verb": trace.grounding.verb,
                    "target": trace.grounding.target,
                    "destination": trace.grounding.destination,
                }
            ),
            "goal": {
                "verb": trace.goal.verb,
                "target": trace.goal.target,
                "destination": trace.goal.destination,
            },
            "n_steps": len(trace.steps),
            "success": trace.success,
            "timings_ms": trace.timings_ms,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(body, f)
        return path
