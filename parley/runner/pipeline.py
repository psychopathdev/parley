"""A speech+grounding+policy pipeline plus its episode rollout.

This is the bit that "runs an episode": given an Episode (audio + scene +
goal), plus an optional Perturbation, it walks the env to termination
and emits a fully-populated :class:`Trace`. Stage timings are recorded
in ``trace.timings_ms`` so the latency metric can pick them up.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from parley.core.errors import ConfigError
from parley.core.registry import registry
from parley.core.rng import RngManager
from parley.core.types import (
    Audio,
    Frame,
    Grounding,
    Observation,
    StepRecord,
    Trace,
    Transcript,
)
from parley.data.schema import Episode
from parley.env.base import Environment
from parley.grounding.base import Grounder
from parley.perturb.base import Perturbation
from parley.policy.base import VLAPolicy
from parley.speech.base import SpeechFrontend


@dataclass
class Pipeline:
    """Bundle of a speech frontend, grounder, and policy.

    Held by name so traces can record exactly which plugins ran, useful
    when a suite mixes multiple pipelines.
    """

    name: str
    speech: SpeechFrontend
    grounder: Grounder
    policy: VLAPolicy


def build_pipeline(
    name: str,
    speech: dict[str, object],
    grounding: dict[str, object],
    policy: dict[str, object],
    *,
    extra_kwargs: dict[str, dict[str, object]] | None = None,
) -> Pipeline:
    """Materialize a :class:`Pipeline` from registry-style dict specs.

    Each spec is ``{"name": registered_name, "params": {...}}``. Extra
    kwargs (e.g. injecting the dataset vocab into the codec frontend)
    can be supplied via ``extra_kwargs[component_name]``.
    """
    extra = extra_kwargs or {}

    def _build(kind_registry: Any, spec: dict[str, object], component: str) -> Any:
        n = spec.get("name")
        if not isinstance(n, str):
            raise ConfigError(f"{component} spec missing 'name'")
        params = dict(cast(dict[str, Any], spec.get("params") or {}))
        params.update(extra.get(component, {}))
        cls = kind_registry.get(n)
        return cls(**params) if params else cls()

    return Pipeline(
        name=name,
        speech=cast(SpeechFrontend, _build(registry.speech, speech, "speech")),
        grounder=cast(Grounder, _build(registry.grounding, grounding, "grounding")),
        policy=cast(VLAPolicy, _build(registry.policy, policy, "policy")),
    )


def run_episode(
    *,
    pipeline: Pipeline,
    env: Environment,
    episode: Episode,
    perturbation: Perturbation | None,
    rng_mgr: RngManager,
    max_steps: int,
    perturbation_name: str = "clean",
) -> Trace:
    """Run one episode end-to-end and return its :class:`Trace`."""

    pert_rng = rng_mgr.fresh(f"perturb.{episode.episode_id}")
    env_rng = rng_mgr.fresh(f"env.{episode.episode_id}")
    policy_rng = rng_mgr.fresh(f"policy.{episode.episode_id}")

    audio = Audio(samples=episode.audio, sample_rate=episode.sample_rate)
    instr = episode.instruction
    if perturbation is not None:
        audio, instr = perturbation.apply(audio, instr, pert_rng)

    speech_ms, transcript_obj = _stopwatch(lambda: pipeline.speech.transcribe(audio, reference=instr))
    transcript = cast(Transcript, transcript_obj)

    ground_ms, grounding_obj = _stopwatch(lambda: pipeline.grounder.ground(transcript))
    grounding = cast(Grounding, grounding_obj)

    frame: Frame = env.reset(episode.scene, episode.goal, env_rng)
    pipeline.policy.reset(policy_rng)
    steps: list[StepRecord] = []
    policy_total_ms = 0.0
    info: dict[str, object] = {}
    done = False
    while not done and len(steps) < max_steps:
        obs = Observation(frame=frame, transcript=transcript, grounding=grounding)
        # Time the policy step inline to avoid B023 closure-over-loop-var warning.
        _start = time.perf_counter()
        action = pipeline.policy.act(obs)
        act_ms = (time.perf_counter() - _start) * 1_000.0
        policy_total_ms += act_ms
        frame, reward, done, info = env.step(action)
        steps.append(StepRecord(step=len(steps), action=action, reward=reward, done=done))

    success = bool(info.get("success", False))
    return Trace(
        episode_id=episode.episode_id,
        instruction=instr,
        audio=audio,
        transcript=transcript,
        grounding=grounding,
        goal=episode.goal,
        scene=episode.scene,
        steps=steps,
        success=success,
        timings_ms={
            "speech_ms": float(speech_ms),
            "grounding_ms": float(ground_ms),
            "policy_ms": float(policy_total_ms),
        },
        metadata={
            "pipeline": pipeline.name,
            "perturbation": perturbation_name,
        },
    )


def _stopwatch(thunk: Callable[[], object]) -> tuple[float, object]:
    """Wall-clock time the thunk and return ``(ms, result)``."""
    start = time.perf_counter()
    result = thunk()
    return (time.perf_counter() - start) * 1_000.0, result


def cache_key(pipeline_name: str, perturbation_name: str, episode_id: str, seed: int) -> str:
    """Stable cache key for a (pipeline, perturbation, episode, seed) tuple."""
    return f"{pipeline_name}::{perturbation_name}::{episode_id}::seed={seed}"
