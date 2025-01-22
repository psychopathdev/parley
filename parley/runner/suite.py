"""Suite expansion: a config -> the cartesian product of (pipeline, pert, episode) runs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

from parley.core.config import BenchmarkConfig, PerturbationGroup, PipelineConfig
from parley.core.registry import registry
from parley.data.schema import Episode
from parley.perturb.base import Compose, Perturbation


@dataclass(frozen=True)
class RunSpec:
    """One unit of work for the engine."""

    pipeline: PipelineConfig
    perturbation_name: str
    perturbation: Perturbation | None  # None -> "clean" baseline
    episode: Episode


def _build_perturbation(group: PerturbationGroup) -> Perturbation:
    """Materialize a :class:`PerturbationGroup` from its registered names."""
    steps: list[Perturbation] = []
    for spec in group.steps:
        cls = registry.perturbation.get(spec.name)
        params = dict(spec.params or {})
        instance = cls(**params) if params else cls()
        steps.append(cast(Perturbation, instance))
    return Compose(steps, name=group.name)


def expand_suite(cfg: BenchmarkConfig, episodes: Iterable[Episode]) -> list[RunSpec]:
    """Cartesian product of pipelines x (clean + perturbations) x episodes."""

    pert_objs: list[tuple[str, Perturbation | None]] = [("clean", None)]
    for group in cfg.perturbations:
        pert_objs.append((group.name, _build_perturbation(group)))

    runs: list[RunSpec] = []
    eps = list(episodes)
    for pipeline in cfg.pipelines:
        for pname, pert in pert_objs:
            for ep in eps:
                runs.append(RunSpec(pipeline=pipeline, perturbation_name=pname,
                                    perturbation=pert, episode=ep))
    return runs
