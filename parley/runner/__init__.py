"""Runner — turns a config + dataset into traces and EpisodeResults.

Three layers:

* :class:`Pipeline` composes a speech frontend + grounder + policy
  (everything from the inputs of an episode up to its actions).
* :class:`BenchmarkEngine` orchestrates: dataset x perturbation_groups x
  pipelines, optional thread-pool parallelism, content-addressed cache.
* :func:`expand_suite` generates the cartesian product of runs from a
  :class:`BenchmarkConfig`.

The engine writes one ``EpisodeResult`` per (pipeline, perturbation,
episode), and persists the full ``Trace`` JSON to ``output_dir`` so
reports can be regenerated without re-running.
"""

from __future__ import annotations

from parley.runner.cache import ContentCache
from parley.runner.engine import BenchmarkEngine, EngineConfig
from parley.runner.pipeline import Pipeline, build_pipeline
from parley.runner.suite import RunSpec, expand_suite

__all__ = [
    "BenchmarkEngine",
    "ContentCache",
    "EngineConfig",
    "Pipeline",
    "RunSpec",
    "build_pipeline",
    "expand_suite",
]
