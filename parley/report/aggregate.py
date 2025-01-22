"""Aggregation: per-(pipeline, perturbation) summaries with bootstrap CIs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from parley.core.types import EpisodeResult
from parley.metrics.aggregate import Summary, summarize


@dataclass
class ReportRow:
    """One row of the report table: a (pipeline, perturbation) cell."""

    pipeline: str
    perturbation: str
    n_episodes: int
    success_rate: float
    metrics: dict[str, Summary] = field(default_factory=dict)


def aggregate_results(
    results: Sequence[EpisodeResult],
    *,
    bootstraps: int = 1_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> list[ReportRow]:
    """Group ``results`` by (pipeline, perturbation) and summarize each metric."""

    grouped: dict[tuple[str, str], list[EpisodeResult]] = defaultdict(list)
    for r in results:
        grouped[(r.pipeline, r.perturbation)].append(r)

    rows: list[ReportRow] = []
    for (pipeline, perturbation), bucket in sorted(grouped.items()):
        n = len(bucket)
        success_rate = sum(1 for r in bucket if r.success) / n if n else 0.0
        # Union of metric names across the bucket (some metrics return {}).
        metric_names = {k for r in bucket for k in r.metrics}
        metrics: dict[str, Summary] = {}
        for name in sorted(metric_names):
            values = [r.metrics[name] for r in bucket if name in r.metrics]
            metrics[name] = summarize(
                values,
                bootstraps=bootstraps,
                seed=seed,
                confidence=confidence,
            )
        rows.append(
            ReportRow(
                pipeline=pipeline,
                perturbation=perturbation,
                n_episodes=n,
                success_rate=success_rate,
                metrics=metrics,
            )
        )
    return rows
