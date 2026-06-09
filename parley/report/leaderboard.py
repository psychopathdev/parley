"""Leaderboard: rank pipelines by clean success-rate and report robustness deltas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from parley.metrics.robustness import RobustnessDelta
from parley.report.aggregate import ReportRow


@dataclass
class LeaderboardEntry:
    pipeline: str
    clean_success_rate: float
    mean_degradation: float
    max_degradation: float
    perturbations: dict[str, float] = field(default_factory=dict)
    rank: int = 0


def build_leaderboard(rows: Sequence[ReportRow]) -> list[LeaderboardEntry]:
    """Group ``rows`` by pipeline and rank by clean ``success_rate``.

    Ties are broken by lower mean degradation across perturbations, then
    alphabetically by pipeline name. The result is sorted with rank=1 at
    the top.
    """

    by_pipeline: dict[str, list[ReportRow]] = {}
    for row in rows:
        by_pipeline.setdefault(row.pipeline, []).append(row)

    entries: list[LeaderboardEntry] = []
    for pipeline, bucket in by_pipeline.items():
        per_pert = {row.perturbation: row.success_rate for row in bucket}
        if "clean" not in per_pert:
            # No clean baseline -> skip; the metric isn't meaningful.
            continue
        rd = RobustnessDelta(metric="success_rate", baseline="clean")
        deltas = rd.compute_from_grouped(per_pert)
        entries.append(
            LeaderboardEntry(
                pipeline=pipeline,
                clean_success_rate=per_pert["clean"],
                mean_degradation=deltas.get("mean_degradation", 0.0),
                max_degradation=deltas.get("max_degradation", 0.0),
                perturbations=per_pert,
            )
        )

    entries.sort(
        key=lambda e: (-e.clean_success_rate, e.mean_degradation, e.pipeline),
    )
    for i, e in enumerate(entries, start=1):
        e.rank = i
    return entries
