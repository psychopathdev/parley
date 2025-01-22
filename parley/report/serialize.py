"""Versioned JSON report serialization.

The schema version lets later toolkits read older reports (or refuse
gracefully). It bumps with any breaking change to the on-disk format.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from parley._version import __version__
from parley.core.errors import ValidationError
from parley.metrics.aggregate import Summary
from parley.report.aggregate import ReportRow
from parley.report.leaderboard import LeaderboardEntry, build_leaderboard

REPORT_SCHEMA_VERSION = 1


def _summary_dict(s: Summary) -> dict[str, Any]:
    return s.as_dict()


def dump_report(
    rows: Sequence[ReportRow],
    path: str | Path,
    *,
    suite_name: str | None = None,
) -> Path:
    """Write rows + leaderboard + provenance to a JSON file."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard(rows)
    body = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "parley_version": __version__,
        "suite_name": suite_name,
        "rows": [
            {
                "pipeline": r.pipeline,
                "perturbation": r.perturbation,
                "n_episodes": r.n_episodes,
                "success_rate": r.success_rate,
                "metrics": {name: _summary_dict(s) for name, s in r.metrics.items()},
            }
            for r in rows
        ],
        "leaderboard": [
            {
                "rank": e.rank,
                "pipeline": e.pipeline,
                "clean_success_rate": e.clean_success_rate,
                "mean_degradation": e.mean_degradation,
                "max_degradation": e.max_degradation,
                "perturbations": e.perturbations,
            }
            for e in leaderboard
        ],
    }
    with p.open("w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return p


def load_report(path: str | Path) -> dict[str, Any]:
    """Read a report back, validating the schema version."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        body = json.load(f)
    if not isinstance(body, dict):
        raise ValidationError(f"{p}: top-level must be a JSON object")
    schema = body.get("schema_version")
    if schema != REPORT_SCHEMA_VERSION:
        raise ValidationError(
            f"{p}: unsupported schema_version {schema!r}; expected {REPORT_SCHEMA_VERSION}"
        )
    return body


# Re-exposed so callers can build a leaderboard view without re-importing.
__all__ = [
    "REPORT_SCHEMA_VERSION",
    "LeaderboardEntry",
    "dump_report",
    "load_report",
]
