"""Markdown / CSV table rendering for ReportRows.

The markdown variant is what ``parley report --format markdown`` emits;
the CSV variant feeds spreadsheet tooling and the JSON dump is the
canonical machine format (see :mod:`parley.report.serialize`).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from parley.report.aggregate import ReportRow

# Metrics worth surfacing as their own column in the headline table.
DEFAULT_COLUMNS: tuple[str, ...] = (
    "wer",
    "grounding_f1",
    "success_rate",
    "latency_total_ms",
)


def render_markdown(
    rows: Sequence[ReportRow],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> str:
    """Render a GitHub-flavored markdown table.

    Each cell shows ``mean [low, high]`` for the bootstrap CI when
    available, otherwise just the mean. ``n`` is included as the first
    metric column so it's always visible.
    """

    if not rows:
        return "_(no results)_"

    header = ["pipeline", "perturbation", "n", *columns]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for row in rows:
        cells = [row.pipeline, row.perturbation, str(row.n_episodes)]
        for col in columns:
            cells.append(_format_cell(row, col))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_csv(rows: Sequence[ReportRow], *, columns: Sequence[str] = DEFAULT_COLUMNS) -> str:
    """Render the same table as CSV (one row per (pipeline, perturbation))."""

    buf = io.StringIO()
    writer = csv.writer(buf)
    header = ["pipeline", "perturbation", "n", "success_rate"]
    for col in columns:
        header += [f"{col}_mean", f"{col}_ci_low", f"{col}_ci_high"]
    writer.writerow(header)
    for row in rows:
        out = [row.pipeline, row.perturbation, row.n_episodes, f"{row.success_rate:.4f}"]
        for col in columns:
            summary = row.metrics.get(col)
            if summary is None:
                out += ["", "", ""]
            else:
                out += [f"{summary.mean:.4f}", f"{summary.ci_low:.4f}", f"{summary.ci_high:.4f}"]
        writer.writerow(out)
    return buf.getvalue()


def _format_cell(row: ReportRow, col: str) -> str:
    if col == "success_rate":
        return f"{row.success_rate:.2%}"
    summary = row.metrics.get(col)
    if summary is None:
        return "—"
    if summary.n <= 1 or summary.ci_low == summary.ci_high == summary.mean:
        return f"{summary.mean:.3f}"
    return f"{summary.mean:.3f} [{summary.ci_low:.3f}, {summary.ci_high:.3f}]"
