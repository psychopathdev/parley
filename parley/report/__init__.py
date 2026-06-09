"""Reporting — turn EpisodeResults into tables + leaderboards + JSON dumps."""

from __future__ import annotations

from parley.report.aggregate import ReportRow, aggregate_results
from parley.report.analysis import (
    SensitivityRow,
    WorstGroupResult,
    sensitivity_index,
    worst_group_report,
)
from parley.report.leaderboard import build_leaderboard
from parley.report.serialize import REPORT_SCHEMA_VERSION, dump_report, load_report
from parley.report.table import render_csv, render_markdown

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportRow",
    "SensitivityRow",
    "WorstGroupResult",
    "aggregate_results",
    "build_leaderboard",
    "dump_report",
    "load_report",
    "render_csv",
    "render_markdown",
    "sensitivity_index",
    "worst_group_report",
]
