"""Reporting — turn EpisodeResults into tables + leaderboards + JSON dumps."""

from __future__ import annotations

from parley.report.aggregate import ReportRow, aggregate_results
from parley.report.leaderboard import build_leaderboard
from parley.report.serialize import REPORT_SCHEMA_VERSION, dump_report, load_report
from parley.report.table import render_csv, render_markdown

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportRow",
    "aggregate_results",
    "build_leaderboard",
    "dump_report",
    "load_report",
    "render_csv",
    "render_markdown",
]
