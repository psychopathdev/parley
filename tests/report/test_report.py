"""Tests for report aggregation, table rendering, leaderboard, serialize."""

from __future__ import annotations

from pathlib import Path

import pytest

from parley.core.types import EpisodeResult
from parley.report import (
    REPORT_SCHEMA_VERSION,
    aggregate_results,
    build_leaderboard,
    dump_report,
    load_report,
    render_csv,
    render_markdown,
)


def _result(pipeline: str, pert: str, ep: str, success: bool, **metrics: float) -> EpisodeResult:
    return EpisodeResult(
        pipeline=pipeline,
        perturbation=pert,
        episode_id=ep,
        success=success,
        metrics={**metrics, "success_rate": 1.0 if success else 0.0},
    )


def _bench_results() -> list[EpisodeResult]:
    return [
        _result("A", "clean", "e0", True, wer=0.0, grounding_f1=1.0),
        _result("A", "clean", "e1", True, wer=0.0, grounding_f1=1.0),
        _result("A", "noise", "e0", True, wer=0.1, grounding_f1=0.8),
        _result("A", "noise", "e1", False, wer=0.3, grounding_f1=0.5),
        _result("B", "clean", "e0", True, wer=0.0),
        _result("B", "clean", "e1", False, wer=0.0),
        _result("B", "noise", "e0", False, wer=0.4),
        _result("B", "noise", "e1", False, wer=0.5),
    ]


# ---- aggregate ----------------------------------------------------------


def test_aggregate_groups_correctly() -> None:
    rows = aggregate_results(_bench_results(), bootstraps=200, seed=0)
    keys = [(r.pipeline, r.perturbation) for r in rows]
    assert keys == [("A", "clean"), ("A", "noise"), ("B", "clean"), ("B", "noise")]
    a_clean = next(r for r in rows if r.pipeline == "A" and r.perturbation == "clean")
    assert a_clean.success_rate == 1.0
    assert a_clean.metrics["wer"].mean == 0.0


def test_aggregate_empty() -> None:
    assert aggregate_results([]) == []


def test_aggregate_skips_missing_metrics() -> None:
    """A metric only present on some episodes is summarized over what exists."""
    results = [
        _result("A", "clean", "e0", True, wer=0.0),
        _result("A", "clean", "e1", True),
    ]
    rows = aggregate_results(results, bootstraps=100, seed=0)
    assert rows[0].metrics["wer"].n == 1
    assert rows[0].metrics["wer"].mean == 0.0


# ---- table ---------------------------------------------------------------


def test_render_markdown_no_rows() -> None:
    assert render_markdown([]) == "_(no results)_"


def test_render_markdown_basic() -> None:
    rows = aggregate_results(_bench_results(), bootstraps=200, seed=0)
    md = render_markdown(rows)
    assert "| pipeline " in md
    assert "A " in md
    assert "100.00%" in md  # A's clean success


def test_render_csv_columns() -> None:
    rows = aggregate_results(_bench_results(), bootstraps=200, seed=0)
    csv = render_csv(rows)
    assert "pipeline,perturbation,n,success_rate" in csv
    assert "A,clean,2" in csv


# ---- leaderboard ---------------------------------------------------------


def test_leaderboard_ranks_by_clean_success() -> None:
    rows = aggregate_results(_bench_results(), bootstraps=200, seed=0)
    board = build_leaderboard(rows)
    assert [e.pipeline for e in board] == ["A", "B"]
    assert board[0].rank == 1
    assert board[1].rank == 2
    # A degrades from 1.0 -> 0.5 under noise; mean degradation = 0.5
    assert board[0].mean_degradation == pytest.approx(0.5)


def test_leaderboard_skips_pipeline_without_clean_baseline() -> None:
    results = [_result("X", "noise", "e0", True)]  # no clean
    rows = aggregate_results(results, bootstraps=100, seed=0)
    assert build_leaderboard(rows) == []


def test_leaderboard_tie_break_by_mean_degradation() -> None:
    """Two pipelines tied on clean success-rate are ordered by lower degradation."""
    results = [
        # A: clean=1.0, noise=0.5 -> mean_deg = 0.5
        _result("A", "clean", "e0", True),
        _result("A", "noise", "e0", False),
        # B: clean=1.0, noise=1.0 -> mean_deg = 0.0  (more robust)
        _result("B", "clean", "e0", True),
        _result("B", "noise", "e0", True),
    ]
    rows = aggregate_results(results, bootstraps=100, seed=0)
    board = build_leaderboard(rows)
    assert [e.pipeline for e in board] == ["B", "A"]


# ---- serialize -----------------------------------------------------------


def test_dump_and_load_round_trip(tmp_path: Path) -> None:
    rows = aggregate_results(_bench_results(), bootstraps=200, seed=0)
    out = dump_report(rows, tmp_path / "r.json", suite_name="t")
    body = load_report(out)
    assert body["schema_version"] == REPORT_SCHEMA_VERSION
    assert body["suite_name"] == "t"
    assert len(body["rows"]) == len(rows)
    assert body["leaderboard"][0]["rank"] == 1


def test_load_rejects_unknown_schema(tmp_path: Path) -> None:
    p = tmp_path / "r.json"
    p.write_text('{"schema_version": 999, "rows": []}', encoding="utf-8")
    with pytest.raises(Exception, match="schema_version"):
        load_report(p)
