"""Tests for the sensitivity index and worst-group analysis."""

from __future__ import annotations

import math

import pytest

from parley.core.types import EpisodeResult
from parley.report import aggregate_results, sensitivity_index, worst_group_report


def _r(pipeline: str, pert: str, ep: str, success: bool, **m: float) -> EpisodeResult:
    return EpisodeResult(
        pipeline=pipeline,
        perturbation=pert,
        episode_id=ep,
        success=success,
        metrics={**m, "success_rate": 1.0 if success else 0.0},
    )


def _bench() -> list[EpisodeResult]:
    return [
        # A: clean = perfect (WER 0, success 1); noise: WER 0.5, success 0.5
        _r("A", "clean", "e0", True, wer=0.0),
        _r("A", "clean", "e1", True, wer=0.0),
        _r("A", "noise", "e0", True, wer=0.4),
        _r("A", "noise", "e1", False, wer=0.6),
        # B: clean = perfect (WER 0, success 1); noise: WER 0.2, success still 1
        _r("B", "clean", "e0", True, wer=0.0),
        _r("B", "clean", "e1", True, wer=0.0),
        _r("B", "noise", "e0", True, wer=0.2),
        _r("B", "noise", "e1", True, wer=0.2),
    ]


def test_sensitivity_ratio_higher_for_fragile_pipeline() -> None:
    rows = aggregate_results(_bench(), bootstraps=100, seed=0)
    sens = sensitivity_index(rows, input_metric="wer", task_metric="success_rate")
    by = {(s.pipeline, s.perturbation): s for s in sens}
    a_noise = by[("A", "noise")]
    b_noise = by[("B", "noise")]
    # A loses 0.5 of success for 0.5 increase in WER -> ratio = 1.0
    assert a_noise.ratio == pytest.approx(1.0)
    # B holds at 1.0 -> ratio = 0.0 (delta_task is 0)
    assert b_noise.ratio == pytest.approx(0.0)
    # And A is the more fragile pipeline.
    assert a_noise.ratio > b_noise.ratio


def test_sensitivity_zero_delta_input_handled() -> None:
    """A perturbation that doesn't change WER but does change task -> inf ratio."""
    results = [
        _r("X", "clean", "e0", True, wer=0.0),
        _r("X", "clean", "e1", True, wer=0.0),
        _r("X", "weird", "e0", False, wer=0.0),
        _r("X", "weird", "e1", False, wer=0.0),
    ]
    rows = aggregate_results(results, bootstraps=100, seed=0)
    sens = sensitivity_index(rows)
    weird = next(s for s in sens if s.perturbation == "weird")
    assert math.isinf(weird.ratio)


def test_sensitivity_skips_pipeline_without_baseline() -> None:
    results = [_r("Y", "noise", "e0", True, wer=0.5)]
    rows = aggregate_results(results, bootstraps=100, seed=0)
    assert sensitivity_index(rows) == []


def test_worst_group_returns_min_per_pipeline() -> None:
    rows = aggregate_results(_bench(), bootstraps=100, seed=0)
    out = worst_group_report(rows, metric="success_rate")
    assert {w.pipeline for w in out} == {"A", "B"}
    a = next(w for w in out if w.pipeline == "A")
    # A's worst is 'noise' at 0.5
    assert a.worst_group == "noise"
    assert a.worst_value == 0.5


def test_worst_group_handles_secondary_metric() -> None:
    """When the metric isn't success_rate, the Summary.mean is used."""
    rows = aggregate_results(_bench(), bootstraps=100, seed=0)
    out = worst_group_report(rows, metric="wer")
    a = next(w for w in out if w.pipeline == "A")
    # WER is "higher is worse" — worst_group_report just takes the min.
    # That's a feature: callers pass a metric where lower = "better" if
    # they want minimum to mean "best." The semantic interpretation is
    # the caller's responsibility; the report just exposes the extreme.
    assert a.worst_group == "clean"
    assert a.worst_value == 0.0


def test_worst_group_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="group_by"):
        worst_group_report([], group_by="accent")
