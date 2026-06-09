"""Tests for ASR / grounding / action / efficiency / robustness / aggregate metrics."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.types import (
    Action,
    Audio,
    GoalSpec,
    Grounding,
    Instruction,
    SceneSpec,
    StepRecord,
    Trace,
    Transcript,
)
from parley.metrics import (
    CER,
    WER,
    DTWDistance,
    GroundingExactMatch,
    GroundingSlotF1,
    KeywordRecall,
    LatencyPercentiles,
    RobustnessDelta,
    SuccessRate,
    bootstrap_ci,
    paired_bootstrap_pvalue,
    summarize,
)
from parley.metrics.action import ActionMSE


def _trace(
    *,
    ref: str = "pick the red cube",
    hyp: str = "pick the red cube",
    success: bool = True,
    grounding: Grounding | None = None,
    timings: dict[str, float] | None = None,
    audio_dur: float = 1.0,
) -> Trace:
    sr = 16_000
    return Trace(
        episode_id="ep0",
        instruction=Instruction(text=ref, reference=ref),
        audio=Audio(samples=np.zeros(int(sr * audio_dur), dtype=np.float32), sample_rate=sr),
        transcript=Transcript(text=hyp, tokens=tuple(hyp.split())),
        grounding=grounding,
        goal=GoalSpec(verb="pick", target="red cube"),
        scene=SceneSpec(objects=()),
        steps=[
            StepRecord(step=0, action=Action(vec=np.zeros(2, dtype=np.float32)),
                       reward=0.0, done=False)
        ],
        success=success,
        timings_ms=timings or {},
    )


# ---- WER / CER / keyword recall -----------------------------------------


def test_wer_zero_when_identical() -> None:
    out = WER().compute(_trace())
    assert out["wer"] == 0.0
    assert out["wer_subs"] == 0.0
    assert out["wer_ins"] == 0.0
    assert out["wer_del"] == 0.0


def test_wer_substitution() -> None:
    out = WER().compute(_trace(ref="pick the red cube", hyp="pick the blue cube"))
    assert out["wer_subs"] == 1.0
    assert out["wer"] == pytest.approx(0.25)


def test_wer_insertion() -> None:
    out = WER().compute(_trace(ref="pick the cube", hyp="pick the red cube"))
    assert out["wer_ins"] == 1.0


def test_wer_deletion() -> None:
    out = WER().compute(_trace(ref="pick the red cube", hyp="pick red cube"))
    assert out["wer_del"] == 1.0


def test_wer_empty_reference_returns_zero_division_safe() -> None:
    out = WER().compute(_trace(ref="", hyp="picked it up"))
    assert out["wer_ref_len"] == 0.0
    # safe div: 0/0 → 0
    assert out["wer"] == 0.0


def test_cer_smaller_for_close_typo() -> None:
    out = CER().compute(_trace(ref="cube", hyp="cobe"))
    assert 0.0 < out["cer"] < 0.5


def test_keyword_recall_full_when_all_present() -> None:
    out = KeywordRecall().compute(_trace())
    assert out["keyword_recall"] == 1.0


def test_keyword_recall_drops_on_missing_keyword() -> None:
    out = KeywordRecall().compute(_trace(ref="pick the red cube", hyp="pick the cube"))
    # 1 of 3 keywords missing (red)
    assert out["keyword_recall"] == pytest.approx(2 / 3)


# ---- Grounding ----------------------------------------------------------


def test_grounding_exact_match() -> None:
    g = Grounding(verb="pick", target="red cube", destination=None)
    out = GroundingExactMatch().compute(_trace(grounding=g))
    assert out["grounding_exact_match"] == 1.0


def test_grounding_exact_match_fails_on_wrong_target() -> None:
    g = Grounding(verb="pick", target="blue sphere")
    out = GroundingExactMatch().compute(_trace(grounding=g))
    assert out["grounding_exact_match"] == 0.0


def test_grounding_f1_perfect() -> None:
    g = Grounding(verb="pick", target="red cube")
    out = GroundingSlotF1().compute(_trace(grounding=g))
    assert out["grounding_f1"] == 1.0


def test_grounding_f1_partial_credit() -> None:
    """Wrong destination but right verb+target => slot F1 must be > 0."""
    g = Grounding(verb="pick", target="red cube", destination="left")  # truth has none
    out = GroundingSlotF1().compute(_trace(grounding=g))
    assert 0.5 < out["grounding_f1"] < 1.0


def test_grounding_f1_no_grounding() -> None:
    out = GroundingSlotF1().compute(_trace(grounding=None))
    assert out["grounding_f1"] == 0.0


# ---- Action -------------------------------------------------------------


def test_success_rate() -> None:
    assert SuccessRate().compute(_trace(success=True))["success_rate"] == 1.0
    assert SuccessRate().compute(_trace(success=False))["success_rate"] == 0.0


def test_action_mse_skipped_without_reference() -> None:
    out = ActionMSE().compute(_trace())
    assert out == {}


def test_action_mse_with_reference() -> None:
    trace = _trace()
    trace.metadata["reference_actions"] = [[0.0, 0.0]]
    trace.steps[0] = StepRecord(
        step=0, action=Action(vec=np.array([1.0, 1.0], dtype=np.float32)), reward=0.0, done=False
    )
    out = ActionMSE().compute(trace)
    assert out["action_mse"] == 1.0
    assert out["action_mae"] == 1.0


def test_dtw_distance_zero_for_identical() -> None:
    trace = _trace()
    ref = [[0.0, 0.0]]
    trace.metadata["reference_actions"] = ref
    out = DTWDistance().compute(trace)
    assert out["dtw"] == 0.0


# ---- Efficiency ---------------------------------------------------------


def test_latency_percentiles_and_rtf() -> None:
    trace = _trace(timings={"asr": 50.0, "ground": 1.0, "policy": 30.0}, audio_dur=2.0)
    out = LatencyPercentiles().compute(trace)
    assert out["latency_total_ms"] == pytest.approx(81.0)
    assert out["latency_p50_ms"] == 30.0
    assert out["latency_rtf"] == pytest.approx(81.0 / 2_000.0)


def test_latency_empty_timings_skipped() -> None:
    out = LatencyPercentiles().compute(_trace(timings={}))
    assert out == {}


# ---- Robustness ---------------------------------------------------------


def test_robustness_delta_basic() -> None:
    rd = RobustnessDelta(metric="success_rate", baseline="clean")
    out = rd.compute_from_grouped({"clean": 1.0, "noisy": 0.7, "reverb": 0.5})
    assert out["delta__noisy"] == pytest.approx(0.3)
    assert out["delta__reverb"] == pytest.approx(0.5)
    assert out["mean_degradation"] == pytest.approx(0.4)
    assert out["max_degradation"] == pytest.approx(0.5)


def test_robustness_delta_missing_baseline() -> None:
    rd = RobustnessDelta(metric="success_rate")
    with pytest.raises(KeyError, match="baseline"):
        rd.compute_from_grouped({"noise": 0.5})


# ---- Aggregate ----------------------------------------------------------


def test_summarize_constant_series() -> None:
    s = summarize([0.5] * 10, bootstraps=200, seed=1)
    assert s.mean == 0.5
    assert s.sem == 0.0
    assert s.ci_low == s.ci_high == 0.5


def test_summarize_random_series() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0.5, 0.1, size=100).tolist()
    s = summarize(series, bootstraps=500, seed=7)
    assert s.n == 100
    assert s.ci_low < s.mean < s.ci_high


def test_bootstrap_ci_invalid_confidence() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        bootstrap_ci([1.0, 2.0], confidence=0.0)


def test_paired_bootstrap_pvalue_is_high_for_no_difference() -> None:
    # identical series → p-value should be ~1
    p = paired_bootstrap_pvalue([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], bootstraps=200, seed=0)
    assert p >= 0.9


def test_paired_bootstrap_pvalue_is_low_for_clear_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.05, size=200).tolist()
    b = rng.normal(0.4, 0.05, size=200).tolist()
    p = paired_bootstrap_pvalue(a, b, bootstraps=500, seed=0)
    assert p < 0.05


def test_paired_bootstrap_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="equal shapes"):
        paired_bootstrap_pvalue([1.0, 2.0], [1.0])
