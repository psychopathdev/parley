"""Grounding metrics: how well the parsed intent matches the goal."""

from __future__ import annotations

from parley.core.registry import registry
from parley.core.types import Trace
from parley.metrics.base import Metric

# Slots the synth tasks have a ground-truth value for. Keeps the F1
# denominator stable across episodes.
_GROUND_TRUTH_SLOTS: tuple[str, ...] = ("verb", "target", "destination")


def _ground_truth_view(trace: Trace) -> dict[str, str | None]:
    """Project the goal onto the same slot vocabulary the grounder uses."""
    return {
        "verb": trace.goal.verb,
        "target": trace.goal.target,
        "destination": trace.goal.destination,
    }


def _grounding_view(trace: Trace) -> dict[str, str | None]:
    g = trace.grounding
    if g is None:
        return dict.fromkeys(_GROUND_TRUTH_SLOTS)
    return {"verb": g.verb, "target": g.target, "destination": g.destination}


@registry.metric.register("grounding_exact_match")
class GroundingExactMatch(Metric):
    """1.0 if every slot matches the goal, else 0.0."""

    name = "grounding_exact_match"

    def compute(self, trace: Trace) -> dict[str, float]:
        truth = _ground_truth_view(trace)
        pred = _grounding_view(trace)
        ok = all((truth[k] or "") == (pred[k] or "") for k in _GROUND_TRUTH_SLOTS)
        return {"grounding_exact_match": 1.0 if ok else 0.0}


@registry.metric.register("grounding_f1")
class GroundingSlotF1(Metric):
    """Per-slot precision/recall/F1 over the present-vs-correct slots.

    A "correct" slot is one whose ground-truth value is non-null AND whose
    prediction matches; "predicted" is non-null prediction; "expected" is
    non-null truth. F1 is the harmonic mean.
    """

    name = "grounding_f1"

    def compute(self, trace: Trace) -> dict[str, float]:
        truth = _ground_truth_view(trace)
        pred = _grounding_view(trace)
        tp = fp = fn = 0
        for slot in _GROUND_TRUTH_SLOTS:
            t = truth[slot]
            p = pred[slot]
            if t is None and p is None:
                continue
            if t is None and p is not None:
                fp += 1
            elif t is not None and p is None:
                fn += 1
            elif t == p:
                tp += 1
            else:
                fp += 1
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"grounding_precision": prec, "grounding_recall": rec, "grounding_f1": f1}
