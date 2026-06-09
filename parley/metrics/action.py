"""Action-side metrics: success, MSE, dynamic time warping."""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Trace
from parley.metrics.base import Metric


@registry.metric.register("success_rate")
class SuccessRate(Metric):
    """1.0 / 0.0 from the trace's ``success`` flag.

    The aggregator (mean across episodes) turns this into a real
    success rate — the headline number a benchmark suite reports.
    """

    name = "success_rate"

    def compute(self, trace: Trace) -> dict[str, float]:
        return {"success_rate": 1.0 if trace.success else 0.0}


@registry.metric.register("action_mse")
class ActionMSE(Metric):
    """Mean squared error between actions and a reference action sequence.

    The reference sequence lives on the trace's metadata under
    ``"reference_actions"``. If absent (no oracle reference for this
    episode), the metric is skipped (returns an empty dict).
    """

    name = "action_mse"

    def compute(self, trace: Trace) -> dict[str, float]:
        ref = trace.metadata.get("reference_actions")
        if ref is None:
            return {}
        actions = np.array([s.action.vec for s in trace.steps], dtype=np.float64)
        ref_arr = np.array(ref, dtype=np.float64)
        n = min(actions.shape[0], ref_arr.shape[0])
        if n == 0:
            return {"action_mse": 0.0, "action_mae": 0.0}
        diff = actions[:n] - ref_arr[:n]
        return {
            "action_mse": float(np.mean(diff * diff)),
            "action_mae": float(np.mean(np.abs(diff))),
        }


@registry.metric.register("dtw")
class DTWDistance(Metric):
    """Dynamic-time-warping distance between actions and reference actions.

    Cheaper than aligning by index when policies move at different
    cadences. Implemented in pure numpy; O(n*m) memory which is fine for
    sub-200-step episodes.
    """

    name = "dtw"

    def compute(self, trace: Trace) -> dict[str, float]:
        ref = trace.metadata.get("reference_actions")
        if ref is None:
            return {}
        actions = np.array([s.action.vec for s in trace.steps], dtype=np.float64)
        ref_arr = np.array(ref, dtype=np.float64)
        if actions.shape[0] == 0 or ref_arr.shape[0] == 0:
            return {"dtw": 0.0}
        n, m = actions.shape[0], ref_arr.shape[0]
        cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        cost[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d = float(np.linalg.norm(actions[i - 1] - ref_arr[j - 1]))
                cost[i, j] = d + min(cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
        # Normalize by path length so it's comparable across different
        # episode lengths.
        return {"dtw": float(cost[n, m]) / max(n + m, 1)}
