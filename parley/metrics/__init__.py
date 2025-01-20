"""Metrics — every number Parley reports.

Metrics are grouped by stage:

* :mod:`asr`         — speech recognition: WER, CER, keyword recall.
* :mod:`grounding`   — instruction parse: slot F1, exact-match.
* :mod:`action`      — action quality: MSE/MAE/cosine, DTW, success rate.
* :mod:`efficiency`  — wall-clock latency percentiles, real-time factor.
* :mod:`robustness`  — degradation curves and clean-vs-perturbed deltas.
* :mod:`aggregate`   — bootstrap CIs, paired significance tests, mean+sem.

Each metric is a callable registered under ``registry.metric``. Aggregation
operates on the metric *outputs* (a flat ``dict[str, float]``) and is
computed on the host side after all episodes have finished.
"""

from __future__ import annotations

from parley.metrics.action import ActionMSE, DTWDistance, SuccessRate
from parley.metrics.aggregate import bootstrap_ci, paired_bootstrap_pvalue, summarize
from parley.metrics.asr import CER, WER, KeywordRecall
from parley.metrics.efficiency import LatencyPercentiles
from parley.metrics.grounding import GroundingExactMatch, GroundingSlotF1
from parley.metrics.robustness import RobustnessDelta

__all__ = [
    "CER",
    "WER",
    "ActionMSE",
    "DTWDistance",
    "GroundingExactMatch",
    "GroundingSlotF1",
    "KeywordRecall",
    "LatencyPercentiles",
    "RobustnessDelta",
    "SuccessRate",
    "bootstrap_ci",
    "paired_bootstrap_pvalue",
    "summarize",
]
