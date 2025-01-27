"""Aggregation: bootstrap CIs, paired significance tests, mean ± sem.

These operate on flat ``Sequence[float]`` inputs (the per-episode values
of one metric). Reports compute one summary per (metric, pipeline,
perturbation) tuple.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Summary:
    """Result of aggregating one metric over a set of episodes."""

    n: int
    mean: float
    sem: float
    ci_low: float
    ci_high: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "sem": self.sem,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
        }


def summarize(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstraps: int = 1_000,
    seed: int = 0,
) -> Summary:
    """Mean + SEM + bootstrap CI of a 1-D series of values."""

    if not values:
        return Summary(n=0, mean=0.0, sem=0.0, ci_low=0.0, ci_high=0.0, confidence=confidence)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if n == 1:
        return Summary(n=n, mean=mean, sem=sem, ci_low=mean, ci_high=mean, confidence=confidence)
    lo, hi = bootstrap_ci(arr.tolist(), confidence=confidence, bootstraps=bootstraps, seed=seed)
    return Summary(n=n, mean=mean, sem=sem, ci_low=lo, ci_high=hi, confidence=confidence)


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstraps: int = 1_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic given ``seed``."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return 0.0, 0.0
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.shape[0], size=(bootstraps, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return lo, hi


def paired_bootstrap_pvalue(
    a: Sequence[float],
    b: Sequence[float],
    *,
    bootstraps: int = 1_000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value for ``mean(a) - mean(b)``.

    Both inputs must have the same length and represent paired episodes
    (same dataset row, different pipeline/perturbation). Used by the
    leaderboard to mark "significantly different" pairs.
    """

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"paired bootstrap requires equal shapes, got {arr_a.shape} vs {arr_b.shape}"
        )
    if arr_a.shape[0] == 0:
        return 1.0
    diffs = arr_a - arr_b
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    # Resample residuals around zero null
    centered = diffs - observed
    idx = rng.integers(0, diffs.shape[0], size=(bootstraps, diffs.shape[0]))
    boot_means = centered[idx].mean(axis=1)
    # Two-sided: how often does |boot| meet or exceed |observed|?
    p = float(np.mean(np.abs(boot_means) >= abs(observed)))
    return p
