"""Robustness: how a base metric degrades under perturbation.

Implemented as a *post-hoc* aggregator rather than a per-trace metric:
once you have per-(pipeline, perturbation) means for an underlying
metric, the degradation curve and clean-vs-perturbed delta fall out by
subtraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DegradationPoint:
    """One row of a degradation curve: a perturbation -> mean metric value."""

    perturbation: str
    value: float


class RobustnessDelta:
    """Compute clean-vs-perturbed deltas for a base metric.

    Usage::

        rd = RobustnessDelta(metric="success_rate", baseline="clean")
        out = rd.compute_from_grouped(per_perturbation_means)

    where ``per_perturbation_means`` is ``{perturbation_name: mean}``.
    The result is a dict with the absolute degradation per perturbation
    (clean - perturbed) and the area-under-degradation-curve when there
    are at least 3 perturbations available.
    """

    name = "robustness"

    def __init__(self, metric: str, baseline: str = "clean") -> None:
        self.metric = metric
        self.baseline = baseline

    def compute_from_grouped(self, per_pert: Mapping[str, float]) -> dict[str, float]:
        if self.baseline not in per_pert:
            raise KeyError(
                f"baseline {self.baseline!r} missing from per-perturbation map "
                f"(have: {sorted(per_pert)})"
            )
        base = per_pert[self.baseline]
        out: dict[str, float] = {}
        deltas: list[float] = []
        for name, value in per_pert.items():
            if name == self.baseline:
                continue
            delta = base - value  # positive = degradation for higher-is-better metrics
            out[f"delta__{name}"] = delta
            deltas.append(delta)
        if deltas:
            out["mean_degradation"] = sum(deltas) / len(deltas)
            out["max_degradation"] = max(deltas)
        return out
