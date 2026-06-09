"""Analysis primitives: sensitivity index and worst-group reporting.

These operate over the same per-(pipeline, perturbation) summary rows
the standard report uses. They sit in :mod:`parley.report` rather than
:mod:`parley.metrics` because they're aggregator-level — they reason
about deltas *across* rows rather than computing a number from a
single trace.

The two patterns implemented here are the two most-cited "robustness"
reports in the literature surveyed in ``docs/design-notes.md``:

* **Sensitivity index** — ``ΔTask / ΔWER``. The slope of the
  task-success curve against the ASR error curve. High sensitivity =
  fragile pipeline (small ASR error blows up task success).
* **Worst-group metric** — the minimum, across subgroups (e.g. accent,
  L1 stratum, perturbation level), of a target metric. Standard in
  fairness / robustness reporting; reflects the experience of the
  *worst-served* population rather than the average.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from parley.report.aggregate import ReportRow


@dataclass(frozen=True)
class SensitivityRow:
    """One (pipeline, perturbation) row of the sensitivity table.

    ``delta_task`` is positive when the perturbation degrades success
    (clean - perturbed); ``delta_input`` is positive when the upstream
    metric (WER, etc.) increases under perturbation. ``ratio`` is their
    quotient — interpretable as: a 1-point increase in the input metric
    costs ``ratio`` points of task success.
    """

    pipeline: str
    perturbation: str
    delta_input: float
    delta_task: float
    ratio: float


def sensitivity_index(
    rows: Sequence[ReportRow],
    *,
    input_metric: str = "wer",
    task_metric: str = "success_rate",
    baseline: str = "clean",
) -> list[SensitivityRow]:
    """Compute ΔTask / ΔInput per (pipeline, perturbation).

    Looks up the ``baseline`` row for each pipeline and produces one
    :class:`SensitivityRow` per non-baseline perturbation. Pipelines
    without a baseline are silently skipped — the metric only makes
    sense relative to a reference point.

    A degenerate denominator (no change in the input metric) becomes
    ``inf`` for non-zero numerators and ``0.0`` for zero numerators.
    Callers can filter these out or treat them as "perturbation didn't
    actually affect the input we asked about."
    """

    # Per-pipeline lookup: {pipeline -> {perturbation -> (input, task)}}
    by_pipeline: dict[str, dict[str, tuple[float, float]]] = {}
    for row in rows:
        if task_metric == "success_rate":
            task_value = row.success_rate
        else:
            s = row.metrics.get(task_metric)
            task_value = s.mean if s is not None else 0.0
        s_in = row.metrics.get(input_metric)
        input_value = s_in.mean if s_in is not None else 0.0
        by_pipeline.setdefault(row.pipeline, {})[row.perturbation] = (input_value, task_value)

    out: list[SensitivityRow] = []
    for pipeline, per_pert in by_pipeline.items():
        base = per_pert.get(baseline)
        if base is None:
            continue
        base_input, base_task = base
        for pert, (cur_input, cur_task) in per_pert.items():
            if pert == baseline:
                continue
            delta_input = cur_input - base_input
            delta_task = base_task - cur_task  # positive = task got worse
            if delta_input == 0.0:
                ratio = 0.0 if delta_task == 0.0 else float("inf")
            else:
                ratio = delta_task / delta_input
            out.append(
                SensitivityRow(
                    pipeline=pipeline,
                    perturbation=pert,
                    delta_input=delta_input,
                    delta_task=delta_task,
                    ratio=ratio,
                )
            )
    return out


@dataclass(frozen=True)
class WorstGroupResult:
    """Worst (minimum) value of a metric across a set of grouped rows.

    ``groups`` is a list of ``(group_name, value)`` so callers can both
    see the worst value and identify the group it came from.
    """

    pipeline: str
    metric: str
    worst_group: str
    worst_value: float
    groups: list[tuple[str, float]]


def worst_group_report(
    rows: Sequence[ReportRow],
    *,
    metric: str = "success_rate",
    group_by: str = "perturbation",
) -> list[WorstGroupResult]:
    """Per pipeline, return the lowest ``metric`` over the grouping axis.

    ``group_by`` is currently ``"perturbation"`` (the column already in
    ReportRow). The function is shaped so a future ``"accent"`` or
    ``"speaker_id"`` axis fits without redesigning the surface; we'd
    extend the row schema with extra group columns then.
    """

    if group_by != "perturbation":
        raise ValueError(
            f"worst_group_report: only group_by='perturbation' is supported, got {group_by!r}"
        )

    by_pipeline: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        if metric == "success_rate":
            value = row.success_rate
        else:
            summary = row.metrics.get(metric)
            if summary is None:
                continue
            value = summary.mean
        by_pipeline.setdefault(row.pipeline, []).append((row.perturbation, value))

    out: list[WorstGroupResult] = []
    for pipeline, pairs in by_pipeline.items():
        if not pairs:
            continue
        worst_group, worst_value = min(pairs, key=lambda kv: kv[1])
        out.append(
            WorstGroupResult(
                pipeline=pipeline,
                metric=metric,
                worst_group=worst_group,
                worst_value=worst_value,
                groups=sorted(pairs),
            )
        )
    return sorted(out, key=lambda w: w.pipeline)
