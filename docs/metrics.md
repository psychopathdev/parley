# Metrics reference

What each metric measures, what scale it's on, and when to use it.

## Speech recognition

### `wer` — word error rate
Per-trace, lower is better.

Standard `(substitutions + insertions + deletions) / |reference|`,
computed via a textbook minimum-edit-distance recurrence with
backpointer reconstruction so we can also report `wer_subs`, `wer_ins`,
`wer_del`. Reference text is the `instruction.reference` (or the
instruction text if no reference is set). Both sides are lower-cased
and whitespace-split. `0 / 0` falls back to `0.0` rather than NaN.

### `cer` — character error rate
Same shape, character-level. Useful when the tokenization disagrees
between reference and hypothesis (e.g. the codec ASR sometimes emits
extra spaces).

### `keyword_recall`
Fraction of "content" reference words also present in the hypothesis.
Default keyword set covers the synthetic grammar (verbs, colors, shapes,
directions); pass `keywords=` to override.

## Grounding / intent

### `grounding_exact_match`
1.0 if every slot in `(verb, target, destination)` matches, else 0.0.
The strictest reading of "did the parser get the right intent?".

### `grounding_f1`
Token-style precision / recall / F1 over slot values. Reports
`grounding_precision`, `grounding_recall`, `grounding_f1`. Partial
credit when some slots match.

## Action / task

### `success_rate`
The headline number. 1.0 / 0.0 from the trace's `success` flag,
averaged across episodes by the aggregator. The env decides what
"success" means (see `parley.env.tabletop`: `pick` → holding the right
object; `place/push` → target object within `PLACE_TOLERANCE` of the
named direction zone).

### `action_mse` / `action_mae`
Only emitted when `trace.metadata["reference_actions"]` is set (oracle
imitation-learning style). MSE / MAE between the policy's action vectors
and the reference, truncated to the shorter length.

### `dtw`
Dynamic-time-warping distance, normalized by path length so it's
comparable across different episode horizons. Also gated on
`reference_actions`. Cheaper / fairer than indexed L2 when policies move
at different cadences.

## Efficiency

### `latency`
Computes `latency_total_ms`, `latency_p50_ms`, `latency_p95_ms`,
`latency_p99_ms`, `latency_rtf`. RTF = `total_ms / (audio_duration *
1000)` — values below 1 mean the pipeline runs faster than real time.

## Robustness (aggregator level)

These don't run per-trace; they sit on top of `aggregate_results()`
output:

### `RobustnessDelta`
Clean-vs-perturbed deltas of an underlying metric, per perturbation
group plus a mean / max across the perturbation panel.
Higher-is-better metrics → positive delta means degradation.

### `sensitivity_index`
`ΔTask / ΔInput` per (pipeline, perturbation). Default: `success_rate`
vs `wer`. A 1-point WER increase costs N points of task success.

### `worst_group_report`
Per pipeline, the lowest value of the target metric across the grouping
axis. Standard fairness/robustness reporting — captures the worst-served
condition rather than averaging it out.

## Statistical bookkeeping

Used by the aggregator and exposed for direct use:

- `summarize(values, confidence=0.95, bootstraps=1000, seed=0)` →
  `Summary(n, mean, sem, ci_low, ci_high, confidence)`.
- `bootstrap_ci(values, …)` — just the CI.
- `paired_bootstrap_pvalue(a, b, …)` — two-sided p-value on
  `mean(a) - mean(b)` for matched episodes. Used to mark significantly-
  different pipeline pairs in the leaderboard.
