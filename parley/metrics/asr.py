"""Speech-recognition metrics.

The implementations here use a textbook minimum-edit-distance recurrence
rather than pulling in jiwer or similar — the inputs are short
instructions (a dozen tokens) so the cost is negligible and we keep the
toolkit dependency-free.

WER and CER both expose error counts (substitutions / insertions /
deletions) alongside the headline rate so reports can show breakdowns.
"""

from __future__ import annotations

from collections.abc import Iterable

from parley.core.registry import registry
from parley.core.types import Trace
from parley.metrics.base import Metric


def _edit_distance(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    """Levenshtein with operation counts.

    Returns ``(distance, substitutions, insertions, deletions)``. Standard
    DP with backpointer reconstruction; O(|ref|*|hyp|) which is fine for
    sentence-length inputs.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0, m, 0
    if m == 0:
        return n, 0, 0, n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[" "] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        bp[i][0] = "D"  # deletion
    for j in range(m + 1):
        dp[0][j] = j
        bp[0][j] = "I"  # insertion
    bp[0][0] = " "
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bp[i][j] = "M"
                continue
            sub = dp[i - 1][j - 1] + 1
            dele = dp[i - 1][j] + 1
            ins = dp[i][j - 1] + 1
            best = min(sub, dele, ins)
            dp[i][j] = best
            bp[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    # Walk the backpointer table to count ops
    i, j = n, m
    sub = ins = dele = 0
    while i > 0 or j > 0:
        op = bp[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            sub += 1
            i -= 1
            j -= 1
        elif op == "I":
            ins += 1
            j -= 1
        elif op == "D":
            dele += 1
            i -= 1
        else:  # pragma: no cover - bp is always set above
            break
    return dp[n][m], sub, ins, dele


def _safe_div(num: float, denom: float) -> float:
    """0/0 → 0; otherwise num/denom."""
    return 0.0 if denom == 0 else num / denom


@registry.metric.register("wer")
class WER(Metric):
    """Word error rate over the reference instruction text vs the transcript."""

    name = "wer"

    def compute(self, trace: Trace) -> dict[str, float]:
        ref = (trace.instruction.reference or trace.instruction.text).lower().split()
        hyp = trace.transcript.text.lower().split()
        d, sub, ins, dele = _edit_distance(ref, hyp)
        rate = _safe_div(d, len(ref))
        return {
            "wer": rate,
            "wer_subs": float(sub),
            "wer_ins": float(ins),
            "wer_del": float(dele),
            "wer_ref_len": float(len(ref)),
        }


@registry.metric.register("cer")
class CER(Metric):
    """Character error rate. Robust to tokenization disagreements between
    reference and hypothesis — useful when the codec ASR produces extra spaces."""

    name = "cer"

    def compute(self, trace: Trace) -> dict[str, float]:
        ref = list((trace.instruction.reference or trace.instruction.text).lower())
        hyp = list(trace.transcript.text.lower())
        d, _, _, _ = _edit_distance(ref, hyp)
        return {"cer": _safe_div(d, len(ref))}


@registry.metric.register("keyword_recall")
class KeywordRecall(Metric):
    """Fraction of "content" reference words also present in the transcript.

    Content words are the ones that affect downstream grounding (verbs,
    colors, shapes, directions). We pass them in as a parameter rather
    than guess; defaults match the synthetic grammar.
    """

    name = "keyword_recall"

    DEFAULT_KEYWORDS: tuple[str, ...] = (
        "pick",
        "place",
        "push",
        "red",
        "blue",
        "green",
        "yellow",
        "cube",
        "sphere",
        "block",
        "ball",
        "left",
        "right",
        "center",
    )

    def __init__(self, keywords: Iterable[str] | None = None) -> None:
        self._keywords = frozenset(k.lower() for k in (keywords or self.DEFAULT_KEYWORDS))

    def compute(self, trace: Trace) -> dict[str, float]:
        ref_words = set((trace.instruction.reference or trace.instruction.text).lower().split())
        hyp_words = set(trace.transcript.text.lower().split())
        ref_keywords = ref_words & self._keywords
        if not ref_keywords:
            return {"keyword_recall": 1.0}
        hit = len(ref_keywords & hyp_words)
        return {"keyword_recall": hit / len(ref_keywords)}
