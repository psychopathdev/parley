"""Efficiency / latency metrics."""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Trace
from parley.metrics.base import Metric


@registry.metric.register("latency")
class LatencyPercentiles(Metric):
    """Stage-level latency percentiles read from ``trace.timings_ms``.

    Reports p50/p95/p99 across the populated stages, plus a real-time
    factor (RTF) when audio duration is known: ``total_ms / audio_ms``.
    A pipeline that runs faster than real-time has RTF < 1.
    """

    name = "latency"

    def compute(self, trace: Trace) -> dict[str, float]:
        timings = trace.timings_ms
        if not timings:
            return {}
        values = np.array(list(timings.values()), dtype=np.float64)
        out = {
            "latency_p50_ms": float(np.percentile(values, 50)),
            "latency_p95_ms": float(np.percentile(values, 95)),
            "latency_p99_ms": float(np.percentile(values, 99)),
            "latency_total_ms": float(values.sum()),
        }
        # RTF — total processing time / audio length. Audio duration is
        # the canonical denominator in ASR papers and toolkits.
        audio_ms = trace.audio.duration * 1_000.0
        if audio_ms > 0:
            out["latency_rtf"] = out["latency_total_ms"] / audio_ms
        return out
