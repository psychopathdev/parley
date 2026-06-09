"""Metric protocol shared by ASR/grounding/action/efficiency metrics.

A metric reads a :class:`Trace` and returns a dict of named scalar values.
Returning a dict (rather than a single float) lets one class emit several
numbers at once — common for things like WER, which naturally pairs with
substitution / insertion / deletion counts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parley.core.types import Trace


@runtime_checkable
class Metric(Protocol):
    """Compute one or more named scalar values from a :class:`Trace`."""

    name: str

    def compute(self, trace: Trace) -> dict[str, float]: ...
