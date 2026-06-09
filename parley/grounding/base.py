"""Grounder protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parley.core.types import Grounding, Transcript


@runtime_checkable
class Grounder(Protocol):
    """Convert a recognized :class:`Transcript` into a :class:`Grounding`."""

    name: str

    def ground(self, transcript: Transcript) -> Grounding: ...
