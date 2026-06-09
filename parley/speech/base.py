"""The :class:`SpeechFrontend` protocol every ASR adapter implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from parley.core.types import Audio, Instruction, Transcript


@runtime_checkable
class SpeechFrontend(Protocol):
    """Convert audio into a recognized :class:`Transcript`.

    The ``reference`` argument carries the ground-truth :class:`Instruction`
    *only* so a mock frontend can replay it verbatim — a real frontend
    must ignore it. The benchmark engine passes it for symmetry; never
    leak it into your recognized output unless you are explicitly the
    perfect-ASR baseline.
    """

    name: str

    def transcribe(self, audio: Audio, *, reference: Instruction | None = None) -> Transcript: ...
