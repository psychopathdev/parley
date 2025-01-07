"""Perfect-ASR baseline. Useful for isolating downstream pipeline errors."""

from __future__ import annotations

from parley.core.registry import registry
from parley.core.types import Audio, Instruction, Transcript


@registry.speech.register("mock")
class MockSpeechFrontend:
    """Passes the reference instruction through verbatim.

    Lets you A/B compare a real ASR pipeline against an upper bound: any
    success-rate drop versus the mock frontend is attributable to the
    speech layer.
    """

    name = "mock"

    def transcribe(self, audio: Audio, *, reference: Instruction | None = None) -> Transcript:
        text = reference.text if reference is not None else ""
        tokens = tuple(text.split()) if text else ()
        return Transcript(text=text, tokens=tokens, confidence=1.0, metadata={"frontend": "mock"})
