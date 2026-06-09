"""Speech frontends — anything that turns :class:`Audio` into a :class:`Transcript`.

The toolkit ships three frontends:

* :class:`MockSpeechFrontend` — copies the reference text through verbatim.
  Useful as a "perfect-ASR" baseline that isolates downstream errors.

* :class:`CodecSpeechFrontend` — the self-contained reference. Pairs with
  :class:`parley.data.synth.SynthSpeechCodec` so we can generate audio
  *from* text deterministically and recover it. Audio perturbations alter
  the recovered text in physically-plausible ways.

* :class:`WhisperSpeechFrontend` — optional adapter behind the ``whisper``
  extra. Defined so adapter shape is documented; runtime import is lazy.

A frontend is any callable implementing :class:`SpeechFrontend`, registered
under ``parley.core.registry.registry.speech``.
"""

from __future__ import annotations

from parley.speech.base import SpeechFrontend
from parley.speech.codec import CodecSpeechFrontend
from parley.speech.mock import MockSpeechFrontend
from parley.speech.whisper_adapter import WhisperSpeechFrontend

__all__ = [
    "CodecSpeechFrontend",
    "MockSpeechFrontend",
    "SpeechFrontend",
    "WhisperSpeechFrontend",
]
