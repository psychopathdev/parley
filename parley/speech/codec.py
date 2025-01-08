"""The codec speech frontend.

Wraps :mod:`parley.speech._codec` as a registered ``SpeechFrontend``.
The vocab must be supplied (the synth dataset will pass it through), so
the user does not have to maintain a vocabulary list by hand.
"""

from __future__ import annotations

from collections.abc import Iterable

from parley.core.registry import registry
from parley.core.types import Audio, Instruction, Transcript
from parley.speech._codec import CodecConfig, decode


@registry.speech.register("codec")
class CodecSpeechFrontend:
    """Closed-vocab spectral codec.

    Parameters
    ----------
    vocab:
        Words the codec knows about. Must match the encoder used to
        produce the dataset audio. The synthetic dataset exposes its
        vocab as ``DatasetMetadata.vocab``; pass it through.
    sample_rate:
        Must match the dataset's sample rate. Default 16 kHz.
    """

    name = "codec"

    def __init__(
        self,
        vocab: Iterable[str],
        sample_rate: int = 16_000,
        symbol_duration: float = 0.08,
        gap_duration: float = 0.02,
    ) -> None:
        self._cfg = CodecConfig(
            vocab=tuple(vocab),
            sample_rate=sample_rate,
            symbol_duration=symbol_duration,
            gap_duration=gap_duration,
        )

    def transcribe(self, audio: Audio, *, reference: Instruction | None = None) -> Transcript:
        if audio.sample_rate != self._cfg.sample_rate:
            raise ValueError(
                f"CodecSpeechFrontend: sample rate mismatch: audio={audio.sample_rate} "
                f"cfg={self._cfg.sample_rate}"
            )
        words = decode(audio.samples, self._cfg)
        text = " ".join(words)
        return Transcript(
            text=text,
            tokens=tuple(words),
            confidence=None,
            metadata={"frontend": "codec", "vocab_size": len(self._cfg.vocab)},
        )
