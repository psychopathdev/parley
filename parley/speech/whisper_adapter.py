"""Optional Whisper adapter.

Importing ``whisper`` is gated behind the ``whisper`` install extra. We
keep the adapter file importable even when whisper isn't installed by
deferring the heavy import to construction time. Calling code that
references this class without the dep gets a clear pip-install hint.
"""

from __future__ import annotations

from typing import Any

from parley.core.registry import registry
from parley.core.types import Audio, Instruction, Transcript


@registry.speech.register("whisper")
class WhisperSpeechFrontend:
    """Thin adapter around ``openai-whisper`` (CPU or GPU).

    Install with ``pip install 'parley-bench[whisper]'``. The model is
    loaded once at construction time. The toolkit treats Whisper as
    opaque: we feed it samples and read back ``result["text"]``.

    Lazy import keeps ``parley.speech`` import-cheap and avoids a circular
    failure when whisper itself is unavailable in CI.
    """

    name = "whisper"

    def __init__(self, model: str = "tiny", device: str | None = None) -> None:
        try:
            import whisper  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "WhisperSpeechFrontend requires the 'whisper' extra: "
                "pip install 'parley-bench[whisper]'"
            ) from exc
        self._model: Any = whisper.load_model(model, device=device)
        self._device = device

    def transcribe(self, audio: Audio, *, reference: Instruction | None = None) -> Transcript:
        # Whisper expects mono float32 PCM at 16 kHz.
        if audio.sample_rate != 16_000:
            raise ValueError(f"WhisperSpeechFrontend expects 16kHz audio, got {audio.sample_rate}")
        result = self._model.transcribe(audio.samples)
        text = str(result.get("text", "")).strip()
        return Transcript(
            text=text,
            tokens=tuple(text.split()),
            confidence=None,
            metadata={"frontend": "whisper", "device": self._device or "default"},
        )
