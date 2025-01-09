"""Tests for the closed-vocab spectral codec and its frontend wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.types import Audio, Instruction
from parley.speech import CodecSpeechFrontend, MockSpeechFrontend
from parley.speech._codec import CodecConfig, decode, encode

VOCAB = (
    "pick", "place", "push", "the", "a",
    "red", "blue", "green", "yellow",
    "cube", "sphere", "block", "ball",
    "left", "right", "to",
)


def test_codec_round_trip_clean() -> None:
    cfg = CodecConfig(vocab=VOCAB)
    text = "pick the red cube"
    wave = encode(text, cfg)
    assert wave.dtype == np.float32
    assert decode(wave, cfg) == text.split()


def test_codec_handles_full_vocab() -> None:
    """Every vocab word must round-trip on its own."""
    cfg = CodecConfig(vocab=VOCAB)
    for word in VOCAB:
        assert decode(encode(word, cfg), cfg) == [word]


def test_codec_rejects_empty_vocab() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CodecConfig(vocab=())


def test_codec_rejects_duplicate_vocab() -> None:
    with pytest.raises(ValueError, match="unique"):
        CodecConfig(vocab=("a", "b", "a"))


def test_decode_rejects_stereo() -> None:
    cfg = CodecConfig(vocab=VOCAB)
    with pytest.raises(ValueError, match="1-D"):
        decode(np.zeros((2, 16), dtype=np.float32), cfg)


def test_codec_frontend_round_trip() -> None:
    fe = CodecSpeechFrontend(vocab=VOCAB)
    wave = encode("pick the blue sphere to the left", CodecConfig(vocab=VOCAB))
    t = fe.transcribe(Audio(samples=wave, sample_rate=16_000))
    assert t.text == "pick the blue sphere to the left"
    assert t.tokens == ("pick", "the", "blue", "sphere", "to", "the", "left")
    assert t.metadata["frontend"] == "codec"


def test_codec_frontend_sample_rate_mismatch() -> None:
    fe = CodecSpeechFrontend(vocab=VOCAB, sample_rate=16_000)
    wave = encode("pick", CodecConfig(vocab=VOCAB))
    with pytest.raises(ValueError, match="sample rate mismatch"):
        fe.transcribe(Audio(samples=wave, sample_rate=22_050))


def test_codec_degrades_under_heavy_noise() -> None:
    """High-noise audio must produce a WER greater than zero.

    This guards the design property: real perturbations need to actually
    perturb. We don't care which word flips — just that something does.
    """
    cfg = CodecConfig(vocab=VOCAB)
    text = "pick the red cube to the left"
    wave = encode(text, cfg)
    rng = np.random.default_rng(0)
    # noise amplitude roughly 4x the tone amplitude — well into the
    # regime where bin SNR collapses
    heavy_noise = (wave + 2.4 * rng.standard_normal(wave.shape).astype(np.float32)).astype(
        np.float32
    )
    recovered = decode(heavy_noise, cfg)
    assert recovered != text.split()


def test_mock_frontend_passes_reference() -> None:
    fe = MockSpeechFrontend()
    t = fe.transcribe(
        Audio(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000),
        reference=Instruction(text="pick the cube"),
    )
    assert t.text == "pick the cube"
    assert t.confidence == 1.0


def test_mock_frontend_without_reference() -> None:
    fe = MockSpeechFrontend()
    t = fe.transcribe(Audio(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000))
    assert t.text == ""
