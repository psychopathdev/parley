"""Tests for audio + linguistic perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.types import Audio, Instruction
from parley.perturb import (
    AccentSubstitution,
    AdditiveNoise,
    Clip,
    Compose,
    Disfluency,
    FillerInsertion,
    Gain,
    MuLawCodec,
    PitchShift,
    Reverb,
    TimeStretch,
)


def _tone(sr: int = 16_000, freq: float = 1000.0, dur: float = 0.5) -> Audio:
    t = np.arange(int(sr * dur)) / sr
    return Audio(samples=(0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate=sr)


def test_gain_changes_rms() -> None:
    audio = _tone()
    rng = np.random.default_rng(0)
    out = Gain(db=-6.0).apply_audio(audio, rng)
    in_rms = float(np.sqrt(np.mean(audio.samples**2)))
    out_rms = float(np.sqrt(np.mean(out.samples**2)))
    assert out_rms == pytest.approx(in_rms * 10 ** (-6 / 20.0), rel=1e-3)


def test_clip_bounds() -> None:
    audio = _tone()
    rng = np.random.default_rng(0)
    out = Clip(threshold=0.1).apply_audio(audio, rng)
    assert float(out.samples.max()) <= 0.1 + 1e-6
    assert float(out.samples.min()) >= -0.1 - 1e-6


def test_clip_invalid_threshold() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        Clip(threshold=0.0)


def test_additive_noise_snr() -> None:
    """Verify the empirical SNR roughly matches the requested target.

    With a long-enough clip the empirical estimate is tight; we use 2 s.
    """
    audio = _tone(dur=2.0)
    rng = np.random.default_rng(0)
    out = AdditiveNoise(snr_db=10.0).apply_audio(audio, rng)
    sp = float(np.mean(audio.samples**2))
    noise = out.samples - audio.samples
    np_power = float(np.mean(noise**2))
    snr = 10 * np.log10(sp / np_power)
    assert abs(snr - 10.0) < 0.6


def test_additive_noise_on_silence() -> None:
    """Silent input must round-trip unchanged (no division by zero)."""
    silence = Audio(samples=np.zeros(1600, dtype=np.float32), sample_rate=16_000)
    out = AdditiveNoise(snr_db=0.0).apply_audio(silence, np.random.default_rng(0))
    assert np.all(out.samples == 0.0)


def test_mu_law_round_trip_preserves_shape() -> None:
    audio = _tone()
    out = MuLawCodec().apply_audio(audio, np.random.default_rng(0))
    assert out.samples.shape == audio.samples.shape
    # Quantization should perturb but not destroy the signal entirely.
    assert float(np.corrcoef(audio.samples, out.samples)[0, 1]) > 0.99


def test_reverb_smears_energy() -> None:
    """Energy from a short tone should leak past the original duration."""
    sr = 16_000
    audio = Audio(
        samples=np.concatenate(
            [_tone(sr, 1000.0, 0.1).samples, np.zeros(sr // 5, dtype=np.float32)]
        ),
        sample_rate=sr,
    )
    out = Reverb(decay_ms=100.0, wet=0.8).apply_audio(audio, np.random.default_rng(0))
    # post-tone region should now carry energy
    tail_energy = float(np.sum(out.samples[sr // 10 :] ** 2))
    assert tail_energy > 1e-3


def test_reverb_invalid_params() -> None:
    with pytest.raises(ValueError, match="positive"):
        Reverb(decay_ms=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Reverb(wet=2.0)


def test_time_stretch_changes_length() -> None:
    audio = _tone()
    out = TimeStretch(rate=2.0).apply_audio(audio, np.random.default_rng(0))
    assert out.samples.shape[0] == audio.samples.shape[0] // 2


def test_pitch_shift_preserves_length() -> None:
    audio = _tone()
    out = PitchShift(semitones=3.0).apply_audio(audio, np.random.default_rng(0))
    assert out.samples.shape == audio.samples.shape


def test_disfluency_inserts_repeats_deterministically() -> None:
    perturb = Disfluency(rate=1.0)  # always stutter
    rng = np.random.default_rng(0)
    out = perturb.apply_text("pick the red cube", rng)
    # rate=1.0 ⇒ every word followed by itself
    assert out.split() == ["pick", "pick", "the", "the", "red", "red", "cube", "cube"]


def test_disfluency_identity_at_zero_rate() -> None:
    out = Disfluency(rate=0.0).apply_text("pick", np.random.default_rng(0))
    assert out == "pick"


def test_filler_invalid_rate() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FillerInsertion(rate=2.0)


def test_filler_inserts_known_token() -> None:
    out = FillerInsertion(rate=1.0, fillers=("uhm",)).apply_text("pick", np.random.default_rng(0))
    assert out.startswith("uhm")


def test_accent_substitution_only_remaps_keys() -> None:
    perturb = AccentSubstitution(rate=1.0, mapping={"the": ("da",)})
    out = perturb.apply_text("pick the red cube", np.random.default_rng(0))
    assert out == "pick da red cube"


def test_compose_chains_audio_and_linguistic() -> None:
    audio = _tone()
    instr = Instruction(text="pick the cube")
    composed = Compose(
        [
            AdditiveNoise(snr_db=20.0),
            FillerInsertion(rate=1.0, fillers=("um",)),
        ],
        name="noise+filler",
    )
    new_audio, new_instr = composed.apply(audio, instr, np.random.default_rng(0))
    assert new_audio.samples.shape == audio.samples.shape
    assert "um" in new_instr.text


def test_compose_empty_is_identity() -> None:
    audio = _tone()
    instr = Instruction(text="pick")
    new_audio, new_instr = Compose([]).apply(audio, instr, np.random.default_rng(0))
    assert new_audio is audio
    assert new_instr is instr


def test_perturbation_seed_replay_is_deterministic() -> None:
    """Two runs with the same RNG seed must produce identical outputs."""
    audio = _tone(dur=1.0)
    p = AdditiveNoise(snr_db=5.0)
    a = p.apply_audio(audio, np.random.default_rng(42)).samples
    b = p.apply_audio(audio, np.random.default_rng(42)).samples
    assert np.array_equal(a, b)
