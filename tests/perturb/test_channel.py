"""Tests for the channel perturbations and the sweep helpers."""

from __future__ import annotations

import numpy as np

from parley.core.types import Audio
from parley.perturb import BandLimit, PacketLoss, SpectralDecimate
from parley.perturb.suites import codec_sweep, linguistic_sweep, snr_sweep


def _tone(sr: int = 16_000, dur: float = 1.0, freq: float = 1000.0) -> Audio:
    t = np.arange(int(sr * dur)) / sr
    return Audio(samples=(0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate=sr)


def test_packet_loss_zeros_proportional_to_rate() -> None:
    audio = _tone(dur=2.0)
    out = PacketLoss(loss_rate=0.5, packet_ms=20.0).apply_audio(audio, np.random.default_rng(0))
    zeros = int(np.sum(out.samples == 0.0))
    # Roughly half should be zero. Bound loosely to absorb RNG variance.
    assert 0.3 * out.samples.shape[0] <= zeros <= 0.7 * out.samples.shape[0]


def test_packet_loss_rejects_invalid_params() -> None:
    import pytest

    with pytest.raises(ValueError):
        PacketLoss(loss_rate=-0.1)
    with pytest.raises(ValueError):
        PacketLoss(packet_ms=0.0)


def test_band_limit_attenuates_out_of_band() -> None:
    """A 5 kHz tone must lose nearly all energy when restricted to 300-3400 Hz."""
    audio = _tone(dur=1.0, freq=5_000.0)
    out = BandLimit(low_hz=300.0, high_hz=3400.0).apply_audio(audio, np.random.default_rng(0))
    in_rms = float(np.sqrt(np.mean(audio.samples**2)))
    out_rms = float(np.sqrt(np.mean(out.samples**2)))
    assert out_rms < 0.05 * in_rms  # > 26 dB attenuation


def test_band_limit_preserves_in_band() -> None:
    audio = _tone(dur=1.0, freq=1_000.0)
    out = BandLimit().apply_audio(audio, np.random.default_rng(0))
    # Should preserve in-band energy within rounding
    assert float(np.corrcoef(audio.samples, out.samples)[0, 1]) > 0.99


def test_band_limit_invalid_params() -> None:
    import pytest

    with pytest.raises(ValueError):
        BandLimit(low_hz=4000.0, high_hz=300.0)


def test_spectral_decimate_drops_high_bins() -> None:
    audio = _tone(dur=0.5, freq=6_000.0)
    # Drop top 50% of bins -> a 6 kHz tone in a 0..8kHz spectrum is just above
    # the threshold and should be heavily attenuated.
    out = SpectralDecimate(drop_fraction=0.5).apply_audio(audio, np.random.default_rng(0))
    in_rms = float(np.sqrt(np.mean(audio.samples**2)))
    out_rms = float(np.sqrt(np.mean(out.samples**2)))
    assert out_rms < 0.5 * in_rms


def test_snr_sweep_names_sort_naturally() -> None:
    groups = snr_sweep(snr_dbs=(10.0, 0.0, -10.0))
    assert [g.name for g in groups] == ["snr_p10db", "snr_p0db", "snr_n10db"]
    assert groups[0].steps[0].name == "additive_noise"
    assert groups[0].steps[0].params == {"snr_db": 10.0}


def test_codec_sweep_has_canonical_set() -> None:
    names = [g.name for g in codec_sweep()]
    assert names == ["mu_law", "telephone", "spectral_decimate_40", "packet_loss_10"]


def test_linguistic_sweep_three_axes() -> None:
    names = [g.name for g in linguistic_sweep()]
    assert set(names) == {"disfluency", "filler", "accent_subst"}
