"""Audio perturbations.

All operate on float32 PCM in ``[-1, 1]`` and leave shape and sample rate
unchanged unless explicitly noted (TimeStretch / PitchShift change length).
Stochastic ones consume the supplied numpy ``Generator`` exclusively — no
hidden global state — so seed-replay is exact.
"""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Audio
from parley.perturb.base import AudioPerturbation


def _wrap(samples: np.ndarray, audio: Audio) -> Audio:
    """Re-pack a float32 mono array as :class:`Audio` matching ``audio``'s SR."""
    return Audio(samples=np.asarray(samples, dtype=np.float32), sample_rate=audio.sample_rate)


# ---------------------------------------------------------------------------
# Amplitude / dynamic-range
# ---------------------------------------------------------------------------


@registry.perturbation.register("gain")
class Gain(AudioPerturbation):
    """Multiply the waveform by a fixed dB gain.

    Negative gain is a "quiet talker" surrogate; positive gain past about
    +6 dB starts clipping the codec tones and degrading recognition.
    """

    def __init__(self, db: float = -6.0) -> None:
        self.db = float(db)
        self.name = f"gain({db:+.1f}dB)"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        factor = 10.0 ** (self.db / 20.0)
        return _wrap(audio.samples * factor, audio)


@registry.perturbation.register("clip")
class Clip(AudioPerturbation):
    """Hard clipping: |x| <= threshold. Models cheap microphone front-ends."""

    def __init__(self, threshold: float = 0.2) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("Clip.threshold must be in (0, 1]")
        self.threshold = float(threshold)
        self.name = f"clip({threshold:.2f})"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        return _wrap(np.clip(audio.samples, -self.threshold, self.threshold), audio)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


@registry.perturbation.register("additive_noise")
class AdditiveNoise(AudioPerturbation):
    """Add Gaussian noise calibrated to a target signal-to-noise ratio (dB).

    SNR is computed from the *RMS* of the signal, which is the convention
    used in CHiME, MUSAN, and most speech-noise mixing tooling.
    """

    def __init__(self, snr_db: float = 10.0) -> None:
        self.snr_db = float(snr_db)
        self.name = f"noise(SNR={snr_db:+.0f}dB)"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        x = audio.samples.astype(np.float64)
        signal_power = float(np.mean(x * x))
        if signal_power <= 0.0:
            return audio  # nothing to add noise to
        noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
        noise = rng.standard_normal(x.shape) * np.sqrt(noise_power)
        return _wrap(x + noise, audio)


# ---------------------------------------------------------------------------
# Codec / channel
# ---------------------------------------------------------------------------


@registry.perturbation.register("mu_law")
class MuLawCodec(AudioPerturbation):
    """Round-trip the signal through G.711 μ-law (8-bit telephony codec).

    Quantization noise concentrates in low-amplitude regions, which is
    exactly where the codec frontend's silence-gate operates — so this
    perturbation tends to *insert* spurious symbols rather than drop them.
    """

    MU = 255.0

    def __init__(self, mu: float = MU) -> None:
        self.mu = float(mu)
        self.name = f"mu_law(mu={int(mu)})"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        x = np.clip(audio.samples.astype(np.float64), -1.0, 1.0)
        # Compand
        compressed = np.sign(x) * np.log1p(self.mu * np.abs(x)) / np.log1p(self.mu)
        # 8-bit quantize to 256 levels
        quantized = np.round((compressed + 1.0) * 127.5) / 127.5 - 1.0
        # Expand
        expanded = (
            np.sign(quantized)
            * (1.0 / self.mu)
            * (np.power(1.0 + self.mu, np.abs(quantized)) - 1.0)
        )
        return _wrap(expanded, audio)


# ---------------------------------------------------------------------------
# Room acoustics
# ---------------------------------------------------------------------------


@registry.perturbation.register("reverb")
class Reverb(AudioPerturbation):
    """Convolve with a synthetic exponentially-decaying impulse response.

    Not as realistic as a measured room impulse, but cheap, deterministic,
    and good enough to smear adjacent codec tones — which is the failure
    mode we care about benchmarking.
    """

    def __init__(self, decay_ms: float = 80.0, wet: float = 0.5) -> None:
        if decay_ms <= 0.0:
            raise ValueError("Reverb.decay_ms must be positive")
        if not 0.0 <= wet <= 1.0:
            raise ValueError("Reverb.wet must be in [0, 1]")
        self.decay_ms = float(decay_ms)
        self.wet = float(wet)
        self.name = f"reverb({decay_ms:.0f}ms, wet={wet:.2f})"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        sr = audio.sample_rate
        n = int(self.decay_ms * 1e-3 * sr)
        if n <= 1:
            return audio
        ir = np.exp(-3.0 * np.arange(n) / n)
        ir = ir / ir.sum()
        wet = np.convolve(audio.samples.astype(np.float64), ir, mode="full")[
            : audio.samples.shape[0]
        ]
        out = (1.0 - self.wet) * audio.samples.astype(np.float64) + self.wet * wet
        return _wrap(out, audio)


# ---------------------------------------------------------------------------
# Time/pitch — implemented via cheap resampling so we keep zero deps
# ---------------------------------------------------------------------------


def _resample_linear(x: np.ndarray, factor: float) -> np.ndarray:
    """Cheap fractional-rate resampler using linear interpolation.

    ``factor > 1`` produces a *shorter* output (faster); ``factor < 1``
    produces a longer one. Linear interp is awful for high-quality
    pitch-shifting but fine for our needs: we just want a controlled
    time-stretch / pitch-shift signal for the benchmark, not great audio.
    """
    if factor == 1.0:
        return x
    n_in = x.shape[0]
    n_out = max(round(n_in / factor), 1)
    src = np.linspace(0.0, n_in - 1.0, n_out)
    i = np.floor(src).astype(np.int64)
    j = np.minimum(i + 1, n_in - 1)
    frac = src - i
    out: np.ndarray = ((1.0 - frac) * x[i] + frac * x[j]).astype(np.float64)
    return out


@registry.perturbation.register("time_stretch")
class TimeStretch(AudioPerturbation):
    """Resample to ``rate * sample_rate`` then play back at ``sample_rate``.

    Implementation note: this *changes the length* of the waveform, which
    is exactly the property we want — codec symbols no longer align to
    the decoder's stride and start spilling across boundaries.
    """

    def __init__(self, rate: float = 1.1) -> None:
        if rate <= 0.0:
            raise ValueError("TimeStretch.rate must be positive")
        self.rate = float(rate)
        self.name = f"time_stretch(rate={rate:.2f})"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        return _wrap(_resample_linear(audio.samples.astype(np.float64), self.rate), audio)


@registry.perturbation.register("pitch_shift")
class PitchShift(AudioPerturbation):
    """Shift pitch by ``semitones`` while keeping the duration approximately fixed.

    Implementation: resample (changes pitch + length) then linearly re-pad
    or truncate to the original length. The duration recovery is naive on
    purpose — formant preservation is out of scope for the toolkit.
    """

    def __init__(self, semitones: float = 4.0) -> None:
        self.semitones = float(semitones)
        self.name = f"pitch_shift({semitones:+.0f}st)"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        n = audio.samples.shape[0]
        factor = 2.0 ** (self.semitones / 12.0)
        shifted = _resample_linear(audio.samples.astype(np.float64), factor)
        # Re-pad / truncate to the original length so downstream length
        # invariants stay intact.
        if shifted.shape[0] >= n:
            out = shifted[:n]
        else:
            out = np.zeros(n, dtype=np.float64)
            out[: shifted.shape[0]] = shifted
        return _wrap(out, audio)
