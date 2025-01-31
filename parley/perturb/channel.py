"""Channel / dropout perturbations.

Live in their own module to keep ``audio.py`` from sprawling. These model
network-style failures rather than acoustic ones:

* :class:`PacketLoss` — drops contiguous blocks of samples, mimicking
  a lossy VoIP transport.
* :class:`BandLimit` — applies a brick-wall lowpass + highpass band, the
  spectral signature of telephony / cheap codecs.
* :class:`SpectralDecimate` — a poor-man's MP3-style perturbation: zero
  out a fraction of the high-frequency FFT bins. Cheap and
  deterministic; not a replacement for a real perceptual codec but
  exhibits qualitatively similar failure modes.
"""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.core.types import Audio
from parley.perturb.base import AudioPerturbation


def _wrap(samples: np.ndarray, audio: Audio) -> Audio:
    return Audio(samples=np.asarray(samples, dtype=np.float32), sample_rate=audio.sample_rate)


@registry.perturbation.register("packet_loss")
class PacketLoss(AudioPerturbation):
    """Drop contiguous ``loss_rate`` of the signal in ``packet_ms``-sized chunks.

    The dropped chunks are replaced with zeros rather than concealed
    (we don't pretend to implement PLC). Low ``loss_rate`` typically
    inserts spurious silence-symbols in the codec decode; higher rates
    drop real words.
    """

    def __init__(self, loss_rate: float = 0.1, packet_ms: float = 20.0) -> None:
        if not 0.0 <= loss_rate <= 1.0:
            raise ValueError("PacketLoss.loss_rate must be in [0, 1]")
        if packet_ms <= 0.0:
            raise ValueError("PacketLoss.packet_ms must be positive")
        self.loss_rate = float(loss_rate)
        self.packet_ms = float(packet_ms)
        self.name = f"packet_loss(rate={loss_rate:.2f}, pkt={packet_ms:.0f}ms)"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        sr = audio.sample_rate
        pkt = max(int(self.packet_ms * 1e-3 * sr), 1)
        n = audio.samples.shape[0]
        n_packets = max(n // pkt, 1)
        drop_mask = rng.random(n_packets) < self.loss_rate
        out = audio.samples.copy()
        for i, drop in enumerate(drop_mask):
            if not drop:
                continue
            start = i * pkt
            out[start : start + pkt] = 0.0
        return _wrap(out, audio)


@registry.perturbation.register("band_limit")
class BandLimit(AudioPerturbation):
    """Brick-wall band-pass via FFT zeroing.

    Default cut-offs (300 Hz / 3400 Hz) match the ITU-T G.712 narrowband
    telephony passband. Cheap and good enough to clip the highest and
    lowest codec tones.
    """

    def __init__(self, low_hz: float = 300.0, high_hz: float = 3400.0) -> None:
        if low_hz < 0 or high_hz <= low_hz:
            raise ValueError("BandLimit needs 0 <= low_hz < high_hz")
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.name = f"band_limit({low_hz:.0f}-{high_hz:.0f}Hz)"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        n = audio.samples.shape[0]
        if n == 0:
            return audio
        spec = np.fft.rfft(audio.samples.astype(np.float64))
        freqs = np.fft.rfftfreq(n, d=1.0 / audio.sample_rate)
        mask = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        spec = spec * mask
        return _wrap(np.fft.irfft(spec, n=n), audio)


@registry.perturbation.register("spectral_decimate")
class SpectralDecimate(AudioPerturbation):
    """Zero out a deterministic fraction of high-frequency FFT bins.

    A poor-man's perceptual-codec degradation: ``drop_fraction=0.3`` zeros
    out the top 30% of bins. Cheap, deterministic (no RNG), and exhibits
    the kind of spectral hole pattern real lossy codecs introduce when
    bitrate drops.
    """

    def __init__(self, drop_fraction: float = 0.4) -> None:
        if not 0.0 <= drop_fraction <= 1.0:
            raise ValueError("SpectralDecimate.drop_fraction must be in [0, 1]")
        self.drop_fraction = float(drop_fraction)
        self.name = f"spectral_decimate(drop={drop_fraction:.2f})"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        n = audio.samples.shape[0]
        if n == 0:
            return audio
        spec = np.fft.rfft(audio.samples.astype(np.float64))
        n_bins = spec.shape[0]
        keep = max(int(n_bins * (1.0 - self.drop_fraction)), 1)
        out = np.zeros_like(spec)
        out[:keep] = spec[:keep]
        return _wrap(np.fft.irfft(out, n=n), audio)
