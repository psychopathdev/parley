"""Self-contained codec encoder/decoder.

The codec is the trick that makes Parley's CI honest: we encode the text
of an instruction into an audio waveform using a deterministic
word-id → tone-burst scheme, and decode the recovered audio with peak FFT
detection. With clean audio the round trip is perfect (WER ≈ 0). With
audio perturbations (noise, mu-law, clipping) the SNR per FFT bin drops
and the decoder mis-picks peaks, producing realistic word substitutions
and insertions — and therefore a meaningful WER.

This is *not* a real acoustic model and does not pretend to be. It is a
defensible synthetic substitute for use in tests, CI, and as a controlled
baseline against which a real Whisper adapter can be compared.

Encoder/decoder live in the same module so they share constants. They are
intentionally NOT importable from the ASR frontend file; the frontend
calls into here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---- Tunable constants ----------------------------------------------------
DEFAULT_SR = 16_000
SYMBOL_DURATION = 0.08  # 80 ms per word symbol — short enough to keep audio < 5s
GAP_DURATION = 0.02  # 20 ms gap between symbols
# Frequency grid: place symbols on a log-spaced grid between 200 Hz and 3400 Hz
# (telephony passband) so mu-law / lowpass perturbations behave realistically.
FREQ_MIN = 220.0
FREQ_MAX = 3300.0


@dataclass
class CodecConfig:
    """Parameters shared by encoder and decoder.

    ``vocab`` is the *closed-set* lexicon the codec round-trips. The synth
    dataset builds a vocab from its template space; callers pass that
    vocab here so the frontend can decode it. A real ASR has an open
    vocabulary, but for a benchmark toolkit where instructions follow a
    template the closed-set assumption is fine and keeps the math simple.
    """

    vocab: tuple[str, ...]
    sample_rate: int = DEFAULT_SR
    symbol_duration: float = SYMBOL_DURATION
    gap_duration: float = GAP_DURATION

    _freq_table: np.ndarray = field(default_factory=lambda: np.empty(0), init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.vocab:
            raise ValueError("CodecConfig.vocab must be non-empty")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("CodecConfig.vocab must be unique")
        # log-spaced grid so neighbour distances are perceptually constant
        n = len(self.vocab)
        self._freq_table = np.geomspace(FREQ_MIN, FREQ_MAX, num=n).astype(np.float64)

    @property
    def freq_table(self) -> np.ndarray:
        return self._freq_table

    @property
    def samples_per_symbol(self) -> int:
        return round(self.symbol_duration * self.sample_rate)

    @property
    def samples_per_gap(self) -> int:
        return round(self.gap_duration * self.sample_rate)

    def word_to_freq(self, word: str) -> float | None:
        try:
            idx = self.vocab.index(word)
        except ValueError:
            return None
        return float(self._freq_table[idx])


def encode(text: str, cfg: CodecConfig) -> np.ndarray:
    """Encode ``text`` into a float32 PCM waveform using ``cfg``'s vocab.

    Unknown words are silently skipped (they emit a silent gap). The
    benchmark dataset generates instructions from the closed vocab so
    this path only fires under deliberately-malformed input.
    """

    sps = cfg.samples_per_symbol
    gap = cfg.samples_per_gap
    t = np.arange(sps, dtype=np.float64) / cfg.sample_rate
    # Smooth in/out so spectral leakage is reasonable; Hann window is cheap.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(sps) / max(sps - 1, 1))

    words = text.lower().split()
    chunks: list[np.ndarray] = []
    for w in words:
        freq = cfg.word_to_freq(w)
        if freq is None:
            chunks.append(np.zeros(sps, dtype=np.float64))
        else:
            tone = 0.6 * np.sin(2.0 * np.pi * freq * t) * window
            chunks.append(tone)
        chunks.append(np.zeros(gap, dtype=np.float64))

    if not chunks:
        return np.zeros(sps, dtype=np.float32)
    audio = np.concatenate(chunks).astype(np.float32)
    # Defensive: prevent any chance of clipping.
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = (audio * (0.99 / peak)).astype(np.float32)
    return audio


def decode(audio: np.ndarray, cfg: CodecConfig) -> list[str]:
    """Decode a waveform back into the most-likely sequence of vocab words.

    The decoder slides over the waveform in (symbol + gap) sized chunks,
    computes a real FFT of each symbol, and picks the vocab frequency
    whose bin has the highest magnitude. A simple energy gate suppresses
    chunks that look like pure silence (decoded to ``<sil>`` and dropped).
    """

    if audio.ndim != 1:
        raise ValueError(f"decode expects 1-D audio, got shape {audio.shape}")

    sps = cfg.samples_per_symbol
    gap = cfg.samples_per_gap
    stride = sps + gap
    n_symbols = max(audio.shape[0] // stride, 1)
    freqs = cfg.freq_table

    # Pre-compute FFT bin indices for vocab frequencies (rounded). For
    # well-separated log-spaced tones this is far cheaper than np.argmax
    # over the full spectrum.
    fft_freqs = np.fft.rfftfreq(sps, d=1.0 / cfg.sample_rate)
    bin_idx = np.searchsorted(fft_freqs, freqs)
    # clip in case the highest tone lands at the Nyquist edge
    bin_idx = np.clip(bin_idx, 0, fft_freqs.shape[0] - 1)

    out: list[str] = []
    # A very loose energy threshold: anything below 1% of the peak symbol
    # energy looks like silence.
    energies = np.zeros(n_symbols)
    symbol_specs: list[np.ndarray] = []
    for s in range(n_symbols):
        chunk = audio[s * stride : s * stride + sps]
        if chunk.shape[0] < sps:
            chunk = np.pad(chunk, (0, sps - chunk.shape[0]))
        spec = np.abs(np.fft.rfft(chunk))
        symbol_specs.append(spec)
        energies[s] = float(np.sum(chunk * chunk))

    threshold = 0.01 * float(np.max(energies)) if energies.size else 0.0

    for s, spec in enumerate(symbol_specs):
        if energies[s] < threshold:
            continue
        # Compare vocab bins; take the strongest.
        idx = int(np.argmax(spec[bin_idx]))
        out.append(cfg.vocab[idx])

    return out
