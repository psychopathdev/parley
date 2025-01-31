"""Perturbations applied to audio or instruction text.

A perturbation is a deterministic, seedable transform over either an
:class:`~parley.core.types.Audio` or an :class:`~parley.core.types.Instruction`.
The runner composes them into named groups; each group is a row in the
final robustness leaderboard.

We separate audio perturbations (additive noise, gain, mu-law codec,
clipping, reverb, time-stretch, pitch shift) from linguistic perturbations
(disfluency insertion, filler injection, lexical accent substitution) so
they can be stacked independently — `compose(noise_at_snr_5, disfluency)`
is a perfectly meaningful combination.
"""

from __future__ import annotations

from parley.perturb.audio import (
    AdditiveNoise,
    Clip,
    Gain,
    MuLawCodec,
    PitchShift,
    Reverb,
    TimeStretch,
)
from parley.perturb.base import AudioPerturbation, Compose, LinguisticPerturbation, Perturbation
from parley.perturb.channel import BandLimit, PacketLoss, SpectralDecimate
from parley.perturb.linguistic import AccentSubstitution, Disfluency, FillerInsertion

__all__ = [
    "AccentSubstitution",
    "AdditiveNoise",
    "AudioPerturbation",
    "BandLimit",
    "Clip",
    "Compose",
    "Disfluency",
    "FillerInsertion",
    "Gain",
    "LinguisticPerturbation",
    "MuLawCodec",
    "PacketLoss",
    "Perturbation",
    "PitchShift",
    "Reverb",
    "SpectralDecimate",
    "TimeStretch",
]
