"""Linguistic perturbations.

These rewrite the *text* of an instruction before it reaches a speech
frontend. They model the kinds of variation a real spoken instruction
would carry: disfluencies ("uhm, pick…"), self-corrections ("the red—
the blue cube"), and lexical accent variation.

A perturbation that returns the input unchanged still records itself in
the trace metadata (via :meth:`base.LinguisticPerturbation.apply`), so
zero-yield perturbations are observable in the report.
"""

from __future__ import annotations

import numpy as np

from parley.core.registry import registry
from parley.perturb.base import LinguisticPerturbation

DEFAULT_FILLERS: tuple[str, ...] = ("uhm", "uh", "er", "like", "you know")

# A tiny English-flavoured lexical-substitution table.  These are deliberately
# *not* phonetic substitutions — that would require a pronunciation lexicon —
# but rather "spoken-style" lexical variations the codec vocab will
# round-trip through identity (unchanged) or pull off-grid (=> misdecoded).
DEFAULT_ACCENT_MAP: dict[str, tuple[str, ...]] = {
    "the": ("da",),
    "to": ("ta",),
    "a": ("uh",),
    "and": ("an",),
    "you": ("ya",),
    "going": ("gonna",),
}


@registry.perturbation.register("disfluency")
class Disfluency(LinguisticPerturbation):
    """Insert a word repetition or restart with the given probability per slot.

    Example output for "pick the red cube" with rate=0.5 might be
    "pick pick the red cube" or "pick the the red cube". We never insert
    more than one stutter per slot to keep the perturbation interpretable.
    """

    def __init__(self, rate: float = 0.2) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("Disfluency.rate must be in [0, 1]")
        self.rate = float(rate)
        self.name = f"disfluency(rate={rate:.2f})"

    def apply_text(self, text: str, rng: np.random.Generator) -> str:
        words = text.split()
        if not words:
            return text
        out: list[str] = []
        for w in words:
            out.append(w)
            if rng.random() < self.rate:
                out.append(w)
        return " ".join(out)


@registry.perturbation.register("filler")
class FillerInsertion(LinguisticPerturbation):
    """Insert filler words ("uhm", "uh", ...) at random positions."""

    def __init__(
        self,
        rate: float = 0.15,
        fillers: tuple[str, ...] = DEFAULT_FILLERS,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("FillerInsertion.rate must be in [0, 1]")
        if not fillers:
            raise ValueError("FillerInsertion.fillers must be non-empty")
        self.rate = float(rate)
        self.fillers = tuple(fillers)
        self.name = f"filler(rate={rate:.2f})"

    def apply_text(self, text: str, rng: np.random.Generator) -> str:
        words = text.split()
        if not words:
            return text
        out: list[str] = []
        for w in words:
            if rng.random() < self.rate:
                out.append(self.fillers[int(rng.integers(0, len(self.fillers)))])
            out.append(w)
        return " ".join(out)


@registry.perturbation.register("accent_subst")
class AccentSubstitution(LinguisticPerturbation):
    """Replace selected words with spoken-style variants.

    Words *not* in the substitution table are passed through unchanged.
    A custom mapping can be supplied; the default targets common English
    function words.
    """

    def __init__(
        self,
        rate: float = 0.5,
        mapping: dict[str, tuple[str, ...]] = DEFAULT_ACCENT_MAP,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError("AccentSubstitution.rate must be in [0, 1]")
        self.rate = float(rate)
        self.mapping = {k.lower(): tuple(v) for k, v in mapping.items()}
        self.name = f"accent_subst(rate={rate:.2f})"

    def apply_text(self, text: str, rng: np.random.Generator) -> str:
        words = text.split()
        out: list[str] = []
        for w in words:
            lw = w.lower()
            choices = self.mapping.get(lw)
            if choices and rng.random() < self.rate:
                out.append(choices[int(rng.integers(0, len(choices)))])
            else:
                out.append(w)
        return " ".join(out)
