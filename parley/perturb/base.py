"""Perturbation protocol and the :class:`Compose` adapter.

A :class:`Perturbation` is one of two flavours:

* :class:`AudioPerturbation` — mutates the audio in-place semantically but
  leaves the reference instruction untouched. Examples: additive noise,
  mu-law, reverb.

* :class:`LinguisticPerturbation` — mutates the *instruction text* before
  it is encoded into audio (or before it reaches a text-only frontend).
  Examples: insert "uhm", swap a word for an accent-y variant.

Both share the :meth:`apply` signature: they take an ``(audio,
instruction)`` pair and a per-call RNG, and return a (possibly new) pair.
The :class:`Compose` adapter chains them while threading the RNG.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from parley.core.types import Audio, Instruction


@runtime_checkable
class Perturbation(Protocol):
    """Base perturbation protocol.

    ``rng`` is a per-episode numpy ``Generator``; stochastic perturbations
    must use it (never ``np.random`` or a module-level generator) so the
    suite stays deterministic across episodes.
    """

    name: str

    def apply(
        self,
        audio: Audio,
        instruction: Instruction,
        rng: np.random.Generator,
    ) -> tuple[Audio, Instruction]: ...


class AudioPerturbation:
    """Convenience base for perturbations that only touch audio."""

    name: str = "audio_perturbation"

    def apply_audio(self, audio: Audio, rng: np.random.Generator) -> Audio:
        raise NotImplementedError

    def apply(
        self,
        audio: Audio,
        instruction: Instruction,
        rng: np.random.Generator,
    ) -> tuple[Audio, Instruction]:
        return self.apply_audio(audio, rng), instruction


class LinguisticPerturbation:
    """Convenience base for perturbations that only touch the instruction."""

    name: str = "linguistic_perturbation"

    def apply_text(self, text: str, rng: np.random.Generator) -> str:
        raise NotImplementedError

    def apply(
        self,
        audio: Audio,
        instruction: Instruction,
        rng: np.random.Generator,
    ) -> tuple[Audio, Instruction]:
        new_text = self.apply_text(instruction.text, rng)
        if new_text == instruction.text:
            return audio, instruction
        return audio, Instruction(
            text=new_text,
            reference=instruction.reference,  # keep gold reference intact
            language=instruction.language,
            metadata={**instruction.metadata, "perturbed_from": instruction.text},
        )


class Compose:
    """Apply several perturbations in order, sharing a single RNG.

    Empty composition acts as identity — useful as the "clean" baseline
    row in the robustness table.
    """

    def __init__(self, steps: list[Perturbation], name: str = "compose") -> None:
        self.steps = steps
        self.name = name

    def apply(
        self,
        audio: Audio,
        instruction: Instruction,
        rng: np.random.Generator,
    ) -> tuple[Audio, Instruction]:
        for step in self.steps:
            audio, instruction = step.apply(audio, instruction, rng)
        return audio, instruction
