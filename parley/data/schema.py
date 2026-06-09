"""Episode dataset schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from parley.core.types import GoalSpec, Instruction, SceneSpec


@dataclass
class Episode:
    """One row of a Parley dataset.

    Audio is kept as a float32 numpy array. The on-disk format pairs a
    metadata jsonl with an npz blob keyed by ``episode_id`` to keep
    write/read fast for large datasets.
    """

    episode_id: str
    instruction: Instruction
    audio: np.ndarray  # float32 mono
    sample_rate: int
    scene: SceneSpec
    goal: GoalSpec
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMetadata:
    """Index entry persisted alongside the audio blob.

    Used by :func:`parley.data.loader.load_episodes` to materialize
    :class:`Episode` instances without forcing the audio to be in-memory
    before it is needed.
    """

    episode_id: str
    instruction_text: str
    instruction_reference: str | None
    sample_rate: int
    scene: dict[str, Any]
    goal: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)
