"""Datasets — episodes a benchmark suite is run over.

Episodes carry: a spoken instruction (audio + reference text), a scene
description, and a goal predicate. Parley's synthetic generator builds
all of these procedurally so the toolkit ships with a CI-runnable dataset.
"""

from __future__ import annotations

from parley.data.loader import load_episodes, save_episodes
from parley.data.schema import Episode, EpisodeMetadata
from parley.data.synth import SynthConfig, generate_dataset

__all__ = [
    "Episode",
    "EpisodeMetadata",
    "SynthConfig",
    "generate_dataset",
    "load_episodes",
    "save_episodes",
]
