"""Shared dataclasses passed between modules.

These types are the *wire format* between Speech, Grounding, Policy, Env,
and Metrics. Keep them small and stable. Numpy arrays are the only "fat"
field; everything else is plain Python so traces JSON-serialize cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Audio + speech
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Audio:
    """A mono PCM waveform.

    ``samples`` is float32 in ``[-1, 1]``; ``sample_rate`` is in Hz. We keep
    only mono for v0 because every supported frontend downmixes anyway.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"Audio.samples must be 1-D, got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise ValueError(f"Audio.sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate)


@dataclass(frozen=True)
class Transcript:
    """Output of a :class:`SpeechFrontend`.

    ``text`` is the recognized transcript. ``tokens`` is the optional
    space-separated word list (kept separate so frontends that emit
    subwords can normalize before reporting). ``confidence`` is in [0,1]
    when available, otherwise ``None``.
    """

    text: str
    tokens: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Instruction + grounding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instruction:
    """A natural-language instruction with an optional canonical reference.

    ``reference`` is the gold transcript (when known), used for ASR
    metrics. It is *not* given to the policy.
    """

    text: str
    reference: str | None = None
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Grounding:
    """Parsed structured intent extracted from a transcript.

    Slot values are kept as plain strings; numeric ones (counts, indices)
    can be parsed downstream. ``confidence`` is a free-form parser score.
    """

    verb: str
    target: str | None = None
    modifier: str | None = None
    destination: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    confidence: float | None = None


# ---------------------------------------------------------------------------
# Scene + observation + action
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneSpec:
    """Procedural description of a synthetic scene.

    Object positions are 2-D (x, y) in a unit-square workspace. Colors are
    free strings so the synth generator can use any palette; the
    environment treats them as opaque labels.
    """

    objects: tuple[dict[str, Any], ...]
    workspace: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalSpec:
    """Structured goal an episode is judged against.

    Mirrors :class:`Grounding` because the success predicate compares the
    two. We keep them as separate types so a downstream policy never sees
    the goal directly.
    """

    verb: str
    target: str | None = None
    modifier: str | None = None
    destination: str | None = None
    extras: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """One environment frame given to the policy.

    A real env emits images and proprioception; the bundled synth env emits
    a low-dimensional state vector and the scene. We keep both fields
    optional so adapters can fill what they have.
    """

    state: np.ndarray
    scene: SceneSpec | None = None
    image: np.ndarray | None = None
    proprio: np.ndarray | None = None
    step: int = 0


ActionSpace = Literal["xy_delta", "xy_pick_place", "discrete"]


@dataclass(frozen=True)
class Action:
    """A policy output.

    Continuous actions live in ``vec`` (float32). Discrete actions live in
    ``label``. Pick-place actions use both: ``label`` for the verb,
    ``vec`` for the (x, y, dx, dy) coordinates. The :attr:`space` tag tells
    the environment how to interpret it.
    """

    vec: np.ndarray
    space: ActionSpace = "xy_delta"
    label: str | None = None


@dataclass(frozen=True)
class Observation:
    """Sugar bundle handed to a policy each step."""

    frame: Frame
    transcript: Transcript
    grounding: Grounding | None = None


# ---------------------------------------------------------------------------
# Episode trace + result
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One row of an episode trace. Lightweight — no images by default."""

    step: int
    action: Action
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """All the data a single episode produces.

    Carries the inputs (audio + reference instruction), the intermediate
    artifacts (transcript, grounding), the step-by-step actions, and the
    final outcome. Metrics consume ``Trace`` exclusively.
    """

    episode_id: str
    instruction: Instruction
    audio: Audio
    transcript: Transcript
    grounding: Grounding | None
    goal: GoalSpec
    scene: SceneSpec
    steps: list[StepRecord] = field(default_factory=list)
    success: bool = False
    timings_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """A single (pipeline, perturbation, episode) outcome with metrics.

    Produced by the engine and consumed by reporting. ``metrics`` is a
    free-form dict so different pipelines can attach extras without
    breaking the schema.
    """

    pipeline: str
    perturbation: str
    episode_id: str
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    trace_path: str | None = None
