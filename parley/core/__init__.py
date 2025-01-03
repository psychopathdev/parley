"""Shared core contracts and utilities used across Parley.

This subpackage is deliberately small and stable: every other module imports
from `parley.core`, so churn here is expensive. Public re-exports below.
"""

from parley.core.errors import (
    ConfigError,
    ParleyError,
    RegistryError,
    ValidationError,
)
from parley.core.registry import Registry, registry
from parley.core.rng import RngManager, derive_seed
from parley.core.types import (
    Action,
    ActionSpace,
    Audio,
    EpisodeResult,
    Frame,
    GoalSpec,
    Grounding,
    Instruction,
    Observation,
    SceneSpec,
    StepRecord,
    Trace,
    Transcript,
)

__all__ = [
    # types
    "Action",
    "ActionSpace",
    "Audio",
    # errors
    "ConfigError",
    "EpisodeResult",
    "Frame",
    "GoalSpec",
    "Grounding",
    "Instruction",
    "Observation",
    "ParleyError",
    # registry
    "Registry",
    "RegistryError",
    # rng
    "RngManager",
    "SceneSpec",
    "StepRecord",
    "Trace",
    "Transcript",
    "ValidationError",
    "derive_seed",
    "registry",
]
