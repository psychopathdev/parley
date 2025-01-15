"""Environments — synthetic worlds the policies act in.

Parley ships a single dependency-free environment, :class:`TabletopEnv`,
that models a 2-D tabletop with colored shape objects. It exposes the
:class:`Environment` protocol so adapters for real simulators (LIBERO,
ManiSkill, RLBench) can be plugged in later without touching the runner.
"""

from __future__ import annotations

from parley.env.base import Environment
from parley.env.tabletop import TabletopEnv

__all__ = ["Environment", "TabletopEnv"]
