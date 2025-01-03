"""Pytest configuration shared by every test module."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.rng import RngManager


@pytest.fixture
def rng() -> RngManager:
    """A deterministically-seeded RNG for tests that need randomness."""
    return RngManager(seed=12345)


@pytest.fixture
def small_wave() -> np.ndarray:
    """A 0.1s 440 Hz tone at 16 kHz — convenient for audio-shape assertions."""
    sr = 16_000
    t = np.linspace(0.0, 0.1, sr // 10, endpoint=False, dtype=np.float32)
    wave: np.ndarray = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    return wave
