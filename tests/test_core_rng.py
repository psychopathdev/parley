"""Tests for the seeded RNG manager."""

from __future__ import annotations

import numpy as np
import pytest

from parley.core.rng import RngManager, derive_seed


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(42, "abc") == derive_seed(42, "abc")


def test_derive_seed_differs_by_name() -> None:
    assert derive_seed(42, "abc") != derive_seed(42, "def")


def test_derive_seed_differs_by_parent() -> None:
    assert derive_seed(0, "x") != derive_seed(1, "x")


def test_derive_seed_rejects_negative_parent() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed(-1, "x")


def test_stream_reuses_generator() -> None:
    mgr = RngManager(seed=7)
    a = mgr.stream("noise")
    a.standard_normal(3)
    b = mgr.stream("noise")
    assert a is b  # advancing 'a' must also advance 'b'


def test_two_managers_with_same_seed_match() -> None:
    a = RngManager(seed=7).stream("noise").standard_normal(5)
    b = RngManager(seed=7).stream("noise").standard_normal(5)
    assert np.allclose(a, b)


def test_fresh_resets_stream() -> None:
    mgr = RngManager(seed=7)
    a1 = mgr.stream("x").standard_normal(3)
    mgr.fresh("x")
    a2 = mgr.stream("x").standard_normal(3)
    assert np.allclose(a1, a2)
