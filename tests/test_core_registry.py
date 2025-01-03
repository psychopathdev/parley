"""Tests for the plugin :mod:`parley.core.registry`."""

from __future__ import annotations

import pytest

from parley.core.errors import RegistryError
from parley.core.registry import Registry


def test_register_and_get() -> None:
    reg: Registry[type] = Registry("widget")

    @reg.register("foo")
    class Foo:
        pass

    assert reg.get("foo") is Foo
    assert "foo" in reg
    assert reg.names() == ["foo"]
    assert len(reg) == 1


def test_duplicate_registration_raises() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register_value("a", 1)
    with pytest.raises(RegistryError, match="duplicate"):
        reg.register_value("a", 2)


def test_replace_allows_overwrite() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register_value("a", 1)
    reg.register_value("a", 2, replace=True)
    assert reg.get("a") == 2


def test_unknown_name_lists_available() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register_value("alpha", 1)
    reg.register_value("beta", 2)
    with pytest.raises(RegistryError, match=r"available: alpha, beta"):
        reg.get("gamma")


def test_iteration_is_sorted() -> None:
    reg: Registry[int] = Registry("widget")
    reg.register_value("zeta", 1)
    reg.register_value("alpha", 2)
    assert list(reg) == ["alpha", "zeta"]
