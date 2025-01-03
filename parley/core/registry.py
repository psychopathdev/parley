"""Plugin registry.

Modules register implementations (speech frontends, perturbations, policies,
metrics, ...) by *kind* and *name*; configs reference them by string. This
is the same pattern used by lm-evaluation-harness and HELM: it lets the
toolkit be config-driven without losing type safety at the call site.

The registry is intentionally tiny and dependency-free. It does not import
anything from the rest of Parley to stay safe to import at startup.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

from parley.core.errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """A typed, named registry for one *kind* of plugin.

    Plugins register themselves via :meth:`register` (typically as a
    decorator on the class or factory). Consumers look them up by name via
    :meth:`get`; iteration is supported for "list available X" CLI
    subcommands.

    Names must be unique within a kind. Re-registering raises
    :class:`RegistryError` unless ``replace=True``, which is intended for
    test fixtures only.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(
        self,
        name: str,
        *,
        replace: bool = False,
    ) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if not replace and name in self._items:
                raise RegistryError(
                    f"duplicate {self.kind} registration: {name!r} already exists"
                )
            self._items[name] = obj
            return obj

        return decorator

    def register_value(self, name: str, value: T, *, replace: bool = False) -> T:
        """Non-decorator form for programmatic registration."""
        if not replace and name in self._items:
            raise RegistryError(
                f"duplicate {self.kind} registration: {name!r} already exists"
            )
        self._items[name] = value
        return value

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise RegistryError(
                f"unknown {self.kind} {name!r} (available: {available})"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)


class _RegistryHub:
    """Container for the per-kind registries Parley uses globally."""

    def __init__(self) -> None:
        self.speech: Registry[Callable[..., object]] = Registry("speech_frontend")
        self.grounding: Registry[Callable[..., object]] = Registry("grounding")
        self.perturbation: Registry[Callable[..., object]] = Registry("perturbation")
        self.policy: Registry[Callable[..., object]] = Registry("policy")
        self.env: Registry[Callable[..., object]] = Registry("env")
        self.metric: Registry[Callable[..., object]] = Registry("metric")
        self.task: Registry[Callable[..., object]] = Registry("task")


registry = _RegistryHub()
"""Global registry hub. Import as ``from parley.core import registry``."""
