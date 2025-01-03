"""Exception hierarchy.

Everything Parley raises that callers might catch is a subclass of
:class:`ParleyError`. Code that uses a third-party library (numpy, pydantic)
lets that library's exceptions propagate unless a higher-level wrapper
makes sense.
"""

from __future__ import annotations


class ParleyError(Exception):
    """Base class for all errors raised by Parley itself."""


class ConfigError(ParleyError):
    """Raised when a config file or programmatic config is invalid.

    Wraps pydantic validation failures with a friendlier message that
    includes the file path when available.
    """


class RegistryError(ParleyError):
    """Raised by the plugin registry on unknown or duplicate names."""


class ValidationError(ParleyError):
    """Raised when a runtime artifact (dataset row, trace, action) is malformed.

    Distinct from :class:`ConfigError` (which is about user input) — this is
    about internal invariants and tends to indicate a bug or a broken plugin.
    """
