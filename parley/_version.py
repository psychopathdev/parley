"""Single source of truth for the package version.

Kept in its own module so tooling that reads the version (CLI, report schema,
release scripts) does not pay the cost of importing the rest of the package.
"""

__version__ = "0.1.0"
