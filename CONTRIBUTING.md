# Contributing to Parley

Thanks for taking the time. This page collects the housekeeping rules so
you can spend the rest of it on the actual change.

## Quick start

```bash
git clone https://github.com/psychopathdev/parley.git
cd parley
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install      # optional but recommended
```

## The four gates

A PR is mergeable when:

1. `ruff check parley tests` is clean.
2. `ruff format --check parley tests` is clean.
3. `mypy parley tests` is clean.
4. `pytest -q` is green (and coverage didn't drop without a reason).

CI runs the same four gates on Python 3.10–3.13. If you change anything
non-trivial, please add a test that would fail without your change.

## Adding a plugin

The four pluggable kinds (`speech`, `grounding`, `perturbation`,
`policy`) all follow the same shape:

```python
from parley.core.registry import registry

@registry.<kind>.register("my_name")
class MyThing:
    name = "my_name"
    def <protocol method>(self, ...): ...
```

Don't forget:

- A test under `tests/<subsystem>/`.
- A line in `docs/api-reference.md` so it's discoverable.
- A `CHANGELOG.md` entry under `[Unreleased]`.

## Commit / PR style

- Conventional-Commits-friendly subject lines (`feat:`, `fix:`,
  `chore:`, `test:`, `docs:`, `ci:`) — not required, but they make the
  history skim well.
- Keep commits small and single-purpose. A 200-line PR is much easier to
  review than a 2000-line one.
- If you touch the report JSON shape, bump `REPORT_SCHEMA_VERSION` and
  describe the migration in `CHANGELOG.md`.

## Code style

- Python ≥3.10 syntax (`X | None`, structural pattern matching ok).
- Type hints everywhere in `parley/`; less strict in `tests/`.
- Docstrings on every public function/class — the *why*, not the *what*
  if the *what* is obvious from the signature.
- No `print()` in library code; the CLI uses `rich.console.Console`.
- Heavy/optional deps go behind `[project.optional-dependencies]` extras
  with lazy imports inside the adapter.

## Security

For security issues please follow [`SECURITY.md`](SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
