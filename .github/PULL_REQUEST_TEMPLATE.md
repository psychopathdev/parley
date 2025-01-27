## Summary

What does this change and why?

## Checklist

- [ ] Tests added or updated for new behavior
- [ ] `ruff check parley tests` is clean
- [ ] `mypy parley tests` is clean
- [ ] `pytest -q` is green locally
- [ ] If a new plugin is added, it's registered and listed in `docs/api-reference.md`
- [ ] If a metric, perturbation, or report field changed: bumped `REPORT_SCHEMA_VERSION` (if breaking) or added a CHANGELOG entry

## Notes for reviewers

Anything subtle (numeric tolerance, RNG seed, performance regression, etc.) worth flagging.
