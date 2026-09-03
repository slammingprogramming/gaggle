## Summary

<!-- What does this change, and why? Link any related issue. -->

## Checklist

- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] `mypy src` passes
- [ ] `pytest` passes locally (including any new tests for this change)
- [ ] Any new/changed config field is documented in `examples/config.yaml`
      and, if user-facing, in `docs/local-ai.md`
- [ ] Any schema change is additive (safe defaults) and, if it touches
      SQLite tables, ships a new Alembic migration under
      `src/gaggle/storage/migrations/versions/` -- see AGENTS.md invariant 5
- [ ] No real personal data, credentials, or machine-specific paths are
      included in this diff (sample media, fixtures, and docs only)

## Test plan

<!-- How did you verify this works? Real CLI output, screenshots of the
review UI, or the specific tests that cover it. -->
