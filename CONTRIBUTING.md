# Contributing to Gaggle

Thanks for considering a contribution. This is a small guide for human
contributors; **`AGENTS.md`** is the deeper working reference (module map,
non-negotiable invariants, conventions for future changes) and applies to
every contributor, human or AI-assisted -- read it before a non-trivial
change.

**Security vulnerability?** Don't open an issue or PR describing it --
see [`SECURITY.md`](SECURITY.md) for the private reporting process instead.

## Before you start

For anything beyond a small fix, open an issue first (or comment on an
existing one) describing what you want to change and why. This avoids
duplicated work and lets us discuss design before code is written --
several existing subsystems (recognition review, event splitting, the
plugin system) went through real design discussion before implementation.

## Development setup

See [`docs/developer-setup.md`](docs/developer-setup.md) for full details,
including Docker/devcontainer options. Summary:

```bash
pip install -e .[dev,vision,cloud]    # requires ffmpeg/ffprobe/tesseract on PATH
pre-commit install                    # runs ruff + mypy on every commit
```

## Making a change

```bash
gaggle workspace init --workspace ./workspace
gaggle ingest examples/sample_media --workspace ./workspace
gaggle analyze --workspace ./workspace
gaggle enrich --workspace ./workspace

ruff check .
ruff format --check .
mypy src                              # strict mode, scoped to src/ not tests/
pytest                                # some tests skip automatically without ffmpeg/tesseract
pytest --cov=gaggle --cov-report=term-missing
```

All four (`ruff check`, `ruff format --check`, `mypy src`, `pytest`) must
pass before a PR is reviewed -- CI runs the same checks (plus a Docker
build/smoke test) on every push and PR.

## Ground rules that shape review

These are the ones that come up most; the full list is AGENTS.md's
"Non-negotiable invariants" section.

- **Never mutate original media or edit an already-written `event.json` in
  place.** Every change is a new revision (`storage/repository.py::save_event_revision`)
  or a new append-only log entry. This is a forensic tool -- the audit
  trail matters as much as the feature.
- **Schema changes are additive, always.** New Pydantic fields get a
  default (`Field(default_factory=...)` or `| None = None`); a widened
  `Literal` never removes an existing value. A new SQLite column/table
  goes through a new Alembic migration under
  `src/gaggle/storage/migrations/versions/`, verified to upgrade an
  existing (pre-migration) database in place -- never assume a rebuild is
  acceptable. Bump the relevant `*_SCHEMA_VERSION` constant.
- **No single weak signal reaches high severity alone.** A new detector or
  inference rule must preserve the corroboration requirement --
  `ScoringService` already enforces "at least two distinct signal types"
  for medium/high severity; don't build a path around that for one
  signal type.
- **Recognition (face/plate/vehicle/person) never resolves identity.** No
  name lookup, no external database correlation, no networking with other
  cameras/users. See `docs/forensic-considerations.md`.
- **New optional dependencies degrade gracefully.** If a capability needs
  a new pip extra or a downloaded model, missing it must log once and
  skip that capability -- never crash the pipeline. See
  `enrichment/face_auraface.py::insightface_available()` for the pattern
  (including the broadened `except Exception`, not just `ImportError`,
  for real native-extension failure modes).
- **Tests never touch the real network.** The CI/test sandbox has no
  network access. Any test involving a downloaded model mocks the
  download and, where useful, the inference call too (see
  `tests/unit/test_model_registry.py`, `tests/unit/test_gunshot_analysis.py`)
  -- real end-to-end verification of a new model/dependency happens
  interactively during development, not in the committed test suite.
- **Be honest about accuracy in docs, not just in code.** If a detector or
  classifier is unvalidated against real-world data, or has a known
  confusion (e.g. gunshot-vs-firework), that goes in `docs/limitations.md`
  in plain language, not buried in a code comment.

## Adding a new capability without touching core

Many things that look like "core changes" are actually a good fit for the
plugin system (`DetectorPlugin` / `InferenceRulePlugin` / `ExporterPlugin` /
review extensions) -- see
[`docs/plugin-authoring.md`](docs/plugin-authoring.md) before adding a new
built-in.

## Commit style

Commit messages should explain *why*, not just restate the diff. Look at
recent `git log` output for the house style. Sign off is not required, but
GPG/SSH-signed commits are appreciated.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
