"""Programmatic Alembic driver for the SQLite timeline index.

Called exclusively from `TimelineDatabase.initialize()`. Never shells out
to the `alembic` CLI and never depends on finding `alembic.ini` relative
to the current working directory -- the repo-root `alembic.ini` exists
purely for development-time `alembic revision --autogenerate`. Production
code builds an in-memory `alembic.config.Config()` with `script_location`
resolved package-relative (`migrations/`, next to this file), so
migrations work identically whether gaggle is run from a checkout or a
`pip install`ed copy.

Three-way branch on what the sqlite file looks like when `initialize()`
runs (see `ensure_schema_up_to_date`):

1. No `alembic_version` table, no app tables either -- brand-new
   workspace. Fast path: `Base.metadata.create_all()` then stamp straight
   to head (a fresh database already has every column any migration would
   add; replaying history it never needed would be pure overhead).
2. No `alembic_version` table, but app tables present -- a workspace from
   before this feature shipped. Stamp to `0001_baseline` (defined to be
   exactly that pre-Alembic 14-table shape) then upgrade to head, so
   anything after baseline gets replayed correctly.
3. `alembic_version` table present -- normal path, just upgrade to head.

`TimelineDatabase.initialize()` runs on nearly every CLI invocation, so
the common case (already at head) must stay cheap: read the stamped
revision directly via one scalar SQL query and compare against a
once-computed head revision id, skipping the real `alembic.command.upgrade`
call (which reparses the whole migration script directory) entirely on a
match.

`TimelineDatabase.check_schema_drift()` (a separate, older mechanism)
stays as a fallback diagnostic -- with Alembic authoritative, drift would
mean a migration was skipped or a workspace was hand-edited, not the
routine upgrade path this module handles automatically.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_BASELINE_REVISION = "0001_baseline"

_head_revision_cache: str | None = None


def _build_config(sqlite_path: Path) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{sqlite_path}")
    return cfg


def _head_revision(cfg: Config) -> str:
    global _head_revision_cache
    if _head_revision_cache is None:
        script = ScriptDirectory.from_config(cfg)
        head = script.get_current_head()
        if head is None:
            raise RuntimeError(f"no migrations found under {_MIGRATIONS_DIR}")
        _head_revision_cache = head
    return _head_revision_cache


def ensure_schema_up_to_date(engine: Engine, sqlite_path: Path) -> None:
    """Bring `sqlite_path`'s schema to the latest migration, in place,
    without data loss. See the module docstring for the three-way branch."""

    from gaggle.storage.database import Base  # local import: avoids a module-level cycle

    cfg = _build_config(sqlite_path)
    head = _head_revision(cfg)
    inspector = inspect(engine)

    if inspector.has_table("alembic_version"):
        with engine.connect() as connection:
            row = connection.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
        current = row[0] if row is not None else None
        if current == head:
            return  # fast path: already at head, skip the real upgrade() call
        LOGGER.info("sqlite_schema_migrating", from_revision=current, to_revision=head)
        command.upgrade(cfg, "head")
        LOGGER.info("sqlite_schema_migrated", from_revision=current, to_revision=head)
        return

    has_app_tables = any(inspector.has_table(name) for name in Base.metadata.tables)
    if has_app_tables:
        LOGGER.info(
            "sqlite_schema_legacy_workspace_detected",
            stamped_to=_BASELINE_REVISION,
            target_revision=head,
        )
        command.stamp(cfg, _BASELINE_REVISION)
        command.upgrade(cfg, "head")
        LOGGER.info("sqlite_schema_migrated", from_revision=_BASELINE_REVISION, to_revision=head)
    else:
        Base.metadata.create_all(engine)
        command.stamp(cfg, "head")
