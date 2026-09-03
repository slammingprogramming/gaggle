from __future__ import annotations

import sqlite3
from pathlib import Path

from gaggle.storage import migrate
from gaggle.storage.database import Base, TimelineDatabase
from gaggle.storage.repository import Repository


def _alembic_version(sqlite_path: Path) -> str | None:
    con = sqlite3.connect(sqlite_path)
    try:
        row = con.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        con.close()
    return row[0] if row else None


def _table_names(sqlite_path: Path) -> set[str]:
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


def test_fresh_workspace_stamps_to_head(tmp_path: Path) -> None:
    database = TimelineDatabase(tmp_path / "index.sqlite3")
    database.initialize()

    tables = _table_names(database.path)
    assert "alembic_version" in tables
    for table_name in Base.metadata.tables:
        assert table_name in tables
    assert _alembic_version(database.path) is not None


def test_legacy_workspace_is_stamped_to_baseline_then_upgraded(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "index.sqlite3"
    # Build the exact pre-Alembic baseline shape via the real baseline
    # migration (not Base.metadata.create_all(), which now reflects head --
    # i.e. includes tables added after baseline, like `camera` -- so it can
    # no longer stand in for what a workspace predating this feature
    # actually looked like). Then drop alembic_version: a true legacy
    # workspace never had that table at all.
    cfg = migrate._build_config(sqlite_path)
    migrate.command.upgrade(cfg, "0001_baseline")
    con = sqlite3.connect(sqlite_path)
    con.execute("DROP TABLE alembic_version")
    con.commit()
    con.close()
    assert "alembic_version" not in _table_names(sqlite_path)

    database = TimelineDatabase(sqlite_path)
    database.initialize()

    assert "alembic_version" in _table_names(sqlite_path)
    assert _alembic_version(sqlite_path) is not None
    assert database.check_schema_drift() == []


def test_a_workspace_already_at_head_skips_the_real_upgrade_call(
    tmp_path: Path, monkeypatch
) -> None:
    database = TimelineDatabase(tmp_path / "index.sqlite3")
    database.initialize()  # first call: stamps to head

    calls: list[str] = []
    monkeypatch.setattr(migrate.command, "upgrade", lambda *args, **kwargs: calls.append("upgrade"))

    database.initialize()  # second call: already at head

    assert calls == []


def test_workspace_reindex_rebuild_round_trips_after_migration(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    database_path = repository.workspace.database
    repository.close()
    database_path.unlink()

    rebuilt = Repository(tmp_path / "workspace")
    rebuilt.initialize()

    assert "alembic_version" in _table_names(rebuilt.database.path)
    assert rebuilt.database.check_schema_drift() == []
