from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid


def _make_event() -> EventRecord:
    start = datetime(2026, 5, 12, 12, 0, tzinfo=UTC)
    return EventRecord(
        event_id=new_uuid(),
        created_at=start,
        pipeline_version="test",
        event_start=start,
        event_end=start,
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def test_check_schema_drift_reports_nothing_on_a_freshly_initialized_workspace(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    assert repository.database.check_schema_drift() == []


def test_reindex_repopulates_the_index_from_the_filesystem(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    count = repository.reindex()

    assert count == 1
    assert len(repository.database.list_media()) == 0  # no media ingested in this test
    reloaded = repository.load_event(event.event_id)
    assert reloaded.event_id == event.event_id


def test_check_schema_drift_detects_a_missing_column(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    # Simulate an old workspace: drop a column that the current model
    # expects on an existing table (SQLite 3.35+ supports DROP COLUMN).
    with repository.database.engine.begin() as connection:
        connection.execute(text("ALTER TABLE event_index DROP COLUMN revision"))

    drift = repository.database.check_schema_drift()

    assert ("event_index", ["revision"]) in drift


def test_workspace_reindex_rebuild_matches_a_normal_reindex(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    repository = Repository(workspace_root)
    repository.initialize()
    events = [_make_event() for _ in range(3)]
    for event in events:
        repository.save_event(event)
    repository.reindex()
    normal_count = len(repository.list_events())

    database_path = repository.workspace.database
    repository.close()
    database_path.unlink()

    rebuilt_repository = Repository(workspace_root)
    rebuilt_repository.initialize()
    rebuilt_count = rebuilt_repository.reindex()

    assert rebuilt_count == normal_count == 3
    assert rebuilt_repository.database.check_schema_drift() == []
    for event in events:
        assert rebuilt_repository.load_event(event.event_id).event_id == event.event_id
