from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaggle.core.triage import TriageService
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.media import MediaClip
from gaggle.schemas.signal import Signal
from gaggle.storage.repository import Repository
from gaggle.utils.filesystem import set_read_only
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_clip(
    workspace_root: Path, camera_id: str, content: bytes, ingest_mode: str = "copy"
) -> MediaClip:
    stored_dir = workspace_root / "originals" / camera_id
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / f"{camera_id}.mp4"
    stored_path.write_bytes(content)
    return MediaClip(
        clip_id=new_uuid(),
        camera_id=camera_id,
        source_path=str(stored_path),
        stored_path=str(stored_path),
        filename=stored_path.name,
        media_type="video",
        byte_size=len(content),
        sha256=hash_file(stored_path),
        observed_start=BASE,
        observed_end=BASE,
        original_timestamp_source="filename",
        timestamp_confidence=0.7,
        duration_seconds=10.0,
        ingest_mode=ingest_mode,  # type: ignore[arg-type]
    )


def _make_event_referencing(clip: MediaClip) -> EventRecord:
    signal = Signal(
        id=new_uuid(),
        source="test",
        signal_type="motion",
        timestamp_start=BASE,
        timestamp_end=BASE,
        confidence=0.5,
        camera_id=clip.camera_id,
        evidence_references=[
            ArtifactReference(
                path=clip.stored_path,
                artifact_type="source_media",
                created_at=BASE,
                sha256=clip.sha256,
            )
        ],
    )
    return EventRecord(
        event_id=new_uuid(),
        created_at=BASE,
        pipeline_version="test",
        event_start=BASE,
        event_end=BASE,
        involved_cameras=[clip.camera_id],
        signals=[signal],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def test_clip_with_signal_is_classified_reviewable(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "front", b"reviewable content")
    repository.index_media_clip(clip)
    event = _make_event_referencing(clip)
    repository.save_event(event)

    records = TriageService(repository).classify_all()

    assert len(records) == 1
    assert records[0].state == "reviewable"
    assert records[0].signal_count == 1
    # reviewable originals are never moved
    assert Path(clip.stored_path).exists()


def test_clip_with_no_signals_is_moved_to_pending_deletion(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "rear", b"benign content")
    repository.index_media_clip(clip)
    # no events reference this clip at all

    records = TriageService(repository).classify_all()

    assert len(records) == 1
    assert records[0].state == "benign_pending_deletion"
    assert not Path(clip.stored_path).exists()  # moved out of originals/
    moved_path = repository.workspace.pending_deletion / Path(clip.stored_path).name
    assert moved_path.exists()


def test_confirm_deletion_writes_append_only_record_and_removes_bytes(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "rear", b"benign content")
    repository.index_media_clip(clip)
    TriageService(repository).classify_all()

    service = TriageService(repository)
    record = service.confirm_deletion(clip.clip_id, actor="tester", notes="confirmed benign")

    assert record.confirmed_by == "tester"
    assert record.sha256 == clip.sha256
    moved_path = repository.workspace.pending_deletion / Path(clip.stored_path).name
    assert not moved_path.exists()

    log_lines = repository.workspace.deletion_log_path.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1

    final_state = service.list_state("deleted")
    assert len(final_state) == 1
    assert final_state[0].clip_id == clip.clip_id


def test_confirm_deletion_succeeds_on_a_read_only_file(tmp_path: Path) -> None:
    """A real bug hit on Windows: `set_read_only` (the `storage.set_read_only:
    true` default) marks originals read-only on ingest, and Windows'
    `os.unlink()` -- unlike POSIX -- raises `PermissionError: [WinError 5]
    Access is denied` on a read-only file regardless of directory
    permissions. `confirm_deletion` must clear that bit before deleting,
    not just rely on the file happening to be writable."""

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "rear", b"benign content")
    repository.index_media_clip(clip)
    TriageService(repository).classify_all()

    moved_path = repository.workspace.pending_deletion / Path(clip.stored_path).name
    set_read_only(moved_path)

    service = TriageService(repository)
    record = service.confirm_deletion(clip.clip_id, actor="tester", notes="confirmed benign")

    assert record.sha256 == clip.sha256
    assert not moved_path.exists()


def test_cannot_delete_a_reviewable_clip_through_triage(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "front", b"reviewable content")
    repository.index_media_clip(clip)
    event = _make_event_referencing(clip)
    repository.save_event(event)
    TriageService(repository).classify_all()

    with pytest.raises(ValueError):
        TriageService(repository).confirm_deletion(clip.clip_id, actor="tester")


def test_confirm_deletion_all_processes_every_pending_clip(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip_a = _make_clip(repository.workspace.root, "front", b"benign a")
    clip_b = _make_clip(repository.workspace.root, "rear", b"benign b")
    repository.index_media_clip(clip_a)
    repository.index_media_clip(clip_b)
    TriageService(repository).classify_all()

    records = TriageService(repository).confirm_deletion_all(actor="tester")

    assert {r.clip_id for r in records} == {clip_a.clip_id, clip_b.clip_id}
    assert repository.workspace.deletion_log_path.read_text(encoding="utf-8").count("\n") == 2


def test_rerunning_triage_does_not_move_an_already_moved_clip_again(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "rear", b"benign content")
    repository.index_media_clip(clip)

    service = TriageService(repository)
    service.classify_all()
    first_location = repository.database.get_media(clip.clip_id).stored_path
    service.classify_all()
    second_location = repository.database.get_media(clip.clip_id).stored_path

    assert first_location == second_location
    assert Path(second_location).exists()


def test_database_rows_are_readable_after_their_query_session_has_closed(tmp_path: Path) -> None:
    """Regression test for a real crash: DetachedInstanceError.

    ``TimelineDatabase.list_media()``/``get_media()`` (and every other
    ``list_*``/``get_*`` method) open a session, run a query, and close the
    session before returning -- by design, so nothing holds a long-lived
    session across calls. Every caller then reads attributes on the
    returned row *after* that session has closed. Without
    ``expire_on_commit=False`` on the session factory, SQLAlchemy expires
    every loaded attribute at commit time, and reading an expired attribute
    on a detached instance raises ``DetachedInstanceError`` -- exactly what
    happened when a real user ran ``analyze`` against real footage and
    ``TriageService.classify_all()`` tried to read ``row.clip_id``. This
    test exercises that exact call shape directly against the database
    layer, independent of the triage logic built on top of it.
    """

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository.workspace.root, "front", b"content")
    repository.index_media_clip(clip)

    # The bug required reading attributes strictly *after* the query call
    # returns -- i.e. after its internal `with self.session()` block has
    # already run session.commit() and session.close().
    rows = repository.database.list_media()
    row = rows[0]
    assert row.clip_id == str(clip.clip_id)
    assert row.camera_id == clip.camera_id
    assert row.sha256 == clip.sha256
    assert row.stored_path == clip.stored_path

    single = repository.database.get_media(clip.clip_id)
    assert single is not None
    assert single.clip_id == str(clip.clip_id)


def test_reference_mode_clip_is_not_physically_moved_when_classified_benign(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    # A "reference"-mode clip lives outside originals/ entirely, at some
    # external location the workspace never copied it from.
    external_dir = tmp_path / "external_sd_card" / "rear"
    external_dir.mkdir(parents=True)
    external_path = external_dir / "rear.mp4"
    external_path.write_bytes(b"benign external content")
    clip = MediaClip(
        clip_id=new_uuid(),
        camera_id="rear",
        source_path=str(external_path),
        stored_path=str(external_path),
        filename=external_path.name,
        media_type="video",
        byte_size=external_path.stat().st_size,
        sha256=hash_file(external_path),
        observed_start=BASE,
        observed_end=BASE,
        original_timestamp_source="filename",
        timestamp_confidence=0.7,
        duration_seconds=10.0,
        ingest_mode="reference",
    )
    repository.index_media_clip(clip)

    TriageService(repository).classify_all()

    # Still exactly where it always was -- never copied into pending_deletion/.
    assert external_path.exists()
    assert not any(repository.workspace.pending_deletion.iterdir())
    triage = repository.database.get_triage(clip.clip_id)
    assert triage is not None
    assert triage.state == "benign_pending_deletion"


def test_confirm_deletion_refuses_reference_mode_clip_without_acknowledgement(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    external_dir = tmp_path / "external_sd_card" / "rear"
    external_dir.mkdir(parents=True)
    external_path = external_dir / "rear.mp4"
    external_path.write_bytes(b"benign external content")
    clip = MediaClip(
        clip_id=new_uuid(),
        camera_id="rear",
        source_path=str(external_path),
        stored_path=str(external_path),
        filename=external_path.name,
        media_type="video",
        byte_size=external_path.stat().st_size,
        sha256=hash_file(external_path),
        observed_start=BASE,
        observed_end=BASE,
        original_timestamp_source="filename",
        timestamp_confidence=0.7,
        duration_seconds=10.0,
        ingest_mode="reference",
    )
    repository.index_media_clip(clip)
    TriageService(repository).classify_all()

    with pytest.raises(ValueError):
        TriageService(repository).confirm_deletion(clip.clip_id, actor="tester")

    # Refused -- the external file must still exist.
    assert external_path.exists()


def test_confirm_deletion_allows_reference_mode_clip_with_acknowledgement(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    external_dir = tmp_path / "external_sd_card" / "rear"
    external_dir.mkdir(parents=True)
    external_path = external_dir / "rear.mp4"
    external_path.write_bytes(b"benign external content")
    clip = MediaClip(
        clip_id=new_uuid(),
        camera_id="rear",
        source_path=str(external_path),
        stored_path=str(external_path),
        filename=external_path.name,
        media_type="video",
        byte_size=external_path.stat().st_size,
        sha256=hash_file(external_path),
        observed_start=BASE,
        observed_end=BASE,
        original_timestamp_source="filename",
        timestamp_confidence=0.7,
        duration_seconds=10.0,
        ingest_mode="reference",
    )
    repository.index_media_clip(clip)
    TriageService(repository).classify_all()

    record = TriageService(repository).confirm_deletion(
        clip.clip_id, actor="tester", acknowledge_external_deletion=True
    )

    assert not external_path.exists()
    assert record.metadata.get("ingest_mode") == "reference"
