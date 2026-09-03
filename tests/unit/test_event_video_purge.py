from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaggle.core.triage import TriageService
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.media import MediaClip
from gaggle.schemas.signal import Signal
from gaggle.storage.database import TimelineQuery
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_clip(repository: Repository, camera_id: str, content: bytes) -> MediaClip:
    stored_dir = repository.workspace.originals / camera_id
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / f"{camera_id}.mp4"
    stored_path.write_bytes(content)
    clip = MediaClip(
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
    )
    repository.index_media_clip(clip)
    return clip


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


def _make_derived_clip_file(repository: Repository, event_id, content: bytes) -> Path:
    clips_dir = repository.workspace.event_clips_dir(event_id)
    clips_dir.mkdir(parents=True, exist_ok=True)
    path = clips_dir / "front__abcd1234.mp4"
    path.write_bytes(content)
    return path


def test_purge_refuses_unpreserved_event_without_force(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    with pytest.raises(ValueError):
        TriageService(repository).purge_event_video(event.event_id, actor="tester")


def test_purge_with_force_deletes_derived_clips_and_marks_event(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    derived_path = _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    record = TriageService(repository).purge_event_video(event.event_id, actor="tester", force=True)

    assert not derived_path.exists()
    assert str(derived_path) in record.deleted_derived_clip_paths
    assert record.was_preserved_at_time_of_purge is False

    updated = repository.load_event(event.event_id)
    assert updated.video_purged_at is not None
    # Nothing else about the event changed.
    assert updated.signals == event.signals
    assert updated.scoring == event.scoring


def test_purge_succeeds_without_force_when_event_is_preserved(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    # Simulate having already preserved the event (without exercising the
    # full PreservationService here -- just the state this method checks).
    repository.save_event_revision(
        event.event_id,
        reason="preserved",
        update={
            "preservation_status": PreservationStatus(
                state="preserved", immutable=True, bundle_path="/tmp/fake-bundle"
            )
        },
    )

    record = TriageService(repository).purge_event_video(event.event_id, actor="tester")
    assert record.was_preserved_at_time_of_purge is True


def test_purge_cannot_run_twice(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    TriageService(repository).purge_event_video(event.event_id, actor="tester", force=True)
    with pytest.raises(ValueError):
        TriageService(repository).purge_event_video(event.event_id, actor="tester", force=True)


def test_purge_cascades_to_original_when_no_other_event_needs_it(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    original_path = Path(clip.stored_path)
    assert original_path.exists()

    record = TriageService(repository).purge_event_video(event.event_id, actor="tester", force=True)

    assert clip.clip_id in record.cascaded_original_clip_ids
    assert not original_path.exists()


def test_purge_retains_original_still_needed_by_another_unpurged_event(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event_a = _make_event_referencing(clip)
    event_b = _make_event_referencing(clip)  # different event, same original clip
    repository.save_event(event_a)
    repository.save_event(event_b)
    _make_derived_clip_file(repository, event_a.event_id, b"derived clip a")
    _make_derived_clip_file(repository, event_b.event_id, b"derived clip b")

    original_path = Path(clip.stored_path)

    record = TriageService(repository).purge_event_video(
        event_a.event_id, actor="tester", force=True
    )

    # event_b hasn't been purged, so the shared original must be retained.
    assert clip.clip_id in record.retained_original_clip_ids
    assert clip.clip_id not in record.cascaded_original_clip_ids
    assert original_path.exists()

    # Now purge event_b too -- the original should finally be eligible.
    record_b = TriageService(repository).purge_event_video(
        event_b.event_id, actor="tester", force=True
    )
    assert clip.clip_id in record_b.cascaded_original_clip_ids
    assert not original_path.exists()


def test_purge_reviewed_bulk_skips_unpreserved_events_without_force(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip = _make_clip(repository, "front", b"original content")
    event = _make_event_referencing(clip)
    repository.save_event(event)
    _make_derived_clip_file(repository, event.event_id, b"derived clip content")

    records = TriageService(repository).purge_event_video_bulk(TimelineQuery(), actor="tester")

    assert records == []
    reloaded = repository.load_event(event.event_id)
    assert reloaded.video_purged_at is None
