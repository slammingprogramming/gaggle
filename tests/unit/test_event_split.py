from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from gaggle.core.config import RuntimeConfig
from gaggle.core.events import EventSplitError, EventSplitService
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, Hypothesis, PreservationStatus, SeverityAssessment
from gaggle.schemas.signal import Signal
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_mismerged_event(repository: Repository, tmp_path: Path) -> tuple[EventRecord, UUID, UUID]:
    """A synthetic real-shaped repro of the actual bug: one event whose
    signals/derived clips actually came from two unrelated recording
    sessions that normalize/sync.py's pure time-overlap heuristic merged
    together (see core/events.py's module docstring)."""

    clip_a_id, clip_b_id = new_uuid(), new_uuid()
    clip_a_path = tmp_path / "clip_a.mp4"
    clip_b_path = tmp_path / "clip_b.mp4"
    clip_a_path.write_bytes(b"clip a content")
    clip_b_path.write_bytes(b"clip b content")
    clip_a_sha = hash_file(clip_a_path)
    clip_b_sha = hash_file(clip_b_path)

    signal_a = Signal(
        id=new_uuid(),
        source="test",
        signal_type="motion",
        timestamp_start=BASE,
        timestamp_end=BASE + timedelta(seconds=1),
        confidence=0.5,
        camera_id="front",
        evidence_references=[
            ArtifactReference(
                path=str(clip_a_path),
                artifact_type="source_media",
                created_at=BASE,
                sha256=clip_a_sha,
            )
        ],
    )
    signal_b = Signal(
        id=new_uuid(),
        source="test",
        signal_type="motion",
        timestamp_start=BASE + timedelta(minutes=10),
        timestamp_end=BASE + timedelta(minutes=10, seconds=1),
        confidence=0.6,
        camera_id="rear",
        evidence_references=[
            ArtifactReference(
                path=str(clip_b_path),
                artifact_type="source_media",
                created_at=BASE,
                sha256=clip_b_sha,
            )
        ],
    )

    derived_a_path = tmp_path / "derived_a.mp4"
    derived_b_path = tmp_path / "derived_b.mp4"
    derived_a_path.write_bytes(b"derived a")
    derived_b_path.write_bytes(b"derived b")

    event = EventRecord(
        event_id=new_uuid(),
        created_at=BASE,
        pipeline_version="test",
        event_start=signal_a.timestamp_start,
        event_end=signal_b.timestamp_end,
        involved_cameras=["front", "rear"],
        signals=[signal_a, signal_b],
        hypotheses=[
            Hypothesis(
                hypothesis_id=new_uuid(),
                rule_name="test_rule",
                label="mixed",
                confidence=0.5,
                contributing_signal_ids=[signal_a.id, signal_b.id],
                confidence_math="test",
            )
        ],
        scoring=SeverityAssessment(confidence=0.5, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        derived_artifacts=[
            ArtifactReference(
                path=str(derived_a_path),
                artifact_type="derived_clip",
                created_at=BASE,
                sha256=hash_file(derived_a_path),
                metadata={
                    "camera_id": "front",
                    "source_clip_id": str(clip_a_id),
                    "source_sha256": clip_a_sha,
                },
            ),
            ArtifactReference(
                path=str(derived_b_path),
                artifact_type="derived_clip",
                created_at=BASE,
                sha256=hash_file(derived_b_path),
                metadata={
                    "camera_id": "rear",
                    "source_clip_id": str(clip_b_id),
                    "source_sha256": clip_b_sha,
                },
            ),
        ],
        evidence_summary="mismerged test event",
    )
    repository.save_event(event)
    return event, clip_a_id, clip_b_id


def test_split_event_produces_two_independent_events(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, clip_b_id = _make_mismerged_event(repository, tmp_path)

    new_events = EventSplitService(repository, RuntimeConfig()).split_event(
        event.event_id, [[clip_a_id], [clip_b_id]], actor="tester"
    )

    assert len(new_events) == 2
    front_event = next(e for e in new_events if e.involved_cameras == ["front"])
    rear_event = next(e for e in new_events if e.involved_cameras == ["rear"])
    assert len(front_event.signals) == 1
    assert len(rear_event.signals) == 1
    assert front_event.signals[0].camera_id == "front"
    assert rear_event.signals[0].camera_id == "rear"
    # a hypothesis mixing signals from both sides of the split is dropped
    assert front_event.hypotheses == []
    assert rear_event.hypotheses == []

    # derived clip files were actually copied, not just repointed
    front_clip_path = Path(front_event.derived_artifacts[0].path)
    assert front_clip_path.exists()
    assert front_clip_path.read_bytes() == b"derived a"
    assert front_clip_path.parent == repository.workspace.event_clips_dir(front_event.event_id)


def test_split_event_marks_the_original_as_superseded(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, clip_b_id = _make_mismerged_event(repository, tmp_path)

    new_events = EventSplitService(repository, RuntimeConfig()).split_event(
        event.event_id, [[clip_a_id], [clip_b_id]], actor="tester"
    )

    reloaded_original = repository.load_event(event.event_id)
    assert set(reloaded_original.superseded_by_event_ids) == {e.event_id for e in new_events}
    # the original's own signals/derived_artifacts are untouched
    assert len(reloaded_original.signals) == 2
    assert len(reloaded_original.derived_artifacts) == 2


def test_split_event_rejects_fewer_than_two_groups(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, clip_b_id = _make_mismerged_event(repository, tmp_path)

    with pytest.raises(EventSplitError, match="at least 2 groups"):
        EventSplitService(repository, RuntimeConfig()).split_event(
            event.event_id, [[clip_a_id, clip_b_id]], actor="tester"
        )


def test_split_event_rejects_a_clip_in_two_groups(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, clip_b_id = _make_mismerged_event(repository, tmp_path)

    with pytest.raises(EventSplitError, match="more than one group"):
        EventSplitService(repository, RuntimeConfig()).split_event(
            event.event_id, [[clip_a_id, clip_b_id], [clip_a_id]], actor="tester"
        )


def test_split_event_rejects_a_partial_partition(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, _clip_b_id = _make_mismerged_event(repository, tmp_path)

    with pytest.raises(EventSplitError, match="exactly partition"):
        EventSplitService(repository, RuntimeConfig()).split_event(
            event.event_id, [[clip_a_id], [new_uuid()]], actor="tester"
        )


def test_split_event_rejects_an_already_split_event(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event, clip_a_id, clip_b_id = _make_mismerged_event(repository, tmp_path)
    service = EventSplitService(repository, RuntimeConfig())
    service.split_event(event.event_id, [[clip_a_id], [clip_b_id]], actor="tester")

    with pytest.raises(EventSplitError, match="already split"):
        service.split_event(event.event_id, [[clip_a_id], [clip_b_id]], actor="tester")
