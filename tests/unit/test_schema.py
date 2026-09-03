from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.media import MediaClip, NormalizedClip
from gaggle.schemas.review import ReviewAction
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid


def test_review_action_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        ReviewAction(
            action_id=new_uuid(),
            event_id=new_uuid(),
            action="annotate",
            actor="tester",
            timestamp=datetime(2026, 5, 12, 12, 0, 0),
        )


def test_review_action_normalizes_to_utc() -> None:
    action = ReviewAction(
        action_id=new_uuid(),
        event_id=new_uuid(),
        action="annotate",
        actor="tester",
        timestamp=datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC),
    )
    assert action.timestamp.tzinfo == UTC


def test_event_record_defaults_to_revision_zero() -> None:
    event = _make_minimal_event()
    assert event.revision == 0
    assert event.revision_reason == "initial_generation"
    assert event.revised_at is None
    assert event.previous_revision_hash is None


def test_event_record_rejects_unknown_fields() -> None:
    payload = _make_minimal_event().model_dump(mode="json")
    payload["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        EventRecord.model_validate(payload)


def test_signal_confidence_must_be_within_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Signal(
            id=new_uuid(),
            source="test",
            signal_type="motion",
            timestamp_start=datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC),
            timestamp_end=datetime(2026, 5, 12, 12, 0, 1, tzinfo=UTC),
            confidence=1.5,
        )


def test_normalized_clip_exposes_camera_id_from_wrapped_clip() -> None:
    clip = MediaClip(
        clip_id=new_uuid(),
        camera_id="front",
        source_path="/tmp/src.mp4",
        stored_path="/tmp/stored.mp4",
        filename="src.mp4",
        media_type="video",
        byte_size=100,
        sha256="a" * 64,
        observed_start=datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC),
        observed_end=datetime(2026, 5, 12, 12, 0, 10, tzinfo=UTC),
        original_timestamp_source="filename",
        timestamp_confidence=0.7,
        duration_seconds=10.0,
    )
    normalized = NormalizedClip(
        clip=clip,
        session_id="front#000",
        corrected_start=clip.observed_start,
        corrected_end=clip.observed_end,
        sync_confidence=0.7,
        sync_rationale="reference session",
    )
    assert normalized.camera_id == "front"
    assert normalized.clip_id == clip.clip_id
    assert normalized.stored_path == clip.stored_path
    assert normalized.sha256 == clip.sha256


def _make_minimal_event() -> EventRecord:
    start = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)
    return EventRecord(
        event_id=new_uuid(),
        created_at=start,
        pipeline_version="test",
        event_start=start,
        event_end=start,
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.1, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )
