from __future__ import annotations

from datetime import UTC, datetime

from gaggle.core.review import ReviewService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid


def test_review_history_is_append_only(tmp_path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = EventRecord(
        event_id=new_uuid(),
        created_at=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        pipeline_version="test",
        event_start=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        event_end=datetime(2026, 5, 12, 12, 0, 1, tzinfo=UTC),
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.2, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )
    repository.save_event(event)
    service = ReviewService(repository)
    first_action, first_event = service.append_action(
        event.event_id, "annotate", "tester", notes="first"
    )
    second_action, second_event = service.append_action(
        event.event_id, "annotate", "tester", notes="second"
    )
    lines = (
        repository.workspace.review_log_path(event.event_id)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(lines) == 2
    assert first_action.action_id != second_action.action_id
    # review_summary must be reflected in the persisted event, never left stale.
    assert first_event.review_summary.action_count == 1
    assert second_event.review_summary.action_count == 2
    assert second_event.revision == 2
    reloaded = repository.load_event(event.event_id)
    assert reloaded.review_summary.action_count == 2
    assert reloaded.revision == 2


def test_review_action_never_mutates_prior_revisions(tmp_path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = EventRecord(
        event_id=new_uuid(),
        created_at=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        pipeline_version="test",
        event_start=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        event_end=datetime(2026, 5, 12, 12, 0, 1, tzinfo=UTC),
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.2, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )
    repository.save_event(event)
    service = ReviewService(repository)
    service.append_action(event.event_id, "accept", "tester")

    revisions = repository.list_event_revisions(event.event_id)
    assert len(revisions) == 2
    assert revisions[0].revision == 0
    assert revisions[0].review_summary.action_count == 0  # original revision untouched
    assert revisions[1].revision == 1
    assert revisions[1].review_summary.action_count == 1
    assert revisions[1].review_summary.latest_decision == "accepted"
    assert revisions[1].previous_revision_hash is not None
