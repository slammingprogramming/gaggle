from __future__ import annotations

from datetime import UTC, datetime

from gaggle.core.review import ReviewService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid


class _FakeEntry:
    def __init__(self, name: str, loader) -> None:
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class _FakeEntryPoints:
    def __init__(self, entries: list[_FakeEntry]) -> None:
        self._entries = entries

    def select(self, group: str) -> list[_FakeEntry]:
        return self._entries


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


def test_review_extension_is_called_after_the_action_is_persisted(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class RecordingExtension:
        name = "recorder"
        version = "1.0.0"

        def on_review_action(self, action, event) -> None:
            calls.append((action.action, event.review_summary.latest_decision))

    monkeypatch.setattr(
        "gaggle.plugins.registry.entry_points",
        lambda: _FakeEntryPoints([_FakeEntry("recorder", lambda: RecordingExtension)]),
    )

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    service = ReviewService(repository)
    service.append_action(event.event_id, "accept", "tester")

    assert calls == [("accept", "accepted")]


def test_a_broken_review_extension_does_not_break_the_review_action(tmp_path, monkeypatch) -> None:
    class BrokenExtension:
        name = "broken"
        version = "1.0.0"

        def on_review_action(self, action, event) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "gaggle.plugins.registry.entry_points",
        lambda: _FakeEntryPoints([_FakeEntry("broken", lambda: BrokenExtension)]),
    )

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    service = ReviewService(repository)
    # Must not raise -- the review action itself is already durably
    # persisted by the time the (broken) extension runs.
    action, updated_event = service.append_action(event.event_id, "accept", "tester")

    assert action.action == "accept"
    assert updated_event.review_summary.latest_decision == "accepted"
    reloaded = repository.load_event(event.event_id)
    assert reloaded.review_summary.latest_decision == "accepted"


def test_review_service_works_with_no_extensions_registered(tmp_path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    service = ReviewService(repository)
    action, _ = service.append_action(event.event_id, "reject", "tester")

    assert action.action == "reject"
