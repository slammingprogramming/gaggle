from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from gaggle.core.recognition import RecognitionService
from gaggle.schemas.recognition import PlateObservation
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_observation(
    repository: Repository,
    event_id,
    normalized_text: str,
    offset_seconds: float,
    ocr_confidence: float,
    review_status: str = "needs_review",
):
    observation = PlateObservation(
        observation_id=new_uuid(),
        signal_id=new_uuid(),
        event_id=event_id,
        clip_id=new_uuid(),
        camera_id="front",
        observed_at=BASE + timedelta(seconds=offset_seconds),
        crop_path=f"/tmp/{new_uuid()}.jpg",
        crop_sha256="a" * 64,
        raw_ocr_text=normalized_text,
        normalized_text=normalized_text,
        ocr_confidence=ocr_confidence,
        detector_confidence=0.5,
        review_status=review_status,  # type: ignore[arg-type]
        detector_version="1.0.0",
    )
    repository.database.insert_plate_observation(observation)
    return observation


def test_cleanup_collapses_a_burst_of_same_plate_within_one_event(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event_id = new_uuid()

    low1 = _make_observation(repository, event_id, "ABC1234", 0.0, 0.4)
    best = _make_observation(repository, event_id, "ABC1234", 1.0, 0.9)
    low2 = _make_observation(repository, event_id, "ABC1234", 2.0, 0.5)

    result = RecognitionService(repository).cleanup_duplicate_plate_observations(window_seconds=5.0)

    assert result.clusters_with_duplicates == 1
    assert best.observation_id in result.kept_observation_ids
    assert low1.observation_id in result.suppressed_observation_ids
    assert low2.observation_id in result.suppressed_observation_ids

    kept_row = repository.database.list_plate_observations(normalized_text="ABC1234")
    statuses = {row.observation_id: row.review_status for row in kept_row}
    assert statuses[str(best.observation_id)] == "needs_review"
    assert statuses[str(low1.observation_id)] == "duplicate_suppressed"
    assert statuses[str(low2.observation_id)] == "duplicate_suppressed"

    duplicate_of = {row.observation_id: row.duplicate_of_observation_id for row in kept_row}
    assert duplicate_of[str(low1.observation_id)] == str(best.observation_id)


def test_cleanup_respects_the_time_window(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event_id = new_uuid()

    first = _make_observation(repository, event_id, "ABC1234", 0.0, 0.6)
    # 20s later -- well outside a 5s window, treated as a separate sighting.
    second = _make_observation(repository, event_id, "ABC1234", 20.0, 0.6)

    result = RecognitionService(repository).cleanup_duplicate_plate_observations(window_seconds=5.0)

    assert result.clusters_with_duplicates == 0
    assert set(result.kept_observation_ids) == {first.observation_id, second.observation_id}
    assert result.suppressed_observation_ids == []


def test_cleanup_never_touches_a_human_decision(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event_id = new_uuid()

    confirmed = _make_observation(
        repository, event_id, "ABC1234", 0.0, 0.9, review_status="user_confirmed"
    )
    rejected = _make_observation(
        repository, event_id, "ABC1234", 1.0, 0.8, review_status="user_rejected"
    )

    result = RecognitionService(repository).cleanup_duplicate_plate_observations(window_seconds=5.0)

    # Neither a confirmed nor a rejected observation is "actionable" --
    # cleanup must leave both exactly as a human left them.
    assert result.suppressed_observation_ids == []
    rows = repository.database.list_plate_observations(normalized_text="ABC1234")
    statuses = {row.observation_id: row.review_status for row in rows}
    assert statuses[str(confirmed.observation_id)] == "user_confirmed"
    assert statuses[str(rejected.observation_id)] == "user_rejected"


def test_cleanup_treats_different_events_as_separate_groups(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event_a = new_uuid()
    event_b = new_uuid()

    obs_a = _make_observation(repository, event_a, "ABC1234", 0.0, 0.6)
    obs_b = _make_observation(repository, event_b, "ABC1234", 0.5, 0.7)

    result = RecognitionService(repository).cleanup_duplicate_plate_observations(window_seconds=5.0)

    # Same plate text, same rough time, but different events -- not the
    # same physical sighting, so both must be kept.
    assert result.clusters_with_duplicates == 0
    assert set(result.kept_observation_ids) == {obs_a.observation_id, obs_b.observation_id}
