from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

from gaggle.core.config import RuntimeConfig
from gaggle.enrichment.service import EnrichmentService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.recognition import (
    FaceObservation,
    PlateObservation,
    VehicleAppearanceObservation,
)
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_event() -> EventRecord:
    return EventRecord(
        event_id=new_uuid(),
        created_at=BASE,
        pipeline_version="test",
        event_start=BASE,
        event_end=BASE,
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def _insert_face(repository: Repository, event: EventRecord, clip_id: UUID, offset: float) -> None:
    repository.database.insert_face_observation(
        FaceObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id="front",
            observed_at=BASE + timedelta(seconds=offset),
            crop_path=f"/tmp/face_{offset}.jpg",
            crop_sha256="a" * 64,
            detector_confidence=0.5,
            detector_version="1.0.0",
        )
    )


def _insert_plate(repository: Repository, event: EventRecord, clip_id: UUID, offset: float) -> None:
    repository.database.insert_plate_observation(
        PlateObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id="front",
            observed_at=BASE + timedelta(seconds=offset),
            crop_path=f"/tmp/plate_{offset}.jpg",
            crop_sha256="b" * 64,
            raw_ocr_text="ABC123",
            normalized_text="ABC123",
            ocr_confidence=0.9,
            detector_confidence=0.9,
            review_status="needs_review",
            detector_version="1.0.0",
        )
    )


def _insert_vehicle(
    repository: Repository, event: EventRecord, clip_id: UUID, offset: float
) -> None:
    repository.database.insert_vehicle_appearance_observation(
        VehicleAppearanceObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id="front",
            observed_at=BASE + timedelta(seconds=offset),
            crop_path=f"/tmp/vehicle_{offset}.jpg",
            crop_sha256="c" * 64,
            fingerprint=[0.1, 0.2, 0.3],
            detector_confidence=0.6,
            detector_version="1.0.0",
        )
    )


def _config_with_only_encounters_enabled() -> RuntimeConfig:
    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vehicle_appearance.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.transcription.enabled = False
    config.enrichment.cloud.enabled = False
    config.enrichment.encounters.enabled = True
    return config


def test_close_observations_across_modalities_group_into_one_encounter(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    clip_id = new_uuid()
    _insert_face(repository, event, clip_id, offset=0.0)
    _insert_plate(repository, event, clip_id, offset=0.5)
    _insert_vehicle(repository, event, clip_id, offset=1.0)

    EnrichmentService(repository, _config_with_only_encounters_enabled()).enrich_event(
        event.event_id
    )

    encounters = repository.database.list_encounters_for_event(event.event_id)
    assert len(encounters) == 1
    encounter = encounters[0]
    assert encounter.face_observation_id is not None
    assert encounter.plate_observation_id is not None
    assert encounter.vehicle_appearance_observation_id is not None
    assert encounter.voice_observation_id is None
    assert encounter.clip_id == str(clip_id)
    assert encounter.camera_id == "front"


def test_far_apart_observations_produce_separate_encounters(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    clip_id = new_uuid()
    _insert_face(repository, event, clip_id, offset=0.0)
    _insert_face(repository, event, clip_id, offset=30.0)  # far beyond the 2s tolerance

    EnrichmentService(repository, _config_with_only_encounters_enabled()).enrich_event(
        event.event_id
    )

    encounters = repository.database.list_encounters_for_event(event.event_id)
    assert len(encounters) == 2


def test_rerunning_enrich_does_not_duplicate_encounters(tmp_path: Path) -> None:
    """Regression test: `_derive_encounters` used to mint fresh `Encounter`
    rows on every call with no cleanup, so calling `enrich_event` twice
    (e.g. `--force`, or a second `enrich` after adding more footage) would
    silently double every encounter forever. `enrich_event` now clears an
    event's encounters before re-deriving them, so a rerun over the same
    observations is a clean replace, not an accumulate."""

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    clip_id = new_uuid()
    _insert_face(repository, event, clip_id, offset=0.0)
    _insert_plate(repository, event, clip_id, offset=0.5)

    config = _config_with_only_encounters_enabled()
    service = EnrichmentService(repository, config)
    service.enrich_event(event.event_id)
    first_count = len(repository.database.list_encounters_for_event(event.event_id))
    assert first_count == 1

    # force=True so encounters is re-derived even though it's already
    # marked complete -- exactly the "someone reruns enrich" scenario the
    # dedup fix protects against.
    service.enrich_event(event.event_id, force=True)
    second_count = len(repository.database.list_encounters_for_event(event.event_id))

    assert second_count == first_count


def test_encounters_disabled_skips_derivation_entirely(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    clip_id = new_uuid()
    _insert_face(repository, event, clip_id, offset=0.0)

    config = _config_with_only_encounters_enabled()
    config.enrichment.encounters.enabled = False
    EnrichmentService(repository, config).enrich_event(event.event_id)

    assert repository.database.list_encounters_for_event(event.event_id) == []


def test_observations_from_different_clips_never_share_an_encounter(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    _insert_face(repository, event, new_uuid(), offset=0.0)
    _insert_plate(repository, event, new_uuid(), offset=0.1)  # different clip, same instant

    EnrichmentService(repository, _config_with_only_encounters_enabled()).enrich_event(
        event.event_id
    )

    encounters = repository.database.list_encounters_for_event(event.event_id)
    assert len(encounters) == 2
    assert all(e.plate_observation_id is None or e.face_observation_id is None for e in encounters)


def test_multiple_same_modality_observations_in_one_window_pair_index_wise(
    tmp_path: Path,
) -> None:
    """Two faces and one plate within one time window must not be
    combined combinatorially (which would fabricate a second face-plate
    pairing that never happened) -- every observation lands in exactly
    one Encounter, index-aligned within the window."""

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    clip_id = new_uuid()
    _insert_face(repository, event, clip_id, offset=0.0)
    _insert_face(repository, event, clip_id, offset=0.5)
    _insert_plate(repository, event, clip_id, offset=1.0)

    EnrichmentService(repository, _config_with_only_encounters_enabled()).enrich_event(
        event.event_id
    )

    encounters = repository.database.list_encounters_for_event(event.event_id)
    assert len(encounters) == 2
    with_plate = [e for e in encounters if e.plate_observation_id is not None]
    without_plate = [e for e in encounters if e.plate_observation_id is None]
    assert len(with_plate) == 1
    assert len(without_plate) == 1
    assert with_plate[0].face_observation_id is not None
    assert without_plate[0].face_observation_id is not None
