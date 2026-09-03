from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

from gaggle.review_ui.app import create_app
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.recognition import (
    FaceCluster,
    FaceObservation,
    PlateObservation,
    VehicleAppearanceObservation,
    VoiceObservation,
)
from gaggle.schemas.signal import Signal
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
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
        evidence_summary="test event",
    )


def _build_workspace_with_enrichment_data(tmp_path: Path) -> tuple[Repository, EventRecord]:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    face_crop = repository.workspace.face_crops / "a.jpg"
    face_crop.parent.mkdir(parents=True, exist_ok=True)
    face_crop.write_bytes(b"not a real jpeg, just a fixture")
    repository.database.insert_face_observation(
        FaceObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(face_crop),
            crop_sha256="a" * 64,
            detector_confidence=0.8,
            detector_version="1.0.0",
        )
    )

    plate_crop = repository.workspace.plate_crops / "a.jpg"
    plate_crop.parent.mkdir(parents=True, exist_ok=True)
    plate_crop.write_bytes(b"not a real jpeg either")
    repository.database.insert_plate_observation(
        PlateObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(plate_crop),
            crop_sha256="b" * 64,
            raw_ocr_text="ABC1234",
            normalized_text="ABC1234",
            ocr_confidence=0.9,
            detector_confidence=0.7,
            review_status="auto_accepted",
            detector_version="1.0.0",
        )
    )

    repository.database.insert_voice_observation(
        VoiceObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            segment_start_seconds=0.0,
            segment_end_seconds=1.5,
            voiceprint=[0.1, 0.2, 0.3],
            energy_confidence=0.6,
            detector_version="1.0.0",
        )
    )

    vehicle_crop = repository.workspace.vehicle_appearance_crops / "a.jpg"
    vehicle_crop.parent.mkdir(parents=True, exist_ok=True)
    vehicle_crop.write_bytes(b"not a real jpeg, a third time")
    repository.database.insert_vehicle_appearance_observation(
        VehicleAppearanceObservation(
            observation_id=new_uuid(),
            signal_id=new_uuid(),
            event_id=event.event_id,
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(vehicle_crop),
            crop_sha256="c" * 64,
            fingerprint=[0.4, 0.5, 0.6],
            detector_confidence=0.5,
            detector_version="1.0.0",
        )
    )

    transcript_path = repository.workspace.transcripts / f"{event.event_id}.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start_offset_seconds": 0.0,
                        "end_offset_seconds": 1.2,
                        "text": "watch out for that car",
                        "confidence": 0.85,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    return repository, event


def test_event_detail_page_renders_enrichment_data(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/events/{event.event_id}")
    assert response.status_code == 200
    body = response.text
    assert "Faces (1)" in body
    assert "Plates (1)" in body
    assert "ABC1234" in body
    assert "Voices (1)" in body
    assert "Vehicle appearance (1)" in body
    assert "Transcript" in body
    assert "watch out for that car" in body
    assert "class='crop'" in body


def test_event_detail_page_omits_enrichment_section_when_empty(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/events/{event.event_id}")
    assert response.status_code == 200
    assert "Faces (" not in response.text
    assert "Transcript" not in response.text


def test_crop_route_serves_the_face_crop_file(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    observations = repository.database.list_face_observations_for_event(event.event_id)
    observation_id = observations[0].observation_id

    response = client.get(f"/api/events/{event.event_id}/crop/face/{observation_id}")
    assert response.status_code == 200
    assert response.content == b"not a real jpeg, just a fixture"


def test_crop_route_rejects_an_observation_from_a_different_event(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    other_event = _make_event()
    repository.save_event(other_event)
    client = TestClient(create_app(repository.workspace.root))

    observations = repository.database.list_face_observations_for_event(event.event_id)
    observation_id = observations[0].observation_id

    response = client.get(f"/api/events/{other_event.event_id}/crop/face/{observation_id}")
    assert response.status_code == 404


def test_crop_route_rejects_an_unknown_entity_type(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/api/events/{event.event_id}/crop/voice/{new_uuid()}")
    assert response.status_code == 404


def test_transcript_api_route_returns_the_transcript(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/api/events/{event.event_id}/transcript")
    assert response.status_code == 200
    payload = response.json()
    assert payload["segments"][0]["text"] == "watch out for that car"


def test_transcript_api_route_404s_when_absent(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/api/events/{event.event_id}/transcript")
    assert response.status_code == 404


def test_plate_reject_route_marks_the_observation_rejected(tmp_path: Path) -> None:
    """A real gap this closes: plate was the only observation table with
    no reject action in review_ui -- the generic
    `/api/recognition/{entity_type}/observations/{id}/reject` route
    excludes plate (it has no cluster concept), so this dedicated route
    must actually be reachable and not swallowed by route registration
    order."""

    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))
    observation_id = repository.database.list_plate_observations_for_event(event.event_id)[
        0
    ].observation_id

    response = client.post(
        f"/api/recognition/plate/observations/{observation_id}/reject",
        json={"actor": "tester"},
    )
    assert response.status_code == 200

    updated = repository.database.get_plate_observation(observation_id)
    assert updated is not None
    assert updated.review_status == "user_rejected"


def test_plate_confirm_route_corrects_the_ocr_text(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))
    observation_id = repository.database.list_plate_observations_for_event(event.event_id)[
        0
    ].observation_id

    response = client.post(
        f"/api/recognition/plate/observations/{observation_id}/confirm",
        json={"corrected_text": "xyz9876", "actor": "tester"},
    )
    assert response.status_code == 200

    updated = repository.database.get_plate_observation(observation_id)
    assert updated is not None
    assert updated.user_corrected_text == "XYZ9876"
    assert updated.review_status == "user_confirmed"


def test_event_detail_page_renders_plate_review_controls(tmp_path: Path) -> None:
    repository, event = _build_workspace_with_enrichment_data(tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/events/{event.event_id}")
    assert response.status_code == 200
    body = response.text
    assert "data-plate-text-input=" in body
    assert "data-confirm-plate-text-inline=" in body
    assert 'data-entity-type="plate">not a plate</button>' in body


def _make_face_cluster_with_observation(
    repository: Repository, tmp_path: Path, name: str
) -> tuple[UUID, UUID]:
    cluster_id = new_uuid()
    repository.database.upsert_face_cluster(
        FaceCluster(
            cluster_id=cluster_id,
            created_at=BASE,
            updated_at=BASE,
            observation_count=0,
            model_version="1.0.0",
        )
    )
    crop_path = tmp_path / f"{name}.jpg"
    crop_path.write_bytes(f"crop-{name}".encode())
    observation_id = new_uuid()
    repository.database.insert_face_observation(
        FaceObservation(
            observation_id=observation_id,
            signal_id=new_uuid(),
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(crop_path),
            crop_sha256=hash_file(crop_path),
            detector_confidence=0.5,
            cluster_id=cluster_id,
            detector_version="1.0.0",
        )
    )
    return cluster_id, observation_id


def test_detach_route_clears_the_observations_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _cluster_id, observation_id = _make_face_cluster_with_observation(repository, tmp_path, "a")
    client = TestClient(create_app(repository.workspace.root))

    response = client.post(
        f"/api/recognition/face/observations/{observation_id}/detach",
        json={"actor": "tester"},
    )
    assert response.status_code == 200

    updated = repository.database.get_face_observation(observation_id)
    assert updated is not None
    assert updated.cluster_id is None


def test_move_route_reassigns_the_observation_to_the_target_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _source_id, observation_id = _make_face_cluster_with_observation(repository, tmp_path, "a")
    target_cluster_id, _other = _make_face_cluster_with_observation(repository, tmp_path, "b")
    client = TestClient(create_app(repository.workspace.root))

    response = client.post(
        f"/api/recognition/face/observations/{observation_id}/move",
        json={"target_cluster_id": str(target_cluster_id), "actor": "tester"},
    )
    assert response.status_code == 200

    updated = repository.database.get_face_observation(observation_id)
    assert updated is not None
    assert updated.cluster_id == str(target_cluster_id)


def _make_mismerged_event_for_review_ui(repository: Repository, tmp_path: Path) -> EventRecord:
    clip_a_id, clip_b_id = new_uuid(), new_uuid()
    clip_a_path = tmp_path / "clip_a.mp4"
    clip_b_path = tmp_path / "clip_b.mp4"
    clip_a_path.write_bytes(b"clip a")
    clip_b_path.write_bytes(b"clip b")
    clip_a_sha, clip_b_sha = hash_file(clip_a_path), hash_file(clip_b_path)

    signal_a = Signal(
        id=new_uuid(),
        source="test",
        signal_type="motion",
        timestamp_start=BASE,
        timestamp_end=BASE,
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
        timestamp_start=BASE,
        timestamp_end=BASE,
        confidence=0.5,
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
    derived_a, derived_b = tmp_path / "derived_a.mp4", tmp_path / "derived_b.mp4"
    derived_a.write_bytes(b"derived a")
    derived_b.write_bytes(b"derived b")

    event = EventRecord(
        event_id=new_uuid(),
        created_at=BASE,
        pipeline_version="test",
        event_start=BASE,
        event_end=BASE,
        involved_cameras=["front", "rear"],
        signals=[signal_a, signal_b],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.5, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        derived_artifacts=[
            ArtifactReference(
                path=str(derived_a),
                artifact_type="derived_clip",
                created_at=BASE,
                sha256=hash_file(derived_a),
                metadata={
                    "camera_id": "front",
                    "source_clip_id": str(clip_a_id),
                    "source_sha256": clip_a_sha,
                },
            ),
            ArtifactReference(
                path=str(derived_b),
                artifact_type="derived_clip",
                created_at=BASE,
                sha256=hash_file(derived_b),
                metadata={
                    "camera_id": "rear",
                    "source_clip_id": str(clip_b_id),
                    "source_sha256": clip_b_sha,
                },
            ),
        ],
        evidence_summary="mismerged",
    )
    repository.save_event(event)
    return event


def test_split_route_creates_two_new_events_and_supersedes_the_original(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_mismerged_event_for_review_ui(repository, tmp_path)
    clip_a_id = event.derived_artifacts[0].metadata["source_clip_id"]
    clip_b_id = event.derived_artifacts[1].metadata["source_clip_id"]
    client = TestClient(create_app(repository.workspace.root))

    response = client.post(
        f"/api/events/{event.event_id}/split",
        json={"clip_id_groups": [[clip_a_id], [clip_b_id]], "actor": "tester"},
    )
    assert response.status_code == 200
    new_event_ids = response.json()["new_event_ids"]
    assert len(new_event_ids) == 2

    reloaded = repository.load_event(event.event_id)
    assert set(reloaded.superseded_by_event_ids) == {UUID(i) for i in new_event_ids}


def test_event_detail_page_shows_the_split_banner_after_splitting(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_mismerged_event_for_review_ui(repository, tmp_path)
    clip_a_id = event.derived_artifacts[0].metadata["source_clip_id"]
    clip_b_id = event.derived_artifacts[1].metadata["source_clip_id"]
    client = TestClient(create_app(repository.workspace.root))

    client.post(
        f"/api/events/{event.event_id}/split",
        json={"clip_id_groups": [[clip_a_id], [clip_b_id]], "actor": "tester"},
    )

    response = client.get(f"/events/{event.event_id}")
    assert response.status_code == 200
    assert "This event was split into" in response.text
    assert "Split into groups" not in response.text  # split section hidden once already split


def test_event_detail_page_offers_the_split_control_for_a_multi_clip_event(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_mismerged_event_for_review_ui(repository, tmp_path)
    client = TestClient(create_app(repository.workspace.root))

    response = client.get(f"/events/{event.event_id}")
    assert response.status_code == 200
    body = response.text
    assert "Split into groups" in body
    assert "data-split-group-input" in body
    assert "This event was split into" not in body
