from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from gaggle.core.recognition import MergeError, RecognitionService
from gaggle.schemas.recognition import (
    FaceCluster,
    PersonAppearanceCluster,
    PlateRecord,
    VehicleAppearanceCluster,
)
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_face_cluster(
    repository: Repository, *, observation_count: int = 1, label: str | None = None
) -> UUID:
    cluster_id = new_uuid()
    cluster = FaceCluster(
        cluster_id=cluster_id,
        created_at=BASE,
        updated_at=BASE,
        label=label,
        representative_crop_paths=[f"/tmp/{cluster_id}.jpg"],
        observation_count=observation_count,
        first_seen_at=BASE,
        last_seen_at=BASE,
        model_version="1.0.0",
    )
    repository.database.upsert_face_cluster(cluster)
    return cluster_id


def _make_plate_record(
    repository: Repository, normalized_text: str, *, observation_count: int = 1
) -> UUID:
    plate_id = new_uuid()
    record = PlateRecord(
        plate_id=plate_id,
        normalized_text=normalized_text,
        created_at=BASE,
        updated_at=BASE,
        observation_count=observation_count,
        first_seen_at=BASE,
        last_seen_at=BASE,
    )
    repository.database.upsert_plate_record(record)
    return plate_id


def _make_vehicle_appearance_cluster(
    repository: Repository, *, observation_count: int = 1, label: str | None = None
) -> UUID:
    cluster_id = new_uuid()
    cluster = VehicleAppearanceCluster(
        cluster_id=cluster_id,
        created_at=BASE,
        updated_at=BASE,
        label=label,
        representative_crop_paths=[f"/tmp/{cluster_id}.jpg"],
        observation_count=observation_count,
        first_seen_at=BASE,
        last_seen_at=BASE,
        model_version="1.0.0",
    )
    repository.database.upsert_vehicle_appearance_cluster(cluster)
    return cluster_id


def _make_person_appearance_cluster(
    repository: Repository, *, observation_count: int = 1, label: str | None = None
) -> UUID:
    cluster_id = new_uuid()
    cluster = PersonAppearanceCluster(
        cluster_id=cluster_id,
        created_at=BASE,
        updated_at=BASE,
        label=label,
        representative_crop_paths=[f"/tmp/{cluster_id}.jpg"],
        observation_count=observation_count,
        first_seen_at=BASE,
        last_seen_at=BASE,
        model_version="1.0.0",
    )
    repository.database.upsert_person_appearance_cluster(cluster)
    return cluster_id


def test_merging_two_face_clusters_marks_source_as_alias(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)
    b = _make_face_cluster(repository)

    RecognitionService(repository).merge_faces(a, b, actor="tester", notes="same person")

    row_a = repository.database.get_face_cluster(a)
    assert row_a is not None
    assert row_a.merged_into == str(b)
    # the alias is never deleted or renamed -- it still exists as its own row
    row_b = repository.database.get_face_cluster(b)
    assert row_b is not None
    assert row_b.merged_into is None


def test_merge_is_permanently_logged(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)
    b = _make_face_cluster(repository)

    RecognitionService(repository).merge_faces(a, b, actor="jane", notes="confirmed via crops")

    log_lines = repository.workspace.identity_merge_log_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(log_lines) == 1
    assert "jane" in log_lines[0]
    assert str(a) in log_lines[0]
    assert str(b) in log_lines[0]


def test_resolve_face_identity_follows_chain(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)
    b = _make_face_cluster(repository)
    c = _make_face_cluster(repository)
    service = RecognitionService(repository)

    service.merge_faces(a, b, actor="tester")
    service.merge_faces(b, c, actor="tester")

    assert service.resolve_face_identity(a) == c
    assert service.resolve_face_identity(b) == c
    assert service.resolve_face_identity(c) == c


def test_merge_rejects_self_merge(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)

    with pytest.raises(MergeError):
        RecognitionService(repository).merge_faces(a, a, actor="tester")


def test_merge_rejects_unknown_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)

    with pytest.raises(MergeError):
        RecognitionService(repository).merge_faces(a, new_uuid(), actor="tester")


def test_merge_rejects_creating_a_cycle(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)
    b = _make_face_cluster(repository)
    service = RecognitionService(repository)
    service.merge_faces(a, b, actor="tester")

    # b already resolves through to... itself is the root; merging b back
    # into a (which now resolves to b) would create a 2-cycle.
    with pytest.raises(MergeError):
        service.merge_faces(b, a, actor="tester")


def test_get_face_identity_aggregates_across_merge_group(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository, observation_count=3, label="delivery driver")
    b = _make_face_cluster(repository, observation_count=5)
    service = RecognitionService(repository)
    service.merge_faces(a, b, actor="tester")

    identity = service.get_face_identity(a)
    assert identity.identity_id == b
    assert set(identity.member_cluster_ids) == {a, b}
    assert identity.observation_count == 8
    assert identity.label == "delivery driver"  # picked up from the merged-away member


def test_plate_merge_and_identity_aggregation(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    p1 = _make_plate_record(repository, "ABC1234", observation_count=2)
    p2 = _make_plate_record(repository, "ABC1Z34", observation_count=1)  # likely OCR misread
    service = RecognitionService(repository)

    service.merge_plates(p1, p2, actor="tester", notes="same vehicle, OCR misread the 2")

    identity = service.get_plate_identity(p1)
    assert identity.identity_id == p2
    assert set(identity.normalized_texts) == {"ABC1234", "ABC1Z34"}
    assert identity.observation_count == 3


def test_search_plates_exact_substring_match(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "XYZ9999")

    result = RecognitionService(repository).search_plates("ABC")
    assert len(result.exact_matches) == 1
    assert result.exact_matches[0].normalized_text == "ABC1234"  # type: ignore[attr-defined]
    assert result.fuzzy_suggestions == []


def test_search_plates_falls_back_to_fuzzy_suggestions(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")

    # "ABC1Z34" is close to "ABC1234" but not a substring match either way
    result = RecognitionService(repository).search_plates("ABC1Z34")
    assert result.exact_matches == []
    assert len(result.fuzzy_suggestions) == 1
    assert result.fuzzy_suggestions[0].normalized_text == "ABC1234"  # type: ignore[attr-defined]


def test_resolve_plate_input_accepts_id_or_text(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    plate_id = _make_plate_record(repository, "ABC1234")
    service = RecognitionService(repository)

    by_id = service.resolve_plate_input(str(plate_id))
    by_text = service.resolve_plate_input("abc1234")
    assert by_id is not None
    assert by_text is not None
    assert by_id.plate_id == by_text.plate_id == str(plate_id)


def test_list_face_sightings_exact_vs_merged(tmp_path: Path) -> None:
    from gaggle.schemas.recognition import FaceObservation

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_face_cluster(repository)
    b = _make_face_cluster(repository)

    def _make_observation(cluster_id: UUID, offset_seconds: float) -> None:
        repository.database.insert_face_observation(
            FaceObservation(
                observation_id=new_uuid(),
                signal_id=new_uuid(),
                clip_id=new_uuid(),
                camera_id="front",
                observed_at=BASE + timedelta(seconds=offset_seconds),
                crop_path=f"/tmp/{cluster_id}.jpg",
                crop_sha256="a" * 64,
                detector_confidence=0.5,
                cluster_id=cluster_id,
                detector_version="1.0.0",
            )
        )

    _make_observation(a, 0)
    _make_observation(b, 10)

    service = RecognitionService(repository)
    service.merge_faces(a, b, actor="tester")

    exact = service.list_face_sightings(a, follow_merges=False)
    assert len(exact) == 1

    merged = service.list_face_sightings(a, follow_merges=True)
    assert len(merged) == 2


def test_merging_two_vehicle_appearance_clusters_marks_source_as_alias(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_vehicle_appearance_cluster(repository)
    b = _make_vehicle_appearance_cluster(repository)

    RecognitionService(repository).merge_vehicle_appearances(
        a, b, actor="tester", notes="same truck"
    )

    row_a = repository.database.get_vehicle_appearance_cluster(a)
    assert row_a is not None
    assert row_a.merged_into == str(b)
    row_b = repository.database.get_vehicle_appearance_cluster(b)
    assert row_b is not None
    assert row_b.merged_into is None


def test_resolve_vehicle_appearance_identity_follows_chain(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_vehicle_appearance_cluster(repository)
    b = _make_vehicle_appearance_cluster(repository)
    c = _make_vehicle_appearance_cluster(repository)
    service = RecognitionService(repository)

    service.merge_vehicle_appearances(a, b, actor="tester")
    service.merge_vehicle_appearances(b, c, actor="tester")

    assert service.resolve_vehicle_appearance_identity(a) == c
    assert service.resolve_vehicle_appearance_identity(b) == c
    assert service.resolve_vehicle_appearance_identity(c) == c


def test_vehicle_appearance_merge_rejects_creating_a_cycle(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_vehicle_appearance_cluster(repository)
    b = _make_vehicle_appearance_cluster(repository)
    service = RecognitionService(repository)
    service.merge_vehicle_appearances(a, b, actor="tester")

    with pytest.raises(MergeError):
        service.merge_vehicle_appearances(b, a, actor="tester")


def test_get_vehicle_appearance_identity_aggregates_across_merge_group(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_vehicle_appearance_cluster(repository, observation_count=3, label="neighbor's van")
    b = _make_vehicle_appearance_cluster(repository, observation_count=5)
    service = RecognitionService(repository)
    service.merge_vehicle_appearances(a, b, actor="tester")

    identity = service.get_vehicle_appearance_identity(a)
    assert identity.identity_id == b
    assert set(identity.member_cluster_ids) == {a, b}
    assert identity.observation_count == 8
    assert identity.label == "neighbor's van"
    assert identity.representative_crop_paths  # crops carried through, unlike voice


def test_search_vehicle_appearances_exact_and_fuzzy(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_vehicle_appearance_cluster(repository, label="red pickup")
    _make_vehicle_appearance_cluster(repository, label="blue sedan")

    result = RecognitionService(repository).search_vehicle_appearances("red")
    assert len(result.exact_matches) == 1
    assert result.exact_matches[0].label == "red pickup"  # type: ignore[attr-defined]


def test_list_vehicle_appearance_sightings_exact_vs_merged(tmp_path: Path) -> None:
    from gaggle.schemas.recognition import VehicleAppearanceObservation

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_vehicle_appearance_cluster(repository)
    b = _make_vehicle_appearance_cluster(repository)

    def _make_observation(cluster_id: UUID, offset_seconds: float) -> None:
        repository.database.insert_vehicle_appearance_observation(
            VehicleAppearanceObservation(
                observation_id=new_uuid(),
                signal_id=new_uuid(),
                clip_id=new_uuid(),
                camera_id="front",
                observed_at=BASE + timedelta(seconds=offset_seconds),
                crop_path=f"/tmp/{cluster_id}.jpg",
                crop_sha256="a" * 64,
                fingerprint=[0.1, 0.2, 0.3],
                detector_confidence=0.5,
                cluster_id=cluster_id,
                detector_version="1.0.0",
            )
        )

    _make_observation(a, 0)
    _make_observation(b, 10)

    service = RecognitionService(repository)
    service.merge_vehicle_appearances(a, b, actor="tester")

    exact = service.list_vehicle_appearance_sightings(a, follow_merges=False)
    assert len(exact) == 1

    merged = service.list_vehicle_appearance_sightings(a, follow_merges=True)
    assert len(merged) == 2


def test_merging_two_person_appearance_clusters_marks_source_as_alias(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_person_appearance_cluster(repository)
    b = _make_person_appearance_cluster(repository)

    RecognitionService(repository).merge_person_appearances(
        a, b, actor="tester", notes="same jacket"
    )

    row_a = repository.database.get_person_appearance_cluster(a)
    assert row_a is not None
    assert row_a.merged_into == str(b)
    row_b = repository.database.get_person_appearance_cluster(b)
    assert row_b is not None
    assert row_b.merged_into is None


def test_resolve_person_appearance_identity_follows_chain(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_person_appearance_cluster(repository)
    b = _make_person_appearance_cluster(repository)
    c = _make_person_appearance_cluster(repository)
    service = RecognitionService(repository)

    service.merge_person_appearances(a, b, actor="tester")
    service.merge_person_appearances(b, c, actor="tester")

    assert service.resolve_person_appearance_identity(a) == c
    assert service.resolve_person_appearance_identity(b) == c
    assert service.resolve_person_appearance_identity(c) == c


def test_person_appearance_merge_rejects_creating_a_cycle(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_person_appearance_cluster(repository)
    b = _make_person_appearance_cluster(repository)
    service = RecognitionService(repository)
    service.merge_person_appearances(a, b, actor="tester")

    with pytest.raises(MergeError):
        service.merge_person_appearances(b, a, actor="tester")


def test_get_person_appearance_identity_aggregates_across_merge_group(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_person_appearance_cluster(repository, observation_count=3, label="mail carrier")
    b = _make_person_appearance_cluster(repository, observation_count=5)
    service = RecognitionService(repository)
    service.merge_person_appearances(a, b, actor="tester")

    identity = service.get_person_appearance_identity(a)
    assert identity.identity_id == b
    assert set(identity.member_cluster_ids) == {a, b}
    assert identity.observation_count == 8
    assert identity.label == "mail carrier"
    assert identity.representative_crop_paths  # crops carried through, unlike voice


def test_search_person_appearances_exact_and_fuzzy(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_person_appearance_cluster(repository, label="red jacket")
    _make_person_appearance_cluster(repository, label="blue hoodie")

    result = RecognitionService(repository).search_person_appearances("red")
    assert len(result.exact_matches) == 1
    assert result.exact_matches[0].label == "red jacket"  # type: ignore[attr-defined]


def test_list_person_appearance_sightings_exact_vs_merged(tmp_path: Path) -> None:
    from gaggle.schemas.recognition import PersonAppearanceObservation

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    a = _make_person_appearance_cluster(repository)
    b = _make_person_appearance_cluster(repository)

    def _make_observation(cluster_id: UUID, offset_seconds: float) -> None:
        repository.database.insert_person_appearance_observation(
            PersonAppearanceObservation(
                observation_id=new_uuid(),
                signal_id=new_uuid(),
                clip_id=new_uuid(),
                camera_id="front",
                observed_at=BASE + timedelta(seconds=offset_seconds),
                crop_path=f"/tmp/{cluster_id}.jpg",
                crop_sha256="a" * 64,
                fingerprint=[0.1, 0.2, 0.3],
                detector_confidence=0.5,
                cluster_id=cluster_id,
                detector_version="1.0.0",
            )
        )

    _make_observation(a, 0)
    _make_observation(b, 10)

    service = RecognitionService(repository)
    service.merge_person_appearances(a, b, actor="tester")

    exact = service.list_person_appearance_sightings(a, follow_merges=False)
    assert len(exact) == 1

    merged = service.list_person_appearance_sightings(a, follow_merges=True)
    assert len(merged) == 2
