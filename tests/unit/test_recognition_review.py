from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from gaggle.core.recognition import RecognitionService, ReviewError
from gaggle.schemas.recognition import (
    FaceCluster,
    FaceObservation,
    PersonAppearanceCluster,
    PersonAppearanceObservation,
    PlateObservation,
    VehicleAppearanceCluster,
    VehicleAppearanceObservation,
)
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_face_cluster(repository: Repository) -> UUID:
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
    return cluster_id


def _make_vehicle_appearance_cluster(repository: Repository) -> UUID:
    cluster_id = new_uuid()
    repository.database.upsert_vehicle_appearance_cluster(
        VehicleAppearanceCluster(
            cluster_id=cluster_id,
            created_at=BASE,
            updated_at=BASE,
            observation_count=0,
            model_version="1.0.0",
        )
    )
    return cluster_id


def _make_face_observation(
    repository: Repository, tmp_path: Path, cluster_id: UUID, name: str
) -> UUID:
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
    return observation_id


def _make_vehicle_observation(
    repository: Repository, tmp_path: Path, cluster_id: UUID, name: str
) -> UUID:
    crop_path = tmp_path / f"{name}.jpg"
    crop_path.write_bytes(f"crop-{name}".encode())
    observation_id = new_uuid()
    repository.database.insert_vehicle_appearance_observation(
        VehicleAppearanceObservation(
            observation_id=observation_id,
            signal_id=new_uuid(),
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(crop_path),
            crop_sha256=hash_file(crop_path),
            fingerprint=[0.1, 0.2],
            detector_confidence=0.5,
            cluster_id=cluster_id,
            detector_version="1.0.0",
        )
    )
    return observation_id


def _make_person_appearance_cluster(repository: Repository) -> UUID:
    cluster_id = new_uuid()
    repository.database.upsert_person_appearance_cluster(
        PersonAppearanceCluster(
            cluster_id=cluster_id,
            created_at=BASE,
            updated_at=BASE,
            observation_count=0,
            model_version="1.0.0",
        )
    )
    return cluster_id


def _make_person_observation(
    repository: Repository, tmp_path: Path, cluster_id: UUID, name: str
) -> UUID:
    crop_path = tmp_path / f"{name}.jpg"
    crop_path.write_bytes(f"crop-{name}".encode())
    observation_id = new_uuid()
    repository.database.insert_person_appearance_observation(
        PersonAppearanceObservation(
            observation_id=observation_id,
            signal_id=new_uuid(),
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(crop_path),
            crop_sha256=hash_file(crop_path),
            fingerprint=[0.1, 0.2],
            detector_confidence=0.5,
            cluster_id=cluster_id,
            detector_version="1.0.0",
        )
    )
    return observation_id


def _make_plate_observation(repository: Repository, tmp_path: Path, name: str) -> UUID:
    crop_path = tmp_path / f"{name}.jpg"
    crop_path.write_bytes(f"crop-{name}".encode())
    observation_id = new_uuid()
    repository.database.insert_plate_observation(
        PlateObservation(
            observation_id=observation_id,
            signal_id=new_uuid(),
            clip_id=new_uuid(),
            camera_id="front",
            observed_at=BASE,
            crop_path=str(crop_path),
            crop_sha256=hash_file(crop_path),
            raw_ocr_text="ABC123",
            normalized_text="ABC123",
            ocr_confidence=0.9,
            detector_confidence=0.9,
            review_status="needs_review",
            detector_version="1.0.0",
        )
    )
    return observation_id


def test_confirm_identity_sets_representative_and_confirms_all_observations(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")
    b = _make_face_observation(repository, tmp_path, cluster_id, "b")

    record = RecognitionService(repository).confirm_identity(
        cluster_id, [a], actor="tester", entity_type="face", label="mail carrier"
    )

    assert record.action == "confirmed"
    assert sorted(record.observation_ids) == sorted([a, b])
    cluster = repository.database.get_face_cluster(cluster_id)
    assert cluster is not None
    assert cluster.label == "mail carrier"
    assert cluster.representative_observation_ids_csv == str(a)
    obs_a = repository.database.get_face_observation(a)
    obs_b = repository.database.get_face_observation(b)
    assert obs_a is not None and obs_a.review_status == "user_confirmed"
    assert obs_b is not None and obs_b.review_status == "user_confirmed"


def test_confirm_identity_rejects_a_representative_not_in_the_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    _make_face_observation(repository, tmp_path, cluster_id, "a")
    other_cluster = _make_face_cluster(repository)
    outsider = _make_face_observation(repository, tmp_path, other_cluster, "outsider")

    with pytest.raises(ReviewError):
        RecognitionService(repository).confirm_identity(
            cluster_id, [outsider], actor="tester", entity_type="face"
        )


def test_reject_cluster_marks_every_observation_rejected(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_vehicle_appearance_cluster(repository)
    a = _make_vehicle_observation(repository, tmp_path, cluster_id, "a")
    b = _make_vehicle_observation(repository, tmp_path, cluster_id, "b")

    RecognitionService(repository).reject_cluster(
        cluster_id, actor="tester", entity_type="vehicle_appearance"
    )

    obs_a = repository.database.get_vehicle_appearance_observation(a)
    obs_b = repository.database.get_vehicle_appearance_observation(b)
    assert obs_a is not None and obs_a.review_status == "user_rejected"
    assert obs_b is not None and obs_b.review_status == "user_rejected"


def test_reject_observation_only_affects_the_one_observation(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")
    b = _make_face_observation(repository, tmp_path, cluster_id, "b")

    RecognitionService(repository).reject_observation(a, actor="tester", entity_type="face")

    obs_a = repository.database.get_face_observation(a)
    obs_b = repository.database.get_face_observation(b)
    assert obs_a is not None and obs_a.review_status == "user_rejected"
    assert obs_b is not None and obs_b.review_status == "needs_review"


def test_reject_observation_raises_for_unknown_observation(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    with pytest.raises(ReviewError):
        RecognitionService(repository).reject_observation(
            new_uuid(), actor="tester", entity_type="face"
        )


def test_detach_observation_clears_cluster_and_recomputes_count(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")
    b = _make_face_observation(repository, tmp_path, cluster_id, "b")

    record = RecognitionService(repository).detach_observation(
        a, actor="tester", entity_type="face"
    )

    assert record.action == "detached"
    assert record.cluster_id == cluster_id
    detached = repository.database.get_face_observation(a)
    assert detached is not None
    assert detached.cluster_id is None
    assert detached.review_status == "user_rejected"
    # the observation row itself is never deleted
    assert repository.database.get_face_observation(a) is not None

    remaining = repository.database.list_face_observations_by_cluster_ids([cluster_id])
    assert [o.observation_id for o in remaining] == [str(b)]
    updated_cluster = repository.database.get_face_cluster(cluster_id)
    assert updated_cluster is not None
    assert updated_cluster.observation_count == 1


def test_detach_observation_raises_for_unknown_observation(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    with pytest.raises(ReviewError):
        RecognitionService(repository).detach_observation(
            new_uuid(), actor="tester", entity_type="face"
        )


def test_move_observation_reassigns_cluster_and_recomputes_both_counts(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    source_cluster_id = _make_vehicle_appearance_cluster(repository)
    target_cluster_id = _make_vehicle_appearance_cluster(repository)
    a = _make_vehicle_observation(repository, tmp_path, source_cluster_id, "a")
    _make_vehicle_observation(repository, tmp_path, source_cluster_id, "b")
    _make_vehicle_observation(repository, tmp_path, target_cluster_id, "c")

    record = RecognitionService(repository).move_observation(
        a, target_cluster_id, actor="tester", entity_type="vehicle_appearance"
    )

    assert record.action == "moved"
    assert record.cluster_id == target_cluster_id
    assert str(source_cluster_id) in record.notes
    moved = repository.database.get_vehicle_appearance_observation(a)
    assert moved is not None
    assert moved.cluster_id == str(target_cluster_id)

    source_cluster = repository.database.get_vehicle_appearance_cluster(source_cluster_id)
    target_cluster = repository.database.get_vehicle_appearance_cluster(target_cluster_id)
    assert source_cluster is not None and source_cluster.observation_count == 1
    assert target_cluster is not None and target_cluster.observation_count == 2


def test_move_observation_reassigns_person_appearance_cluster_and_recomputes_both_counts(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    source_cluster_id = _make_person_appearance_cluster(repository)
    target_cluster_id = _make_person_appearance_cluster(repository)
    a = _make_person_observation(repository, tmp_path, source_cluster_id, "a")
    _make_person_observation(repository, tmp_path, source_cluster_id, "b")
    _make_person_observation(repository, tmp_path, target_cluster_id, "c")

    record = RecognitionService(repository).move_observation(
        a, target_cluster_id, actor="tester", entity_type="person_appearance"
    )

    assert record.action == "moved"
    assert record.cluster_id == target_cluster_id
    assert str(source_cluster_id) in record.notes
    moved = repository.database.get_person_appearance_observation(a)
    assert moved is not None
    assert moved.cluster_id == str(target_cluster_id)

    source_cluster = repository.database.get_person_appearance_cluster(source_cluster_id)
    target_cluster = repository.database.get_person_appearance_cluster(target_cluster_id)
    assert source_cluster is not None and source_cluster.observation_count == 1
    assert target_cluster is not None and target_cluster.observation_count == 2


def test_detach_observation_clears_person_appearance_cluster_and_recomputes_count(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_person_appearance_cluster(repository)
    a = _make_person_observation(repository, tmp_path, cluster_id, "a")
    b = _make_person_observation(repository, tmp_path, cluster_id, "b")

    record = RecognitionService(repository).detach_observation(
        a, actor="tester", entity_type="person_appearance"
    )

    assert record.action == "detached"
    assert record.cluster_id == cluster_id
    detached = repository.database.get_person_appearance_observation(a)
    assert detached is not None
    assert detached.cluster_id is None
    assert detached.review_status == "user_rejected"

    remaining = repository.database.list_person_appearance_observations_by_cluster_ids([cluster_id])
    assert [o.observation_id for o in remaining] == [str(b)]
    updated_cluster = repository.database.get_person_appearance_cluster(cluster_id)
    assert updated_cluster is not None
    assert updated_cluster.observation_count == 1


def test_confirm_identity_confirms_person_appearance_observations(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_person_appearance_cluster(repository)
    a = _make_person_observation(repository, tmp_path, cluster_id, "a")

    record = RecognitionService(repository).confirm_identity(
        cluster_id,
        [a],
        actor="tester",
        entity_type="person_appearance",
        label="mail carrier",
    )

    assert record.action == "confirmed"
    confirmed = repository.database.get_person_appearance_observation(a)
    assert confirmed is not None
    assert confirmed.review_status == "user_confirmed"
    updated_cluster = repository.database.get_person_appearance_cluster(cluster_id)
    assert updated_cluster is not None
    assert updated_cluster.label == "mail carrier"


def test_move_observation_raises_for_unknown_target_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")

    with pytest.raises(ReviewError):
        RecognitionService(repository).move_observation(
            a, new_uuid(), actor="tester", entity_type="face"
        )


def test_move_observation_raises_when_already_in_that_cluster(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")

    with pytest.raises(ReviewError):
        RecognitionService(repository).move_observation(
            a, cluster_id, actor="tester", entity_type="face"
        )


def test_purge_reviewed_crops_deletes_only_non_representative_confirmed_and_rejected(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    representative = _make_face_observation(repository, tmp_path, cluster_id, "rep")
    non_representative = _make_face_observation(repository, tmp_path, cluster_id, "nonrep")
    other_cluster = _make_face_cluster(repository)
    rejected = _make_face_observation(repository, tmp_path, other_cluster, "rejected")

    service = RecognitionService(repository)
    service.confirm_identity(cluster_id, [representative], actor="tester", entity_type="face")
    service.reject_observation(rejected, actor="tester", entity_type="face")

    rep_row = repository.database.get_face_observation(representative)
    nonrep_row = repository.database.get_face_observation(non_representative)
    rejected_row = repository.database.get_face_observation(rejected)
    assert rep_row is not None and nonrep_row is not None and rejected_row is not None
    rep_path = Path(rep_row.crop_path)
    nonrep_path = Path(nonrep_row.crop_path)
    rejected_path = Path(rejected_row.crop_path)
    assert rep_path.exists() and nonrep_path.exists() and rejected_path.exists()

    record = service.purge_reviewed_crops("face", actor="tester")

    assert sorted(record.purged_observation_ids) == sorted([non_representative, rejected])
    assert rep_path.exists()  # representative crop is kept
    assert not nonrep_path.exists()  # confirmed but non-representative -> purged
    assert not rejected_path.exists()  # rejected -> purged

    nonrep_row_after = repository.database.get_face_observation(non_representative)
    assert nonrep_row_after is not None
    assert nonrep_row_after.crop_purged_at is not None
    assert nonrep_row_after.crop_path == str(nonrep_path)  # historical pointer kept


def test_purge_reviewed_crops_dry_run_does_not_delete_anything(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    a = _make_face_observation(repository, tmp_path, cluster_id, "a")

    service = RecognitionService(repository)
    service.reject_observation(a, actor="tester", entity_type="face")
    obs_row = repository.database.get_face_observation(a)
    assert obs_row is not None
    crop_path = Path(obs_row.crop_path)

    record = service.purge_reviewed_crops("face", actor="tester", dry_run=True)

    assert a in record.purged_observation_ids
    assert crop_path.exists()  # dry run never deletes
    obs_row_after = repository.database.get_face_observation(a)
    assert obs_row_after is not None
    assert obs_row_after.crop_purged_at is None  # dry run never marks purged either


def test_purge_flag_purges_immediately_on_confirm(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    cluster_id = _make_face_cluster(repository)
    representative = _make_face_observation(repository, tmp_path, cluster_id, "rep")
    non_representative = _make_face_observation(repository, tmp_path, cluster_id, "nonrep")

    RecognitionService(repository).confirm_identity(
        cluster_id, [representative], actor="tester", entity_type="face", purge=True
    )

    nonrep_row = repository.database.get_face_observation(non_representative)
    assert nonrep_row is not None
    assert not Path(nonrep_row.crop_path).exists()
    assert nonrep_row.crop_purged_at is not None


def test_plate_confirm_and_purge_workflow(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    observation_id = _make_plate_observation(repository, tmp_path, "plate")

    service = RecognitionService(repository)
    service.confirm_plate_observation(observation_id, "abc123", actor="tester")
    row = repository.database.get_plate_observation(observation_id)
    assert row is not None
    assert row.review_status == "user_confirmed"
    assert row.user_corrected_text == "ABC123"
    crop_path = Path(row.crop_path)
    assert crop_path.exists()

    record = service.purge_reviewed_crops("plate", actor="tester")

    assert observation_id in record.purged_observation_ids
    assert not crop_path.exists()


def test_confirm_plate_observation_raises_for_unknown_observation(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    with pytest.raises(ReviewError):
        RecognitionService(repository).confirm_plate_observation(
            new_uuid(), "abc123", actor="tester"
        )
