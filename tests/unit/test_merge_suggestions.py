from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import cv2
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
FIXTURE_FACE = Path(__file__).parent.parent / "fixtures" / "sample_face.jpg"


def _make_plate_record(repository: Repository, normalized_text: str) -> UUID:
    plate_id = new_uuid()
    record = PlateRecord(
        plate_id=plate_id,
        normalized_text=normalized_text,
        created_at=BASE,
        updated_at=BASE,
        observation_count=1,
        first_seen_at=BASE,
        last_seen_at=BASE,
    )
    repository.database.upsert_plate_record(record)
    return plate_id


def test_suggest_plate_merges_flags_likely_ocr_misreads(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ABC1Z34")  # 1-character OCR-style misread

    suggestions = RecognitionService(repository).suggest_plate_merges(similarity_threshold=0.75)

    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "plate"
    assert suggestions[0].status == "pending"
    assert suggestions[0].similarity_score >= 0.75


def test_suggest_plate_merges_does_not_flag_unrelated_plates(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ZZZ9999")

    suggestions = RecognitionService(repository).suggest_plate_merges(similarity_threshold=0.75)
    assert suggestions == []


def test_suggest_plate_merges_does_not_duplicate_a_pending_suggestion(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ABC1Z34")
    service = RecognitionService(repository)

    first_run = service.suggest_plate_merges(similarity_threshold=0.75)
    second_run = service.suggest_plate_merges(similarity_threshold=0.75)

    assert len(first_run) == 1
    assert second_run == []  # already pending, not suggested again
    all_pending = repository.database.list_merge_suggestions(entity_type="plate", status="pending")
    assert len(all_pending) == 1


def test_suggest_plate_merges_skips_already_merged_records(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    first = _make_plate_record(repository, "ABC1234")
    second = _make_plate_record(repository, "ABC1Z34")
    service = RecognitionService(repository)
    service.merge_plates(first, second, actor="tester")

    suggestions = service.suggest_plate_merges(similarity_threshold=0.75)
    assert suggestions == []  # already merged, nothing left to suggest


def test_confirm_merge_suggestion_performs_the_actual_merge(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ABC1Z34")
    service = RecognitionService(repository)

    suggestions = service.suggest_plate_merges(similarity_threshold=0.75)
    suggestion = suggestions[0]

    service.confirm_merge_suggestion(suggestion.suggestion_id, actor="jane", notes="confirmed")

    resolved = repository.database.get_merge_suggestion(suggestion.suggestion_id)
    assert resolved is not None
    assert resolved.status == "confirmed"
    assert resolved.resolved_by == "jane"

    # the actual merge really happened
    source_row = repository.database.get_plate_record(suggestion.source_id)
    assert source_row is not None
    assert source_row.merged_into == str(suggestion.target_id)

    # and it was logged in the regular identity_merge_log, same as a manual merge
    log_content = repository.workspace.identity_merge_log_path.read_text(encoding="utf-8")
    assert "jane" in log_content


def test_reject_merge_suggestion_performs_no_merge(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ABC1Z34")
    service = RecognitionService(repository)

    suggestion = service.suggest_plate_merges(similarity_threshold=0.75)[0]
    service.reject_merge_suggestion(suggestion.suggestion_id, actor="jane")

    resolved = repository.database.get_merge_suggestion(suggestion.suggestion_id)
    assert resolved is not None
    assert resolved.status == "rejected"

    source_row = repository.database.get_plate_record(suggestion.source_id)
    assert source_row is not None
    assert source_row.merged_into is None  # no merge happened


def test_cannot_resolve_a_suggestion_twice(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _make_plate_record(repository, "ABC1234")
    _make_plate_record(repository, "ABC1Z34")
    service = RecognitionService(repository)

    suggestion = service.suggest_plate_merges(similarity_threshold=0.75)[0]
    service.reject_merge_suggestion(suggestion.suggestion_id, actor="jane")

    with pytest.raises(MergeError):
        service.reject_merge_suggestion(suggestion.suggestion_id, actor="jane")
    with pytest.raises(MergeError):
        service.confirm_merge_suggestion(suggestion.suggestion_id, actor="jane")


def test_confirm_nonexistent_suggestion_raises(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    with pytest.raises(MergeError):
        RecognitionService(repository).confirm_merge_suggestion(new_uuid(), actor="jane")


@pytest.mark.skipif(not FIXTURE_FACE.exists(), reason="face fixture not available")
def test_suggest_face_merges_uses_the_representative_crop_correctly(tmp_path: Path) -> None:
    """Exercises the real plumbing (load a cluster's representative crop,
    predict against the trained model, apply the threshold band) end to
    end using the real face fixture, rather than only hand-tracing it."""

    from gaggle.enrichment.face import (
        IncrementalFaceClusterer,
        crop_and_normalize,
        detect_faces,
    )

    image = cv2.imread(str(FIXTURE_FACE))
    faces = detect_faces(image)
    assert faces, "fixture must contain a detectable face"
    crop = crop_and_normalize(image, faces[0].bbox)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    crop_dir = repository.workspace.face_crops
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / "a.jpg"
    cv2.imwrite(str(crop_path), crop)

    cluster = FaceCluster(
        cluster_id=new_uuid(),
        created_at=BASE,
        updated_at=BASE,
        representative_crop_paths=[str(crop_path)],
        observation_count=1,
        first_seen_at=BASE,
        last_seen_at=BASE,
        model_version="1.0.0",
    )
    repository.database.upsert_face_cluster(cluster)

    clusterer = IncrementalFaceClusterer(
        repository.workspace.face_model_path, distance_threshold=70.0
    )
    clusterer.match_or_create_cluster(crop)
    clusterer.save()

    # Only one cluster exists, so there's nothing to suggest merging with --
    # this confirms the "no other cluster" path doesn't crash or false-suggest.
    suggestions = RecognitionService(repository).suggest_face_merges(
        cluster_distance_threshold=70.0, suggestion_multiplier=1.6
    )
    assert suggestions == []


@pytest.mark.skipif(not FIXTURE_FACE.exists(), reason="face fixture not available")
def test_suggest_face_merges_flags_a_second_close_cluster(tmp_path: Path) -> None:
    """Regression test for a real bug (see
    test_face_recognition.py::test_predict_nearest_cluster_excludes_the_queried_clusters_own_match):
    with 2+ trained clusters, `suggest_face_merges` must find the real
    nearest *other* cluster, not just discard every query as a self-match.
    A lightly blurred copy of the same real fixture face lands at LBPH
    distance ~93.5 -- verified numerically, inside the (70, 112] suggestion
    band for the default threshold/multiplier -- close enough to suggest,
    not close enough to have already auto-merged."""

    from gaggle.enrichment.face import (
        IncrementalFaceClusterer,
        crop_and_normalize,
        detect_faces,
    )

    image = cv2.imread(str(FIXTURE_FACE))
    faces = detect_faces(image)
    assert faces, "fixture must contain a detectable face"
    crop_a = crop_and_normalize(image, faces[0].bbox)
    crop_b = cv2.GaussianBlur(crop_a, (3, 3), 0)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    crop_dir = repository.workspace.face_crops
    crop_dir.mkdir(parents=True, exist_ok=True)

    clusterer = IncrementalFaceClusterer(
        repository.workspace.face_model_path, distance_threshold=70.0
    )
    cluster_ids = []
    for suffix, crop in (("a", crop_a), ("b", crop_b)):
        crop_path = crop_dir / f"{suffix}.jpg"
        cv2.imwrite(str(crop_path), crop)
        cluster_id, _distance, is_new = clusterer.match_or_create_cluster(crop)
        assert is_new  # confirm they did NOT already auto-merge
        cluster_ids.append(cluster_id)
        repository.database.upsert_face_cluster(
            FaceCluster(
                cluster_id=UUID(cluster_id),
                created_at=BASE,
                updated_at=BASE,
                representative_crop_paths=[str(crop_path)],
                observation_count=1,
                first_seen_at=BASE,
                last_seen_at=BASE,
                model_version="1.0.0",
            )
        )
    clusterer.save()

    suggestions = RecognitionService(repository).suggest_face_merges(
        cluster_distance_threshold=70.0, suggestion_multiplier=1.6
    )
    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "face"
    assert {str(suggestions[0].source_id), str(suggestions[0].target_id)} == set(cluster_ids)


def _seed_two_close_vehicle_appearance_clusters(repository: Repository) -> tuple[UUID, UUID]:
    """Two fingerprints constructed to sit at cosine distance ~0.130 apart --
    beyond the 0.10 match threshold (so they land in separate clusters) but
    within the 1.6x suggestion band (0.10, 0.16] -- verified numerically
    rather than guessed (see this test module's docstring discipline
    elsewhere in this project for why: `enrichment/voice.py`'s threshold
    validation story)."""

    import numpy as np

    from gaggle.enrichment.vehicle_appearance import (
        VEHICLE_FINGERPRINT_DIMENSIONS,
        IncrementalVehicleAppearanceClusterer,
    )

    clusterer = IncrementalVehicleAppearanceClusterer(
        repository.workspace.vehicle_appearance_model_path, distance_threshold=0.10
    )
    fingerprint_a = np.zeros(VEHICLE_FINGERPRINT_DIMENSIONS)
    fingerprint_a[0] = 1.0
    fingerprint_b = np.zeros(VEHICLE_FINGERPRINT_DIMENSIONS)
    fingerprint_b[0] = 0.87
    fingerprint_b[1] = 0.493

    cluster_a, _distance_a, is_new_a = clusterer.match_or_create_cluster(fingerprint_a)
    cluster_b, _distance_b, is_new_b = clusterer.match_or_create_cluster(fingerprint_b)
    assert is_new_a
    assert is_new_b  # confirm they did NOT already auto-merge
    clusterer.save()

    for cluster_id in (cluster_a, cluster_b):
        repository.database.upsert_vehicle_appearance_cluster(
            VehicleAppearanceCluster(
                cluster_id=UUID(cluster_id),
                created_at=BASE,
                updated_at=BASE,
                observation_count=1,
                first_seen_at=BASE,
                last_seen_at=BASE,
                model_version="1.0.0",
            )
        )
    return UUID(cluster_a), UUID(cluster_b)


def test_suggest_vehicle_appearance_merges_flags_close_clusters(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _seed_two_close_vehicle_appearance_clusters(repository)

    suggestions = RecognitionService(repository).suggest_vehicle_appearance_merges(
        cluster_distance_threshold=0.10, suggestion_multiplier=1.6
    )

    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "vehicle_appearance"
    assert suggestions[0].status == "pending"


def test_confirm_vehicle_appearance_merge_suggestion_performs_the_actual_merge(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _seed_two_close_vehicle_appearance_clusters(repository)
    service = RecognitionService(repository)

    suggestion = service.suggest_vehicle_appearance_merges(
        cluster_distance_threshold=0.10, suggestion_multiplier=1.6
    )[0]
    service.confirm_merge_suggestion(suggestion.suggestion_id, actor="jane", notes="same van")

    resolved = repository.database.get_merge_suggestion(suggestion.suggestion_id)
    assert resolved is not None
    assert resolved.status == "confirmed"

    source_row = repository.database.get_vehicle_appearance_cluster(suggestion.source_id)
    assert source_row is not None
    assert source_row.merged_into == str(suggestion.target_id)


def _seed_two_close_face_embedding_clusters(repository: Repository) -> tuple[UUID, UUID]:
    """Two embeddings at cosine distance ~0.130 apart -- same numeric
    construction as `_seed_two_close_vehicle_appearance_clusters` (beyond
    a 0.10 match threshold, so they land in separate clusters, but within
    the 1.6x suggestion band (0.10, 0.16]). Populates
    `IncrementalFaceEmbeddingClusterer` directly and creates matching
    `FaceCluster` rows with no crop files at all -- unlike the LBPH path,
    `suggest_face_merges(embedding_model="auraface")` compares stored
    centroids directly and never touches a crop image."""

    import numpy as np

    from gaggle.enrichment.face_auraface import IncrementalFaceEmbeddingClusterer

    clusterer = IncrementalFaceEmbeddingClusterer(
        repository.workspace.face_embedding_model_path, distance_threshold=0.10
    )
    embedding_a = np.zeros(512)
    embedding_a[0] = 1.0
    embedding_b = np.zeros(512)
    embedding_b[0] = 0.87
    embedding_b[1] = 0.493

    cluster_a, _distance_a, is_new_a = clusterer.match_or_create_cluster(embedding_a)
    cluster_b, _distance_b, is_new_b = clusterer.match_or_create_cluster(embedding_b)
    assert is_new_a
    assert is_new_b  # confirm they did NOT already auto-merge
    clusterer.save()

    for cluster_id in (cluster_a, cluster_b):
        repository.database.upsert_face_cluster(
            FaceCluster(
                cluster_id=UUID(cluster_id),
                created_at=BASE,
                updated_at=BASE,
                observation_count=1,
                first_seen_at=BASE,
                last_seen_at=BASE,
                model_version="auraface-v1",
            )
        )
    return UUID(cluster_a), UUID(cluster_b)


def test_suggest_face_merges_dispatches_to_auraface_embedding_comparison(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _seed_two_close_face_embedding_clusters(repository)

    suggestions = RecognitionService(repository).suggest_face_merges(
        cluster_distance_threshold=0.10,
        suggestion_multiplier=1.6,
        embedding_model="auraface",
    )

    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "face"
    assert suggestions[0].status == "pending"
    assert "AuraFace" in suggestions[0].basis


def test_suggest_face_merges_auraface_ignores_clusters_with_no_stored_centroid(
    tmp_path: Path,
) -> None:
    """A face cluster that predates switching to `embedding_model:
    auraface` has no entry in the embedding clusterer's model at all --
    must be silently skipped, not crash."""

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    for _ in range(2):
        repository.database.upsert_face_cluster(
            FaceCluster(
                cluster_id=new_uuid(),
                created_at=BASE,
                updated_at=BASE,
                observation_count=1,
                first_seen_at=BASE,
                last_seen_at=BASE,
                model_version="1.0.0",
            )
        )

    suggestions = RecognitionService(repository).suggest_face_merges(
        cluster_distance_threshold=0.10,
        suggestion_multiplier=1.6,
        embedding_model="auraface",
    )

    assert suggestions == []


def _seed_two_close_person_appearance_clusters(repository: Repository) -> tuple[UUID, UUID]:
    """Two fingerprints constructed to sit at cosine distance ~0.130 apart --
    beyond the 0.10 match threshold (so they land in separate clusters) but
    within the 1.6x suggestion band (0.10, 0.16]. Same numeric construction
    as `_seed_two_close_vehicle_appearance_clusters`."""

    import numpy as np

    from gaggle.enrichment.person_appearance import (
        PERSON_FINGERPRINT_DIMENSIONS,
        IncrementalPersonAppearanceClusterer,
    )

    clusterer = IncrementalPersonAppearanceClusterer(
        repository.workspace.person_appearance_model_path, distance_threshold=0.10
    )
    fingerprint_a = np.zeros(PERSON_FINGERPRINT_DIMENSIONS)
    fingerprint_a[0] = 1.0
    fingerprint_b = np.zeros(PERSON_FINGERPRINT_DIMENSIONS)
    fingerprint_b[0] = 0.87
    fingerprint_b[1] = 0.493

    cluster_a, _distance_a, is_new_a = clusterer.match_or_create_cluster(fingerprint_a)
    cluster_b, _distance_b, is_new_b = clusterer.match_or_create_cluster(fingerprint_b)
    assert is_new_a
    assert is_new_b  # confirm they did NOT already auto-merge
    clusterer.save()

    for cluster_id in (cluster_a, cluster_b):
        repository.database.upsert_person_appearance_cluster(
            PersonAppearanceCluster(
                cluster_id=UUID(cluster_id),
                created_at=BASE,
                updated_at=BASE,
                observation_count=1,
                first_seen_at=BASE,
                last_seen_at=BASE,
                model_version="1.0.0",
            )
        )
    return UUID(cluster_a), UUID(cluster_b)


def test_suggest_person_appearance_merges_flags_close_clusters(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _seed_two_close_person_appearance_clusters(repository)

    suggestions = RecognitionService(repository).suggest_person_appearance_merges(
        cluster_distance_threshold=0.10, suggestion_multiplier=1.6
    )

    assert len(suggestions) == 1
    assert suggestions[0].entity_type == "person_appearance"
    assert suggestions[0].status == "pending"


def test_confirm_person_appearance_merge_suggestion_performs_the_actual_merge(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    _seed_two_close_person_appearance_clusters(repository)
    service = RecognitionService(repository)

    suggestion = service.suggest_person_appearance_merges(
        cluster_distance_threshold=0.10, suggestion_multiplier=1.6
    )[0]
    service.confirm_merge_suggestion(suggestion.suggestion_id, actor="jane", notes="same jacket")

    resolved = repository.database.get_merge_suggestion(suggestion.suggestion_id)
    assert resolved is not None
    assert resolved.status == "confirmed"

    source_row = repository.database.get_person_appearance_cluster(suggestion.source_id)
    assert source_row is not None
    assert source_row.merged_into == str(suggestion.target_id)
