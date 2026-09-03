from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gaggle.enrichment.face import (
    IncrementalFaceClusterer,
    crop_and_normalize,
    detect_faces,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_face.jpg"


def test_detects_the_real_face_in_the_fixture_photo() -> None:
    image = cv2.imread(str(FIXTURE_PATH))
    faces = detect_faces(image)
    assert len(faces) == 1
    assert faces[0].confidence > 0
    _x, _y, w, h = faces[0].bbox
    assert w > 50 and h > 50


def test_blank_image_has_no_faces() -> None:
    blank = np.full((200, 200, 3), 200, dtype="uint8")
    assert detect_faces(blank) == []


def test_clusterer_matches_the_same_face_and_separates_a_different_one(tmp_path: Path) -> None:
    image = cv2.imread(str(FIXTURE_PATH))
    faces = detect_faces(image)
    crop = crop_and_normalize(image, faces[0].bbox)

    clusterer = IncrementalFaceClusterer(tmp_path / "model.yml")
    first_id, first_distance, first_is_new = clusterer.match_or_create_cluster(crop)
    assert first_is_new is True
    assert first_distance == 0.0

    second_id, second_distance, second_is_new = clusterer.match_or_create_cluster(crop)
    assert second_is_new is False
    assert second_id == first_id
    assert second_distance < 30

    noise = np.random.default_rng(42).integers(0, 255, size=(100, 100), dtype="uint8")
    third_id, third_distance, third_is_new = clusterer.match_or_create_cluster(noise)
    assert third_is_new is True
    assert third_id != first_id
    assert third_distance > clusterer.distance_threshold


def test_clusterer_persists_across_reload(tmp_path: Path) -> None:
    image = cv2.imread(str(FIXTURE_PATH))
    faces = detect_faces(image)
    crop = crop_and_normalize(image, faces[0].bbox)
    model_path = tmp_path / "model.yml"

    first = IncrementalFaceClusterer(model_path)
    cluster_id, _distance, _is_new = first.match_or_create_cluster(crop)
    first.save()

    second = IncrementalFaceClusterer(model_path)
    reloaded_id, _distance2, is_new2 = second.match_or_create_cluster(crop)
    assert is_new2 is False
    assert reloaded_id == cluster_id


def test_fixture_photo_exists() -> None:
    # Guards against the fixture file being accidentally removed -- every
    # other test in this module silently no-ops (via cv2.imread returning
    # None -> a downstream crash) without a clear message otherwise.
    assert FIXTURE_PATH.exists(), f"missing test fixture: {FIXTURE_PATH}"


@pytest.mark.parametrize("min_size", [(30, 30), (10, 10)])
def test_detect_faces_accepts_custom_min_size(min_size: tuple[int, int]) -> None:
    image = cv2.imread(str(FIXTURE_PATH))
    faces = detect_faces(image, min_size=min_size)
    assert len(faces) >= 1


def test_predict_nearest_cluster_excludes_the_queried_clusters_own_match(
    tmp_path: Path,
) -> None:
    """Regression test for a real bug: querying `predict_nearest_cluster`
    with a cluster's own trained crop always matched that same cluster at
    distance 0 (confirmed: OpenCV's LBPH `predict()` reliably returns the
    label that trained on the exact queried image), so
    `suggest_face_merges`'s "discard a self-match" check discarded every
    real query and could never surface an actual cross-cluster suggestion
    -- caught via real cross-entity-type testing (the same bug was found
    first in the new vehicle-appearance/voice centroid clusterers, then
    confirmed here by symmetry), fixed by having
    `predict_nearest_cluster` exclude the query's own cluster from the
    search itself, using `predict_collect`/`StandardCollector` to see
    every trained label's distance in one pass rather than only the
    single closest one `predict()` exposes."""

    image = cv2.imread(str(FIXTURE_PATH))
    faces = detect_faces(image)
    crop_a = crop_and_normalize(image, faces[0].bbox)
    crop_b = cv2.bitwise_not(crop_a)  # a distinctly different texture

    clusterer = IncrementalFaceClusterer(tmp_path / "model.yml")
    cluster_a, _distance, _is_new = clusterer.match_or_create_cluster(crop_a)
    cluster_b, _distance, _is_new = clusterer.match_or_create_cluster(crop_b)

    # Without exclusion, querying with a cluster's own crop always
    # self-matches at distance 0.
    self_matched_id, self_distance = clusterer.predict_nearest_cluster(crop_a)
    assert self_matched_id == cluster_a
    assert self_distance == 0.0

    # With its own cluster excluded, the real nearest *other* cluster is found.
    other_matched_id, _other_distance = clusterer.predict_nearest_cluster(
        crop_a, exclude_cluster_id=cluster_a
    )
    assert other_matched_id == cluster_b
