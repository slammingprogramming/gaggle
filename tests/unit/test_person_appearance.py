from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaggle.enrichment.person_appearance import (
    PERSON_FINGERPRINT_DIMENSIONS,
    IncrementalPersonAppearanceClusterer,
    PersonAppearanceError,
    PersonRegion,
    _cosine_distance,
    compute_fingerprint,
)


def _make_synthetic_person_scene(
    color_bgr: tuple[int, int, int], seed: int = 0, aspect: float = 0.4
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A synthetic scene with one colored, person-tall rectangle standing
    in for a pedestrian's clothing against a contrasting background --
    same construction style as test_vehicle_appearance.py's
    `_make_synthetic_vehicle_scene`, but narrow-and-tall (aspect < 1)
    instead of wide-and-short."""

    rng = np.random.default_rng(seed)
    canvas = np.full((400, 300, 3), (60, 60, 60), dtype=np.uint8)
    height = 240
    width = int(height * aspect)
    x, y = 100, 80
    body = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    noise = rng.integers(-12, 12, size=body.shape, dtype=np.int16)
    body = np.clip(body.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    canvas[y : y + height, x : x + width] = body
    return canvas, (x, y, width, height)


def test_compute_fingerprint_rejects_an_empty_crop() -> None:
    canvas = np.full((400, 300, 3), (60, 60, 60), dtype=np.uint8)
    region = PersonRegion(bbox=(0, 0, 0, 0), confidence=0.5, source="yolo")
    with pytest.raises(PersonAppearanceError):
        compute_fingerprint(canvas, region)


def test_fingerprint_has_the_documented_fixed_length() -> None:
    canvas, bbox = _make_synthetic_person_scene((30, 30, 200))
    region = PersonRegion(bbox=bbox, confidence=0.8, source="yolo")
    result = compute_fingerprint(canvas, region)
    assert result.fingerprint.shape == (PERSON_FINGERPRINT_DIMENSIONS,)


def test_fingerprint_carries_structured_attributes_not_a_description() -> None:
    canvas, bbox = _make_synthetic_person_scene((30, 30, 200))
    region = PersonRegion(bbox=bbox, confidence=0.8, source="yolo")
    result = compute_fingerprint(canvas, region)
    assert isinstance(result.dominant_hue_bin, int)
    assert 0.0 <= result.height_to_frame_ratio <= 1.0
    # bbox height (240) over frame height (400)
    assert result.height_to_frame_ratio == pytest.approx(240 / 400, abs=1e-6)


def test_measured_distance_gap_between_same_and_different_clothing_colors() -> None:
    """Reproduces the same style of empirical measurement
    `test_vehicle_appearance.py` makes for vehicles: same-clothing-color
    (different noise realization) distances should sit far below
    different-clothing-color distances -- the evidence behind the shipped
    default threshold (0.10), not an assertion of it in isolation."""

    colors = {
        "red": (30, 30, 200),
        "blue": (200, 30, 30),
        "green": (30, 150, 30),
        "yellow": (30, 200, 200),
        "orange": (0, 120, 230),
    }

    same_person_distances: list[float] = []
    for color in colors.values():
        fingerprints = []
        for seed in range(8):
            canvas, bbox = _make_synthetic_person_scene(color, seed=seed * 17 + 1)
            region = PersonRegion(bbox=bbox, confidence=0.8, source="yolo")
            fingerprints.append(compute_fingerprint(canvas, region).fingerprint)
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                same_person_distances.append(_cosine_distance(fingerprints[i], fingerprints[j]))

    representative_fingerprints = {}
    for name, color in colors.items():
        canvas, bbox = _make_synthetic_person_scene(color, seed=999)
        region = PersonRegion(bbox=bbox, confidence=0.8, source="yolo")
        representative_fingerprints[name] = compute_fingerprint(canvas, region).fingerprint

    names = list(colors.keys())
    different_person_distances: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            different_person_distances.append(
                _cosine_distance(
                    representative_fingerprints[names[i]], representative_fingerprints[names[j]]
                )
            )

    assert max(same_person_distances) < 0.05
    assert min(different_person_distances) > 0.3
    default_threshold = IncrementalPersonAppearanceClusterer(Path("unused")).distance_threshold
    assert max(same_person_distances) < default_threshold < min(different_person_distances)


def test_clusterer_matches_the_same_person_and_separates_a_different_one(
    tmp_path: Path,
) -> None:
    clusterer = IncrementalPersonAppearanceClusterer(tmp_path / "model.json")
    canvas_a1, bbox_a1 = _make_synthetic_person_scene((30, 30, 200), seed=1)
    canvas_a2, bbox_a2 = _make_synthetic_person_scene((30, 30, 200), seed=2)
    canvas_b, bbox_b = _make_synthetic_person_scene((200, 30, 30), seed=3)

    fp_a1 = compute_fingerprint(
        canvas_a1, PersonRegion(bbox=bbox_a1, confidence=0.8, source="yolo")
    ).fingerprint
    fp_a2 = compute_fingerprint(
        canvas_a2, PersonRegion(bbox=bbox_a2, confidence=0.8, source="yolo")
    ).fingerprint
    fp_b = compute_fingerprint(
        canvas_b, PersonRegion(bbox=bbox_b, confidence=0.8, source="yolo")
    ).fingerprint

    cluster_a1, _distance, is_new_a1 = clusterer.match_or_create_cluster(fp_a1)
    assert is_new_a1

    cluster_a2, distance_a2, is_new_a2 = clusterer.match_or_create_cluster(fp_a2)
    assert not is_new_a2
    assert cluster_a2 == cluster_a1
    assert distance_a2 < clusterer.distance_threshold

    cluster_b, _distance, is_new_b = clusterer.match_or_create_cluster(fp_b)
    assert is_new_b
    assert cluster_b != cluster_a1


def test_clusterer_model_survives_a_save_reload_cycle(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    clusterer = IncrementalPersonAppearanceClusterer(model_path)
    canvas, bbox = _make_synthetic_person_scene((30, 30, 200), seed=1)
    fingerprint = compute_fingerprint(
        canvas, PersonRegion(bbox=bbox, confidence=0.8, source="yolo")
    ).fingerprint
    cluster_id, _distance, _is_new = clusterer.match_or_create_cluster(fingerprint)
    clusterer.save()

    reloaded = IncrementalPersonAppearanceClusterer(model_path)
    same_color_canvas, same_color_bbox = _make_synthetic_person_scene((30, 30, 200), seed=2)
    same_fingerprint = compute_fingerprint(
        same_color_canvas,
        PersonRegion(bbox=same_color_bbox, confidence=0.8, source="yolo"),
    ).fingerprint
    predicted_id, distance = reloaded.predict_nearest_cluster(same_fingerprint)
    assert predicted_id == cluster_id
    assert distance < reloaded.distance_threshold


def test_get_cluster_centroid_returns_none_for_an_unknown_cluster(tmp_path: Path) -> None:
    clusterer = IncrementalPersonAppearanceClusterer(tmp_path / "model.json")
    assert clusterer.get_cluster_centroid("does-not-exist") is None


def test_predict_nearest_cluster_on_an_empty_model_returns_none(tmp_path: Path) -> None:
    clusterer = IncrementalPersonAppearanceClusterer(tmp_path / "model.json")
    predicted_id, distance = clusterer.predict_nearest_cluster(
        np.zeros(PERSON_FINGERPRINT_DIMENSIONS)
    )
    assert predicted_id is None
    assert distance == 0.0
