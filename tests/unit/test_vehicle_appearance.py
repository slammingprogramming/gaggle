from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gaggle.enrichment.vehicle_appearance import (
    VEHICLE_FINGERPRINT_DIMENSIONS,
    IncrementalVehicleAppearanceClusterer,
    VehicleAppearanceError,
    VehicleRegion,
    _cosine_distance,
    compute_fingerprint,
    detect_vehicle_regions,
)


def _make_synthetic_vehicle_scene(
    color_bgr: tuple[int, int, int], seed: int = 0, aspect: float = 2.2
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A synthetic scene with one colored rectangle standing in for a
    vehicle body against a contrasting road/background -- same
    construction style as test_plate_recognition.py's
    `_make_synthetic_car_scene`, tuned for a vehicle-scale target with a
    darker "window band" for internal edge texture (a flat single-color
    rectangle alone produces weak internal contours)."""

    rng = np.random.default_rng(seed)
    canvas = np.full((300, 500, 3), (60, 60, 60), dtype=np.uint8)
    width, height = 220, int(220 / aspect)
    x, y = 140, 90
    body = np.full((height, width, 3), color_bgr, dtype=np.uint8)
    noise = rng.integers(-12, 12, size=body.shape, dtype=np.int16)
    body = np.clip(body.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    canvas[y : y + height, x : x + width] = body
    window_h = max(5, height // 4)
    cv2.rectangle(canvas, (x + 10, y + 5), (x + width - 10, y + 5 + window_h), (20, 20, 20), -1)
    return canvas, (x, y, width, height)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def test_heuristic_detector_localizes_a_synthetic_vehicle() -> None:
    canvas, target_bbox = _make_synthetic_vehicle_scene((30, 30, 200))
    regions = detect_vehicle_regions(canvas)
    assert regions
    best = max(regions, key=lambda r: _iou(r.bbox, target_bbox))
    assert _iou(best.bbox, target_bbox) > 0.3


def test_no_vehicle_present_yields_no_confident_region() -> None:
    canvas = np.full((300, 500, 3), (60, 60, 60), dtype=np.uint8)
    regions = detect_vehicle_regions(canvas)
    assert all(r.confidence < 0.5 for r in regions)


def test_compute_fingerprint_rejects_an_empty_crop() -> None:
    canvas = np.full((300, 500, 3), (60, 60, 60), dtype=np.uint8)
    region = VehicleRegion(bbox=(0, 0, 0, 0), confidence=0.5, source="heuristic_contour")
    with pytest.raises(VehicleAppearanceError):
        compute_fingerprint(canvas, region)


def test_fingerprint_has_the_documented_fixed_length() -> None:
    canvas, bbox = _make_synthetic_vehicle_scene((30, 30, 200))
    region = VehicleRegion(bbox=bbox, confidence=0.8, source="heuristic_contour")
    result = compute_fingerprint(canvas, region)
    assert result.fingerprint.shape == (VEHICLE_FINGERPRINT_DIMENSIONS,)


def test_measured_distance_gap_between_same_and_different_vehicles() -> None:
    """Reproduces the empirical measurement documented in this module's
    docstring: same-vehicle (same hue, different noise realization)
    distances should sit far below different-vehicle (different hue)
    distances, across several colors and several seeds -- not just one
    lucky sample. This is the evidence behind the shipped default
    threshold (0.10), not an assertion of it in isolation."""

    colors = {
        "red": (30, 30, 200),
        "blue": (200, 30, 30),
        "green": (30, 150, 30),
        "yellow": (30, 200, 200),
        "orange": (0, 120, 230),
    }

    same_vehicle_distances: list[float] = []
    for color in colors.values():
        fingerprints = []
        for seed in range(8):
            canvas, bbox = _make_synthetic_vehicle_scene(color, seed=seed * 17 + 1)
            region = VehicleRegion(bbox=bbox, confidence=0.8, source="heuristic_contour")
            fingerprints.append(compute_fingerprint(canvas, region).fingerprint)
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                same_vehicle_distances.append(_cosine_distance(fingerprints[i], fingerprints[j]))

    representative_fingerprints = {}
    for name, color in colors.items():
        canvas, bbox = _make_synthetic_vehicle_scene(color, seed=999)
        region = VehicleRegion(bbox=bbox, confidence=0.8, source="heuristic_contour")
        representative_fingerprints[name] = compute_fingerprint(canvas, region).fingerprint

    names = list(colors.keys())
    different_vehicle_distances: list[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            different_vehicle_distances.append(
                _cosine_distance(
                    representative_fingerprints[names[i]], representative_fingerprints[names[j]]
                )
            )

    assert max(same_vehicle_distances) < 0.05
    assert min(different_vehicle_distances) > 0.3
    # The shipped default sits comfortably inside the measured gap, biased
    # toward the same-vehicle end (conservative -- prefer a missed merge
    # over a false one).
    default_threshold = IncrementalVehicleAppearanceClusterer(Path("unused")).distance_threshold
    assert max(same_vehicle_distances) < default_threshold < min(different_vehicle_distances)


def test_clusterer_matches_the_same_vehicle_and_separates_a_different_one(
    tmp_path: Path,
) -> None:
    clusterer = IncrementalVehicleAppearanceClusterer(tmp_path / "model.json")
    canvas_a1, bbox_a1 = _make_synthetic_vehicle_scene((30, 30, 200), seed=1)
    canvas_a2, bbox_a2 = _make_synthetic_vehicle_scene((30, 30, 200), seed=2)
    canvas_b, bbox_b = _make_synthetic_vehicle_scene((200, 30, 30), seed=3)

    fp_a1 = compute_fingerprint(
        canvas_a1, VehicleRegion(bbox=bbox_a1, confidence=0.8, source="heuristic_contour")
    ).fingerprint
    fp_a2 = compute_fingerprint(
        canvas_a2, VehicleRegion(bbox=bbox_a2, confidence=0.8, source="heuristic_contour")
    ).fingerprint
    fp_b = compute_fingerprint(
        canvas_b, VehicleRegion(bbox=bbox_b, confidence=0.8, source="heuristic_contour")
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
    clusterer = IncrementalVehicleAppearanceClusterer(model_path)
    canvas, bbox = _make_synthetic_vehicle_scene((30, 30, 200), seed=1)
    fingerprint = compute_fingerprint(
        canvas, VehicleRegion(bbox=bbox, confidence=0.8, source="heuristic_contour")
    ).fingerprint
    cluster_id, _distance, _is_new = clusterer.match_or_create_cluster(fingerprint)
    clusterer.save()

    reloaded = IncrementalVehicleAppearanceClusterer(model_path)
    same_color_canvas, same_color_bbox = _make_synthetic_vehicle_scene((30, 30, 200), seed=2)
    same_fingerprint = compute_fingerprint(
        same_color_canvas,
        VehicleRegion(bbox=same_color_bbox, confidence=0.8, source="heuristic_contour"),
    ).fingerprint
    predicted_id, distance = reloaded.predict_nearest_cluster(same_fingerprint)
    assert predicted_id == cluster_id
    assert distance < reloaded.distance_threshold


def test_get_cluster_centroid_returns_none_for_an_unknown_cluster(tmp_path: Path) -> None:
    clusterer = IncrementalVehicleAppearanceClusterer(tmp_path / "model.json")
    assert clusterer.get_cluster_centroid("does-not-exist") is None


def test_predict_nearest_cluster_on_an_empty_model_returns_none(tmp_path: Path) -> None:
    clusterer = IncrementalVehicleAppearanceClusterer(tmp_path / "model.json")
    predicted_id, distance = clusterer.predict_nearest_cluster(
        np.zeros(VEHICLE_FINGERPRINT_DIMENSIONS)
    )
    assert predicted_id is None
    assert distance == 0.0
