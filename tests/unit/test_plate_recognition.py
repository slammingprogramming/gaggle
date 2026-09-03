from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from gaggle.enrichment.plate import (
    detect_plate_regions,
    ocr_plate_text,
    render_debug_frame,
    tesseract_available,
)

pytestmark = pytest.mark.skipif(not tesseract_available(), reason="tesseract not available")


def _make_synthetic_car_scene(
    plate_text: str = "ABC1234",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((300, 400, 3), 120, dtype="uint8")
    cv2.rectangle(canvas, (50, 50), (350, 250), (90, 90, 90), -1)
    plate_bbox = (120, 180, 160, 40)
    x, y, w, h = plate_bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.putText(canvas, plate_text, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return canvas, plate_bbox


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def test_heuristic_detector_localizes_a_synthetic_plate() -> None:
    canvas, target_bbox = _make_synthetic_car_scene()
    regions = detect_plate_regions(canvas)
    assert regions
    best = max(regions, key=lambda r: _iou(r.bbox, target_bbox))
    assert _iou(best.bbox, target_bbox) > 0.3


def test_ocr_reads_the_synthetic_plate_text() -> None:
    canvas, target_bbox = _make_synthetic_car_scene("ABC1234")
    x, y, w, h = target_bbox
    crop = canvas[y : y + h, x : x + w]
    result = ocr_plate_text(crop)
    assert "ABC1234" in result.normalized_text
    assert result.confidence > 0.5


def test_no_plate_present_yields_no_high_confidence_region() -> None:
    canvas = np.full((300, 400, 3), 120, dtype="uint8")
    regions = detect_plate_regions(canvas)
    # A perfectly uniform image should not produce a confident detection.
    assert all(r.confidence < 0.5 for r in regions)


def test_deduplicates_overlapping_regions() -> None:
    canvas, _target_bbox = _make_synthetic_car_scene()
    regions = detect_plate_regions(canvas)
    # No two returned regions should overlap heavily -- dedup should have
    # collapsed anything the cascade and heuristic both found.
    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            assert _iou(a.bbox, b.bbox) <= 0.4


def test_tesseract_available_reports_true_when_binary_on_path() -> None:
    assert tesseract_available() == (shutil.which("tesseract") is not None)


def _make_junk_heavy_scene(
    plate_text: str = "XYZ9081",
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """A scene with several plate-aspect-ratio decoy rectangles (grille
    slats, trim lines) -- the scenario that originally exposed the old
    contour-only detector missing a real plate entirely (0.00 IoU) before
    MSER-based detection was added."""

    rng = np.random.default_rng(7)
    canvas = np.full((300, 400, 3), 130, dtype="uint8")
    cv2.rectangle(canvas, (50, 50), (350, 250), (100, 100, 100), -1)
    junk_rects = [(70, 90, 120, 25), (200, 70, 90, 20), (60, 210, 100, 22), (230, 205, 110, 24)]
    for x, y, w, h in junk_rects:
        shade = int(rng.integers(60, 200))
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (shade, shade, shade), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 30, 30), 1)
    plate_bbox = (130, 190, 150, 38)
    x, y, w, h = plate_bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), -1)
    cv2.putText(canvas, plate_text, (x + 8, y + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return canvas, plate_bbox


def test_mser_finds_the_plate_in_a_junk_heavy_scene_the_old_detector_missed() -> None:
    """Regression test: this exact scenario produced a 0.00 IoU (complete
    miss) from the pre-MSER contour-only heuristic during development, and
    was the concrete evidence used to justify adding MSER detection."""

    canvas, target_bbox = _make_junk_heavy_scene()
    regions = detect_plate_regions(canvas)
    assert regions
    best = max(regions, key=lambda r: _iou(r.bbox, target_bbox))
    assert _iou(best.bbox, target_bbox) > 0.7
    # Confirm MSER specifically is what found it, not a lucky contour hit.
    mser_regions = [r for r in regions if r.source == "mser"]
    assert any(_iou(r.bbox, target_bbox) > 0.7 for r in mser_regions)


def test_ocr_reads_the_plate_found_in_a_junk_heavy_scene() -> None:
    canvas, target_bbox = _make_junk_heavy_scene("XYZ9081")
    regions = detect_plate_regions(canvas)
    best = max(regions, key=lambda r: _iou(r.bbox, target_bbox))
    x, y, w, h = best.bbox
    crop = canvas[y : y + h, x : x + w]
    result = ocr_plate_text(crop)
    assert "XYZ9081" in result.normalized_text


def test_rotated_plate_is_still_detected() -> None:
    """A plate viewed at an angle -- the other specific accuracy complaint
    this pass addressed, via minAreaRect-based (rotation-aware) aspect
    checking instead of an axis-aligned bounding box."""

    plate_img = np.full((50, 180, 3), 255, dtype="uint8")
    cv2.putText(plate_img, "ANG5521", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    canvas = np.full((300, 400, 3), 130, dtype="uint8")
    cv2.rectangle(canvas, (50, 50), (350, 250), (100, 100, 100), -1)
    center = (90, 25)
    rotation_matrix = cv2.getRotationMatrix2D(center, 20, 1.0)
    rotated_plate = cv2.warpAffine(
        plate_img, rotation_matrix, (180, 50), borderValue=(100, 100, 100)
    )
    canvas[170:220, 110:290] = rotated_plate

    regions = detect_plate_regions(canvas)
    assert regions
    assert any(r.source == "heuristic_contour" for r in regions)


def test_blank_scene_yields_no_regions_at_all() -> None:
    canvas = np.full((300, 400, 3), 120, dtype="uint8")
    regions = detect_plate_regions(canvas)
    assert regions == []


def test_render_debug_frame_draws_every_region_without_crashing() -> None:
    canvas, _target_bbox = _make_junk_heavy_scene()
    regions = detect_plate_regions(canvas)
    annotated = render_debug_frame(canvas, regions)
    assert annotated.shape == canvas.shape
    # The debug frame should differ from the original (something was drawn),
    # unless there happened to be zero regions -- there are regions here.
    assert not np.array_equal(annotated, canvas)


def test_render_debug_frame_handles_zero_regions() -> None:
    canvas = np.full((300, 400, 3), 120, dtype="uint8")
    annotated = render_debug_frame(canvas, [])
    assert np.array_equal(annotated, canvas)
