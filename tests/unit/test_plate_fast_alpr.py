from __future__ import annotations

from dataclasses import dataclass

import pytest

from gaggle.enrichment.plate_fast_alpr import (
    FastAlprDetector,
    FastAlprUnavailableError,
    fast_alpr_available,
)

# The test sandbox has no network access (see AGENTS.md), and `fast-alpr`
# is an optional extra not installed by default in `dev` -- these tests
# degrade cleanly either way. `FastAlprDetector.detect_and_ocr`'s logic is
# exercised via `object.__new__` with duck-typed fakes (mirroring
# test_face_auraface.py's `_FakeRecognitionModel` pattern) rather than
# real `fast_alpr` result objects, so no import of `fast_alpr` itself is
# needed to test the conversion logic.


def test_raises_when_fast_alpr_not_installed() -> None:
    if fast_alpr_available():
        pytest.skip("fast-alpr is installed in this environment")
    with pytest.raises(FastAlprUnavailableError):
        FastAlprDetector()


@dataclass(frozen=True)
class _FakeBoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class _FakeDetectionResult:
    label: str
    confidence: float
    bounding_box: _FakeBoundingBox


@dataclass(frozen=True)
class _FakeOcrResult:
    text: str
    confidence: float | list[float]
    region: str | None = None
    region_confidence: float | None = None


@dataclass(frozen=True)
class _FakeAlprResult:
    detection: _FakeDetectionResult
    ocr: _FakeOcrResult | None


class _FakeAlpr:
    def __init__(self, results: list[_FakeAlprResult]) -> None:
        self._results = results

    def predict(self, frame: object) -> list[_FakeAlprResult]:
        return self._results


def _make_detector(results: list[_FakeAlprResult]) -> FastAlprDetector:
    detector = object.__new__(FastAlprDetector)
    detector._alpr = _FakeAlpr(results)  # type: ignore[attr-defined]
    return detector


def test_detect_and_ocr_converts_a_real_reading() -> None:
    detector = _make_detector(
        [
            _FakeAlprResult(
                detection=_FakeDetectionResult(
                    label="plate",
                    confidence=0.9,
                    bounding_box=_FakeBoundingBox(x1=10, y1=20, x2=110, y2=60),
                ),
                ocr=_FakeOcrResult(
                    text="abc-1234",
                    confidence=0.85,
                    region="US-CA",
                    region_confidence=0.7,
                ),
            )
        ]
    )

    pairs = detector.detect_and_ocr(object())

    assert len(pairs) == 1
    region, ocr = pairs[0]
    assert region.bbox == (10, 20, 100, 40)
    assert region.confidence == pytest.approx(0.9)
    assert region.source == "fast_alpr"
    assert ocr is not None
    assert ocr.raw_text == "abc-1234"
    assert ocr.normalized_text == "ABC1234"
    assert ocr.confidence == pytest.approx(0.85)
    assert ocr.region == "US-CA"
    assert ocr.region_confidence == pytest.approx(0.7)


def test_detect_and_ocr_averages_a_per_character_confidence_list() -> None:
    detector = _make_detector(
        [
            _FakeAlprResult(
                detection=_FakeDetectionResult(
                    label="plate",
                    confidence=0.5,
                    bounding_box=_FakeBoundingBox(x1=0, y1=0, x2=10, y2=10),
                ),
                ocr=_FakeOcrResult(text="AB", confidence=[0.6, 1.0]),
            )
        ]
    )

    pairs = detector.detect_and_ocr(object())

    ocr = pairs[0][1]
    assert ocr is not None
    assert ocr.confidence == pytest.approx(0.8)
    assert ocr.region is None
    assert ocr.region_confidence is None


def test_detect_and_ocr_passes_through_a_detection_with_no_ocr_result() -> None:
    """A detected plate region with no OCR reading (e.g. an unreadable
    crop) is a real, expected outcome -- not an error -- and must reach
    the caller as `(region, None)`, not be silently dropped here (the
    caller decides what to do with a None OCR result)."""

    detector = _make_detector(
        [
            _FakeAlprResult(
                detection=_FakeDetectionResult(
                    label="plate",
                    confidence=0.6,
                    bounding_box=_FakeBoundingBox(x1=0, y1=0, x2=10, y2=10),
                ),
                ocr=None,
            )
        ]
    )

    pairs = detector.detect_and_ocr(object())

    assert len(pairs) == 1
    assert pairs[0][1] is None
