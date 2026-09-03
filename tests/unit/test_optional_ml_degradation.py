from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaggle.enrichment.llm_analysis import analyze_transcript
from gaggle.enrichment.transcription import (
    TranscriptionUnavailableError,
    WhisperTranscriber,
    faster_whisper_available,
    load_transcriber_if_available,
)
from gaggle.enrichment.vehicle_yolo import (
    VisionModelUnavailableError,
    YoloOnnxDetector,
    load_detector_if_available,
)


def test_vehicle_detector_raises_when_model_file_missing(tmp_path: Path) -> None:
    with pytest.raises(VisionModelUnavailableError):
        YoloOnnxDetector(tmp_path / "does-not-exist.onnx")


def test_vehicle_detector_loader_degrades_gracefully(tmp_path: Path) -> None:
    assert load_detector_if_available(tmp_path / "does-not-exist.onnx") is None


def test_vehicle_decode_recovers_confident_box_and_filters_noise() -> None:
    detector = object.__new__(YoloOnnxDetector)
    detector.confidence_threshold = 0.35
    detector.iou_threshold = 0.45

    num_classes = 80
    raw = np.zeros((1, 4 + num_classes, 2), dtype=np.float32)
    raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = 320, 320, 100, 60
    raw[0, 4 + 2, 0] = 0.9  # class 2 = "car"
    raw[0, 0, 1], raw[0, 1, 1], raw[0, 2, 1], raw[0, 3, 1] = 100, 100, 20, 20
    raw[0, 4 + 0, 1] = 0.1  # below threshold

    detections = detector._decode(raw, orig_width=640, orig_height=640)
    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].confidence > 0.85


def test_transcriber_raises_when_faster_whisper_not_installed() -> None:
    if faster_whisper_available():
        pytest.skip("faster-whisper is installed in this environment")
    with pytest.raises(TranscriptionUnavailableError):
        WhisperTranscriber()


def test_transcriber_loader_degrades_gracefully() -> None:
    if faster_whisper_available():
        pytest.skip("faster-whisper is installed in this environment")
    assert load_transcriber_if_available() is None


def test_faster_whisper_available_returns_false_instead_of_raising_on_a_non_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real failure mode hit on Windows: `faster_whisper` imports
    `ctranslate2`, which unconditionally imports `torch` -- and a broken/
    conflicting local torch CUDA install can raise a plain OSError (not
    ImportError) partway through that chain. This must degrade to "not
    available" (transcription skipped for the run), not crash the run --
    see `test_face_auraface.py`'s matching regression test for
    `insightface_available()`, the same real bug hit twice."""
    import builtins

    real_import = builtins.__import__

    def _broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "faster_whisper":
            raise OSError(
                "[WinError 127] The specified procedure could not be found. "
                'Error loading "torch\\lib\\cudnn_cnn64_9.dll" or one of its dependencies.'
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    assert faster_whisper_available() is False


def test_llm_analysis_short_circuits_on_empty_transcript() -> None:
    result = analyze_transcript(
        "   ", endpoint="http://example.invalid/v1/chat/completions", api_key="x", model="m"
    )
    assert result.importance_score == 0.0
    assert result.extracted_events == []
