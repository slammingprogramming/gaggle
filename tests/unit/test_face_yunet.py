from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaggle.core.models import ModelUnavailableError
from gaggle.enrichment import face_yunet as face_yunet_module
from gaggle.enrichment.face_yunet import YuNetDetector, YuNetUnavailableError

# The test sandbox has no network access (see AGENTS.md), so these tests
# never let `YuNetDetector.__init__` reach a real download or construct a
# real `cv2.FaceDetectorYN` -- they exercise the error-wrapping and
# post-processing logic in isolation, the same way
# `test_optional_ml_degradation.py::test_vehicle_decode_recovers_confident_box_and_filters_noise`
# exercises `YoloOnnxDetector._decode` via `object.__new__` without a real
# ONNX session.


def test_raises_when_model_registry_cannot_provide_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(self: object, name: str, device: str = "cpu") -> Path:
        raise ModelUnavailableError("no network")

    monkeypatch.setattr(face_yunet_module.ModelRegistry, "ensure_model", _fail)

    with pytest.raises(YuNetUnavailableError):
        YuNetDetector()


def test_raises_when_the_cached_model_file_is_not_a_real_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bogus_model = tmp_path / "not-a-real-model.onnx"
    bogus_model.write_bytes(b"not an onnx model")

    def _fake_ensure_model(self: object, name: str, device: str = "cpu") -> Path:
        return bogus_model

    monkeypatch.setattr(face_yunet_module.ModelRegistry, "ensure_model", _fake_ensure_model)

    with pytest.raises(YuNetUnavailableError):
        YuNetDetector()


def _mock_model_and_create(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> list[tuple]:
    """Mocks both `ModelRegistry.ensure_model` (avoid a real download) and
    `cv2.FaceDetectorYN_create` (avoid needing a real, valid model file --
    only the arguments it's called with matter for these tests), and
    returns the list `cv2.FaceDetectorYN_create` calls get appended to."""

    def _fake_ensure_model(self: object, name: str, device: str = "cpu") -> Path:
        return model_path

    monkeypatch.setattr(face_yunet_module.ModelRegistry, "ensure_model", _fake_ensure_model)

    calls: list[tuple] = []

    def _fake_create(*args: object) -> object:
        calls.append(args)
        return _FakeCvDetector(None)

    monkeypatch.setattr(face_yunet_module.cv2, "FaceDetectorYN_create", _fake_create)
    return calls


def test_device_cuda_falls_back_to_cpu_backend_when_opencv_has_no_cuda_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real, verified gap: `cv2.dnn.DNN_BACKEND_CUDA`/
    `DNN_TARGET_CUDA` exist in the Python bindings regardless of whether
    the installed OpenCV binary was actually built with CUDA support --
    the standard pip `opencv-contrib-python-headless` wheel never is
    (confirmed: `cv2.cuda.getCudaEnabledDeviceCount()` is 0 on it even
    with a real, working GPU). Requesting `device: cuda` must fall back
    to the CPU backend/target honestly (with a clear log explaining why)
    rather than silently doing nothing or crashing."""

    calls = _mock_model_and_create(monkeypatch, tmp_path / "model.onnx")
    monkeypatch.setattr(face_yunet_module.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)

    YuNetDetector(device="cuda")

    assert len(calls) == 1
    backend_id, target_id = calls[0][-2], calls[0][-1]
    assert (backend_id, target_id) == (0, 0)  # DNN_BACKEND_DEFAULT / DNN_TARGET_CPU


def test_device_cuda_uses_cuda_backend_when_opencv_actually_has_cuda_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_model_and_create(monkeypatch, tmp_path / "model.onnx")
    monkeypatch.setattr(face_yunet_module.cv2.cuda, "getCudaEnabledDeviceCount", lambda: 1)

    YuNetDetector(device="cuda")

    assert len(calls) == 1
    backend_id, target_id = calls[0][-2], calls[0][-1]
    assert backend_id == face_yunet_module.cv2.dnn.DNN_BACKEND_CUDA
    assert target_id == face_yunet_module.cv2.dnn.DNN_TARGET_CUDA


def test_device_cpu_never_checks_cuda_availability_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _mock_model_and_create(monkeypatch, tmp_path / "model.onnx")

    def _fail_if_called() -> int:
        raise AssertionError("getCudaEnabledDeviceCount should not be called for device=cpu")

    monkeypatch.setattr(face_yunet_module.cv2.cuda, "getCudaEnabledDeviceCount", _fail_if_called)

    YuNetDetector(device="cpu")

    assert len(calls) == 1
    assert (calls[0][-2], calls[0][-1]) == (0, 0)


def test_detect_clamps_negative_coordinates_and_confidence_and_sorts_by_area() -> None:
    detector = object.__new__(YuNetDetector)
    fake_cv2_detector = _FakeCvDetector(
        np.array(
            [
                # small face, negative x/y (near-edge detection), confidence > 1
                [-5, -5, 20, 20, *([0.0] * 10), 1.5],
                # larger face, well-formed, confidence within range
                [10, 10, 50, 60, *([0.0] * 10), 0.42],
            ],
            dtype=np.float32,
        )
    )
    detector._detector = fake_cv2_detector  # type: ignore[attr-defined]

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    faces = detector.detect(image)

    assert len(faces) == 2
    # Larger face (area 3000) sorts before the smaller one (area 400).
    assert faces[0].bbox == (10, 10, 50, 60)
    assert faces[0].confidence == pytest.approx(0.42)
    assert faces[1].bbox == (0, 0, 20, 20)  # negative origin clamped to 0
    assert faces[1].confidence == 1.0  # clamped to the [0, 1] range


def test_detect_returns_empty_list_when_no_faces_found() -> None:
    detector = object.__new__(YuNetDetector)
    detector._detector = _FakeCvDetector(None)  # type: ignore[attr-defined]

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(image) == []


class _FakeCvDetector:
    """Stands in for `cv2.FaceDetectorYN` -- only `setInputSize`/`detect`
    are exercised by `YuNetDetector.detect`."""

    def __init__(self, raw_faces: np.ndarray | None) -> None:
        self._raw_faces = raw_faces

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        pass

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        return 1, self._raw_faces
