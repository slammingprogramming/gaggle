"""Real deep-learning face detection via YuNet (`opencv/opencv_zoo`,
Apache-2.0), the new default `enrichment.face.detector` option --
replacing `enrichment/face.py`'s classical Haar cascade, which remains
available as `detector: haar` and is still what a zero-extra-dependency
install falls back to if the model can't be fetched.

Unlike the Haar cascade (bundled with every OpenCV install, zero setup),
YuNet's weights are not bundled with this package -- they're fetched on
demand via `core/models.py::ModelRegistry` into a per-machine cache, the
first time this detector is actually used (or via `gaggle models
download`). No new pip dependency either way: `cv2.FaceDetectorYN_create`
is part of `opencv-contrib-python-headless`, already a core dependency of
this project for the Haar cascade and LBPH clusterer.

Returns the same `DetectedFace` shape `enrichment/face.py::detect_faces`
does (bounding box + confidence) so `enrichment/service.py` only needs to
dispatch on which detector to call, not on two different result shapes.
YuNet also produces 5-point facial landmarks per detection; discarded for
now since neither LBPH clustering nor crop storage use them today --
landmark-aligned crops would likely improve re-identification accuracy
further, a natural follow-up, not implemented in this pass.
"""

from __future__ import annotations

from typing import Any

import cv2
import cv2.typing
import numpy.typing as npt

from gaggle.core.models import Device, ModelRegistry, ModelUnavailableError
from gaggle.enrichment.face import DetectedFace
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

ImageArray = cv2.typing.MatLike

DETECTOR_VERSION = "yunet-2023mar"


class YuNetUnavailableError(RuntimeError):
    """Raised when the YuNet model can't be obtained (a `ModelUnavailableError`
    from `core/models.py`, e.g. no network on first use) or loaded (a
    corrupt/unreadable file)."""


class YuNetDetector:
    """Wraps `cv2.FaceDetectorYN`. Construction fetches the model (via
    `ModelRegistry`, cached after the first call) and builds the
    detector; expected to be constructed once and reused across many
    frames, mirroring `enrichment/vehicle_yolo.py::YoloOnnxDetector`'s
    shape.
    """

    def __init__(
        self,
        device: Device = "cpu",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        try:
            model_path = ModelRegistry().ensure_model("yunet-detector", device=device)
        except ModelUnavailableError as error:
            raise YuNetUnavailableError(str(error)) from error

        # cv2.dnn.DNN_BACKEND_CUDA/DNN_TARGET_CUDA exist in the Python
        # bindings regardless of how the installed OpenCV binary was
        # built -- calling them on a CPU-only build (which is what the
        # standard `opencv-contrib-python-headless` PyPI wheel always is;
        # confirmed: `cv2.cuda.getCudaEnabledDeviceCount()` is 0 on it,
        # even with a real GPU and working CUDA present) either raises or
        # silently does nothing useful. There is no pip-installable fix
        # for this, unlike onnxruntime's CUDA gap -- it requires a custom
        # OpenCV build compiled with `-DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON`,
        # which this project does not attempt to automate. Real-check
        # device availability first and fall back to the CPU
        # backend/target (cv2's defaults) with a clear, honest log
        # explaining why, rather than silently no-op'ing or crashing.
        backend_id, target_id = 0, 0  # cv2.dnn.DNN_BACKEND_DEFAULT / DNN_TARGET_CPU
        if device == "cuda":
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                backend_id, target_id = cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
            else:
                LOGGER.warning(
                    "yunet_cuda_requested_but_opencv_has_no_cuda_support",
                    message=(
                        "device: cuda was requested, but this OpenCV build has no CUDA "
                        "support compiled in (standard pip opencv-contrib-python-headless "
                        "never does) -- YuNet detection will run on CPU. This is an "
                        "OpenCV build limitation, not fixable via pip; see "
                        "docs/local-ai.md's GPU setup notes. Embedding "
                        "(embedding_model: auraface) and plate detection "
                        "(detector: fast_alpr) still use onnxruntime's real CUDA support "
                        "independently of this."
                    ),
                )
        try:
            # cv2's bundled stubs don't declare FaceDetectorYN_create even
            # though it's real at runtime (same class of gap as cv2.data/
            # cv2.face -- see AGENTS.md's 1.4 pass narrative).
            self._detector: cv2.FaceDetectorYN = cv2.FaceDetectorYN_create(  # type: ignore[attr-defined]
                str(model_path),
                "",
                (0, 0),
                score_threshold,
                nms_threshold,
                top_k,
                backend_id,
                target_id,
            )
        except cv2.error as error:
            raise YuNetUnavailableError(
                f"could not load YuNet model at {model_path}: {error}"
            ) from error

    def detect(self, image: ImageArray) -> list[DetectedFace]:
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _retval, raw_faces = self._detector.detect(image)
        if raw_faces is None:
            return []
        faces: npt.NDArray[Any] = raw_faces
        detected = [
            DetectedFace(
                bbox=(max(0, int(row[0])), max(0, int(row[1])), int(row[2]), int(row[3])),
                # Column 14 is YuNet's own detection confidence; columns
                # 4-13 are the 5-point landmarks (see module docstring).
                confidence=round(min(1.0, max(0.0, float(row[14]))), 6),
            )
            for row in faces
        ]
        detected.sort(key=lambda f: (-(f.bbox[2] * f.bbox[3]), f.bbox))
        return detected
