"""Optional local vehicle/object detection via a user-supplied YOLO ONNX model.

This is the intentional real-ML-classifier extension point the built-in
contour-based "moving region" detector (`detection/object_detection.py`)
deliberately avoids being (see that module's docstring and the project's
ML-avoidance-by-default directive). It is entirely optional:

* Requires the `vision` extra (`pip install gaggle[vision]`,
  i.e. `onnxruntime`) to even import successfully.
* Requires a model file (e.g. a standard `yolov8n.onnx` export) at a path
  set in config (`enrichment.vision.model_path`) -- not bundled, since
  shipping model weights in the package would both bloat it and undermine
  "offline after installation" (the model download is the one-time,
  user-initiated network step, exactly like Whisper below).

If either the dependency or the model file is missing, every function here
degrades to "no detections, logged once" rather than raising -- enrichment
is always additive, never a hard requirement for the core pipeline to run.

Runs on CPU by default; set `enrichment.vision.device: cuda` to use
onnxruntime's CUDAExecutionProvider if available on the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2.typing
import numpy as np
import numpy.typing as npt

from gaggle.utils.logging import get_logger

ImageArray = cv2.typing.MatLike
FloatArray = npt.NDArray[np.float32]

LOGGER = get_logger(__name__)

MODEL_INPUT_SIZE = 640
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45

# Standard 80-class COCO label set used by common pretrained YOLO exports.
# Only vehicle/pedestrian-relevant classes are surfaced by default via
# `RELEVANT_CLASS_IDS`; the full names are kept for transparency in output.
COCO_CLASS_NAMES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
RELEVANT_CLASS_IDS: frozenset[int] = frozenset(
    {0, 1, 2, 3, 5, 7}
)  # person, bicycle, car, motorcycle, bus, truck


class VisionModelUnavailableError(RuntimeError):
    """Raised (and expected to be caught) when onnxruntime or the model file is missing."""


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in source image pixel coordinates
    class_id: int
    class_name: str
    confidence: float


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


class YoloOnnxDetector:
    """Loads a YOLO-family ONNX model once and runs standard box-decoding + NMS.

    Implements the common YOLOv8-style output format: a single
    `[1, 4+num_classes, num_boxes]` tensor of center-x/center-y/w/h plus
    per-class scores. Other export formats (e.g. YOLOv5's `[1, num_boxes,
    5+num_classes]`) are not currently handled -- see `docs/local-ai.md`
    for which exports are supported.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        if not onnxruntime_available():
            raise VisionModelUnavailableError(
                "onnxruntime is not installed; install the 'vision' extra "
                "(pip install gaggle[vision]) to enable local vehicle detection"
            )
        if not model_path.exists():
            raise VisionModelUnavailableError(
                f"no YOLO ONNX model found at {model_path}; see docs/local-ai.md "
                "for how to obtain one (a one-time, user-initiated download)"
            )
        import onnxruntime as ort

        from gaggle.core.models import ensure_cuda_dlls_preloaded

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if device == "cuda":
            ensure_cuda_dlls_preloaded()
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # onnxruntime silently falls back to whatever's next in the
        # providers list (usually CPU) if CUDA can't actually be
        # initialized (missing driver/CUDA-toolkit/cuDNN DLLs) -- it logs
        # a warning of its own but doesn't fail construction, so
        # `device: cuda` in config can silently *not* be honored. Compare
        # what was requested against what `get_providers()` says is
        # actually active and log it plainly rather than leaving this to
        # be inferred from a cryptic onnxruntime warning buried in stdout.
        active_provider = self._session.get_providers()[0]
        if device == "cuda" and active_provider != "CUDAExecutionProvider":
            LOGGER.warning(
                "vehicle_detector_cuda_requested_but_not_active",
                requested="cuda",
                active_provider=active_provider,
                message=(
                    "device: cuda was requested but onnxruntime's CUDAExecutionProvider "
                    f"did not initialize; running on {active_provider} instead. See "
                    "docs/local-ai.md's GPU setup notes."
                ),
            )
        else:
            LOGGER.info("vehicle_detector_provider_active", provider=active_provider)

    def detect(self, image: ImageArray) -> list[VehicleDetection]:
        import cv2

        height, width = image.shape[:2]
        resized = cv2.resize(image, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        blob = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, :, :, :]

        raw_output = self._session.run(None, {self._input_name: blob})[0]
        return self._decode(raw_output, width, height)

    def _decode(
        self, raw_output: FloatArray, orig_width: int, orig_height: int
    ) -> list[VehicleDetection]:
        import cv2

        predictions = np.squeeze(raw_output).T  # -> [num_boxes, 4+num_classes]
        if predictions.ndim != 2 or predictions.shape[1] <= 4:
            return []
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        keep = confidences >= self.confidence_threshold
        predictions, class_ids, confidences = predictions[keep], class_ids[keep], confidences[keep]
        if predictions.shape[0] == 0:
            return []

        scale_x, scale_y = orig_width / MODEL_INPUT_SIZE, orig_height / MODEL_INPUT_SIZE
        boxes = []
        for cx, cy, w, h in predictions[:, :4]:
            x0 = (cx - w / 2) * scale_x
            y0 = (cy - h / 2) * scale_y
            box_w = w * scale_x
            box_h = h * scale_y
            boxes.append([x0, y0, box_w, box_h])

        raw_indices = cv2.dnn.NMSBoxes(
            boxes, confidences.tolist(), self.confidence_threshold, self.iou_threshold
        )
        indices = np.array(raw_indices).flatten() if len(raw_indices) else np.array([], dtype=int)

        detections: list[VehicleDetection] = []
        for i in indices:
            x0, y0, box_w, box_h = boxes[i]
            class_id = int(class_ids[i])
            class_name = (
                COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else str(class_id)
            )
            detections.append(
                VehicleDetection(
                    bbox=(int(x0), int(y0), int(x0 + box_w), int(y0 + box_h)),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=round(float(confidences[i]), 6),
                )
            )
        detections.sort(key=lambda d: (-d.confidence, d.bbox))
        return detections


def load_detector_if_available(model_path: Path, device: str = "cpu") -> YoloOnnxDetector | None:
    """Best-effort loader: returns None (logged, not raised) if unavailable."""

    try:
        return YoloOnnxDetector(model_path, device=device)
    except VisionModelUnavailableError:
        return None
