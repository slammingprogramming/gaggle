"""Real deep-learning license plate detection + OCR via fast-alpr
(`ankandrew/fast-alpr` + `ankandrew/fast-plate-ocr`, MIT), the new
`enrichment.plate.detector: fast_alpr` option -- replacing
`enrichment/plate.py`'s classical cascade+MSER+contour-heuristic detector
and Tesseract OCR (calibrated for Russian-format plate proportions, an
honest documented limitation), which remains available as
`detector: cascade` and is still what a zero-extra-dependency install
falls back to.

fast-alpr's default OCR model (`cct-xs-v2-global-model`) is trained on
international ("global") plate formats rather than one region -- the
right shape for "focus on American plates but also work for other
regions too," unlike the cascade path's Russian-format calibration. It
also optionally reports a region/country guess per plate, surfaced here
as additive `reasoning_metadata`, never replacing the existing fields.
**Not yet independently validated against real American dashcam
footage** -- see docs/limitations.md.

**Model acquisition deliberately does NOT go through
`core/models.py::ModelRegistry`**, unlike YuNet/AuraFace -- a real API
constraint, not an oversight: fast-alpr's own model-hub resolution
(`detector_model`/`ocr_model` constructor arguments) only accepts a
closed set of named presets, which it downloads and caches itself on
first use (via `open_image_models`/`fast_plate_ocr`, both Hugging
Face-backed). This is the exact same one-time, user-initiated,
library-managed download this project already uses for Whisper
transcription (`enrichment/transcription.py`) -- fully offline
afterward, no path this project needs to manage itself.

Requires the `plate_recognition` extra (`pip install
gaggle[plate_recognition]`, i.e. `fast-alpr[onnx]`). CPU by default;
`device: cuda` requires additionally installing `onnxruntime-gpu` in
place of the CPU package, per fast-alpr's own convention (same as
`vision`/`face_recognition`).
"""

from __future__ import annotations

import re

import cv2.typing

from gaggle.core.models import Device, ensure_cuda_dlls_preloaded
from gaggle.enrichment.plate import OcrResult, PlateRegion
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

ImageArray = cv2.typing.MatLike

DETECTOR_VERSION = "fast-alpr-yolo-v9-t-384"
_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
_OCR_MODEL = "cct-xs-v2-global-model"
_NORMALIZE_PATTERN = re.compile(r"[^A-Z0-9]")


def fast_alpr_available() -> bool:
    try:
        import fast_alpr  # noqa: F401
    except ImportError:
        return False
    return True


class FastAlprUnavailableError(RuntimeError):
    """Raised when `fast-alpr` isn't installed, or its models can't be
    obtained (its own hub download failed -- typically no network on
    first use) or loaded."""


class FastAlprDetector:
    """Wraps `fast_alpr.ALPR`, which performs detection and OCR together
    per plate (unlike the cascade path's separate detect-then-OCR
    stages). Construction triggers fast-alpr's own one-time model
    download if nothing is cached yet; expected to be constructed once
    and reused across many frames, mirroring
    `YoloOnnxDetector`/`YuNetDetector`'s shape.
    """

    def __init__(self, device: Device = "cpu", confidence_threshold: float = 0.4) -> None:
        if not fast_alpr_available():
            raise FastAlprUnavailableError(
                "fast-alpr is not installed; install the 'plate_recognition' extra "
                "(pip install gaggle[plate_recognition]) to enable it"
            )
        from fast_alpr import ALPR

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if device == "cuda":
            ensure_cuda_dlls_preloaded()
        try:
            self._alpr = ALPR(
                detector_model=_DETECTOR_MODEL,
                detector_conf_thresh=confidence_threshold,
                detector_providers=providers,
                ocr_model=_OCR_MODEL,
                ocr_device=device,
                ocr_providers=providers,
            )
        except Exception as error:  # fast-alpr/onnxruntime raise plain Exception on a bad model
            raise FastAlprUnavailableError(f"could not load fast-alpr models: {error}") from error

        # Unlike vehicle_yolo.py/face_auraface.py, fast-alpr's detector/OCR
        # wrap their own onnxruntime sessions several layers deep (through
        # open_image_models/fast_plate_ocr) with no stable public
        # attribute to confirm which provider actually initialized -- this
        # only logs what was *requested*, not a confirmed guarantee.
        # `onnxruntime` itself still warns loudly to stderr if
        # CUDAExecutionProvider fails to load, and `gaggle enrich` with
        # `enrichment.face.detector: yunet`/`vision.device: cuda` (both of
        # which do confirm active provider) is a reliable way to check
        # whether CUDA works on this machine at all.
        LOGGER.info("fast_alpr_providers_requested", providers=providers, device=device)

    def detect_and_ocr(self, frame: ImageArray) -> list[tuple[PlateRegion, OcrResult | None]]:
        """Returns one `(PlateRegion, OcrResult | None)` pair per detected
        plate, using this project's own dataclasses (from
        `enrichment/plate.py`) rather than fast-alpr's, so
        `enrichment/service.py::_run_plate_recognition` can dispatch on
        `enrichment.plate.detector` with no further reshaping. `OcrResult`
        is `None` when fast-alpr's OCR stage produced no result for a
        detected plate (e.g. an unreadable crop) -- a detection without a
        reading, not an error."""

        results = []
        for alpr_result in self._alpr.predict(frame):
            box = alpr_result.detection.bounding_box
            region = PlateRegion(
                bbox=(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1),
                confidence=round(float(alpr_result.detection.confidence), 6),
                source="fast_alpr",
            )
            results.append((region, _to_ocr_result(alpr_result.ocr)))
        return results


def _to_ocr_result(raw: object) -> OcrResult | None:
    if raw is None:
        return None
    text = str(raw.text)  # type: ignore[attr-defined]
    confidence = raw.confidence  # type: ignore[attr-defined]
    mean_confidence = (
        sum(confidence) / len(confidence) if isinstance(confidence, list) else float(confidence)
    )
    normalized_text = _NORMALIZE_PATTERN.sub("", text.upper())
    region = raw.region  # type: ignore[attr-defined]
    region_confidence = raw.region_confidence  # type: ignore[attr-defined]
    return OcrResult(
        raw_text=text,
        normalized_text=normalized_text,
        confidence=round(min(1.0, max(0.0, mean_confidence)), 6),
        region=str(region) if region is not None else None,
        region_confidence=round(float(region_confidence), 6)
        if region_confidence is not None
        else None,
    )
