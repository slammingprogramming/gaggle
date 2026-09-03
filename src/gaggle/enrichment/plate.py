"""License plate detection and OCR.

**Detection** combines three approaches, all fully offline:

1. OpenCV's bundled Haar cascades (`haarcascade_russian_plate_number.xml`,
   `haarcascade_license_plate_rus_16stages.xml`) -- real, pretrained,
   zero-download, but calibrated for Russian-format plate proportions.
   Accuracy on other countries' plate formats will be lower; this is an
   honest limitation, not a hidden one (see `docs/local-ai.md`).
2. MSER (Maximally Stable Extremal Regions, `cv2.MSER_create`) -- a
   classic, deterministic blob detector well-suited to text-dense regions
   like a plate's characters. In practice this catches real plates the
   contour heuristic below misses entirely, especially against cluttered
   backgrounds (verified during development: a synthetic scene with
   several plate-aspect-ratio decoy rectangles produced zero usable
   candidates from the contour heuristic alone -- 0.0 IoU with the true
   plate -- versus a 0.97 IoU match from MSER on the same scene). This is
   the primary detector; the others are supplementary.
3. A format-agnostic contour heuristic: Canny edge detection, morphological
   closing to bridge character gaps into a solid blob, then filtering by
   plate-like aspect ratio. Aspect ratio is measured on the *rotated*
   minimum-area rectangle of each contour, not just its axis-aligned
   bounding box, so an angled plate (common in real dashcam footage --
   this was a specific reported accuracy complaint) isn't penalized just
   for not being perfectly horizontal.

All three run and results are merged with IoU-based deduplication so hits
from different sources pointing at the same region don't double-count.
More candidates reaching OCR is an intentional tradeoff -- OCR's own
confidence score and the text-length pre-filter (see
`core/config.py::PlateRecognitionConfig`) are what actually reject junk,
not the detector trying to be maximally precise on its own. See
`recognize plates-debug` (`enrichment/plate.py::render_debug_frame`) for a
way to visually inspect exactly what a given frame's candidates look like
and why each one was kept or dropped -- built specifically to answer "how
do I check whether this is actually working."

**OCR** shells out to the `tesseract` binary (matching this project's
existing pattern of using ffmpeg/ffprobe as external tools rather than
adding heavier pip dependencies), configured for single-line text with a
plate-appropriate character whitelist. Confidence comes directly from
tesseract's own per-word TSV output, never a guessed value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.typing

ImageArray = cv2.typing.MatLike

DETECTOR_VERSION = "2.0.0"
OCR_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_MIN_ASPECT_RATIO = 1.8
_MAX_ASPECT_RATIO = 6.5
_MIN_REGION_AREA = 300
_MSER_MAX_AREA = 40000
_IOU_DEDUPE_THRESHOLD = 0.4


class PlateOcrError(RuntimeError):
    """Raised when tesseract is unavailable or fails to run."""


@dataclass(frozen=True, slots=True)
class PlateRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    source: str  # "cascade_ru_plate" | "cascade_ru_16stages" | "mser" | "heuristic_contour"


@dataclass(frozen=True, slots=True)
class OcrResult:
    raw_text: str
    normalized_text: str
    confidence: float  # 0-1, from tesseract's own per-word confidence
    # Only ever set by enrichment/plate_fast_alpr.py -- an optional
    # region/country guess for the plate, additive context the cascade+
    # Tesseract path has no equivalent of and always leaves as None.
    region: str | None = None
    region_confidence: float | None = None


_cascade_cache: dict[str, cv2.CascadeClassifier] = {}


def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    if filename not in _cascade_cache:
        # cv2.data is a real contrib submodule at runtime; opencv's bundled
        # stubs just don't declare it.
        _cascade_cache[filename] = cv2.CascadeClassifier(
            cv2.data.haarcascades + filename  # type: ignore[attr-defined]
        )
    return _cascade_cache[filename]


def detect_plate_regions(
    image: ImageArray, min_size: tuple[int, int] = (40, 15)
) -> list[PlateRegion]:
    regions: list[PlateRegion] = []
    regions.extend(
        _detect_cascade(image, "haarcascade_russian_plate_number.xml", "cascade_ru_plate", min_size)
    )
    regions.extend(
        _detect_cascade(
            image, "haarcascade_license_plate_rus_16stages.xml", "cascade_ru_16stages", min_size
        )
    )
    regions.extend(_detect_mser(image, min_size))
    regions.extend(_detect_heuristic(image, min_size))
    return _deduplicate(regions)


def _detect_cascade(
    image: ImageArray, cascade_file: str, source: str, min_size: tuple[int, int]
) -> list[PlateRegion]:
    cascade = _load_cascade(cascade_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    boxes, _levels, weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=min_size, outputRejectLevels=True
    )
    results = []
    for box, weight in zip(boxes, weights, strict=True):
        x, y, w, h = (int(v) for v in box)
        confidence = round(min(1.0, max(0.0, float(weight) / 10.0)), 6)
        results.append(PlateRegion(bbox=(x, y, w, h), confidence=confidence, source=source))
    return results


def _detect_mser(image: ImageArray, min_size: tuple[int, int]) -> list[PlateRegion]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # cv2.MSER_create is real at runtime; not declared in opencv's bundled stubs.
    mser = cv2.MSER_create()  # type: ignore[attr-defined]
    mser.setMinArea(_MIN_REGION_AREA)
    mser.setMaxArea(_MSER_MAX_AREA)
    try:
        _regions, bboxes = mser.detectRegions(gray)
    except cv2.error:
        return []

    results = []
    for x, y, w, h in bboxes:
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w < min_size[0] or h < min_size[1]:
            continue
        aspect_ratio = w / h if h else 0.0
        if not (_MIN_ASPECT_RATIO <= aspect_ratio <= _MAX_ASPECT_RATIO):
            continue
        # MSER doesn't score its own regions; a region's stability under
        # this aspect/size filter is the only signal available at this
        # stage, so every surviving candidate gets the same moderate
        # starting confidence -- OCR confidence is what actually
        # distinguishes a real plate from a lucky rectangle later.
        results.append(PlateRegion(bbox=(x, y, w, h), confidence=0.5, source="mser"))
    return results


def _detect_heuristic(image: ImageArray, min_size: tuple[int, int]) -> list[PlateRegion]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_size[0] or h < min_size[1]:
            continue
        area = w * h
        if area < _MIN_REGION_AREA:
            continue
        # The *rotated* minimum-area rectangle's aspect ratio, not the
        # axis-aligned bounding box's -- an angled plate's true width:height
        # ratio is preserved this way even when it isn't axis-aligned in
        # the frame, which the axis-aligned box alone would distort.
        (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
        short_side, long_side = sorted((rw, rh))
        aspect_ratio = long_side / short_side if short_side else 0.0
        if not (_MIN_ASPECT_RATIO <= aspect_ratio <= _MAX_ASPECT_RATIO):
            continue
        fill_ratio = cv2.contourArea(contour) / area if area else 0.0
        confidence = round(min(1.0, fill_ratio * 1.2), 6)
        results.append(
            PlateRegion(bbox=(x, y, w, h), confidence=confidence, source="heuristic_contour")
        )
    return results


def render_debug_frame(image: ImageArray, regions: list[PlateRegion]) -> ImageArray:
    """Draw every candidate region (color-coded by source) with its
    confidence, for visually auditing what the detector actually found on
    a given frame and why. See `recognize plates-debug`."""

    colors = {
        "cascade_ru_plate": (255, 128, 0),
        "cascade_ru_16stages": (255, 0, 128),
        "mser": (0, 200, 0),
        "heuristic_contour": (0, 128, 255),
    }
    annotated = image.copy()
    for region in regions:
        x, y, w, h = region.bbox
        color = colors.get(region.source, (200, 200, 200))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        label = f"{region.source} {region.confidence:.2f}"
        label_y = max(0, y - 6)
        cv2.putText(
            annotated, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
        )
    return annotated


def _deduplicate(regions: list[PlateRegion]) -> list[PlateRegion]:
    ordered = sorted(regions, key=lambda r: (-r.confidence, r.bbox))
    kept: list[PlateRegion] = []
    for region in ordered:
        if not any(_iou(region.bbox, other.bbox) > _IOU_DEDUPE_THRESHOLD for other in kept):
            kept.append(region)
    kept.sort(key=lambda r: r.bbox)
    return kept


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area else 0.0


def ocr_plate_text(crop: ImageArray, timeout_seconds: float = 10.0) -> OcrResult:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    scale = max(1, 200 // max(1, gray.shape[1]))
    if scale > 1:
        gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale))
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    with tempfile.TemporaryDirectory(prefix="gaggle-plate-") as tmp_dir:
        image_path = Path(tmp_dir) / "plate.png"
        cv2.imwrite(str(image_path), binarized)
        command = [
            "tesseract",
            str(image_path),
            "stdout",
            "--psm",
            "7",
            "-c",
            f"tessedit_char_whitelist={OCR_CHAR_WHITELIST}",
            "tsv",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout_seconds, check=True
            )
        except subprocess.TimeoutExpired as exc:
            raise PlateOcrError("tesseract timed out") from exc
        except subprocess.CalledProcessError as exc:
            raise PlateOcrError(f"tesseract failed: {exc.stderr.strip()}") from exc
        except OSError as exc:
            raise PlateOcrError(f"tesseract could not be executed: {exc}") from exc

    return _parse_tsv(completed.stdout)


def _parse_tsv(tsv_output: str) -> OcrResult:
    words: list[str] = []
    confidences: list[float] = []
    lines = tsv_output.strip().splitlines()
    if len(lines) < 2:
        return OcrResult(raw_text="", normalized_text="", confidence=0.0)
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 12:
            continue
        level, conf_str, text = fields[0], fields[10], fields[11]
        if level != "5" or not text.strip():
            continue
        try:
            confidence = float(conf_str)
        except ValueError:
            continue
        if confidence < 0:
            continue
        words.append(text)
        confidences.append(confidence)

    raw_text = " ".join(words)
    normalized_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return OcrResult(
        raw_text=raw_text, normalized_text=normalized_text, confidence=round(avg_confidence, 6)
    )


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None
