"""Vehicle appearance re-identification -- for a vehicle seen without a
legible plate.

**Scope and intent -- read this before extending this module.** Same
boundary as `enrichment/face.py`/`enrichment/plate.py`/`enrichment/voice.py`
(see `docs/forensic-considerations.md`'s "Recognition data: scope and
intent"): local pattern re-identification within the user's own footage --
"does this look like a vehicle I've seen before" -- never vehicle
*identification*. No plate-text link, no name field, no external lookup,
no networking with other cameras or users.

**Why this module exists.** Plate recognition (`enrichment/plate.py`) is
this project's primary, higher-confidence way to re-identify a vehicle,
but it needs a legible plate. This closes the gap
`docs/limitations.md` explicitly named: "vehicle re-identification by
visual description (color, body shape) without a legible plate." Built
from classical, deterministic, zero-download techniques, consistent with
every other local-AI capability here -- not a learned embedding: a
dominant-color histogram (hue + saturation, computed in HSV -- hue is far
more lighting-invariant than raw BGR/RGB for "is this the same color car"
comparisons) plus a normalized aspect ratio, concatenated into one
fixed-length fingerprint vector.

**Detection source.** Two paths, layered the same way `enrichment/plate.py`
layers cascade -> MSER -> heuristic-contour:

1. If the optional YOLO vehicle detector (`enrichment/vehicle_yolo.py`) is
   loaded (`enrichment.vision.enabled` + a model file present), its boxes
   for vehicle classes are categorically more precise and are preferred --
   see `enrichment/service.py::_run_vehicle_appearance_recognition`, which
   reuses the *same* lazily-loaded `YoloOnnxDetector` instance
   `_run_vehicle_detection` already checks for, rather than loading a
   second one.
2. Otherwise (the zero-setup default every user gets), `detect_vehicle_regions`
   below uses the same classical technique as
   `enrichment/plate.py::_detect_heuristic` (Canny edges -> morphological
   closing -> contour extraction -> `cv2.minAreaRect` for a
   rotation-aware aspect ratio), tuned for vehicle-scale regions (a much
   larger minimum area, a wider aspect-ratio band) rather than
   plate-scale ones. This is explicitly a coarse "vehicle-shaped moving
   blob" heuristic, not real vehicle classification -- labeled as such,
   matching `detection/object_detection.py`'s honesty precedent for its
   own `unclassified_moving_region` label.

**Accuracy caveat, stated plainly.** This is a meaningfully weaker
fingerprint than face or plate re-identification. It cannot distinguish
two different vehicles of the same color and body shape, and is far more
sensitive to lighting, angle, and dirt/wear than a plate reading. Unlike
face/plate (validated against real or realistic synthetic imagery), this
was validated only against synthetic test scenes during development, not
real footage -- see `docs/limitations.md`. Treat every match as a
considerably weaker signal than a face or plate match.

**Threshold validation methodology** (the same discipline
`enrichment/voice.py`'s docstring documents -- an empirically measured
value, not a single-test guess): the default clustering threshold
(0.10) was chosen by generating synthetic vehicle-colored crops across
five distinct hues (red/blue/green/yellow/orange) with eight
independent-noise-seed variations each, measuring the cosine distance
between same-vehicle crops (same hue, different noise realization --
standing in for different lighting/frames of the same real vehicle)
versus different-vehicle crops (different hue): same-vehicle distances
measured ~0.00000-0.00009, different-vehicle distances measured
~0.353-0.690 -- a wide, consistent separation, comfortably wide enough
that 0.10 sits close to the same-vehicle end, biased toward *not*
merging when uncertain, the same conservative philosophy
`enrichment/voice.py`'s corrected 0.05 threshold uses. See
`tests/unit/test_vehicle_appearance.py`'s
`test_measured_distance_gap_between_same_and_different_vehicles` for the
exact reproduction of this measurement.
"""

from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cv2
import cv2.typing
import numpy as np
import numpy.typing as npt

ImageArray = cv2.typing.MatLike
FloatArray = npt.NDArray[np.float64]

DETECTOR_VERSION = "1.0.0"
HUE_BINS = 16
SATURATION_BINS = 8
# hue + saturation histograms, plus one scalar for normalized aspect ratio.
VEHICLE_FINGERPRINT_DIMENSIONS = HUE_BINS + SATURATION_BINS + 1
# A vehicle's true aspect ratio rarely exceeds this even at a shallow side
# angle; used only to normalize the aspect-ratio scalar into a sane [0, 1]
# range before concatenating it with the (already 0..1-normalized)
# histogram bins, so it neither dominates nor is dominated by them.
MAX_EXPECTED_ASPECT_RATIO = 4.0

# Vehicle-scale region heuristic tuning -- deliberately much coarser than
# `enrichment/plate.py`'s plate-scale equivalents. A vehicle can appear
# anywhere from a near-square rear/front view to a long, low side profile.
_MIN_ASPECT_RATIO = 0.8
_MAX_ASPECT_RATIO = 4.2
_MIN_REGION_AREA_RATIO = 0.02


class VehicleAppearanceError(RuntimeError):
    """Raised when a region/fingerprint can't be computed (e.g. an empty crop)."""


@dataclass(frozen=True, slots=True)
class VehicleRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    source: str  # "heuristic_contour" | "yolo"


@dataclass(frozen=True, slots=True)
class VehicleAppearanceFingerprint:
    region: VehicleRegion
    fingerprint: FloatArray  # shape (VEHICLE_FINGERPRINT_DIMENSIONS,)
    confidence: float  # carried through from the source region's detection confidence


def detect_vehicle_regions(
    image: ImageArray, min_size: tuple[int, int] = (60, 40)
) -> list[VehicleRegion]:
    """Coarse, classical "vehicle-shaped moving blob" heuristic.

    Canny edges, morphological closing to bridge body-panel/window edges
    into a solid blob, then contour filtering by vehicle-scale area and
    (rotation-aware) aspect ratio. Not real vehicle classification -- see
    the module docstring for why, and for the YOLO-based alternative used
    when available.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Lower thresholds than `enrichment/plate.py::_detect_heuristic` uses
    # (80, 200) -- a plate's white-on-body contrast is much stronger than
    # a vehicle body's contrast against road/background, so a plate-tuned
    # threshold misses real vehicle-body edges entirely. Consistent with
    # this project's over-inclusion-favoring philosophy (see
    # `docs/forensic-considerations.md`): more candidate regions reaching
    # the fingerprint/clustering stage is the intended tradeoff, not the
    # detector trying to be maximally precise on its own.
    edges = cv2.Canny(blurred, 20, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = float(image.shape[0] * image.shape[1])
    results: list[VehicleRegion] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_size[0] or h < min_size[1]:
            continue
        area = w * h
        if not frame_area or area / frame_area < _MIN_REGION_AREA_RATIO:
            continue
        # The *rotated* minimum-area rectangle's aspect ratio, not the
        # axis-aligned bounding box's -- mirrors
        # `enrichment/plate.py::_detect_heuristic`'s rotation-aware
        # technique so an angled vehicle isn't penalized just for not
        # being perfectly parallel to the frame.
        (_cx, _cy), (rw, rh), _angle = cv2.minAreaRect(contour)
        short_side, long_side = sorted((rw, rh))
        aspect_ratio = long_side / short_side if short_side else 0.0
        if not (_MIN_ASPECT_RATIO <= aspect_ratio <= _MAX_ASPECT_RATIO):
            continue
        fill_ratio = cv2.contourArea(contour) / area if area else 0.0
        confidence = round(min(1.0, fill_ratio * 1.2), 6)
        results.append(
            VehicleRegion(bbox=(x, y, w, h), confidence=confidence, source="heuristic_contour")
        )
    results.sort(key=lambda r: (-r.confidence, r.bbox))
    return results


def compute_fingerprint(image: ImageArray, region: VehicleRegion) -> VehicleAppearanceFingerprint:
    """Compute a fixed-length appearance fingerprint for one detected region."""

    x, y, w, h = region.bbox
    crop = image[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0 or crop.ndim != 3:
        raise VehicleAppearanceError("expected a non-empty color (BGR) region crop")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([hsv], [0], None, [HUE_BINS], [0, 180]).flatten().astype(np.float64)
    sat_hist = (
        cv2.calcHist([hsv], [1], None, [SATURATION_BINS], [0, 256]).flatten().astype(np.float64)
    )
    hue_total = hue_hist.sum()
    if hue_total > 0:
        hue_hist = hue_hist / hue_total
    sat_total = sat_hist.sum()
    if sat_total > 0:
        sat_hist = sat_hist / sat_total

    aspect_ratio = w / h if h else 0.0
    normalized_aspect = min(1.0, aspect_ratio / MAX_EXPECTED_ASPECT_RATIO)

    fingerprint = np.concatenate([hue_hist, sat_hist, np.array([normalized_aspect])])
    return VehicleAppearanceFingerprint(
        region=region, fingerprint=fingerprint, confidence=region.confidence
    )


class _ClusterState(TypedDict):
    centroid: FloatArray
    count: int


class IncrementalVehicleAppearanceClusterer:
    """Persistent, incrementally-updated centroid clusterer for vehicle
    appearance fingerprints.

    A near-verbatim structural copy of
    `enrichment/voice.py::IncrementalVoiceClusterer` (running-mean
    centroid per cluster, JSON-persisted, cosine-distance match-or-create)
    -- see that module for why this design was chosen over, e.g., a
    trained classifier: fingerprints are already fixed-length real
    vectors a plain distance metric handles directly, and reusing the
    proven design here minimizes net-new logic to review.
    """

    def __init__(self, model_path: Path, distance_threshold: float = 0.10) -> None:
        self.model_path = model_path
        self.distance_threshold = distance_threshold
        self._clusters: dict[str, _ClusterState] = {}
        if model_path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        self._clusters = {
            cluster_id: _ClusterState(
                centroid=np.array(data["centroid"], dtype=np.float64), count=data["count"]
            )
            for cluster_id, data in payload.items()
        }

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            cluster_id: {"centroid": data["centroid"].tolist(), "count": data["count"]}
            for cluster_id, data in self._clusters.items()
        }
        self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def match_or_create_cluster(self, fingerprint: FloatArray) -> tuple[str, float, bool]:
        """Return (cluster_id, distance, is_new_cluster) for a fingerprint vector.

        ``distance`` is cosine distance (0 = identical direction, 2 =
        opposite) to the nearest existing cluster centroid, or the
        (above-threshold) rejection distance that caused a new cluster to
        be created.
        """

        best_id: str | None = None
        best_distance = float("inf")
        for cluster_id, data in self._clusters.items():
            distance = _cosine_distance(fingerprint, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id

        if best_id is not None and best_distance <= self.distance_threshold:
            self._update_centroid(best_id, fingerprint)
            return best_id, float(best_distance), False

        new_cluster_id = str(_uuid.uuid4())
        self._clusters[new_cluster_id] = _ClusterState(centroid=fingerprint.copy(), count=1)
        return new_cluster_id, float(best_distance if best_id else 0.0), True

    def predict_nearest_cluster(
        self, fingerprint: FloatArray, exclude_cluster_id: str | None = None
    ) -> tuple[str | None, float]:
        """Read-only lookup, no mutation -- mirrors
        `IncrementalVoiceClusterer.predict_nearest_cluster`, used for
        merge-suggestion generation.

        ``exclude_cluster_id`` matters when the query vector *is* a
        cluster's own centroid (exactly what
        `RecognitionService.suggest_vehicle_appearance_merges` passes):
        without excluding it from the search, a cluster's own centroid
        always matches itself at distance 0 and wins, so the real nearest
        *other* cluster is never found -- excluding it from the search
        itself, not just discarding a self-match after the fact, is what
        actually fixes it.
        """

        if not self._clusters:
            return None, 0.0
        best_id, best_distance = None, float("inf")
        for cluster_id, data in self._clusters.items():
            if cluster_id == exclude_cluster_id:
                continue
            distance = _cosine_distance(fingerprint, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id
        return best_id, float(best_distance)

    def get_cluster_centroid(self, cluster_id: str) -> FloatArray | None:
        """The current running-mean fingerprint for a cluster, or ``None``
        if no such cluster exists in this model. Used for merge-suggestion
        generation (`core/recognition.py::RecognitionService.suggest_vehicle_appearance_merges`),
        which needs each cluster's own centroid to compare against every
        *other* cluster."""

        data = self._clusters.get(cluster_id)
        if data is None:
            return None
        return data["centroid"]

    def _update_centroid(self, cluster_id: str, fingerprint: FloatArray) -> None:
        data = self._clusters[cluster_id]
        count = data["count"]
        centroid = data["centroid"]
        new_centroid = (centroid * count + fingerprint) / (count + 1)
        self._clusters[cluster_id] = _ClusterState(centroid=new_centroid, count=count + 1)


def _cosine_distance(a: FloatArray, b: FloatArray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - similarity
