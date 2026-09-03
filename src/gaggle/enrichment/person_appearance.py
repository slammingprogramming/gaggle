"""Pedestrian / full-body appearance re-identification -- structured,
classical attributes (dominant clothing color, rough build), never a
learned embedding or an AI-generated description.

**Scope and intent -- read this before extending this module.** Same
boundary as `enrichment/face.py`/`enrichment/vehicle_appearance.py` (see
`docs/forensic-considerations.md`'s "Recognition data: scope and
intent"): local pattern re-identification within the user's own footage
-- "does this look like a person I've seen before" -- never identity
*resolution*. No name field, no external lookup, no networking with
other cameras or users.

**Why structured attributes, not free-text descriptions.** A real
vision-captioning model (local or cloud) could write a sentence like
"person in a dark jacket" -- explicitly out of scope for this pass: it's
a materially bigger dependency (a new local vision-LLM, or routing
through the existing optional cloud path) and reads closer to an
identifying characteristic than this project is comfortable defaulting
to. What's here instead is the same honest, classical technique
`enrichment/vehicle_appearance.py` already uses for the identical
problem (re-identify without a stronger signal available): a
lighting-invariant HSV hue+saturation histogram over the detected
region (dominant clothing color) plus a normalized aspect ratio,
concatenated into one fixed-length fingerprint vector -- structurally a
near-verbatim copy of that module, since the underlying technique and
its real tradeoffs are identical for "what color/shape is this moving
thing," not something worth re-deriving independently.

**Detection source: YOLO only, no classical fallback.** Unlike
`vehicle_appearance.py` (which has a zero-setup classical detector
fallback for vehicle-shaped blobs), a reliable classical "person-shaped
blob" heuristic is a meaningfully harder, more false-positive-prone
problem than vehicle-shaped or plate-shaped blobs -- not attempted here.
This capability requires the `vision` extra + a real YOLO model file
(`enrichment.vision.enabled` + `model_path`), exactly like
`enrichment/service.py::_run_vehicle_detection`. COCO class 0 ("person")
is already in `enrichment/vehicle_yolo.py::RELEVANT_CLASS_IDS` -- these
boxes already exist today, just weren't surfaced as their own
signal/observation before this module.

**"Facing away from the camera" already works with zero new logic** --
YOLO is a bounding-box detector, not face-dependent; any orientation
already produces a detection. The gap this module closes is purely that
person detections weren't a first-class, re-identifiable signal yet.

**Accuracy caveat, stated plainly**, mirroring
`vehicle_appearance.py`'s: this cannot distinguish two different people
wearing similarly-colored clothing, and is far more sensitive to
lighting, angle, and clothing changes between sightings than face or
plate re-identification. Validated only against synthetic test data
during development, not real footage -- treat every match as a
considerably weaker signal than a face match, and every structured
attribute as a rough description, not a determination.
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
# hue + saturation histograms, plus one scalar for normalized aspect ratio --
# identical shape/rationale to vehicle_appearance.py's fingerprint.
PERSON_FINGERPRINT_DIMENSIONS = HUE_BINS + SATURATION_BINS + 1
# A standing person's bounding box is almost always taller than wide;
# capped generously to still normalize sensibly for a crouching/prone/
# unusual-pose detection.
MAX_EXPECTED_ASPECT_RATIO = 3.0


class PersonAppearanceError(RuntimeError):
    """Raised when a fingerprint can't be computed (e.g. an empty crop)."""


@dataclass(frozen=True, slots=True)
class PersonRegion:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    source: str  # "yolo" -- the only source this module supports today


@dataclass(frozen=True, slots=True)
class PersonAppearanceFingerprint:
    region: PersonRegion
    fingerprint: FloatArray  # shape (PERSON_FINGERPRINT_DIMENSIONS,)
    confidence: float  # carried through from the source region's detection confidence
    dominant_hue_bin: int  # structured attribute: which hue histogram bin dominates
    height_to_frame_ratio: float  # structured attribute: rough build/distance proxy


def compute_fingerprint(image: ImageArray, region: PersonRegion) -> PersonAppearanceFingerprint:
    """Compute a fixed-length appearance fingerprint plus the structured
    attributes (`reasoning_metadata` surfaces these, never a free-text
    description) for one detected person region."""

    x, y, w, h = region.bbox
    crop = image[max(0, y) : y + h, max(0, x) : x + w]
    if crop.size == 0 or crop.ndim != 3:
        raise PersonAppearanceError("expected a non-empty color (BGR) region crop")

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
    frame_height = image.shape[0] or 1
    return PersonAppearanceFingerprint(
        region=region,
        fingerprint=fingerprint,
        confidence=region.confidence,
        dominant_hue_bin=int(np.argmax(hue_hist)),
        height_to_frame_ratio=round(h / frame_height, 6),
    )


class _ClusterState(TypedDict):
    centroid: FloatArray
    count: int


class IncrementalPersonAppearanceClusterer:
    """Persistent, incrementally-updated centroid clusterer for person
    appearance fingerprints -- a near-verbatim structural copy of
    `enrichment/vehicle_appearance.py::IncrementalVehicleAppearanceClusterer`
    (itself a copy of `enrichment/voice.py`'s clusterer); see that module
    for why this design was chosen over, e.g., a trained classifier.
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
        `IncrementalVehicleAppearanceClusterer.predict_nearest_cluster`,
        used for merge-suggestion generation."""

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
