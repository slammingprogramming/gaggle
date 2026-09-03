"""Local face detection and on-device re-identification.

**Detection** uses OpenCV's bundled Haar cascade
(`cv2.data.haarcascades/haarcascade_frontalface_default.xml`) -- a classic,
decades-old, pretrained classifier that ships with every OpenCV install, so
this works fully offline with no model download. It is not as accurate as
a modern deep-learning face detector, especially at odd angles or in low
light; that tradeoff is deliberate (see the module docstring in
`enrichment/__init__.py` and `docs/local-ai.md`) in exchange for zero setup
and zero network dependency. A more accurate detector can be added later as
a `DetectorPlugin`.

**Re-identification/clustering** uses OpenCV's LBPH (Local Binary Pattern
Histogram) face recognizer (`cv2.face`, part of opencv-contrib), trained
*incrementally* per-workspace directly from detected crops -- there is no
pretrained embedding model to download here either. This is explicitly
**not** a deep-learning face embedding; it is a much simpler, classical
texture-histogram comparison. It is good enough to answer "have I seen a
similar-looking face before" for personal review, not to make any strong
identity claim. Because clustering is incremental (each new face is
compared against the current model state, and the model is updated as it
goes), cluster assignment can depend on processing order -- this is an
accepted, documented tradeoff of online/incremental clustering, not a bug.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.typing
import numpy as np

ImageArray = cv2.typing.MatLike

DETECTOR_VERSION = "1.0.0"
RECOGNIZER_VERSION = "1.0.0"
FACE_SIZE = (100, 100)
_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_CONFIDENCE_SCALE = 10.0  # empirical normalization for Haar cascade level_weights -> [0,1]


class FaceModelError(RuntimeError):
    """Raised when the persistent LBPH model cannot be loaded or saved."""


@dataclass(frozen=True, slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]  # x, y, w, h in source image pixel coordinates
    confidence: float


def detect_faces(image: ImageArray, min_size: tuple[int, int] = (30, 30)) -> list[DetectedFace]:
    """Detect faces in a BGR (or grayscale) image using the bundled Haar cascade."""

    cascade = _load_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.equalizeHist(gray)
    boxes, _reject_levels, level_weights = cascade.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size, outputRejectLevels=True
    )
    faces: list[DetectedFace] = []
    for box, weight in zip(boxes, level_weights, strict=True):
        x, y, w, h = (int(v) for v in box)
        confidence = round(min(1.0, max(0.0, float(weight) / _CONFIDENCE_SCALE)), 6)
        faces.append(DetectedFace(bbox=(x, y, w, h), confidence=confidence))
    # Deterministic ordering: largest face first, ties broken by position.
    faces.sort(key=lambda f: (-(f.bbox[2] * f.bbox[3]), f.bbox))
    return faces


def crop_and_normalize(image: ImageArray, bbox: tuple[int, int, int, int]) -> ImageArray:
    """Crop a detected face and normalize it for both storage and LBPH matching."""

    x, y, w, h = bbox
    crop = image[max(0, y) : y + h, max(0, x) : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, FACE_SIZE)


_load_cascade_cache: cv2.CascadeClassifier | None = None


def _load_cascade() -> cv2.CascadeClassifier:
    global _load_cascade_cache
    if _load_cascade_cache is None:
        # cv2.data is a real contrib submodule at runtime; opencv's bundled
        # stubs just don't declare it.
        _load_cascade_cache = cv2.CascadeClassifier(
            cv2.data.haarcascades + _CASCADE_FILENAME  # type: ignore[attr-defined]
        )
    return _load_cascade_cache


class IncrementalFaceClusterer:
    """Persistent, incrementally-trained LBPH clusterer for one workspace.

    Not thread-safe; callers are expected to serialize access (the
    enrichment pipeline processes clips sequentially, consistent with the
    project's determinism-over-parallelism preference for anything that
    mutates shared state).
    """

    def __init__(self, model_path: Path, distance_threshold: float = 70.0) -> None:
        self.model_path = model_path
        self.distance_threshold = distance_threshold
        # cv2.face (opencv-contrib) is real at runtime; not declared in stubs.
        self._recognizer = cv2.face.LBPHFaceRecognizer_create()  # type: ignore[attr-defined]
        self._next_label = 0
        self._label_to_cluster: dict[int, str] = {}
        self._trained = False
        if model_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            self._recognizer.read(str(self.model_path))
            labels_path = self.model_path.with_suffix(".labels.txt")
            if labels_path.exists():
                for line in labels_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    label_str, cluster_id = line.split("\t", 1)
                    label = int(label_str)
                    self._label_to_cluster[label] = cluster_id
                    self._next_label = max(self._next_label, label + 1)
            self._trained = True
        except cv2.error as error:
            raise FaceModelError(
                f"could not load face model at {self.model_path}: {error}"
            ) from error

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self._trained:
            self._recognizer.write(str(self.model_path))
        labels_path = self.model_path.with_suffix(".labels.txt")
        lines = [
            f"{label}\t{cluster_id}" for label, cluster_id in sorted(self._label_to_cluster.items())
        ]
        labels_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def match_or_create_cluster(self, normalized_face: ImageArray) -> tuple[str, float, bool]:
        """Return (cluster_id, distance, is_new_cluster) for a normalized face crop.

        ``distance`` is the raw LBPH distance (0 = identical, larger = less
        similar) to the nearest existing cluster's exemplar. For a newly
        created cluster this is the (above-threshold) rejection distance
        that caused a new cluster to be created, or exactly 0.0 only for
        the very first face ever seen (nothing to compare against yet).
        """

        if self._trained:
            label, distance = self._recognizer.predict(normalized_face)
            if distance <= self.distance_threshold and label in self._label_to_cluster:
                self._recognizer.update([normalized_face], np.array([label]))
                return self._label_to_cluster[label], float(distance), False
        else:
            distance = 0.0

        new_label = self._next_label
        self._next_label += 1
        new_cluster_id = str(_uuid.uuid4())
        self._label_to_cluster[new_label] = new_cluster_id
        if self._trained:
            self._recognizer.update([normalized_face], np.array([new_label]))
        else:
            self._recognizer.train([normalized_face], np.array([new_label]))
            self._trained = True
        return new_cluster_id, float(distance), True

    def predict_nearest_cluster(
        self, normalized_face: ImageArray, exclude_cluster_id: str | None = None
    ) -> tuple[str | None, float]:
        """Read-only lookup: which trained cluster is this face closest to, and
        how far away is it -- without training on it or otherwise mutating
        model state.

        Used for merge-suggestion generation (`core/recognition.py::
        RecognitionService.suggest_face_merges`), where we want to compare
        a cluster's *own* representative crop against every *other*
        trained cluster without accidentally teaching the model that crop
        belongs to whatever it happens to match. Returns ``(None, 0.0)``
        if the model has no trained data yet.

        ``exclude_cluster_id`` matters here specifically because the query
        image passed in is typically one of the exact images that trained
        a label in this same model -- plain ``predict()`` only returns the
        single closest label, which is then always that image's own label
        at distance 0 (confirmed empirically, not assumed: querying with a
        cluster's own trained crop reliably returns that same cluster at
        distance 0.0). Simply discarding a self-match after the fact (the
        original approach) throws away the search entirely, since
        `predict()` never reveals what the real nearest *other* label was.
        `predict_collect` with a `StandardCollector` returns the distance
        to *every* trained label in one pass, which lets this method
        properly exclude the query's own cluster from the search itself
        and return the true nearest remaining one.
        """

        if not self._trained:
            return None, 0.0
        # cv2.face.StandardCollector_create is real at runtime; not
        # declared in opencv's bundled stubs (same gap as
        # cv2.face.LBPHFaceRecognizer_create above).
        collector = cv2.face.StandardCollector_create()  # type: ignore[attr-defined]
        self._recognizer.predict_collect(normalized_face, collector)
        best_label: int | None = None
        best_distance = float("inf")
        for label, distance in collector.getResults():
            cluster_id = self._label_to_cluster.get(label)
            if cluster_id is None or cluster_id == exclude_cluster_id:
                continue
            if distance < best_distance:
                best_distance = distance
                best_label = label
        if best_label is None:
            return None, 0.0
        return self._label_to_cluster.get(best_label), float(best_distance)
