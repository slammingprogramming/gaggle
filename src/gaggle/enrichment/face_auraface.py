"""Real deep-learning face embedding via AuraFace-v1 (`fal/AuraFace-v1`,
Apache-2.0 -- explicitly commercial-use-safe), the new
`enrichment.face.embedding_model: auraface` option, replacing
`enrichment/face.py`'s classical LBPH texture-histogram clusterer, which
remains available as `embedding_model: lbph` and is still what a
zero-extra-dependency install falls back to if `insightface` isn't
installed or the model can't be fetched.

**Why AuraFace and not InsightFace's own `buffalo_l`/ArcFace models**:
those pretrained recognition weights require a separate commercial
license from InsightFace despite the loader code itself being MIT --
a real problem to recommend by default in an AGPL tool other people
deploy. AuraFace is the same architecture (in fact the same file layout
as `buffalo_l` -- `glintr100.onnx` for recognition, plus detection/
landmark/gender-age models this project doesn't use), republished by fal
under Apache-2.0, no strings attached. Only `glintr100.onnx` -- the
recognition/embedding model -- is fetched; detection is already handled
by `enrichment/face_yunet.py`/`enrichment/face.py`, so the rest of that
model pack is never downloaded.

**Why `insightface.model_zoo.get_model()` directly, not the
`FaceAnalysis` app class**: `FaceAnalysis` loads and runs an entire model
pack (detection + landmarks + gender/age + recognition) as one pipeline.
This project only ever wants the recognition model run on a face this
project's own detector already found -- `model_zoo.get_model(onnx_path)`
loads exactly that one ONNX file (real, verified: returns an
`ArcFaceONNX` wrapper whose `get_feat()` takes an already-cropped,
112x112 BGR image and returns a 512-d embedding directly, no
`FaceAnalysis`, no additional detection pass, no landmark-based
alignment). Landmark-based alignment (`ArcFaceONNX.get()`, which expects
5-point keypoints) would likely improve accuracy further -- `YuNet`
produces those keypoints but they're currently discarded (see
`enrichment/face_yunet.py`'s docstring); a documented follow-up, not
implemented in this pass.

**Clustering** mirrors `enrichment/voice.py::IncrementalVoiceClusterer`'s
exact running-centroid, cosine-distance design (already proven twice:
voice, then vehicle-appearance) rather than
`IncrementalFaceClusterer`'s LBPH-specific train/predict API, since a
plain distance metric over stored fixed-length vectors is all an
embedding needs.
"""

from __future__ import annotations

import json
import uuid as _uuid
from pathlib import Path
from typing import TypedDict

import cv2.typing
import numpy as np
import numpy.typing as npt

from gaggle.core.models import (
    Device,
    ModelRegistry,
    ModelUnavailableError,
    ensure_cuda_dlls_preloaded,
)
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

ImageArray = cv2.typing.MatLike
FloatArray = npt.NDArray[np.float64]

RECOGNIZER_VERSION = "auraface-v1"

# A starting point calibrated from published ArcFace-family benchmarks,
# not validated against this project's own real footage -- see
# docs/limitations.md. Cosine distance over normalized embeddings from
# this model family typically separates same/different identities
# somewhere around 0.3-0.4 at a reasonable false-accept rate; retune
# before trusting this in a high-stakes review.
DEFAULT_DISTANCE_THRESHOLD = 0.35


def insightface_available() -> bool:
    try:
        import insightface  # noqa: F401
    except ImportError:
        return False
    except Exception as error:
        # insightface unconditionally imports `app` -> `mask_renderer` ->
        # `albumentations`, which itself tries to import torch whenever
        # torch is installed at all (not gated behind whether *this*
        # project needs it -- we only ever use `insightface.model_zoo`
        # directly, never `FaceAnalysis`). A broken/conflicting local
        # torch CUDA install (e.g. a stale cudnn DLL earlier on PATH than
        # torch's own bundled one) raises a plain OSError here, not an
        # ImportError -- a real failure hit during development on a
        # machine with an unrelated torch install already present.
        # Treated the same as "not installed": fall back to
        # embedding_model: lbph rather than crashing the whole run.
        LOGGER.warning(
            "insightface_import_failed",
            reason=str(error),
            hint="falling back to embedding_model: lbph for this run",
        )
        return False
    return True


class AuraFaceUnavailableError(RuntimeError):
    """Raised when `insightface` isn't installed, or the AuraFace
    recognition model can't be obtained (`ModelUnavailableError`) or
    loaded (a corrupt/unreadable file, or an execution-provider gap --
    see `core/models.py::derive_int8`'s docstring)."""


class AuraFaceEmbedder:
    """Loads the AuraFace recognition model once (via `ModelRegistry`,
    cached after the first call) and computes fixed-length embeddings for
    already-detected, already-cropped faces. Expected to be constructed
    once and reused across many frames, mirroring
    `YoloOnnxDetector`/`YuNetDetector`'s shape.
    """

    def __init__(self, device: Device = "cpu") -> None:
        if not insightface_available():
            raise AuraFaceUnavailableError(
                "insightface is not installed; install the 'face_recognition' "
                "extra (pip install gaggle[face_recognition]) to enable "
                "AuraFace embeddings"
            )
        try:
            model_path = ModelRegistry().ensure_model("auraface-embedding", device=device)
        except ModelUnavailableError as error:
            raise AuraFaceUnavailableError(str(error)) from error

        from insightface import model_zoo

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if device == "cuda":
            ensure_cuda_dlls_preloaded()
        try:
            model = model_zoo.get_model(str(model_path), providers=providers)
        except Exception as error:  # insightface/onnxruntime raise plain Exception on a bad model
            raise AuraFaceUnavailableError(
                f"could not load AuraFace model at {model_path}: {error}"
            ) from error
        if model is None or model.taskname != "recognition":
            raise AuraFaceUnavailableError(
                f"model at {model_path} was not recognized as a face recognition model"
            )
        self._model = model

        # onnxruntime silently falls back to CPU if CUDA can't actually be
        # initialized (missing driver/CUDA-toolkit/cuDNN DLLs) -- it warns
        # on its own but doesn't fail construction, so `device: cuda` can
        # silently not be honored. Log what's actually active rather than
        # leaving this to be inferred from a cryptic onnxruntime warning.
        active_provider = model.session.get_providers()[0]
        if device == "cuda" and active_provider != "CUDAExecutionProvider":
            LOGGER.warning(
                "auraface_cuda_requested_but_not_active",
                requested="cuda",
                active_provider=active_provider,
                message=(
                    "device: cuda was requested but onnxruntime's CUDAExecutionProvider "
                    f"did not initialize; running on {active_provider} instead. See "
                    "docs/local-ai.md's GPU setup notes."
                ),
            )
        else:
            LOGGER.info("auraface_provider_active", provider=active_provider)

    def get_embedding(self, crop: ImageArray) -> FloatArray | None:
        """Returns a unit-normalized embedding vector for `crop` (a BGR
        image already cropped to roughly one face -- typically the same
        bounding box the configured detector just produced), or `None` if
        the crop is empty/degenerate. Resizes to the model's expected
        input size itself; does not re-detect or re-align, so a loosely
        cropped face will embed less accurately than a tight one -- see
        the module docstring on landmark alignment as a future
        improvement."""

        if crop.size == 0:
            return None
        resized = cv2.resize(crop, self._model.input_size)
        raw: FloatArray = np.asarray(self._model.get_feat(resized), dtype=np.float64).flatten()
        norm = float(np.linalg.norm(raw))
        if norm == 0.0:
            return None
        normalized: FloatArray = (raw / norm).astype(np.float64)
        return normalized


class _ClusterState(TypedDict):
    centroid: FloatArray
    count: int


class IncrementalFaceEmbeddingClusterer:
    """Persistent, incrementally-updated centroid clusterer for face
    embeddings -- structurally identical to
    `enrichment.voice.IncrementalVoiceClusterer`, duplicated rather than
    shared because the two are conceptually independent (different vector
    spaces, different persisted model files, different callers) even
    though the algorithm is the same; see that module's docstring for the
    full rationale on why a plain centroid clusterer is the right choice
    for an already-fixed-length vector.
    """

    def __init__(
        self, model_path: Path, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    ) -> None:
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

    def match_or_create_cluster(self, embedding: FloatArray) -> tuple[str, float, bool]:
        """Return (cluster_id, distance, is_new_cluster) for an embedding
        vector. ``distance`` is cosine distance (0 = identical direction,
        2 = opposite) to the nearest existing cluster centroid, or the
        (above-threshold) rejection distance that caused a new cluster to
        be created."""

        best_id: str | None = None
        best_distance = float("inf")
        for cluster_id, data in self._clusters.items():
            distance = _cosine_distance(embedding, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id

        if best_id is not None and best_distance <= self.distance_threshold:
            self._update_centroid(best_id, embedding)
            return best_id, float(best_distance), False

        new_cluster_id = str(_uuid.uuid4())
        self._clusters[new_cluster_id] = _ClusterState(centroid=embedding.copy(), count=1)
        return new_cluster_id, float(best_distance if best_id else 0.0), True

    def predict_nearest_cluster(
        self, embedding: FloatArray, exclude_cluster_id: str | None = None
    ) -> tuple[str | None, float]:
        """Read-only lookup, no mutation -- used for merge-suggestion
        generation. ``exclude_cluster_id`` excludes a cluster's own
        centroid from the search itself (not just discarded after the
        fact), the same fix applied to
        `IncrementalVoiceClusterer.predict_nearest_cluster` and
        `IncrementalFaceClusterer.predict_nearest_cluster` after both hit
        the same real self-match bug."""

        if not self._clusters:
            return None, 0.0
        best_id, best_distance = None, float("inf")
        for cluster_id, data in self._clusters.items():
            if cluster_id == exclude_cluster_id:
                continue
            distance = _cosine_distance(embedding, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id
        return best_id, float(best_distance)

    def get_cluster_centroid(self, cluster_id: str) -> FloatArray | None:
        data = self._clusters.get(cluster_id)
        if data is None:
            return None
        return data["centroid"]

    def _update_centroid(self, cluster_id: str, embedding: FloatArray) -> None:
        data = self._clusters[cluster_id]
        count = data["count"]
        centroid = data["centroid"]
        new_centroid = (centroid * count + embedding) / (count + 1)
        self._clusters[cluster_id] = _ClusterState(centroid=new_centroid, count=count + 1)


def _cosine_distance(a: FloatArray, b: FloatArray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)
