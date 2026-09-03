from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gaggle.core.models import ModelUnavailableError
from gaggle.enrichment import face_auraface as face_auraface_module
from gaggle.enrichment.face_auraface import (
    AuraFaceEmbedder,
    AuraFaceUnavailableError,
    IncrementalFaceEmbeddingClusterer,
    insightface_available,
)

# -- IncrementalFaceEmbeddingClusterer: pure numpy, no network, no
# optional dependency -- exercised fully, mirroring
# test_face_recognition.py's clusterer coverage exactly, but for cosine
# distance over fixed-length vectors instead of LBPH.


def _unit_vector(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_clusterer_matches_the_same_embedding_and_separates_a_different_one(
    tmp_path: Path,
) -> None:
    same_direction = _unit_vector(1.0, 0.0, 0.0, 0.0)
    almost_same_direction = _unit_vector(0.98, 0.02, 0.0, 0.0)
    different_direction = _unit_vector(0.0, 0.0, 1.0, 0.0)

    clusterer = IncrementalFaceEmbeddingClusterer(tmp_path / "model.json", distance_threshold=0.1)
    first_id, first_distance, first_is_new = clusterer.match_or_create_cluster(same_direction)
    assert first_is_new is True
    assert first_distance == 0.0

    second_id, second_distance, second_is_new = clusterer.match_or_create_cluster(
        almost_same_direction
    )
    assert second_is_new is False
    assert second_id == first_id
    assert second_distance < 0.1

    third_id, third_distance, third_is_new = clusterer.match_or_create_cluster(different_direction)
    assert third_is_new is True
    assert third_id != first_id
    assert third_distance > clusterer.distance_threshold


def test_clusterer_persists_across_reload(tmp_path: Path) -> None:
    embedding = _unit_vector(1.0, 0.0, 0.0)
    model_path = tmp_path / "model.json"

    first = IncrementalFaceEmbeddingClusterer(model_path)
    cluster_id, _distance, _is_new = first.match_or_create_cluster(embedding)
    first.save()

    second = IncrementalFaceEmbeddingClusterer(model_path)
    reloaded_id, _distance2, is_new2 = second.match_or_create_cluster(embedding)
    assert is_new2 is False
    assert reloaded_id == cluster_id


def test_predict_nearest_cluster_excludes_the_queried_clusters_own_match(
    tmp_path: Path,
) -> None:
    embedding_a = _unit_vector(1.0, 0.0, 0.0)
    embedding_b = _unit_vector(0.0, 1.0, 0.0)

    clusterer = IncrementalFaceEmbeddingClusterer(tmp_path / "model.json", distance_threshold=0.1)
    cluster_a, _distance, _is_new = clusterer.match_or_create_cluster(embedding_a)
    cluster_b, _distance, _is_new = clusterer.match_or_create_cluster(embedding_b)

    self_matched_id, self_distance = clusterer.predict_nearest_cluster(embedding_a)
    assert self_matched_id == cluster_a
    assert self_distance == 0.0

    other_matched_id, _other_distance = clusterer.predict_nearest_cluster(
        embedding_a, exclude_cluster_id=cluster_a
    )
    assert other_matched_id == cluster_b


def test_predict_nearest_cluster_returns_none_when_no_clusters_exist(tmp_path: Path) -> None:
    clusterer = IncrementalFaceEmbeddingClusterer(tmp_path / "model.json")
    assert clusterer.predict_nearest_cluster(_unit_vector(1.0, 0.0)) == (None, 0.0)


def test_get_cluster_centroid_returns_none_for_unknown_cluster(tmp_path: Path) -> None:
    clusterer = IncrementalFaceEmbeddingClusterer(tmp_path / "model.json")
    assert clusterer.get_cluster_centroid("does-not-exist") is None


# -- AuraFaceEmbedder: the test sandbox has no network access (see
# AGENTS.md), and `insightface` is an optional extra not installed by
# default in `dev` -- these tests degrade cleanly either way rather than
# assuming the extra is present.


def test_raises_when_insightface_not_installed() -> None:
    if insightface_available():
        pytest.skip("insightface is installed in this environment")
    with pytest.raises(AuraFaceUnavailableError):
        AuraFaceEmbedder()


def test_insightface_available_returns_false_instead_of_raising_on_a_non_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real failure mode hit on Windows: `insightface` unconditionally
    imports `albumentations`, which imports `torch` if torch happens to be
    installed at all (unrelated to whether this project needs it) -- and a
    broken/conflicting local torch CUDA install can raise a plain OSError
    (not ImportError) partway through that chain. This must degrade to
    "not available" (embedding_model: lbph fallback), not crash the run."""
    import builtins

    real_import = builtins.__import__

    def _broken_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "insightface":
            raise OSError(
                "[WinError 127] The specified procedure could not be found. "
                'Error loading "torch\\lib\\cudnn_cnn64_9.dll" or one of its dependencies.'
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _broken_import)

    assert insightface_available() is False


def test_raises_when_model_registry_cannot_provide_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not insightface_available():
        pytest.skip("insightface is not installed in this environment")

    def _fail(self: object, name: str, device: str = "cpu") -> Path:
        raise ModelUnavailableError("no network")

    monkeypatch.setattr(face_auraface_module.ModelRegistry, "ensure_model", _fail)

    with pytest.raises(AuraFaceUnavailableError):
        AuraFaceEmbedder()


def test_get_embedding_returns_none_for_an_empty_crop() -> None:
    embedder = object.__new__(AuraFaceEmbedder)
    embedder._model = _FakeRecognitionModel(np.array([1.0, 0.0, 0.0]))  # type: ignore[attr-defined]

    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    assert embedder.get_embedding(empty_crop) is None


def test_get_embedding_normalizes_the_raw_feature_vector() -> None:
    embedder = object.__new__(AuraFaceEmbedder)
    embedder._model = _FakeRecognitionModel(np.array([3.0, 4.0, 0.0]))  # type: ignore[attr-defined]

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    embedding = embedder.get_embedding(crop)

    assert embedding is not None
    assert embedding == pytest.approx(np.array([0.6, 0.8, 0.0]))
    assert np.linalg.norm(embedding) == pytest.approx(1.0)


def test_get_embedding_returns_none_for_a_zero_feature_vector() -> None:
    embedder = object.__new__(AuraFaceEmbedder)
    embedder._model = _FakeRecognitionModel(np.zeros(3))  # type: ignore[attr-defined]

    crop = np.zeros((50, 50, 3), dtype=np.uint8)
    assert embedder.get_embedding(crop) is None


class _FakeRecognitionModel:
    """Stands in for the `ArcFaceONNX` object `insightface.model_zoo.get_model`
    returns -- only `input_size`/`get_feat` are exercised by
    `AuraFaceEmbedder.get_embedding`."""

    input_size = (112, 112)

    def __init__(self, feature_vector: np.ndarray) -> None:
        self._feature_vector = feature_vector

    def get_feat(self, image: np.ndarray) -> np.ndarray:
        return self._feature_vector.reshape(1, -1)
