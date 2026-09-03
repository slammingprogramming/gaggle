from __future__ import annotations

from pathlib import Path

import pytest

from gaggle.core import models as models_module
from gaggle.core.models import (
    ModelFile,
    ModelRegistry,
    ModelSpec,
    ModelUnavailableError,
    precision_for_device,
)


def test_precision_for_device() -> None:
    assert precision_for_device("cpu") == "int8"
    assert precision_for_device("cuda") == "fp16"


@pytest.fixture
def fake_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModelRegistry:
    """A registry pointed at a scratch cache dir, with the module-level
    model table replaced by a small fake spec so tests never touch the
    real network or the real (much larger) model catalog."""

    fake_spec = ModelSpec(
        name="fake-model",
        license="Fake-License",
        variants={
            "fp32": ModelFile(url="https://example.invalid/fake.onnx", filename="fake.onnx"),
            "int8": ModelFile(
                url="https://example.invalid/fake_int8.onnx", filename="fake_int8.onnx"
            ),
        },
    )
    monkeypatch.setattr(models_module, "_REGISTRY", {"fake-model": fake_spec})
    return ModelRegistry(cache_dir=tmp_path / "model-cache")


def _fake_download(monkeypatch: pytest.MonkeyPatch, content: bytes = b"fake-model-bytes") -> None:
    def _write(url: str, destination: Path) -> None:
        destination.write_bytes(content)

    monkeypatch.setattr(models_module.urllib.request, "urlretrieve", _write)


def test_ensure_model_downloads_the_precision_specific_prebuilt_variant(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_download(monkeypatch)
    path = fake_registry.ensure_model("fake-model", device="cpu")
    assert path.exists()
    assert path.name == "fake_int8.onnx"


def test_ensure_model_is_cache_hit_on_second_call(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def _write(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_bytes(b"data")

    monkeypatch.setattr(models_module.urllib.request, "urlretrieve", _write)

    first = fake_registry.ensure_model("fake-model", device="cpu")
    second = fake_registry.ensure_model("fake-model", device="cpu")

    assert first == second
    assert len(calls) == 1  # second call was a cache hit, no re-download


def test_ensure_model_falls_back_to_fp32_and_derives_missing_precision(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_download(monkeypatch)
    derived_paths: list[Path] = []

    def _fake_derive_fp16(source: Path, destination: Path) -> None:
        derived_paths.append(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"derived-fp16")

    monkeypatch.setattr(models_module, "derive_fp16", _fake_derive_fp16)

    # "fake-model" has no fp16 variant registered -- requesting cuda should
    # download fp32 and derive fp16 locally rather than failing.
    path = fake_registry.ensure_model("fake-model", device="cuda")

    assert path.exists()
    assert path in derived_paths
    assert path.read_bytes() == b"derived-fp16"


def test_ensure_model_falls_back_to_fp32_when_derivation_fails(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real failure mode: local dynamic int8
    quantization of a conv-heavy recognition model produced a file that
    then failed to load with a `ConvInteger` kernel this machine's
    onnxruntime CPU execution provider doesn't implement (verified
    against AuraFace's real recognition model, not hypothetical). A
    derivation failure should degrade to serving the fp32 model rather
    than making the whole model unavailable."""

    _fake_download(monkeypatch)

    def _fake_derive_int8(source: Path, destination: Path) -> None:
        raise ModelUnavailableError("derived model failed to load")

    monkeypatch.setattr(models_module, "derive_int8", _fake_derive_int8)

    # "int8-only-source" variant deliberately absent -- forces the
    # derive-from-fp32 path for the cpu/int8 precision.
    spec = ModelSpec(
        name="conv-heavy-model",
        license="Fake-License",
        variants={
            "fp32": ModelFile(url="https://example.invalid/fp32.onnx", filename="fp32.onnx"),
        },
    )
    monkeypatch.setattr(models_module, "_REGISTRY", {"conv-heavy-model": spec})
    registry = ModelRegistry(cache_dir=fake_registry.cache_dir)

    path = registry.ensure_model("conv-heavy-model", device="cpu")

    assert path.name == "fp32.onnx"
    assert path.exists()


def test_ensure_model_raises_for_unknown_model(fake_registry: ModelRegistry) -> None:
    with pytest.raises(ModelUnavailableError):
        fake_registry.ensure_model("does-not-exist", device="cpu")


def test_ensure_model_raises_when_no_variant_and_no_fp32_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = ModelSpec(
        name="int8-only",
        license="Fake-License",
        variants={
            "int8": ModelFile(url="https://example.invalid/x.onnx", filename="x.onnx"),
        },
    )
    monkeypatch.setattr(models_module, "_REGISTRY", {"int8-only": spec})
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    with pytest.raises(ModelUnavailableError):
        registry.ensure_model("int8-only", device="cuda")


def test_download_failure_raises_model_unavailable_error(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(url: str, destination: Path) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(models_module.urllib.request, "urlretrieve", _fail)

    with pytest.raises(ModelUnavailableError):
        fake_registry.ensure_model("fake-model", device="cpu")


def test_remove_model_deletes_cached_files(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_download(monkeypatch)
    path = fake_registry.ensure_model("fake-model", device="cpu")
    assert path.exists()

    fake_registry.remove_model("fake-model")

    assert not path.exists()


def test_status_reports_cached_and_uncached_variants(
    fake_registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_download(monkeypatch)
    fake_registry.ensure_model("fake-model", device="cpu")

    rows = fake_registry.status()

    by_precision = {row["precision"]: row for row in rows}
    assert by_precision["int8"]["cached"] is True
    assert by_precision["fp32"]["cached"] is False


def test_known_models_lists_registered_names(fake_registry: ModelRegistry) -> None:
    assert fake_registry.known_models() == ["fake-model"]
