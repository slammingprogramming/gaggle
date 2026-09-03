"""On-demand ML model acquisition, caching, and precision selection.

The three deep-learning recognition upgrades (YuNet face detection,
AuraFace face embedding, fast-alpr plate detection+OCR -- see
`enrichment/face_yunet.py`/`enrichment/face_auraface.py`/
`enrichment/plate_fast_alpr.py`) all go through this module rather than
bundling model weights in the package or hand-rolling their own download
logic. **No model file is ever committed to the repo** -- every model is
fetched on first use (or via `gaggle models download`) into a per-machine
cache, not a per-workspace one, since a model is shared across every
workspace: `platformdirs.user_cache_dir("gaggle") / "models"`.

**Precision selection**: CUDA benefits from fp16 throughput; CPU
inference is fastest with int8 (no calibration data needed for either
conversion -- see `derive_int8`/`derive_fp16`). `ensure_model` maps
`device="cuda"` -> `precision="fp16"` and `device="cpu"` -> `"int8"`,
preferring a pre-built upstream variant at that precision when one
exists (it has real accuracy validation upstream; a local conversion
hasn't), and falling back to downloading the canonical fp32 source and
deriving the missing precision locally, exactly once per machine
(cached after that).

**Auto-provisioning**: unlike the existing YOLO/Whisper pattern (missing
model = silent "no detections, logged once"), a detector wrapper that
uses this registry calls `ensure_model` right where it loads its model --
if nothing is cached yet, it downloads and prepares it transparently
before continuing. A network failure at that moment still degrades
gracefully (`ModelUnavailableError` is caught by the caller, logged once,
that modality skipped for the run) -- the difference is "missing locally
but a download is possible" is no longer a dead end requiring a separate
manual step.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import platformdirs

from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

Device = Literal["cpu", "cuda"]
Precision = Literal["fp32", "fp16", "int8"]

_cuda_dlls_preloaded = False


def ensure_cuda_dlls_preloaded() -> None:
    """Loads onnxruntime's CUDA/cuDNN DLLs from the `nvidia-cuda-runtime`/
    `nvidia-cudnn-cuXX`/`nvidia-cublas`/etc. pip packages (installed via
    the `onnxruntime-gpu[cuda,cudnn]` extra) before any CUDA-requesting
    `onnxruntime.InferenceSession` is constructed.

    A real, verified requirement, not a defensive guess: even with those
    packages correctly installed, `CUDAExecutionProvider` silently fails
    to initialize (onnxruntime logs a warning and falls back to
    `CPUExecutionProvider`) unless `onnxruntime.preload_dlls()` runs
    first in this same process -- onnxruntime does not search
    pip-installed nvidia site-packages by default, only its own default
    DLL search paths. Confirmed: identical `InferenceSession(...,
    providers=["CUDAExecutionProvider", ...])` call, CPU-only without a
    prior `preload_dlls()` call in-process, CUDA-active with one.

    Idempotent and safe to call from every CUDA-requesting construction
    site (`enrichment/vehicle_yolo.py`, `enrichment/face_auraface.py`,
    `enrichment/plate_fast_alpr.py`, `derive_fp16` below) -- only does
    real work once per process.
    """

    global _cuda_dlls_preloaded
    if _cuda_dlls_preloaded:
        return
    _cuda_dlls_preloaded = True
    import onnxruntime as ort

    ort.preload_dlls()


class ModelUnavailableError(RuntimeError):
    """Raised when a model cannot be downloaded or prepared: a network
    failure, a downloaded file that fails hash verification, or a missing
    optional dependency needed for a local precision conversion."""


def cache_root() -> Path:
    return Path(platformdirs.user_cache_dir("gaggle")) / "models"


def precision_for_device(device: Device) -> Precision:
    return "fp16" if device == "cuda" else "int8"


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One concrete downloadable file for one model, at one precision."""

    url: str
    filename: str
    sha256: str | None = None  # None when upstream doesn't publish a fixed hash


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    license: str
    variants: dict[Precision, ModelFile]


_YUNET_SPEC = ModelSpec(
    name="yunet-detector",
    license="Apache-2.0 (opencv/opencv_zoo)",
    variants={
        "fp32": ModelFile(
            url=(
                "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx"
            ),
            filename="face_detection_yunet_2023mar.onnx",
        ),
        "int8": ModelFile(
            url=(
                "https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar_int8bq.onnx"
            ),
            filename="face_detection_yunet_2023mar_int8bq.onnx",
        ),
    },
)

_AURAFACE_SPEC = ModelSpec(
    name="auraface-embedding",
    license="Apache-2.0 (fal/AuraFace-v1)",
    variants={
        # `fal/AuraFace-v1` on Hugging Face is a full InsightFace-style
        # model pack (detection, landmarks, gender/age, recognition), same
        # file layout as InsightFace's own `buffalo_l` -- but this project
        # only ever wants the recognition/embedding model
        # (`glintr100.onnx`), since detection is already handled by
        # `enrichment/face_yunet.py`/`enrichment/face.py`'s Haar cascade.
        # Downloading just this one file (a plain HTTPS GET against the
        # repo's raw blob URL, no `huggingface_hub` dependency needed) is
        # both leaner and lets it flow through the same single-file
        # download/derive path every other model here uses -- fp32 ships
        # pre-built; int8/fp16 are always derived locally by
        # `ensure_model`. See `enrichment/face_auraface.py`.
        "fp32": ModelFile(
            url="https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
            filename="glintr100.onnx",
        ),
    },
)

_REGISTRY: dict[str, ModelSpec] = {
    "yunet-detector": _YUNET_SPEC,
    "auraface-embedding": _AURAFACE_SPEC,
}


def register_model(spec: ModelSpec) -> None:
    """Extension point for `enrichment/plate_fast_alpr.py`, whose models
    are resolved by name through the `fast_alpr`/`fast_plate_ocr`
    libraries' own loaders rather than a direct download URL -- see that
    module. Kept separate from the static `_REGISTRY` above so
    `gaggle models list` still has a single place to look, without this
    module needing an import-time dependency on `fast_alpr`.
    """

    _REGISTRY[spec.name] = spec


class ModelRegistry:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or cache_root()

    def known_models(self) -> list[str]:
        return sorted(_REGISTRY)

    def status(self) -> list[dict[str, object]]:
        """One row per known (model, precision) pair -- backs
        `gaggle models list`."""

        rows: list[dict[str, object]] = []
        for name in sorted(_REGISTRY):
            spec = _REGISTRY[name]
            for precision, model_file in spec.variants.items():
                path = self._path_for(name, precision, model_file.filename)
                rows.append(
                    {
                        "name": name,
                        "precision": precision,
                        "cached": path.exists(),
                        "path": str(path) if path.exists() else None,
                        "size_bytes": path.stat().st_size if path.exists() else None,
                        "license": spec.license,
                    }
                )
        return rows

    def _path_for(self, name: str, precision: Precision, filename: str) -> Path:
        return self.cache_dir / name / precision / filename

    def ensure_model(self, name: str, device: Device = "cpu") -> Path:
        """Return a local path to `name` at the precision appropriate for
        `device`, downloading and/or deriving it first if not already
        cached. See the module docstring for the full algorithm."""

        spec = _require_spec(name)
        precision = precision_for_device(device)
        if precision in spec.variants:
            return self._ensure_downloaded(spec, precision)

        if "fp32" not in spec.variants:
            raise ModelUnavailableError(
                f"no usable variant of model '{name}' for device '{device}' "
                "(no fp32 source to derive from either)"
            )
        fp32_path = self._ensure_downloaded(spec, "fp32")
        derived_filename = f"{spec.name}.{precision}.onnx"
        derived_path = self._path_for(name, precision, derived_filename)
        if derived_path.exists():
            return derived_path
        try:
            if precision == "int8":
                derive_int8(fp32_path, derived_path)
            else:
                derive_fp16(fp32_path, derived_path)
        except ModelUnavailableError as error:
            # A real failure mode, not hypothetical: local dynamic int8
            # quantization of a conv-heavy model can hit an op kernel
            # (e.g. ConvInteger) that this machine's onnxruntime build
            # doesn't implement on the CPU execution provider -- verified
            # against AuraFace's real recognition model. Falling back to
            # fp32 keeps the modality working (just without the precision
            # win) instead of turning an optimization failure into a hard
            # outage.
            LOGGER.warning(
                "model_precision_derivation_failed_falling_back_to_fp32",
                model=name,
                precision=precision,
                reason=str(error),
            )
            return fp32_path
        return derived_path

    def remove_model(self, name: str) -> None:
        _require_spec(name)
        model_dir = self.cache_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)
            LOGGER.info("model_removed", model=name)

    def _ensure_downloaded(self, spec: ModelSpec, precision: Precision) -> Path:
        model_file = spec.variants[precision]
        destination = self._path_for(spec.name, precision, model_file.filename)
        if destination.exists():
            return destination
        _download_http(model_file, destination)
        return destination


def _require_spec(name: str) -> ModelSpec:
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ModelUnavailableError(
            f"unknown model '{name}' (known models: {', '.join(sorted(_REGISTRY))})"
        )
    return spec


def _download_http(model_file: ModelFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".part")
    LOGGER.info("model_download_started", url=model_file.url, destination=str(destination))
    try:
        urllib.request.urlretrieve(model_file.url, tmp_path)
    except OSError as error:
        tmp_path.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"failed to download model from {model_file.url}: {error}"
        ) from error
    if model_file.sha256 is not None:
        actual = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if actual != model_file.sha256:
            tmp_path.unlink(missing_ok=True)
            raise ModelUnavailableError(f"downloaded model hash mismatch for {model_file.url}")
    tmp_path.replace(destination)
    LOGGER.info("model_ready", destination=str(destination))


_ONNX_TYPE_TO_NUMPY_DTYPE: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
}


def _validate_derived_model_runs(session: Any) -> None:
    """A loaded `InferenceSession` doesn't prove the model is actually
    usable -- confirmed twice for real, not hypothetically: `derive_int8`
    below caught a `ConvInteger` kernel gap only session construction
    surfaces, but `derive_fp16`'s I/O-dtype mismatch (a model converted
    with default settings ends up expecting float16 input, while every
    real caller in this codebase passes float32 arrays from OpenCV) only
    shows up on an actual forward pass. So every derived model gets one
    real `.run()` with zero-filled tensors matching its own declared
    input shapes/dtypes before being reported ready.
    """
    feed = {}
    for input_meta in session.get_inputs():
        dtype = _ONNX_TYPE_TO_NUMPY_DTYPE.get(input_meta.type, np.float32)
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in input_meta.shape]
        feed[input_meta.name] = np.zeros(shape, dtype=dtype)
    session.run(None, feed)


def derive_int8(source_path: Path, destination_path: Path) -> None:
    """Locally quantize an fp32 ONNX model to int8 for CPU inference, via
    `onnxruntime`'s own built-in dynamic quantization -- no calibration
    dataset needed, and no dependency beyond `onnxruntime` itself (already
    required by every ML extra this module serves). Real accuracy loss is
    model-dependent -- see docs/local-ai.md.

    Quantization can *succeed* (produce a file) and still be unusable --
    verified for real against a conv-heavy recognition model, where the
    derived int8 graph loaded fine as a file but failed at
    `InferenceSession` creation with a `ConvInteger` kernel this
    onnxruntime build's CPU execution provider doesn't implement. So this
    always does one real load *and* inference run of what it just
    produced before declaring success, and raises rather than leaving a
    file behind that looks cached but silently can't be used.
    """

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise ModelUnavailableError(
            "onnxruntime is required to derive an int8 model locally -- install "
            "the relevant extra (e.g. gaggle[face_recognition])"
        ) from error
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("model_conversion_started", target_precision="int8", source=str(source_path))
    quantize_dynamic(str(source_path), str(destination_path), weight_type=QuantType.QInt8)
    try:
        session = ort.InferenceSession(str(destination_path), providers=["CPUExecutionProvider"])
        _validate_derived_model_runs(session)
    except Exception as error:
        destination_path.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"derived int8 model at {destination_path} failed to load "
            f"(a real onnxruntime CPU execution provider gap, not a bug in "
            f"the derivation itself): {error}"
        ) from error
    LOGGER.info("model_ready", destination=str(destination_path))


def derive_fp16(source_path: Path, destination_path: Path) -> None:
    """Locally convert an fp32 ONNX model to fp16 for faster CUDA
    inference, via `onnxconverter-common` (only needed on this path --
    part of the `face_recognition`/`plate_recognition` extras, not a core
    dependency, since most installs never touch the GPU conversion path).
    No calibration data needed, unlike int8. Validated with a real load
    before being reported ready -- see `derive_int8`'s docstring for why
    that matters (a real gap was found there, not hypothetical).
    """

    try:
        import onnx
        import onnxruntime as ort
        from onnxconverter_common import float16
    except ImportError as error:
        raise ModelUnavailableError(
            "onnx and onnxconverter-common are required to derive an fp16 model "
            "locally -- install the relevant extra (e.g. gaggle[face_recognition])"
        ) from error
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("model_conversion_started", target_precision="fp16", source=str(source_path))
    model = onnx.load(str(source_path))
    # keep_io_types=True: internal compute still happens in fp16 (the
    # speed win this exists for), but the graph's own input/output
    # tensors stay float32 -- every caller in this codebase (YuNet/
    # AuraFace/vehicle-YOLO preprocessing) builds plain float32 arrays via
    # OpenCV, and without this flag onnxconverter-common also converts the
    # I/O tensors, so the derived model then rejects that same float32
    # input with `onnxruntime.InvalidArgument: Unexpected input data type`
    # -- a real gap found only once this path was actually exercised with
    # real image data (session construction alone doesn't catch it, which
    # is why _validate_derived_model_runs below does a real .run() too).
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(destination_path))
    ensure_cuda_dlls_preloaded()
    try:
        session = ort.InferenceSession(
            str(destination_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _validate_derived_model_runs(session)
    except Exception as error:
        destination_path.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"derived fp16 model at {destination_path} failed to load: {error}"
        ) from error
    LOGGER.info("model_ready", destination=str(destination_path))
