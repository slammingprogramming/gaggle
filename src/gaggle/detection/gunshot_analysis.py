"""Gunshot/gunfire audio-event detection via a local ONNX audio classifier.

**Why a classifier, not a classical impulse-detection heuristic.** A
gunshot's acoustic signature (fast rise time, high peak/RMS crest factor,
broadband energy burst) is real and measurable, but a classical heuristic
built on it cannot reliably tell a gunshot apart from a car door slam, an
engine backfire, a firework, or a construction impact -- all of which
produce a very similar sharp transient. A pretrained classifier, trained
on a large labeled corpus of real-world sound events, has actually seen
enough examples of each to draw a better (though still imperfect)
boundary between them. This was a real, considered tradeoff -- see this
project's plan history for the alternative that was ruled out.

**The model: k2-fsa's zipformer-small AudioSet tagger, via `sherpa-onnx`.**
Both the `sherpa-onnx`/`sherpa-onnx-core` Python packages and the model
itself are Apache-2.0 licensed (confirmed directly: the model archive's
own `README.md` states `license: apache-2.0`; both PyPI packages'
METADATA state `License: Apache-2.0`). The model is a Zipformer2 encoder
trained from scratch on AudioSet's 527 sound-event classes (not
fine-tuned/derived from a third-party checkpoint of unclear provenance),
published by the k2-fsa project (also behind Next-gen Kaldi/icefall) at a
stable, versioned GitHub Releases URL. `sherpa-onnx` bundles its own
onnxruntime statically inside its compiled extension module, so it does
not conflict with a separately-installed `onnxruntime`/`onnxruntime-gpu`
the vision/face_recognition extras use (a real, previously-hit DLL
conflict in this project -- see `core/models.py::ensure_cuda_dlls_preloaded`'s
docstring for that history).

Real, verified end-to-end: this module's model+class-name choices were
validated by actually downloading the archive, hash-verifying it,
inspecting the ONNX graph's real input/output tensor contract, and running
real inference against the archive's own bundled test WAV files through
`sherpa_onnx.AudioTagging` -- every test clip (cat meow, dog bark, siren,
baby cry, smoke alarm, etc.) produced the correct top-1/top-2 label at
high confidence. No real or synthetic *gunshot* audio was available in
this environment to test against directly, though -- unlike
`enrichment/vehicle_appearance.py`/`enrichment/person_appearance.py`,
where a synthetic test scene could be generated for a classical color/
shape fingerprint, there is no equivalent honest way to synthesize a
realistic gunshot waveform for this classifier. Treat every match this
detector produces as an unvalidated-in-this-environment classifier
opinion, not a confirmed identification -- see docs/limitations.md.

**Placement: `analyze` time, not `enrich` time.** Unlike face/plate/
vehicle-appearance recognition (which never affect scoring -- see
`enrichment/service.py`'s module docstring), a gunshot is a
safety-relevant signal that should be able to contribute to severity
scoring the same way motion/audio-spike/telemetry signals do. See
`inference/service.py`'s `isolated_gunshot_retention`/`gunshot_plus_motion`
rules and AGENTS.md invariant 7 -- a lone gunshot-classifier detection,
however confident, is capped and can never alone reach medium/high
severity (scoring requires corroborating signal *types*, not just
confidence).
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import platformdirs

from gaggle.detection.audio_analysis import AudioAnalysisError, extract_normalized_waveform
from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

DETECTOR_VERSION = "1.0.0"

# The archive's own README.md states `license: apache-2.0`; the k2-fsa/
# sherpa-onnx GitHub Releases URL is a stable, versioned, checkable
# source (not a re-uploaded/converted third-party mirror).
_ARCHIVE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2"
)
# Computed directly from the downloaded archive during this feature's
# research spike -- real, not guessed.
_ARCHIVE_SHA256 = "07e2fafcdcbc461f2816188d9b0bbafced12584030cf67d5652e549ef256a2c6"
_ARCHIVE_ROOT_DIR = "sherpa-onnx-zipformer-small-audio-tagging-2024-04-15"
_MODEL_MEMBER = f"{_ARCHIVE_ROOT_DIR}/model.int8.onnx"
_LABELS_MEMBER = f"{_ARCHIVE_ROOT_DIR}/class_labels_indices.csv"
_MODEL_FILENAME = "model.int8.onnx"
_LABELS_FILENAME = "class_labels_indices.csv"
MODEL_LICENSE = "Apache-2.0 (k2-fsa/sherpa-onnx zipformer-small AudioSet tagger)"

# AudioSet class *display names* (from class_labels_indices.csv, index 427
# is "Gunshot, gunfire") treated as "gunshot-like" for this detector's
# purposes -- confirmed present in the real downloaded label file.
# Deliberately excludes the acoustically-adjacent "Fireworks"/
# "Firecracker" classes: conflating those with actual gunfire would be
# actively misleading for a safety signal, not just imprecise, so they
# are never counted here even if the classifier's own confusion between
# them is real and unresolved.
GUNSHOT_LIKE_CLASS_NAMES = frozenset(
    {
        "Gunshot, gunfire",
        "Machine gun",
        "Artillery fire",
        "Cap gun",
    }
)


class GunshotDetectionError(RuntimeError):
    """Raised when the audio classifier can't be prepared or run."""


@dataclass(frozen=True, slots=True)
class GunshotEvent:
    offset_seconds: float
    duration_seconds: float
    class_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class GunshotAnalysisResult:
    events: list[GunshotEvent]
    has_audio: bool
    analyzer_version: str = DETECTOR_VERSION


def sherpa_onnx_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    except Exception as error:
        LOGGER.warning(
            "sherpa_onnx_import_failed",
            reason=str(error),
            hint="gunshot detection disabled for this run",
        )
        return False
    return True


def _cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("gaggle")) / "models" / "gunshot-audio-tagger"


def ensure_gunshot_model(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return ``(model_onnx_path, labels_csv_path)``, downloading and
    extracting the model archive into the per-machine cache first if not
    already present. Verifies the downloaded archive's sha256 before
    extracting anything from it -- see this module's docstring for the
    pinned hash's provenance.
    """

    root = cache_dir or _cache_dir()
    model_path = root / _MODEL_FILENAME
    labels_path = root / _LABELS_FILENAME
    if model_path.exists() and labels_path.exists():
        return model_path, labels_path

    root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("gunshot_model_download_started", url=_ARCHIVE_URL)
    try:
        with urllib.request.urlopen(_ARCHIVE_URL, timeout=300) as response:
            archive_bytes = response.read()
    except OSError as error:
        raise GunshotDetectionError(
            f"failed to download gunshot model archive from {_ARCHIVE_URL}: {error}"
        ) from error

    actual_hash = hashlib.sha256(archive_bytes).hexdigest()
    if actual_hash != _ARCHIVE_SHA256:
        raise GunshotDetectionError(
            f"gunshot model archive hash mismatch (expected {_ARCHIVE_SHA256}, got {actual_hash})"
        )

    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:bz2") as archive:
        for member_name, destination in (
            (_MODEL_MEMBER, model_path),
            (_LABELS_MEMBER, labels_path),
        ):
            try:
                extracted = archive.extractfile(member_name)
            except KeyError as error:
                raise GunshotDetectionError(
                    f"gunshot model archive is missing expected member '{member_name}'"
                ) from error
            if extracted is None:
                raise GunshotDetectionError(
                    f"gunshot model archive member '{member_name}' is not a regular file"
                )
            destination.write_bytes(extracted.read())
    LOGGER.info("gunshot_model_ready", destination=str(root))
    return model_path, labels_path


def load_tagger(model_path: Path, labels_path: Path, top_k: int = 5) -> Any:
    """Build a `sherpa_onnx.AudioTagging` instance for `model_path`.
    Callers should build this once and reuse it across every clip in a
    run -- loading the ONNX graph is real, non-trivial work, unlike this
    module's other (cheap, per-call) functions.
    """

    import sherpa_onnx

    config = sherpa_onnx.AudioTaggingConfig(
        model=sherpa_onnx.AudioTaggingModelConfig(
            zipformer=sherpa_onnx.OfflineZipformerAudioTaggingModelConfig(model=str(model_path)),
            num_threads=1,
        ),
        labels=str(labels_path),
        top_k=top_k,
    )
    return sherpa_onnx.AudioTagging(config)


def analyze_gunshot_events(
    path: Path,
    tagger: Any,
    window_seconds: float = 2.0,
    hop_seconds: float = 1.0,
    confidence_threshold: float = 0.5,
    timeout_seconds: float = 300.0,
) -> GunshotAnalysisResult:
    """Slide a `window_seconds`-long window (stride `hop_seconds`) over
    ``path``'s audio track, classify each window with `tagger`, and
    return every window whose top-matching label is one of
    `GUNSHOT_LIKE_CLASS_NAMES` at or above `confidence_threshold`.

    Windowed rather than one classification for the whole clip: a real
    source clip can run minutes long, and a single whole-clip
    classification would only say "this clip contains a gunshot-like
    sound somewhere," with no usable timestamp for the signal this
    produces.
    """

    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    try:
        extracted = extract_normalized_waveform(path, timeout_seconds)
    except AudioAnalysisError as error:
        raise GunshotDetectionError(str(error)) from error
    if extracted is None:
        return GunshotAnalysisResult(events=[], has_audio=False)
    waveform, sample_rate = extracted
    waveform_f32 = waveform.astype(np.float32)

    window_samples = max(1, round(window_seconds * sample_rate))
    hop_samples = max(1, round(hop_seconds * sample_rate))
    min_usable_samples = max(1, round(sample_rate * 0.1))

    events: list[GunshotEvent] = []
    for start_index in range(0, waveform_f32.size, hop_samples):
        chunk = waveform_f32[start_index : start_index + window_samples]
        if chunk.size < min_usable_samples:
            break
        stream = tagger.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=chunk)
        for entry in tagger.compute(stream):
            if entry.name not in GUNSHOT_LIKE_CLASS_NAMES or entry.prob < confidence_threshold:
                continue
            events.append(
                GunshotEvent(
                    offset_seconds=round(start_index / sample_rate, 3),
                    duration_seconds=round(chunk.size / sample_rate, 3),
                    class_name=entry.name,
                    confidence=round(float(entry.prob), 6),
                )
            )
        if start_index + window_samples >= waveform_f32.size:
            break

    return GunshotAnalysisResult(events=events, has_audio=True)
