from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)
_ENV_VAR_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_duration_seconds: int = 10
    window_stride_seconds: int = 5
    timezone: str = "UTC"
    # Caps how long a single merged event can span (see
    # core/pipeline.py::AnalysisPipeline._cluster_overlapping_windows).
    # Without this, near-continuous real footage (activity present in
    # almost every window, e.g. an actual long stretch of driving) merges
    # into one arbitrarily long event -- a derived clip effectively the
    # length of the entire source recording. A reasoned starting point
    # (long enough for a real multi-signal incident, short enough to keep
    # derived clips and enrichment fast), not empirically validated
    # against every possible driving scenario -- see docs/limitations.md
    # for the tradeoff a forced split introduces. None disables the cap.
    max_event_duration_seconds: float | None = 120.0


class SyncConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_gap_seconds: float = 120.0
    # Manual per-camera correction (camera_id -> seconds, added to every
    # clip's already-computed corrected_start/corrected_end) for the real
    # failure mode where normalize/sync.py's start-alignment heuristic
    # picks a genuinely wrong offset for a camera -- e.g. one whose clock
    # was already off before this project ever saw it, in a direction the
    # heuristic's own confidence-based reference selection can't detect on
    # its own. Empty by default (no correction). Only affects future
    # `analyze` runs, not events that already exist -- an event already
    # built with wrong sync needs `EventSplitService`/manual correction
    # instead (analyze is idempotent by design and won't reprocess
    # already-covered clips just because this changed). See
    # docs/local-ai.md's cross-camera sync section.
    manual_offset_overrides: dict[str, float] = Field(default_factory=dict)


class TelemetryConfig(BaseModel):
    """Thresholds for `detection/telemetry_analysis.py`'s GPS-track-derived
    events. See that module's docstring for the exact formulas.
    """

    model_config = ConfigDict(extra="forbid")

    # ~0.4g, a commonly-cited "hard braking" threshold in telematics
    # literature -- a sane default, not empirically tuned against real
    # dashcam trips in this environment (see docs/limitations.md).
    hard_braking_threshold_mps2: float = 4.0
    speed_spike_threshold_mps: float = 20.0
    heading_change_threshold_deg_per_sec: float = 45.0


class OpticalFlowConfig(BaseModel):
    """Thresholds for `detection/optical_flow_analysis.py`'s "rapid
    approach" (looming) detection. See that module's docstring for the
    exact technique and the empirical measurement behind the default.
    """

    model_config = ConfigDict(extra="forbid")

    sample_rate_hz: float = 2.0
    # Empirically measured (not guessed) against synthetic true-positive
    # (approaching-object) and true-negative (ego-motion/static) scenes --
    # see detection/optical_flow_analysis.py's module docstring for the
    # measured distributions.
    roi_divergence_delta_threshold: float = 0.015


class GunshotDetectionConfig(BaseModel):
    """Thresholds for `detection/gunshot_analysis.py`'s local ONNX
    audio-event classifier (k2-fsa's zipformer-small AudioSet tagger,
    Apache-2.0, via the optional `sherpa-onnx` dependency -- see
    `docs/local-ai.md`'s "Gunshot detection" section for the full
    picture, including why this was chosen over a classical
    impulse-detection heuristic and its real false-positive risks).

    **Off by default, unlike audio_spike/motion** -- unlike those
    zero-dependency detectors, this requires the `gunshot` extra
    (`pip install gaggle[gunshot]`) and downloads a real model file on
    first use. Silently produces no signals (logged once) if the
    dependency isn't installed or the model can't be downloaded --
    never a hard failure of `analyze`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Minimum classifier probability (0-1) for a window's top-matching
    # gunshot-like AudioSet class to count as a detection. Not empirically
    # tuned against real gunshot audio in this environment (none was
    # available to test against) -- expect to retune against your own
    # footage. See docs/limitations.md.
    confidence_threshold: float = 0.5
    # Length of each audio window classified at a time. The underlying
    # model has no fixed input-length requirement, but was trained on
    # ~10s AudioSet excerpts -- shorter windows trade some accuracy for
    # finer-grained timestamp localization of the specific sound.
    window_seconds: float = 2.0
    hop_seconds: float = 1.0


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motion_threshold: float = 0.20
    audio_spike_threshold: float = 0.45
    min_signal_duration_seconds: float = 1.0
    motion_sample_rate_hz: float = 2.0
    audio_window_seconds: float = 0.5
    min_object_region_area_ratio: float = 0.01
    use_fixture_signals_when_available: bool = True
    # ffmpeg's real audio-extraction timeout (detection/audio_analysis.py)
    # for this stage's normalized (not-yet-derived-clip) source media,
    # which can legitimately be a full-length real recording several
    # minutes long. Generous by default rather than tuned for a short
    # synthetic test clip.
    audio_extraction_timeout_seconds: float = 300.0
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    optical_flow: OpticalFlowConfig = Field(default_factory=OpticalFlowConfig)
    gunshot: GunshotDetectionConfig = Field(default_factory=GunshotDetectionConfig)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    low_threshold: float = 0.20
    medium_threshold: float = 0.50
    high_threshold: float = 0.80


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hash_algorithm: str = "sha256"
    set_read_only: bool = True
    # "copy" (default): duplicate every source file into workspace/originals/,
    #   leaving the source untouched. Safest, but needs 2x disk space during
    #   ingest.
    # "move": relocate each source file into workspace/originals/ instead of
    #   copying. Frees the source location (e.g. an SD card) immediately
    #   after ingest, at the cost of a one-way operation -- the source no
    #   longer has the file afterward.
    # "reference": don't touch the source at all. The workspace indexes the
    #   file at its existing location and every downstream stage reads it
    #   from there. Zero extra disk use, but the workspace now depends on
    #   that external location staying available and unmodified -- see
    #   docs/local-ai.md's "Choosing an ingest storage mode" section before
    #   using this for anything you can't re-ingest later.
    ingest_mode: Literal["copy", "move", "reference"] = "copy"
    # When true (default), ingesting a file whose content hash already
    # matches a previously-indexed clip is skipped (logged, not copied
    # again) rather than silently producing a second, redundant copy.
    # Never touches the source file or the existing copy either way -- see
    # invariant 1 in AGENTS.md. Set to false to restore the old
    # always-copy behavior (e.g. for automation that expects one output
    # clip per input file, duplicates included).
    dedupe_on_ingest: bool = True


class FaceRecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # "yunet" (opencv_zoo, Apache-2.0) is a real deep-learning detector,
    # fetched on demand via core/models.py::ModelRegistry the first time
    # it's used; "haar" is the classical cascade bundled with every OpenCV
    # install, zero setup, kept as an always-available fallback. See
    # enrichment/face_yunet.py.
    detector: Literal["haar", "yunet"] = "yunet"
    # "auraface" (fal/AuraFace-v1, Apache-2.0) is a real embedding model,
    # also fetched on demand; "lbph" is the classical texture-histogram
    # clusterer bundled with opencv-contrib, kept as an always-available
    # fallback. See enrichment/face_auraface.py. Defaults to auraface,
    # matching detector: yunet's same "default to the real model, degrade
    # gracefully to the classical one if the extra/model isn't available"
    # choice -- if `insightface` isn't installed (the `face_recognition`
    # extra), this falls back to lbph automatically, logged once.
    embedding_model: Literal["lbph", "auraface"] = "auraface"
    # "cuda" uses onnxruntime's CUDAExecutionProvider (requires
    # onnxruntime-gpu installed in place of the CPU package) and fp16
    # model precision; "cpu" uses int8. Only consulted by the auraface
    # embedding path -- the lbph fallback is always CPU-only, and YuNet
    # detection specifically cannot be GPU-accelerated via the standard
    # pip OpenCV build regardless of this setting (a real OpenCV build
    # limitation, not a gaggle one -- see enrichment/face_yunet.py and
    # docs/local-ai.md). Falls back to CPU gracefully (logged) if CUDA
    # isn't actually available -- see docs/local-ai.md's GPU setup notes
    # before assuming this is engaged; it needs a real, verified working
    # onnxruntime-gpu + CUDA/cuDNN setup, not just this being set to cuda.
    device: str = "cuda"
    detector_min_size: tuple[int, int] = (30, 30)
    # LBPH distance scale (roughly 0-100+); only consulted when
    # embedding_model: lbph.
    cluster_distance_threshold: float = 70.0
    # Cosine distance over normalized AuraFace embeddings (0-2 scale, not
    # comparable to cluster_distance_threshold above); only consulted when
    # embedding_model: auraface. A starting point from published
    # ArcFace-family benchmarks, not validated against this project's own
    # footage -- see enrichment/face_auraface.py and docs/limitations.md.
    embedding_cluster_distance_threshold: float = 0.35
    min_detection_confidence: float = 0.15
    # Used by `recognize faces-cleanup`: observations of the same cluster
    # within this many seconds of each other, in the same event, are
    # treated as the same physical sighting sampled repeatedly.
    duplicate_observation_window_seconds: float = 5.0
    # Used by `recognize suggest-merges --entity-type face`: a distance
    # above cluster_distance_threshold but within this multiplier of it is
    # close enough to suggest a merge for human review (below the
    # threshold, the incremental clusterer would already have auto-merged
    # them; well above it, they're presumably different people).
    merge_suggestion_multiplier: float = 1.6


class PlateRecognitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # "fast_alpr" (ankandrew/fast-alpr, MIT) is a real deep-learning
    # detector+OCR, trained on international plate formats rather than
    # one region; its models are downloaded and cached by the library
    # itself on first use (see enrichment/plate_fast_alpr.py). "cascade"
    # is the classical Haar-cascade/MSER/contour detector + Tesseract
    # OCR, calibrated for Russian-format plates, kept as an
    # always-available fallback. Defaults to fast_alpr, matching
    # `face.detector: yunet`'s same "default to the real model, degrade
    # gracefully if the extra isn't installed" choice -- if `fast-alpr`
    # isn't installed (the `plate_recognition` extra), this falls back
    # to cascade automatically, logged once.
    detector: Literal["cascade", "fast_alpr"] = "fast_alpr"
    # "cuda" requires additionally installing onnxruntime-gpu in place of
    # the CPU package. Only consulted by the fast_alpr path -- the
    # cascade/Tesseract fallback is always CPU-only. Falls back to CPU
    # gracefully (logged) if CUDA isn't actually available -- see
    # docs/local-ai.md's GPU setup notes.
    device: str = "cuda"
    detector_min_size: tuple[int, int] = (40, 15)
    min_detection_confidence: float = 0.10
    auto_accept_ocr_confidence: float = 0.75
    min_ocr_confidence_to_keep: float = 0.20
    # OCR results outside this character-count range are almost always noise
    # (a single stray character, or a run-on misread spanning unrelated
    # background text) and are discarded before ever being stored -- not
    # just hidden from review. Most plates worldwide fall inside 4-8
    # characters; widen this if your plates run shorter/longer.
    min_plate_text_length: int = 4
    max_plate_text_length: int = 9
    # Used by `recognize plates-cleanup`: observations of the same plate
    # text within this many seconds of each other, in the same event, are
    # treated as the same physical sighting sampled repeatedly -- only the
    # highest-confidence one needs a human's attention.
    duplicate_observation_window_seconds: float = 5.0
    # Used by `recognize suggest-merges --entity-type plate`: two plate
    # records whose text similarity (difflib ratio) is at or above this are
    # suggested as a likely OCR-misread pair, for human review.
    merge_suggestion_similarity_threshold: float = 0.75


class VisionConfig(BaseModel):
    """Optional local YOLO-style vehicle/object detection (`vision` extra).

    Still off by default (`enabled: False`) regardless of `device` --
    it always needs a user-supplied `model_path` (see docs/local-ai.md),
    so there's nothing to run out of the box either way.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_path: str | None = None
    # Falls back to CPU gracefully (logged) if CUDA isn't actually
    # available -- see docs/local-ai.md's GPU setup notes.
    device: str = "cuda"
    confidence_threshold: float = 0.35


class TranscriptionConfig(BaseModel):
    """Optional local Whisper transcription (`transcription` extra)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_name: str = "base"
    # faster-whisper/ctranslate2's own device auto-detection (real GPU
    # if genuinely usable, CPU otherwise) -- unlike the onnxruntime-based
    # capabilities above, ctranslate2 does not gracefully degrade from a
    # hard "cuda" request if CUDA turns out to be unavailable at
    # construction time (it raises, which this project treats as
    # "transcription unavailable this run", not a crash -- but that's a
    # worse outcome than just using CPU). "auto" avoids that trap while
    # still using the GPU whenever it's genuinely there.
    device: str = "auto"
    compute_type: str = "int8"


class CloudEnrichmentConfig(BaseModel):
    """Optional, off-by-default OpenAI-compatible transcript analysis (`cloud` extra).

    Disabled unless explicitly turned on AND both endpoint and an API key
    (via the environment variable named by ``api_key_env_var``, never
    written directly into a config file) are set. See docs/local-ai.md.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    endpoint: str | None = None
    model: str = "openai/gpt-4o-mini"
    api_key_env_var: str = "DASHCAM_SENTINEL_LLM_API_KEY"
    timeout_seconds: float = 30.0


class VoiceRecognitionConfig(BaseModel):
    """Local voice-activity-detection + classical MFCC-based voiceprinting.

    On by default like face/plate recognition (zero extra download, just
    numpy/scipy, both core dependencies) -- but see
    `enrichment/voice.py`'s module docstring before relying on this for
    anything beyond casual personal review: it's a meaningfully weaker
    fingerprint than face/plate recognition, validated only against
    synthetic test signals in this project's own development, not real
    recorded speech.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_segment_seconds: float = 0.3
    energy_percentile_threshold: float = 60.0
    merge_gap_seconds: float = 0.2
    cluster_distance_threshold: float = 0.05
    duplicate_observation_window_seconds: float = 5.0
    merge_suggestion_multiplier: float = 1.6
    # ffmpeg's real audio-extraction timeout for this stage's derived
    # clips (see detection/audio_analysis.py). Generous by default since
    # a derived clip's length depends on pipeline.max_event_duration_seconds,
    # not a fixed short window.
    audio_extraction_timeout_seconds: float = 300.0


class VehicleAppearanceConfig(BaseModel):
    """Local, classical vehicle-appearance re-identification (dominant
    color histogram + aspect ratio) -- for a vehicle seen without a
    legible plate. On by default like face/plate/voice recognition (zero
    extra download, just OpenCV/numpy, both core dependencies already
    required), but see `enrichment/vehicle_appearance.py`'s module
    docstring: a meaningfully weaker fingerprint than face or plate
    re-identification -- it can't distinguish two vehicles of the same
    color and body shape -- validated only against synthetic test scenes
    in this project's own development, not real footage.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    detector_min_size: tuple[int, int] = (60, 40)
    min_detection_confidence: float = 0.15
    cluster_distance_threshold: float = 0.10
    # Used by `recognize vehicles-cleanup`: observations of the same
    # cluster within this many seconds of each other, in the same event,
    # are treated as the same physical sighting sampled repeatedly.
    duplicate_observation_window_seconds: float = 5.0
    # Used by `recognize suggest-merges --entity-type vehicle_appearance`:
    # same interpretation as `FaceRecognitionConfig.merge_suggestion_multiplier`.
    merge_suggestion_multiplier: float = 1.6


class PersonAppearanceConfig(BaseModel):
    """Local, classical pedestrian/full-body appearance re-identification
    (dominant clothing-color histogram + aspect ratio) -- see
    `enrichment/person_appearance.py`'s module docstring for the full
    picture: structured attributes only, never a learned embedding or an
    AI-generated description.

    **Off by default, unlike face/plate/voice/vehicle_appearance** --
    unlike those, this has no classical zero-setup detector fallback; it
    requires `enrichment.vision.enabled: true` plus a real YOLO model
    file (same requirement as `vision` itself), so leaving this on by
    default with `vision` off would silently do nothing. Turn both on
    together to use it.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    min_detection_confidence: float = 0.15
    cluster_distance_threshold: float = 0.10
    # Used by `recognize persons-cleanup`: observations of the same
    # cluster within this many seconds of each other, in the same event,
    # are treated as the same physical sighting sampled repeatedly.
    duplicate_observation_window_seconds: float = 5.0
    # Used by `recognize suggest-merges --entity-type person_appearance`:
    # same interpretation as `FaceRecognitionConfig.merge_suggestion_multiplier`.
    merge_suggestion_multiplier: float = 1.6


class EncounterConfig(BaseModel):
    """Cross-modality encounter derivation (`schemas/encounter.py`) -- a
    pure post-processing pass over whatever face/plate/voice/vehicle
    observations the other enrichment passes already persisted, with no
    detection logic or extra dependency of its own. On by default like
    face/plate/voice/vehicle_appearance recognition, for the same reason:
    zero extra cost beyond what already ran.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class EnrichmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # How many frames per second of derived-clip video each image-based
    # capability (face/plate/vehicle detection, vehicle-appearance
    # fingerprinting) actually examines -- shared across all of them since
    # they sample from the same clip in the same pass. Lower this to trade
    # detection density for real wall-clock speed on long/high-resolution
    # footage; 1.0 (once per second) is a reasoned default, not empirically
    # tuned -- see docs/limitations.md. Also sets the time-window grouping
    # tolerance for `enrichment.encounters` (twice this interval).
    frame_sample_rate_hz: float = Field(default=1.0, gt=0.0)
    face: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    plate: PlateRecognitionConfig = Field(default_factory=PlateRecognitionConfig)
    voice: VoiceRecognitionConfig = Field(default_factory=VoiceRecognitionConfig)
    vehicle_appearance: VehicleAppearanceConfig = Field(default_factory=VehicleAppearanceConfig)
    person_appearance: PersonAppearanceConfig = Field(default_factory=PersonAppearanceConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    cloud: CloudEnrichmentConfig = Field(default_factory=CloudEnrichmentConfig)
    encounters: EncounterConfig = Field(default_factory=EncounterConfig)


class SigningConfig(BaseModel):
    """Ed25519 signing of the event revision hash chain (`core/signing.py`).

    Off by default. A storage/integrity concern, not an enrichment stage --
    lives at the top `RuntimeConfig` level rather than nested under
    `enrichment`. Turning this on does not itself generate a key: run
    `workspace signing-init` once first, or every subsequent revision
    write will raise a clear error rather than silently skip signing.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class LifecycleConfig(BaseModel):
    """Storage lifecycle: triage classification and deletion workflow."""

    model_config = ConfigDict(extra="forbid")

    auto_triage_after_analyze: bool = True
    benign_requires_zero_signals: bool = True


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_profile: str = "default"
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    signing: SigningConfig = Field(default_factory=SigningConfig)


def load_config(path: Path | None = None) -> RuntimeConfig:
    if path is None:
        config = RuntimeConfig()
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        profile_name = payload.get("active_profile", "default")
        profiles = payload.get("profiles", {})
        profile_payload = profiles.get(profile_name, {})
        config = RuntimeConfig(active_profile=profile_name, **profile_payload)
    config = _apply_environment_overrides(config)
    _warn_if_api_key_env_var_looks_like_a_secret(config)
    return config


def _warn_if_api_key_env_var_looks_like_a_secret(config: RuntimeConfig) -> None:
    """`enrichment.cloud.api_key_env_var` must hold the *name* of an
    environment variable (e.g. `DASHCAM_SENTINEL_LLM_API_KEY`) that the
    real API key is set in separately -- never the key itself. A real,
    observed mistake: pasting the actual key value into this field
    directly. That doesn't just fail silently (`os.environ.get()` on a
    key-shaped string finds nothing, so cloud analysis is skipped) -- it
    also means the real secret sits in plaintext in a config file (which
    could end up committed to version control) and gets echoed into logs
    every time the lookup is attempted
    (`llm_analysis_skipped_no_api_key`'s `env_var` field). Real
    environment variable names are conventionally `[A-Z_][A-Z0-9_]*`; a
    pasted key (lowercase letters, hyphens, unusual length) reliably
    fails that shape. This is a warning, not a hard validation error --
    an unconventional but technically-valid lowercase env var name would
    still work, just look unusual.
    """

    value = config.enrichment.cloud.api_key_env_var
    if not _ENV_VAR_NAME_PATTERN.match(value):
        LOGGER.warning(
            "api_key_env_var_looks_like_a_secret_not_a_variable_name",
            message=(
                "enrichment.cloud.api_key_env_var should be the NAME of an environment "
                "variable (e.g. DASHCAM_SENTINEL_LLM_API_KEY), not an API key itself. "
                "The value currently configured doesn't look like a variable name -- if "
                "it's actually your real API key, it is sitting in plaintext in a config "
                "file and has likely already been printed to your terminal/logs. Move the "
                "real key into an actual environment variable, put that variable's NAME "
                "here instead, and rotate the exposed key with your provider."
            ),
        )


def _apply_environment_overrides(config: RuntimeConfig) -> RuntimeConfig:
    payload = config.model_dump()
    prefix = "DASHCAM_SENTINEL__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix) :].lower().split("__")
        _nested_set(payload, path, _coerce_env_value(value))
    return RuntimeConfig.model_validate(payload)


def _nested_set(payload: dict[str, Any], path: list[str], value: Any) -> None:
    cursor: dict[str, Any] = payload
    for segment in path[:-1]:
        next_value = cursor.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[segment] = next_value
        cursor = next_value
    cursor[path[-1]] = value


def _coerce_env_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
