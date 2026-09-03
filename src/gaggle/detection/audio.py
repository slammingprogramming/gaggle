from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig
from gaggle.detection.audio_analysis import AudioAnalysisError, analyze_audio_spikes
from gaggle.detection.base import DetectionInputs, Detector, match_window_id
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import NormalizedClip
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class AudioSpikeDetector(Detector):
    """RMS-envelope audio spike detector.

    Like ``MotionDetector``, this prefers a precomputed ``audio_rms_series``
    sidecar when present and otherwise extracts and analyzes the real audio
    track via ``gaggle.detection.audio_analysis``. Clips with no
    audio stream produce no signals -- silence is a normal outcome, not a
    detection failure.
    """

    name = "builtin.audio_spike"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        signals: list[Signal] = []
        for normalized_clip in inputs.clips:
            series, source = _load_audio_series(normalized_clip, self.config)
            for offset_seconds, value in series:
                if value < self.config.detection.audio_spike_threshold:
                    continue
                start = normalized_clip.corrected_start + timedelta(seconds=offset_seconds)
                end = start + timedelta(seconds=self.config.detection.min_signal_duration_seconds)
                window_id = match_window_id(inputs.windows, normalized_clip.camera_id, start, end)
                if window_id is None:
                    continue
                signals.append(
                    Signal(
                        id=new_uuid(),
                        source=self.name,
                        signal_type="audio_spike",
                        timestamp_start=start,
                        timestamp_end=end,
                        confidence=min(1.0, float(value)),
                        camera_id=normalized_clip.camera_id,
                        window_id=window_id,
                        evidence_references=[
                            ArtifactReference(
                                path=normalized_clip.stored_path,
                                artifact_type="source_media",
                                created_at=utc_now(),
                                sha256=normalized_clip.sha256,
                            )
                        ],
                        reasoning_metadata={
                            "detector_version": self.version,
                            "evidence_source": source,
                        },
                    )
                )
        LOGGER.info("audio_detection_completed", signal_count=len(signals))
        return signals


def _load_audio_series(
    normalized_clip: NormalizedClip, config: RuntimeConfig
) -> tuple[list[tuple[float, float]], str]:
    if config.detection.use_fixture_signals_when_available:
        fixture = _load_sidecar_audio_series(normalized_clip)
        if fixture is not None:
            return fixture, "sidecar_fixture"
    if normalized_clip.clip.media_type not in ("video", "audio"):
        return [], "unsupported_media_type"
    try:
        result = analyze_audio_spikes(
            Path(normalized_clip.stored_path),
            window_seconds=config.detection.audio_window_seconds,
            timeout_seconds=config.detection.audio_extraction_timeout_seconds,
        )
    except AudioAnalysisError as error:
        LOGGER.warning(
            "audio_analysis_failed",
            clip_id=str(normalized_clip.clip_id),
            reason=str(error),
        )
        return [], "analysis_failed"
    if not result.has_audio:
        return [], "no_audio_stream"
    return [(sample.offset_seconds, sample.value) for sample in result.samples], "computed"


def _load_sidecar_audio_series(
    normalized_clip: NormalizedClip,
) -> list[tuple[float, float]] | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type != "sample_metrics":
            continue
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        series = payload.get("audio_rms_series", [])
        return [(float(item["offset_seconds"]), float(item["value"])) for item in series]
    return None
