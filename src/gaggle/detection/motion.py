from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs, Detector, match_window_id
from gaggle.detection.video_analysis import VideoAnalysisError, analyze_video_motion
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import NormalizedClip
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class MotionDetector(Detector):
    """Frame-differencing motion detector.

    Prefers a precomputed ``motion_series`` sidecar when one ships alongside
    the source media (useful for deterministic fixtures and for calibrating
    against a known-good reference series). When no sidecar is present, it
    analyzes the real clip via ``gaggle.detection.video_analysis``
    (grayscale frame differencing -- classic, deterministic, and
    explainable, per the project's ML-avoidance-by-default directive).
    """

    name = "builtin.motion"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        signals: list[Signal] = []
        for normalized_clip in inputs.clips:
            series, source = _load_motion_series(normalized_clip, self.config)
            for offset_seconds, value in series:
                if value < self.config.detection.motion_threshold:
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
                        signal_type="motion",
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
        LOGGER.info("motion_detection_completed", signal_count=len(signals))
        return signals


def _load_motion_series(
    normalized_clip: NormalizedClip, config: RuntimeConfig
) -> tuple[list[tuple[float, float]], str]:
    if config.detection.use_fixture_signals_when_available:
        fixture = _load_sidecar_motion_series(normalized_clip)
        if fixture is not None:
            return fixture, "sidecar_fixture"
    if normalized_clip.clip.media_type != "video":
        return [], "not_video"
    try:
        result = analyze_video_motion(
            Path(normalized_clip.stored_path),
            sample_rate_hz=config.detection.motion_sample_rate_hz,
        )
    except VideoAnalysisError as error:
        LOGGER.warning(
            "motion_analysis_failed",
            clip_id=str(normalized_clip.clip_id),
            reason=str(error),
        )
        return [], "analysis_failed"
    return [(sample.offset_seconds, sample.value) for sample in result.motion_series], "computed"


def _load_sidecar_motion_series(
    normalized_clip: NormalizedClip,
) -> list[tuple[float, float]] | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type != "sample_metrics":
            continue
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        series = payload.get("motion_series", [])
        return [(float(item["offset_seconds"]), float(item["value"])) for item in series]
    return None
