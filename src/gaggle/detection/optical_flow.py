from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs, Detector, match_window_id
from gaggle.detection.optical_flow_analysis import (
    OpticalFlowAnalysisError,
    RapidApproachEvent,
    analyze_optical_flow,
    detect_rapid_approach_events,
)
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import NormalizedClip
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class OpticalFlowDetector(Detector):
    """Dense-optical-flow "rapid approach" (looming) detector.

    Mirrors `motion.py`/`telemetry.py`'s exact sidecar-then-real-analysis
    structure: prefers a precomputed `optical_flow_events` sidecar when
    present (deterministic fixtures), otherwise analyzes the real clip via
    `detection/optical_flow_analysis.py`. See that module's docstring for
    the full design rationale (ego-motion rejection via a comparative,
    not absolute, threshold) and the empirical threshold measurement.
    """

    name = "builtin.optical_flow"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        signals: list[Signal] = []
        for normalized_clip in inputs.clips:
            events, source = _load_rapid_approach_events(normalized_clip, self.config)
            for event in events:
                start = normalized_clip.corrected_start + timedelta(seconds=event.offset_seconds)
                end = start + timedelta(seconds=self.config.detection.min_signal_duration_seconds)
                window_id = match_window_id(inputs.windows, normalized_clip.camera_id, start, end)
                if window_id is None:
                    continue
                signals.append(
                    Signal(
                        id=new_uuid(),
                        source=self.name,
                        signal_type="rapid_approach",
                        timestamp_start=start,
                        timestamp_end=end,
                        confidence=min(1.0, float(event.confidence)),
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
                            "roi_divergence": event.roi_divergence,
                            "global_divergence": event.global_divergence,
                            "baseline_global_divergence": event.baseline_global_divergence,
                        },
                    )
                )
        LOGGER.info("optical_flow_detection_completed", signal_count=len(signals))
        return signals


def _load_rapid_approach_events(
    normalized_clip: NormalizedClip, config: RuntimeConfig
) -> tuple[list[RapidApproachEvent], str]:
    if config.detection.use_fixture_signals_when_available:
        fixture = _load_sidecar_rapid_approach_events(normalized_clip)
        if fixture is not None:
            return fixture, "sidecar_fixture"
    if normalized_clip.clip.media_type != "video":
        return [], "not_video"
    optical_flow_config = config.detection.optical_flow
    try:
        result = analyze_optical_flow(
            Path(normalized_clip.stored_path),
            sample_rate_hz=optical_flow_config.sample_rate_hz,
        )
    except OpticalFlowAnalysisError as error:
        LOGGER.warning(
            "optical_flow_analysis_failed",
            clip_id=str(normalized_clip.clip_id),
            reason=str(error),
        )
        return [], "analysis_failed"
    events = detect_rapid_approach_events(
        result.divergence_series,
        roi_divergence_delta_threshold=optical_flow_config.roi_divergence_delta_threshold,
    )
    return events, "computed"


def _load_sidecar_rapid_approach_events(
    normalized_clip: NormalizedClip,
) -> list[RapidApproachEvent] | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type != "sample_metrics":
            continue
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        events = payload.get("optical_flow_events", [])
        return [
            RapidApproachEvent(
                offset_seconds=float(item["offset_seconds"]),
                confidence=float(item["confidence"]),
                roi_divergence=float(item["roi_divergence"]),
                global_divergence=float(item["global_divergence"]),
                baseline_global_divergence=float(item["baseline_global_divergence"]),
            )
            for item in events
        ]
    return None
