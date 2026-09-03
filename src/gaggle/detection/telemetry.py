from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs, Detector, match_window_id
from gaggle.detection.telemetry_analysis import (
    TelemetryAnalysisError,
    TelemetryEvent,
    compute_telemetry_samples,
    detect_telemetry_events,
    parse_gpx,
)
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import NormalizedClip
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class TelemetryDetector(Detector):
    """GPS-track-derived telemetry event detector (hard braking, speed
    spikes, sudden heading changes).

    Mirrors ``MotionDetector``/``AudioSpikeDetector``'s exact
    sidecar-then-real-analysis structure: prefers a precomputed
    ``telemetry_events`` sidecar when present (deterministic fixtures),
    otherwise looks for a real GPX track associated with the clip (a
    ``gps_track``-type sidecar artifact copied in by
    ``ingest/service.py``) and analyzes it for real via
    ``detection/telemetry_analysis.py``. A clip with no associated GPX
    track produces no signals -- missing telemetry is a normal outcome,
    not a detection failure, exactly like a clip with no audio stream.
    """

    name = "builtin.telemetry"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        signals: list[Signal] = []
        for normalized_clip in inputs.clips:
            events, source = _load_telemetry_events(normalized_clip, self.config)
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
                        signal_type="telemetry",
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
                            "event_type": event.event_type,
                            "value": event.value,
                        },
                    )
                )
        LOGGER.info("telemetry_detection_completed", signal_count=len(signals))
        return signals


def _load_telemetry_events(
    normalized_clip: NormalizedClip, config: RuntimeConfig
) -> tuple[list[TelemetryEvent], str]:
    if config.detection.use_fixture_signals_when_available:
        fixture = _load_sidecar_telemetry_events(normalized_clip)
        if fixture is not None:
            return fixture, "sidecar_fixture"

    gpx_path = _find_gpx_sidecar(normalized_clip)
    if gpx_path is None:
        return [], "no_gps_track"
    try:
        points = parse_gpx(gpx_path)
    except TelemetryAnalysisError as error:
        LOGGER.warning(
            "telemetry_gpx_parse_failed", clip_id=str(normalized_clip.clip_id), reason=str(error)
        )
        return [], "gpx_parse_failed"

    # One GPX track can cover an entire ingest session spanning many
    # clips -- restrict to the portion overlapping this clip's own
    # (uncorrected) observed time range.
    clip_start = normalized_clip.clip.observed_start
    clip_end = normalized_clip.clip.observed_end
    relevant_points = [point for point in points if clip_start <= point.time <= clip_end]
    if len(relevant_points) < 2:
        return [], "no_overlapping_track_points"

    try:
        samples = compute_telemetry_samples(relevant_points)
    except TelemetryAnalysisError:
        return [], "insufficient_samples"

    telemetry_config = config.detection.telemetry
    events = detect_telemetry_events(
        samples,
        hard_braking_threshold_mps2=telemetry_config.hard_braking_threshold_mps2,
        speed_spike_threshold_mps=telemetry_config.speed_spike_threshold_mps,
        heading_change_threshold_deg_per_sec=telemetry_config.heading_change_threshold_deg_per_sec,
    )

    # Event offsets are relative to the *track's* first relevant point,
    # but Signal timestamps need to be relative to the clip's own start
    # (matching motion.py/audio.py's convention) -- rebase.
    clip_offset = (relevant_points[0].time - clip_start).total_seconds()
    rebased = [
        TelemetryEvent(
            event_type=event.event_type,
            offset_seconds=event.offset_seconds + clip_offset,
            confidence=event.confidence,
            value=event.value,
        )
        for event in events
    ]
    return rebased, "computed"


def _load_sidecar_telemetry_events(
    normalized_clip: NormalizedClip,
) -> list[TelemetryEvent] | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type != "sample_metrics":
            continue
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        events = payload.get("telemetry_events", [])
        return [
            TelemetryEvent(
                event_type=str(item["event_type"]),
                offset_seconds=float(item["offset_seconds"]),
                confidence=float(item["confidence"]),
                value=float(item["value"]),
            )
            for item in events
        ]
    return None


def _find_gpx_sidecar(normalized_clip: NormalizedClip) -> Path | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type == "gps_track":
            path = Path(artifact.path)
            if path.exists():
                return path
    return None
