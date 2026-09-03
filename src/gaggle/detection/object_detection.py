from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

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

#: Label used for the built-in heuristic blob detector, deliberately not a
#: named object class (car/person/etc). This detector draws bounding boxes
#: around large contiguous moving regions via contour analysis on the same
#: frame-difference mask the motion detector uses -- it does not classify
#: what moved. Real classification is an intentional extension point for a
#: ``DetectorPlugin`` (e.g. wrapping a local ONNX/YOLO model); mixing that in
#: here would violate the project's avoid-ML-first-design directive for the
#: built-in path.
UNCLASSIFIED_REGION_LABEL = "unclassified_moving_region"


class MetadataObjectDetector(Detector):
    """Object-hint detector: sidecar-provided hints, or a heuristic fallback.

    When a ``object_hints`` sidecar is present (e.g. hand-authored fixtures,
    or hints copied in from an external tool/telemetry feed) those are used
    verbatim, including their semantic labels and confidences. Otherwise,
    the built-in fallback reports large moving-region bounding boxes derived
    from contour analysis -- an explainable heuristic "something big moved
    here" signal, not an object classification. This keeps the abstraction
    point open for real ML object detectors to be registered as plugins
    without changing the ``Signal`` shape they must produce.
    """

    name = "builtin.object_hints"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        signals: list[Signal] = []
        config = self.config or inputs.config
        for normalized_clip in inputs.clips:
            hints, source = _load_object_hints(normalized_clip, config)
            for hint in hints:
                start = normalized_clip.corrected_start + timedelta(
                    seconds=float(hint["start_offset_seconds"])
                )
                end = normalized_clip.corrected_start + timedelta(
                    seconds=float(hint["end_offset_seconds"])
                )
                window_id = match_window_id(inputs.windows, normalized_clip.camera_id, start, end)
                if window_id is None:
                    continue
                signals.append(
                    Signal(
                        id=new_uuid(),
                        source=self.name,
                        signal_type="object_hint",
                        timestamp_start=start,
                        timestamp_end=end,
                        confidence=min(1.0, float(hint["confidence"])),
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
                        spatial_metadata={"bbox": hint.get("bbox")},
                        reasoning_metadata={
                            "label": str(hint["label"]),
                            "detector_version": self.version,
                            "evidence_source": source,
                        },
                    )
                )
        LOGGER.info("object_detection_completed", signal_count=len(signals))
        return signals


def _load_object_hints(
    normalized_clip: NormalizedClip, config: RuntimeConfig
) -> tuple[list[dict[str, Any]], str]:
    if config.detection.use_fixture_signals_when_available:
        fixture = _load_sidecar_object_hints(normalized_clip)
        if fixture is not None:
            return fixture, "sidecar_fixture"
    if normalized_clip.clip.media_type != "video":
        return [], "not_video"
    try:
        result = analyze_video_motion(
            Path(normalized_clip.stored_path),
            sample_rate_hz=config.detection.motion_sample_rate_hz,
            min_region_area_ratio=config.detection.min_object_region_area_ratio,
        )
    except VideoAnalysisError as error:
        LOGGER.warning(
            "object_region_analysis_failed",
            clip_id=str(normalized_clip.clip_id),
            reason=str(error),
        )
        return [], "analysis_failed"
    hints = [
        {
            "label": UNCLASSIFIED_REGION_LABEL,
            "confidence": region.confidence,
            "start_offset_seconds": region.offset_seconds,
            "end_offset_seconds": region.end_offset_seconds,
            "bbox": list(region.bbox),
        }
        for region in result.regions
    ]
    return hints, "computed"


def _load_sidecar_object_hints(
    normalized_clip: NormalizedClip,
) -> list[dict[str, Any]] | None:
    for artifact in normalized_clip.clip.sidecar_artifacts:
        if artifact.artifact_type != "sample_metrics":
            continue
        payload = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        raw_items = payload.get("object_hints", [])
        return [dict(item) for item in raw_items]
    return None
