from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs, Detector, match_window_id
from gaggle.detection.gunshot_analysis import (
    GunshotDetectionError,
    analyze_gunshot_events,
    ensure_gunshot_model,
    load_tagger,
    sherpa_onnx_available,
)
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.signal import Signal
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class GunshotDetector(Detector):
    """Local ONNX audio-classifier gunshot/gunfire detector.

    Off unless `detection.gunshot.enabled: true` -- see
    `detection/gunshot_analysis.py`'s module docstring for the full
    picture (model choice, license, honesty caveats). Degrades gracefully
    (no signals, logged once) if the optional `sherpa-onnx` dependency
    isn't installed or the model can't be downloaded, the same pattern
    `enrichment/face_auraface.py`/`enrichment/transcription.py` use for
    their own optional dependencies -- never a hard failure of `analyze`.
    """

    name = "builtin.gunshot"
    version = "1.0.0"

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._tagger: Any | None = None
        self._load_attempted = False

    def detect(self, inputs: DetectionInputs) -> list[Signal]:
        if not self.config.detection.gunshot.enabled:
            return []
        tagger = self._ensure_tagger()
        if tagger is None:
            return []

        signals: list[Signal] = []
        gunshot_config = self.config.detection.gunshot
        for normalized_clip in inputs.clips:
            if normalized_clip.clip.media_type not in ("video", "audio"):
                continue
            try:
                result = analyze_gunshot_events(
                    Path(normalized_clip.stored_path),
                    tagger,
                    window_seconds=gunshot_config.window_seconds,
                    hop_seconds=gunshot_config.hop_seconds,
                    confidence_threshold=gunshot_config.confidence_threshold,
                    timeout_seconds=self.config.detection.audio_extraction_timeout_seconds,
                )
            except GunshotDetectionError as error:
                LOGGER.warning(
                    "gunshot_analysis_failed",
                    clip_id=str(normalized_clip.clip_id),
                    reason=str(error),
                )
                continue
            if not result.has_audio:
                continue
            for event in result.events:
                start = normalized_clip.corrected_start + timedelta(seconds=event.offset_seconds)
                end = start + timedelta(
                    seconds=max(
                        event.duration_seconds, self.config.detection.min_signal_duration_seconds
                    )
                )
                window_id = match_window_id(inputs.windows, normalized_clip.camera_id, start, end)
                if window_id is None:
                    continue
                signals.append(
                    Signal(
                        id=new_uuid(),
                        source=self.name,
                        signal_type="gunshot",
                        timestamp_start=start,
                        timestamp_end=end,
                        confidence=event.confidence,
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
                            "class_name": event.class_name,
                        },
                    )
                )
        LOGGER.info("gunshot_detection_completed", signal_count=len(signals))
        return signals

    def _ensure_tagger(self) -> Any | None:
        if self._tagger is not None:
            return self._tagger
        if self._load_attempted:
            return None
        self._load_attempted = True
        if not sherpa_onnx_available():
            LOGGER.warning(
                "gunshot_detection_unavailable",
                reason="sherpa_onnx is not installed (the 'gunshot' extra)",
            )
            return None
        try:
            model_path, labels_path = ensure_gunshot_model()
            self._tagger = load_tagger(model_path, labels_path)
        except GunshotDetectionError as error:
            LOGGER.warning("gunshot_model_unavailable", reason=str(error))
            return None
        return self._tagger
