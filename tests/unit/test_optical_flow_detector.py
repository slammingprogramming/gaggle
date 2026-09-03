from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs
from gaggle.detection.optical_flow import OpticalFlowDetector
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import EventWindow, MediaClip, NormalizedClip
from gaggle.utils.ids import new_uuid
from gaggle.utils.time import utc_now

CLIP_START = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)
CLIP_END = CLIP_START + timedelta(seconds=2)

WIDTH, HEIGHT = 320, 240
FPS = 10.0
CENTER = (WIDTH // 2, HEIGHT // 2)


def _growing_disc_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i in range(15):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, CENTER, 20 + 8 * i, (255, 255, 255), thickness=-1)
        writer.write(frame)
    writer.release()


def _clip(
    sidecar_artifacts: list[ArtifactReference],
    stored_path: str = "stored.mp4",
    media_type: str = "video",
) -> MediaClip:
    return MediaClip(
        clip_id=new_uuid(),
        camera_id="front",
        source_path="source.mp4",
        stored_path=stored_path,
        filename="stored.mp4",
        media_type=media_type,  # type: ignore[arg-type]
        byte_size=1024,
        sha256="0" * 64,
        observed_start=CLIP_START,
        observed_end=CLIP_END,
        original_timestamp_source="filename",
        timestamp_confidence=1.0,
        duration_seconds=2.0,
        sidecar_artifacts=sidecar_artifacts,
    )


def _normalized(clip: MediaClip) -> NormalizedClip:
    return NormalizedClip(
        clip=clip,
        session_id="session",
        corrected_start=CLIP_START,
        corrected_end=CLIP_END,
        sync_confidence=1.0,
        sync_rationale="test",
    )


def _window() -> EventWindow:
    return EventWindow(
        window_id=new_uuid(),
        start=CLIP_START,
        end=CLIP_END + timedelta(seconds=5),
        involved_cameras=["front"],
        clip_ids=[],
        rationale="test",
    )


def test_detect_uses_sidecar_fixture_when_present(tmp_path: Path) -> None:
    fixture_path = tmp_path / "clip.mp4.samples.json"
    fixture_path.write_text(
        json.dumps(
            {
                "optical_flow_events": [
                    {
                        "offset_seconds": 1.0,
                        "confidence": 0.8,
                        "roi_divergence": 0.05,
                        "global_divergence": 0.01,
                        "baseline_global_divergence": 0.01,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    clip = _clip(
        [
            ArtifactReference(
                path=str(fixture_path),
                artifact_type="sample_metrics",
                created_at=utc_now(),
                sha256="0" * 64,
            )
        ]
    )
    normalized = _normalized(clip)
    config = RuntimeConfig()
    inputs = DetectionInputs(
        workspace_root=tmp_path, windows=[_window()], clips=[normalized], config=config
    )

    signals = OpticalFlowDetector(config).detect(inputs)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "rapid_approach"
    assert signal.camera_id == "front"
    assert signal.timestamp_start == CLIP_START + timedelta(seconds=1.0)
    assert signal.confidence == 0.8
    assert signal.reasoning_metadata["evidence_source"] == "sidecar_fixture"
    assert signal.reasoning_metadata["roi_divergence"] == 0.05


def test_detect_analyzes_a_real_growing_disc_video(tmp_path: Path) -> None:
    video_path = tmp_path / "growing.mp4"
    _growing_disc_video(video_path)
    clip = _clip([], stored_path=str(video_path))
    normalized = _normalized(clip)
    config = RuntimeConfig()
    config.detection.use_fixture_signals_when_available = False
    config.detection.optical_flow.sample_rate_hz = FPS
    window = EventWindow(
        window_id=new_uuid(),
        start=CLIP_START,
        end=CLIP_START + timedelta(seconds=10),
        involved_cameras=["front"],
        clip_ids=[],
        rationale="test",
    )
    inputs = DetectionInputs(
        workspace_root=tmp_path, windows=[window], clips=[normalized], config=config
    )

    signals = OpticalFlowDetector(config).detect(inputs)

    assert len(signals) >= 1
    for signal in signals:
        assert signal.signal_type == "rapid_approach"
        assert signal.reasoning_metadata["evidence_source"] == "computed"
        assert 0.0 <= signal.confidence <= 1.0


def test_detect_produces_nothing_for_a_non_video_clip(tmp_path: Path) -> None:
    clip = _clip([], media_type="audio")
    normalized = _normalized(clip)
    config = RuntimeConfig()
    config.detection.use_fixture_signals_when_available = False
    inputs = DetectionInputs(
        workspace_root=tmp_path, windows=[_window()], clips=[normalized], config=config
    )

    signals = OpticalFlowDetector(config).detect(inputs)

    assert signals == []


def test_detect_drops_events_outside_any_window(tmp_path: Path) -> None:
    fixture_path = tmp_path / "clip.mp4.samples.json"
    fixture_path.write_text(
        json.dumps(
            {
                "optical_flow_events": [
                    {
                        "offset_seconds": 1.0,
                        "confidence": 0.8,
                        "roi_divergence": 0.05,
                        "global_divergence": 0.01,
                        "baseline_global_divergence": 0.01,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    clip = _clip(
        [
            ArtifactReference(
                path=str(fixture_path),
                artifact_type="sample_metrics",
                created_at=utc_now(),
                sha256="0" * 64,
            )
        ]
    )
    normalized = _normalized(clip)
    config = RuntimeConfig()
    tiny_window = EventWindow(
        window_id=new_uuid(),
        start=CLIP_START,
        end=CLIP_START,
        involved_cameras=["front"],
        clip_ids=[],
        rationale="test",
    )
    inputs = DetectionInputs(
        workspace_root=tmp_path, windows=[tiny_window], clips=[normalized], config=config
    )

    signals = OpticalFlowDetector(config).detect(inputs)

    assert signals == []
