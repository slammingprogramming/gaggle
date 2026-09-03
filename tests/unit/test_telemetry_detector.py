from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gaggle.core.config import RuntimeConfig
from gaggle.detection.base import DetectionInputs
from gaggle.detection.telemetry import TelemetryDetector
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import EventWindow, MediaClip, NormalizedClip
from gaggle.utils.ids import new_uuid
from gaggle.utils.time import utc_now

FIXTURE_GPX = Path(__file__).resolve().parents[1] / "fixtures" / "sample_track.gpx"
TRACK_START = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)
TRACK_END = TRACK_START + timedelta(seconds=9)


def _clip(sidecar_artifacts: list[ArtifactReference], camera_id: str = "front") -> MediaClip:
    return MediaClip(
        clip_id=new_uuid(),
        camera_id=camera_id,
        source_path="source.mp4",
        stored_path="stored.mp4",
        filename="stored.mp4",
        media_type="video",
        byte_size=1024,
        sha256="0" * 64,
        observed_start=TRACK_START,
        observed_end=TRACK_END,
        original_timestamp_source="filename",
        timestamp_confidence=1.0,
        duration_seconds=9.0,
        sidecar_artifacts=sidecar_artifacts,
    )


def _normalized(clip: MediaClip) -> NormalizedClip:
    return NormalizedClip(
        clip=clip,
        session_id="session",
        corrected_start=TRACK_START,
        corrected_end=TRACK_END,
        sync_confidence=1.0,
        sync_rationale="test",
    )


def _window(camera_id: str = "front") -> EventWindow:
    return EventWindow(
        window_id=new_uuid(),
        start=TRACK_START,
        end=TRACK_END + timedelta(seconds=5),
        involved_cameras=[camera_id],
        clip_ids=[],
        rationale="test",
    )


def test_detect_uses_sidecar_fixture_when_present(tmp_path: Path) -> None:
    fixture_path = tmp_path / "clip.mp4.samples.json"
    fixture_path.write_text(
        json.dumps(
            {
                "telemetry_events": [
                    {
                        "event_type": "hard_braking",
                        "offset_seconds": 2.0,
                        "confidence": 0.9,
                        "value": -5.5,
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

    signals = TelemetryDetector(config).detect(inputs)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.signal_type == "telemetry"
    assert signal.camera_id == "front"
    assert signal.timestamp_start == TRACK_START + timedelta(seconds=2.0)
    assert signal.reasoning_metadata["evidence_source"] == "sidecar_fixture"
    assert signal.reasoning_metadata["event_type"] == "hard_braking"


def test_detect_parses_real_gpx_when_no_fixture_present() -> None:
    clip = _clip(
        [
            ArtifactReference(
                path=str(FIXTURE_GPX),
                artifact_type="gps_track",
                created_at=utc_now(),
                sha256="0" * 64,
            )
        ]
    )
    normalized = _normalized(clip)
    config = RuntimeConfig()
    config.detection.use_fixture_signals_when_available = False
    inputs = DetectionInputs(
        workspace_root=Path("."), windows=[_window()], clips=[normalized], config=config
    )

    signals = TelemetryDetector(config).detect(inputs)

    event_types = [s.reasoning_metadata["event_type"] for s in signals]
    assert event_types.count("hard_braking") == 2
    assert event_types.count("sudden_heading_change") == 1
    assert event_types.count("speed_spike") == 1
    assert len(signals) == 4
    for signal in signals:
        assert signal.signal_type == "telemetry"
        assert signal.reasoning_metadata["evidence_source"] == "computed"
        assert TRACK_START <= signal.timestamp_start <= TRACK_END + timedelta(seconds=1)


def test_detect_produces_nothing_without_a_gps_track() -> None:
    clip = _clip([])
    normalized = _normalized(clip)
    config = RuntimeConfig()
    inputs = DetectionInputs(
        workspace_root=Path("."), windows=[_window()], clips=[normalized], config=config
    )

    signals = TelemetryDetector(config).detect(inputs)

    assert signals == []


def test_detect_drops_events_outside_any_window() -> None:
    clip = _clip(
        [
            ArtifactReference(
                path=str(FIXTURE_GPX),
                artifact_type="gps_track",
                created_at=utc_now(),
                sha256="0" * 64,
            )
        ]
    )
    normalized = _normalized(clip)
    config = RuntimeConfig()
    config.detection.use_fixture_signals_when_available = False
    # A window that ends immediately -- no signal's [start, end) interval
    # can fit inside it, so every computed event must be dropped rather
    # than raising or inventing a window for it.
    tiny_window = EventWindow(
        window_id=new_uuid(),
        start=TRACK_START,
        end=TRACK_START,
        involved_cameras=["front"],
        clip_ids=[],
        rationale="test",
    )
    inputs = DetectionInputs(
        workspace_root=Path("."), windows=[tiny_window], clips=[normalized], config=config
    )

    signals = TelemetryDetector(config).detect(inputs)

    assert signals == []
