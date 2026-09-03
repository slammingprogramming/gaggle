"""Real, executed tests against synthetic frame sequences with a known,
unambiguous flow pattern -- no ffmpeg dependency (unlike
`test_video_analysis.py`, which shells out to ffmpeg's lavfi sources),
since `cv2.VideoWriter` gives direct frame-level control that's exactly
what's needed here."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gaggle.detection.optical_flow_analysis import (
    OpticalFlowAnalysisError,
    analyze_optical_flow,
    detect_rapid_approach_events,
)

WIDTH, HEIGHT = 320, 240
FPS = 10.0
CENTER = (WIDTH // 2, HEIGHT // 2)


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _growing_disc_frames(radii: list[int]) -> list[np.ndarray]:
    frames = []
    for radius in radii:
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, CENTER, radius, (255, 255, 255), thickness=-1)
        frames.append(frame)
    return frames


def _uniform_zoom_frames(scales: list[float]) -> list[np.ndarray]:
    """A textured base frame uniformly magnified about its own center each
    step -- a synthetic stand-in for a forward-driving dashcam's own
    ego-motion (radial expansion everywhere in frame, not localized to
    one region)."""

    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)
    for _ in range(400):
        x, y = rng.integers(0, WIDTH), rng.integers(0, HEIGHT)
        cv2.circle(base, (int(x), int(y)), 3, (255, 255, 255), thickness=-1)

    frames = []
    for scale in scales:
        matrix = cv2.getRotationMatrix2D(CENTER, angle=0.0, scale=scale)
        frames.append(cv2.warpAffine(base, matrix, (WIDTH, HEIGHT)))
    return frames


def test_analyze_optical_flow_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OpticalFlowAnalysisError):
        analyze_optical_flow(tmp_path / "does_not_exist.mp4")


def test_a_growing_disc_produces_positive_roi_divergence_exceeding_the_baseline(
    tmp_path: Path,
) -> None:
    radii = list(range(20, 20 + 15 * 5, 5))  # 20, 25, ..., 90
    video_path = tmp_path / "growing.mp4"
    _write_video(video_path, _growing_disc_frames(radii))

    result = analyze_optical_flow(video_path, sample_rate_hz=FPS)
    assert len(result.divergence_series) >= 5

    # The expanding boundary is localized to the ROI -- roi_divergence
    # should clearly exceed global_divergence for at least some samples
    # (the disc's growth rate, and thus its flow signature, is not
    # perfectly uniform frame to frame with a fixed-size video codec).
    max_delta = max(s.roi_divergence - s.global_divergence for s in result.divergence_series)
    assert max_delta > 0.005

    events = detect_rapid_approach_events(
        result.divergence_series, roi_divergence_delta_threshold=0.005
    )
    assert len(events) >= 1
    for event in events:
        assert 0.0 <= event.confidence <= 1.0


def test_a_shrinking_disc_does_not_fire_a_rapid_approach_event(tmp_path: Path) -> None:
    radii = list(range(90, 90 - 15 * 5, -5))  # 90, 85, ..., 20
    video_path = tmp_path / "shrinking.mp4"
    _write_video(video_path, _growing_disc_frames(radii))

    result = analyze_optical_flow(video_path, sample_rate_hz=FPS)
    events = detect_rapid_approach_events(
        result.divergence_series, roi_divergence_delta_threshold=0.01
    )
    assert events == []


def test_uniform_ego_motion_zoom_does_not_fire_a_rapid_approach_event(tmp_path: Path) -> None:
    """The comparative (delta-from-baseline) threshold is the whole point
    of this module -- a uniform whole-frame expansion (roi and global
    divergence rising together) must NOT be flagged, or every ordinary
    forward-driving clip would false-positive."""

    scales = [1.0 + 0.02 * i for i in range(20)]
    video_path = tmp_path / "zoom.mp4"
    _write_video(video_path, _uniform_zoom_frames(scales))

    result = analyze_optical_flow(video_path, sample_rate_hz=FPS)
    assert len(result.divergence_series) >= 5

    events = detect_rapid_approach_events(
        result.divergence_series, roi_divergence_delta_threshold=0.01
    )
    assert events == []


def test_detect_rapid_approach_events_never_fires_on_the_first_sample() -> None:
    from gaggle.detection.optical_flow_analysis import DivergenceSample

    samples = [DivergenceSample(offset_seconds=0.0, roi_divergence=10.0, global_divergence=0.0)]
    events = detect_rapid_approach_events(samples, roi_divergence_delta_threshold=0.01)
    assert events == []
