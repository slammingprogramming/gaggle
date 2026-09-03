from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.detection.video_analysis import VideoAnalysisError, analyze_video_motion

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _make_static_then_animated_video(path: Path) -> None:
    """4s static gray, then 4s of an animated test pattern."""

    static_path = path.with_name("static.mp4")
    animated_path = path.with_name("animated.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:rate=15:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(static_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(animated_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(static_path),
            "-i",
            str(animated_path),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[outv]",
            "-map",
            "[outv]",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_motion_analysis_is_zero_during_static_segment(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_static_then_animated_video(video_path)
    result = analyze_video_motion(video_path, sample_rate_hz=2.0)
    early_samples = [s for s in result.motion_series if s.offset_seconds < 3.5]
    assert early_samples
    assert all(s.value == 0.0 for s in early_samples)


def test_motion_analysis_detects_activity_in_animated_segment(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_static_then_animated_video(video_path)
    result = analyze_video_motion(video_path, sample_rate_hz=2.0)
    later_samples = [s for s in result.motion_series if s.offset_seconds > 4.5]
    assert later_samples
    assert any(s.value > 0.02 for s in later_samples)
    assert result.regions


def test_motion_analysis_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoAnalysisError):
        analyze_video_motion(tmp_path / "does-not-exist.mp4")


def test_motion_analysis_is_deterministic(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    _make_static_then_animated_video(video_path)
    first = analyze_video_motion(video_path, sample_rate_hz=2.0)
    second = analyze_video_motion(video_path, sample_rate_hz=2.0)
    assert [s.value for s in first.motion_series] == [s.value for s in second.motion_series]
