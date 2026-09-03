from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.core.derived_clips import ClipExtractionError, extract_clip_segment
from gaggle.ingest.probe import probe_media

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _make_clip(path: Path, duration: int = 10) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=15:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_extract_clip_segment_produces_shorter_clip(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "segment.mp4"
    _make_clip(source, duration=10)
    extract_clip_segment(source, destination, start_offset_seconds=2.0, end_offset_seconds=5.0)
    assert destination.exists()
    result = probe_media(destination)
    # stream-copy cuts to the nearest keyframe, so the segment may be
    # slightly longer than requested but must not be drastically different.
    assert 2.5 <= result.duration_seconds <= 6.0


def test_extract_clip_segment_rejects_invalid_range(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make_clip(source, duration=5)
    with pytest.raises(ValueError):
        extract_clip_segment(
            source, tmp_path / "out.mp4", start_offset_seconds=3.0, end_offset_seconds=1.0
        )


def test_extract_clip_segment_raises_on_bad_source(tmp_path: Path) -> None:
    with pytest.raises(ClipExtractionError):
        extract_clip_segment(
            tmp_path / "missing.mp4",
            tmp_path / "out.mp4",
            start_offset_seconds=0.0,
            end_offset_seconds=1.0,
        )
