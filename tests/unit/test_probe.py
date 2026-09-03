from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.ingest.probe import ProbeError, probe_media

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not available")


def _make_clip(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=15:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def test_probe_extracts_real_metadata(tmp_path: Path) -> None:
    clip_path = tmp_path / "sample.mp4"
    _make_clip(clip_path)
    result = probe_media(clip_path)
    assert result.duration_seconds == pytest.approx(5.0, abs=0.2)
    assert result.fps == pytest.approx(15.0, abs=0.1)
    assert result.width == 320
    assert result.height == 240
    assert result.video_codec == "h264"
    assert result.has_audio is True
    assert result.probe_tool == "ffprobe"


def test_probe_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        probe_media(tmp_path / "does-not-exist.mp4")
