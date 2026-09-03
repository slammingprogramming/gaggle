from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from gaggle.detection.audio_analysis import AudioAnalysisError, analyze_audio_spikes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _make_quiet_loud_quiet_clip(path: Path) -> None:
    quiet1 = path.with_name("quiet1.wav")
    loud = path.with_name("loud.wav")
    quiet2 = path.with_name("quiet2.wav")
    audio = path.with_name("audio.wav")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono:d=3",
            "-c:a",
            "pcm_s16le",
            str(quiet1),
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
            "sine=frequency=1000:sample_rate=16000:duration=1",
            "-af",
            "volume=6.0",
            "-c:a",
            "pcm_s16le",
            str(loud),
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
            "anullsrc=r=16000:cl=mono:d=3",
            "-c:a",
            "pcm_s16le",
            str(quiet2),
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
            str(quiet1),
            "-i",
            str(loud),
            "-i",
            str(quiet2),
            "-filter_complex",
            "[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]",
            "-map",
            "[outa]",
            str(audio),
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
            "color=c=gray:s=320x240:rate=15:duration=7",
            "-i",
            str(audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def test_audio_spike_detected_in_loud_segment(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    _make_quiet_loud_quiet_clip(clip_path)
    result = analyze_audio_spikes(clip_path, window_seconds=0.5)
    assert result.has_audio is True
    loud_samples = [s for s in result.samples if 3.0 <= s.offset_seconds < 4.0]
    quiet_samples = [s for s in result.samples if s.offset_seconds < 2.5]
    assert loud_samples and quiet_samples
    assert max(s.value for s in loud_samples) > 0.4
    assert max(s.value for s in quiet_samples) < 0.05


def test_no_audio_stream_is_not_an_error(tmp_path: Path) -> None:
    clip_path = tmp_path / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=320x240:rate=15:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip_path),
        ],
        check=True,
    )
    result = analyze_audio_spikes(clip_path)
    assert result.has_audio is False
    assert result.samples == []


def test_audio_analysis_raises_on_invalid_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        analyze_audio_spikes(tmp_path / "clip.mp4", window_seconds=0.0)


def test_audio_analysis_raises_when_ffmpeg_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("gaggle.detection.audio_analysis.ffmpeg_available", lambda: False)
    with pytest.raises(AudioAnalysisError):
        analyze_audio_spikes(tmp_path / "clip.mp4")
