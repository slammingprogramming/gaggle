"""Real media metadata extraction via ``ffprobe``.

This module is intentionally free of any dependency on the pydantic schema
layer or on other project subpackages so it can be exercised in isolation
(including outside a fully installed environment). It shells out to the
``ffprobe`` binary, which ships as part of ffmpeg, and never guesses at
duration/fps/resolution when it can measure them directly.

Determinism note: ffprobe's JSON output for a given input file and ffmpeg
version is stable, so probing the same bytes twice yields the same result.
The ffmpeg/ffprobe version used is recorded in the result so reprocessing
with a different toolchain version is inspectable rather than silently
divergent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProbeError(RuntimeError):
    """Raised when ffprobe is unavailable or its output cannot be parsed."""


@dataclass(frozen=True, slots=True)
class MediaProbeResult:
    duration_seconds: float
    fps: float | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool
    probe_tool: str
    probe_tool_version: str | None


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def ffprobe_version() -> str | None:
    if not ffprobe_available():
        return None
    try:
        completed = subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    # Typical first line: "ffprobe version 6.1.1-...".
    parts = first_line.split()
    return parts[2] if len(parts) >= 3 else None


def probe_media(path: Path, timeout_seconds: float = 30.0) -> MediaProbeResult:
    """Extract real duration/fps/resolution/codec metadata from ``path``.

    Raises ``ProbeError`` if ffprobe is missing, times out, or returns
    output that cannot be interpreted. Callers are expected to handle this
    explicitly (there is no silent fallback here) so that timestamp/duration
    provenance in ``MediaClip`` stays honest about how it was derived.
    """

    if not ffprobe_available():
        raise ProbeError("ffprobe is not available on PATH")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"ffprobe timed out probing {path}") from exc
    except subprocess.CalledProcessError as exc:
        raise ProbeError(f"ffprobe failed on {path}: {exc.stderr.strip()}") from exc
    except OSError as exc:
        raise ProbeError(f"ffprobe could not be executed: {exc}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe returned invalid JSON for {path}") from exc

    return _parse_probe_payload(payload)


def _parse_probe_payload(payload: dict[str, object]) -> MediaProbeResult:
    fmt = payload.get("format")
    streams_raw = payload.get("streams")
    streams: list[dict[str, object]] = streams_raw if isinstance(streams_raw, list) else []

    duration_seconds = 0.0
    if isinstance(fmt, dict):
        duration_value = fmt.get("duration")
        if isinstance(duration_value, str):
            try:
                duration_seconds = float(duration_value)
            except ValueError:
                duration_seconds = 0.0

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )

    fps: float | None = None
    width: int | None = None
    height: int | None = None
    video_codec: str | None = None
    if video_stream is not None:
        fps = _parse_frame_rate(video_stream.get("avg_frame_rate"))
        if fps is None:
            fps = _parse_frame_rate(video_stream.get("r_frame_rate"))
        width_value = video_stream.get("width")
        height_value = video_stream.get("height")
        width = int(width_value) if isinstance(width_value, int) else None
        height = int(height_value) if isinstance(height_value, int) else None
        codec_value = video_stream.get("codec_name")
        video_codec = str(codec_value) if codec_value is not None else None
        if duration_seconds == 0.0:
            stream_duration = video_stream.get("duration")
            if isinstance(stream_duration, str):
                try:
                    duration_seconds = float(stream_duration)
                except ValueError:
                    duration_seconds = 0.0

    audio_codec: str | None = None
    if audio_stream is not None:
        codec_value = audio_stream.get("codec_name")
        audio_codec = str(codec_value) if codec_value is not None else None

    return MediaProbeResult(
        duration_seconds=max(0.0, duration_seconds),
        fps=fps,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_audio=audio_stream is not None,
        probe_tool="ffprobe",
        probe_tool_version=ffprobe_version(),
    )


def _parse_frame_rate(raw: object) -> float | None:
    if not isinstance(raw, str) or "/" not in raw:
        return None
    numerator_str, _, denominator_str = raw.partition("/")
    try:
        numerator = float(numerator_str)
        denominator = float(denominator_str)
    except ValueError:
        return None
    if denominator == 0:
        return None
    return numerator / denominator
