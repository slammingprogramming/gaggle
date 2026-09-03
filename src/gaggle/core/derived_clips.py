"""Extracts event-relevant derived clips from normalized source media.

Uses ``ffmpeg -c copy`` (stream copy, no re-encode) so extraction is fast
and never alters pixel/sample data -- the derived clip is a byte-exact
subset of the original stream. Because stream copy can only cut on
keyframe boundaries, an extracted segment may run slightly longer than the
requested window; per the project's false-positive philosophy this is
treated as a feature (a little extra footage around an event) rather than
a bug, and is noted in the artifact's metadata.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class ClipExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract the requested segment."""


def extract_clip_segment(
    source_path: Path,
    destination_path: Path,
    start_offset_seconds: float,
    end_offset_seconds: float,
    timeout_seconds: float = 60.0,
) -> None:
    if end_offset_seconds <= start_offset_seconds:
        raise ValueError("end_offset_seconds must be greater than start_offset_seconds")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_offset_seconds - start_offset_seconds
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_offset_seconds:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        str(destination_path),
    ]
    try:
        subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise ClipExtractionError(
            f"ffmpeg failed extracting segment from {source_path}: {stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClipExtractionError(
            f"ffmpeg timed out extracting segment from {source_path}"
        ) from exc
    except OSError as exc:
        raise ClipExtractionError(f"ffmpeg could not be executed: {exc}") from exc
