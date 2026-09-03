"""Deterministic audio-spike analysis via ffmpeg extraction + scipy.

The project's tech stack allows either ``librosa`` or ``scipy`` for audio
analysis; this module uses ``scipy`` (already a core dependency) together
with the ``ffmpeg`` binary to avoid pulling in a second, heavier audio
library. The approach is intentionally simple and explainable: extract the
audio track to a mono 16 kHz WAV file, compute a rolling RMS envelope, and
report normalized (0-1) energy per window. No spectral/ML feature
extraction is performed, consistent with the project's preference for
heuristics that a human can verify by ear and by eye.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile

# The raw samples read back from a WAV file are whatever PCM dtype the file
# used (commonly int16), not always float -- callers normalize to float64
# themselves (see `extract_normalized_waveform`/`analyze_audio_spikes`).
RawAudioArray = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float64]

ANALYZER_VERSION = "1.0.0"
TARGET_SAMPLE_RATE_HZ = 16_000


class AudioAnalysisError(RuntimeError):
    """Raised when audio cannot be extracted or analyzed."""


@dataclass(frozen=True, slots=True)
class AudioSample:
    offset_seconds: float
    value: float


@dataclass(frozen=True, slots=True)
class AudioAnalysisResult:
    samples: list[AudioSample]
    has_audio: bool
    analyzer_version: str = ANALYZER_VERSION


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_mono_wav(
    path: Path, timeout_seconds: float = 300.0
) -> tuple[int, RawAudioArray] | None:
    """Extract ``path``'s audio track to mono 16kHz PCM and return
    ``(sample_rate, raw_samples)``, or ``None`` if there's no usable audio
    track. Shared by `analyze_audio_spikes` and
    `extract_normalized_waveform` so the ffmpeg invocation lives in
    exactly one place.

    ``timeout_seconds`` defaults generously (300s) rather than a short
    value tuned for a small synthetic clip -- a real source video can
    legitimately run several minutes (see
    ``core/pipeline.py``'s ``max_event_duration_seconds`` for how derived
    clips are now bounded, but the *normalized* clip this function is
    sometimes called on, via ``detection/audio.py``, is the full source
    recording, not a derived excerpt).
    """

    if not ffmpeg_available():
        raise AudioAnalysisError("ffmpeg is not available on PATH")

    with tempfile.TemporaryDirectory(prefix="gaggle-audio-") as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE_HZ),
            "-f",
            "wav",
            str(wav_path),
        ]
        try:
            subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=True)
        except subprocess.TimeoutExpired as exc:
            raise AudioAnalysisError(f"ffmpeg audio extraction timed out for {path}") from exc
        except subprocess.CalledProcessError:
            # No audio stream (or an unreadable one) is a normal outcome for
            # many dashcam configurations, not a pipeline failure.
            return None
        except OSError as exc:
            raise AudioAnalysisError(f"ffmpeg could not be executed: {exc}") from exc

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            return None

        try:
            sample_rate, raw_audio = wavfile.read(wav_path)
        except ValueError as exc:
            raise AudioAnalysisError(f"could not parse extracted audio for {path}") from exc

    if raw_audio.size == 0:
        return None
    return sample_rate, raw_audio


def extract_normalized_waveform(
    path: Path, timeout_seconds: float = 300.0
) -> tuple[FloatArray, int] | None:
    """Extract ``path``'s audio as a normalized (-1..1) float64 waveform,
    for consumers that need the actual samples (e.g.
    `enrichment/voice.py`'s voice-activity-detection and voiceprinting)
    rather than a derived RMS envelope. Returns ``None`` if there's no
    usable audio track -- a normal outcome, not an error.
    """

    extracted = _extract_mono_wav(path, timeout_seconds)
    if extracted is None:
        return None
    sample_rate, raw_audio = extracted
    audio = raw_audio.astype(np.float64)
    max_magnitude = (
        float(np.iinfo(raw_audio.dtype).max) if np.issubdtype(raw_audio.dtype, np.integer) else 1.0
    )
    audio /= max_magnitude if max_magnitude else 1.0
    return audio, sample_rate


def analyze_audio_spikes(
    path: Path,
    window_seconds: float = 0.5,
    timeout_seconds: float = 300.0,
) -> AudioAnalysisResult:
    """Extract the audio track from ``path`` and compute an RMS envelope.

    Returns ``has_audio=False`` (with an empty sample list) when the source
    has no audio stream or extraction yields no data — a normal, expected
    outcome for silent dashcam configurations, not an error condition.
    """

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    extracted = _extract_mono_wav(path, timeout_seconds)
    if extracted is None:
        return AudioAnalysisResult(samples=[], has_audio=False)
    sample_rate, raw_audio = extracted

    audio = raw_audio.astype(np.float64)
    max_magnitude = (
        float(np.iinfo(raw_audio.dtype).max) if np.issubdtype(raw_audio.dtype, np.integer) else 1.0
    )
    audio /= max_magnitude if max_magnitude else 1.0

    window_samples = max(1, round(window_seconds * sample_rate))
    samples: list[AudioSample] = []
    for start_index in range(0, audio.size, window_samples):
        chunk = audio[start_index : start_index + window_samples]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        offset_seconds = start_index / sample_rate
        samples.append(AudioSample(offset_seconds=round(offset_seconds, 3), value=round(rms, 6)))

    return AudioAnalysisResult(samples=samples, has_audio=True)
