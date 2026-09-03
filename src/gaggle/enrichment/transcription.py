"""Optional local audio transcription via faster-whisper (CTranslate2/Whisper).

Fully offline once a model is downloaded -- the one-time model download
(faster-whisper fetches CTranslate2-converted Whisper weights from Hugging
Face on first use of a given model size, or a local model directory can be
pointed at directly) is the only network dependency, exactly like the YOLO
model file in `vehicle_yolo.py`. Requires the `transcription` extra
(`pip install gaggle[transcription]`).

Runs on CPU by default (`compute_type="int8"` for speed); set
`enrichment.transcription.device: cuda` to use a GPU if available.

Transcript *text* is local data like everything else in this pipeline. It
only leaves the machine if the separate, independently-opt-in
`enrichment.llm_analysis` feature is also enabled (see `llm_analysis.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaggle.utils.logging import get_logger

LOGGER = get_logger(__name__)


class TranscriptionUnavailableError(RuntimeError):
    """Raised (and expected to be caught) when faster-whisper isn't installed."""


@dataclass(frozen=True, slots=True)
class TranscribedSegment:
    start_offset_seconds: float
    end_offset_seconds: float
    text: str
    confidence: float | None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    language: str | None
    segments: list[TranscribedSegment]
    full_text: str
    model_name: str
    device: str


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    except Exception as error:
        # `faster_whisper` imports `ctranslate2`, which unconditionally
        # imports `torch` in its own model-spec module -- the same real
        # failure mode documented in `enrichment/face_auraface.py`'s
        # `insightface_available()`: a broken/conflicting local torch
        # CUDA install (e.g. a cuDNN major-version mismatch against
        # onnxruntime-gpu's own CUDA packages, if both are installed) can
        # raise a plain OSError here, not an ImportError. Treated the same
        # as "not installed": skip transcription for this run rather than
        # crashing the whole enrich command.
        LOGGER.warning(
            "faster_whisper_import_failed",
            reason=str(error),
            hint="skipping transcription for this run",
        )
        return False
    return True


class WhisperTranscriber:
    """Thin, lazily-initialized wrapper around `faster_whisper.WhisperModel`.

    The model is loaded once per process (it's the expensive part) and
    reused across `transcribe()` calls -- callers should hold one instance
    per pipeline run rather than constructing a new one per clip.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ) -> None:
        if not faster_whisper_available():
            raise TranscriptionUnavailableError(
                "faster-whisper is not installed; install the 'transcription' extra "
                "(pip install gaggle[transcription]) and download a model "
                "once (see docs/local-ai.md) to enable local transcription"
            )
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.device = device
        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root) if download_root else None,
            )
        except Exception as error:
            raise TranscriptionUnavailableError(
                f"could not load Whisper model '{model_name}': {error}. "
                "If this is the first run, it needs network access once to download "
                "the model; see docs/local-ai.md."
            ) from error
        active_device = getattr(self._model.model, "device", device)
        if device == "cuda" and active_device != "cuda":
            LOGGER.warning(
                "whisper_cuda_requested_but_not_active",
                requested=device,
                active=active_device,
            )
        else:
            LOGGER.info("whisper_device_active", requested=device, active=active_device)

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        segments_iter, info = self._model.transcribe(str(audio_path), vad_filter=True)
        segments = [
            TranscribedSegment(
                start_offset_seconds=round(segment.start, 3),
                end_offset_seconds=round(segment.end, 3),
                text=segment.text.strip(),
                confidence=(
                    round(float(_prob_from_logprob(segment.avg_logprob)), 6)
                    if segment.avg_logprob is not None
                    else None
                ),
            )
            for segment in segments_iter
        ]
        full_text = " ".join(segment.text for segment in segments).strip()
        return TranscriptionResult(
            language=info.language,
            segments=segments,
            full_text=full_text,
            model_name=self.model_name,
            device=self.device,
        )


def _prob_from_logprob(avg_logprob: float) -> float:
    """Map Whisper's average log-probability to a bounded, human-readable 0-1 score.

    This is a heuristic normalization (`exp(avg_logprob)`, clamped), not a
    calibrated confidence -- documented as such rather than presented as a
    precise probability. Whisper's own avg_logprob is typically in the
    range [-1, 0] for reasonable transcriptions.
    """

    import math

    return max(0.0, min(1.0, math.exp(avg_logprob)))


def load_transcriber_if_available(
    model_name: str = "base", device: str = "cpu", compute_type: str = "int8"
) -> WhisperTranscriber | None:
    """Best-effort loader: returns None (logged, not raised) if unavailable."""

    try:
        return WhisperTranscriber(model_name=model_name, device=device, compute_type=compute_type)
    except TranscriptionUnavailableError:
        return None
