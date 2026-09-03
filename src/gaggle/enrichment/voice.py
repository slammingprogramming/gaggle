"""Voice activity detection and classical (non-deep-learning) voiceprinting.

**Scope and intent -- read this before extending this module.** Everything
here follows the exact same boundary as `enrichment/face.py` and
`enrichment/plate.py` (see `docs/forensic-considerations.md`'s
"Recognition data: scope and intent"): local pattern re-identification
within the user's own footage -- "have I heard this voice before" -- never
voice *identification*. There is no speech-to-text-based identity lookup,
no linking a voiceprint to a real name, and this is emphatically **not**
forensic-grade voice identification. Real forensic voice comparison is a
specialized field with rigor (and legal admissibility standards) far
beyond what a classical spectral-statistics fingerprint can offer; treat
every voice match here as a heuristic aid for personal review, not
evidence of who said something.

**Why classical MFCCs instead of a deep speaker-embedding model** (e.g.
d-vectors, x-vectors, or a model like pyannote/resemblyzer): those need
pretrained weights, and this project has no network access to fetch them
at development time and doesn't want to *require* it for a capability that
should work the same way face/plate re-identification do -- fully offline,
zero setup, no download. So this module implements the classical
pre-deep-learning approach from scratch using only numpy/scipy (both
already core dependencies): Mel-Frequency Cepstral Coefficients (MFCCs),
aggregated into a fixed-length "voiceprint" vector per voice segment, with
a simple incremental centroid-based clusterer for re-identification.

This is a meaningfully weaker fingerprint than a modern deep speaker
embedding -- expect more false matches/misses than face or plate
recognition, especially on real, noisy dashcam audio (road noise, multiple
overlapping speakers, engine hum). It was validated during development
against synthetic multi-tone test signals standing in for different
"voices" (distinct fundamental + harmonic structure), repeated across
several noise seeds for a reliable distribution rather than one lucky
sample: same-voice comparisons consistently scored ~0.0002-0.0003 cosine
distance, different-voice comparisons ~0.14-0.143 -- a wide, consistent
separation, which is what `IncrementalVoiceClusterer`'s default
`distance_threshold=0.05` is chosen from (comfortably in the middle of
that gap, biased toward *not* merging when uncertain). An initial,
less-careful choice of 0.15 was caught during this same validation pass --
close enough to the observed different-voice distance that it produced a
real false merge in testing, not a hypothetical one. None of this has been
validated against real recorded human speech in this environment, though;
treat this capability's real-world accuracy with more caution than the
face/plate detectors, which were validated against real and realistic
synthetic imagery, and expect to need to empirically retune the threshold
against your own footage.
"""

from __future__ import annotations

import json
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
from scipy.fft import dct

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

MFCC_COUNT = 13
MEL_FILTER_COUNT = 26
FRAME_LENGTH_MS = 25.0
FRAME_STEP_MS = 10.0
FFT_SIZE = 512
VOICEPRINT_VERSION = "1.0.0"
VAD_VERSION = "1.0.0"

# Voiceprint = concatenated [mean(MFCC), std(MFCC)] across all frames in a
# voice segment -- a fixed-length (2 * MFCC_COUNT)-dimensional vector.
VOICEPRINT_DIMENSIONS = MFCC_COUNT * 2


class VoiceAnalysisError(RuntimeError):
    """Raised when the input audio can't be analyzed (e.g. wrong shape/empty)."""


@dataclass(frozen=True, slots=True)
class VoiceSegment:
    start_offset_seconds: float
    end_offset_seconds: float


@dataclass(frozen=True, slots=True)
class VoicePrintResult:
    segment: VoiceSegment
    voiceprint: FloatArray  # shape (VOICEPRINT_DIMENSIONS,)
    energy_confidence: float  # 0-1, how confidently this was judged voice-like


def detect_voice_segments(
    samples: FloatArray,
    sample_rate: int,
    min_segment_seconds: float = 0.3,
    energy_percentile_threshold: float = 60.0,
    merge_gap_seconds: float = 0.2,
) -> list[VoiceSegment]:
    """Classical energy + spectral-flatness voice activity detection.

    Deliberately simple and explainable, not a trained VAD model: a frame
    is "voice-active" if its short-time energy is above an adaptive
    per-clip threshold (a percentile of that clip's own energy
    distribution, so it adapts to how loud the recording is) *and* its
    spectral flatness is low enough to suggest a broadband, formant-rich
    signal rather than a narrowband tone (e.g. a horn, an engine's
    dominant hum) -- speech is spectrally "peakier" (less flat) than most
    steady mechanical noise. This is a heuristic, not a guarantee; expect
    some road/wind noise to pass through and some quiet speech to be
    missed. Active frames are merged into segments (small gaps bridged,
    per `merge_gap_seconds`) and short segments are dropped as noise
    blips.
    """

    if samples.ndim != 1 or samples.size == 0:
        raise VoiceAnalysisError("expected a non-empty 1-D audio array")

    frame_length = max(1, round(FRAME_LENGTH_MS / 1000 * sample_rate))
    frame_step = max(1, round(FRAME_STEP_MS / 1000 * sample_rate))
    frames, _padded_length = _frame_signal(samples.astype(np.float64), frame_length, frame_step)
    if frames.shape[0] == 0:
        return []

    windowed = frames * np.hamming(frame_length)
    energy = np.sqrt(np.mean(windowed**2, axis=1) + 1e-12)
    spectrum = np.abs(np.fft.rfft(windowed, FFT_SIZE)) + 1e-12
    geometric_mean = np.exp(np.mean(np.log(spectrum), axis=1))
    arithmetic_mean = np.mean(spectrum, axis=1)
    flatness = geometric_mean / arithmetic_mean  # 0 (tonal) .. 1 (noise-like/flat)

    energy_threshold = np.percentile(energy, energy_percentile_threshold)
    # Speech is neither perfectly tonal (flatness near 0, like a pure horn
    # tone) nor perfectly flat/white-noise-like (flatness near 1, like
    # broadband road hiss) -- a mid band captures voiced speech's
    # harmonic-but-broadband character reasonably well.
    active = (energy >= energy_threshold) & (flatness > 0.15) & (flatness < 0.75)

    segments = _frames_to_segments(active, frame_step, sample_rate, merge_gap_seconds)
    return [
        s
        for s in segments
        if (s.end_offset_seconds - s.start_offset_seconds) >= min_segment_seconds
    ]


def compute_voiceprint(
    samples: FloatArray, sample_rate: int, segment: VoiceSegment | None = None
) -> VoicePrintResult:
    """Compute a fixed-length voiceprint for one voice segment (or the whole
    clip, if `segment` is None)."""

    if segment is not None:
        start = max(0, int(segment.start_offset_seconds * sample_rate))
        end = min(len(samples), int(segment.end_offset_seconds * sample_rate))
        clip = samples[start:end]
    else:
        clip = samples
        segment = VoiceSegment(0.0, len(samples) / sample_rate)

    if clip.size < int(FRAME_LENGTH_MS / 1000 * sample_rate):
        raise VoiceAnalysisError("segment too short to compute a voiceprint")

    mfcc = _compute_mfcc(clip.astype(np.float64), sample_rate)
    if mfcc.shape[0] == 0:
        raise VoiceAnalysisError("no frames produced for this segment")

    voiceprint = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0)])
    energy = np.sqrt(np.mean(clip.astype(np.float64) ** 2) + 1e-12)
    # A simple, bounded confidence proxy from RMS energy -- not a
    # calibrated probability, just "how much signal was actually here."
    confidence = round(float(min(1.0, energy * 8.0)), 6)
    return VoicePrintResult(segment=segment, voiceprint=voiceprint, energy_confidence=confidence)


def _frame_signal(signal: FloatArray, frame_length: int, frame_step: int) -> tuple[FloatArray, int]:
    signal_length = len(signal)
    if signal_length < frame_length:
        return np.zeros((0, frame_length)), signal_length
    num_frames = 1 + (signal_length - frame_length) // frame_step
    pad_length = (num_frames - 1) * frame_step + frame_length
    padded = np.append(signal, np.zeros(max(0, pad_length - signal_length)))
    indices = np.arange(frame_length)[None, :] + np.arange(num_frames)[:, None] * frame_step
    return padded[indices], pad_length


def _frames_to_segments(
    active: BoolArray, frame_step: int, sample_rate: int, merge_gap_seconds: float
) -> list[VoiceSegment]:
    if not active.any():
        return []
    merge_gap_frames = max(1, round(merge_gap_seconds * sample_rate / frame_step))
    segments: list[VoiceSegment] = []
    start_frame: int | None = None
    last_active_frame = -1
    for index, is_active in enumerate(active):
        if is_active:
            if start_frame is None:
                start_frame = index
            elif index - last_active_frame > merge_gap_frames:
                segments.append(
                    _frame_range_to_segment(start_frame, last_active_frame, frame_step, sample_rate)
                )
                start_frame = index
            last_active_frame = index
    if start_frame is not None:
        segments.append(
            _frame_range_to_segment(start_frame, last_active_frame, frame_step, sample_rate)
        )
    return segments


def _frame_range_to_segment(
    start_frame: int, end_frame: int, frame_step: int, sample_rate: int
) -> VoiceSegment:
    start_seconds = (start_frame * frame_step) / sample_rate
    end_seconds = ((end_frame + 1) * frame_step) / sample_rate
    return VoiceSegment(round(start_seconds, 3), round(end_seconds, 3))


def _hz_to_mel(hz: FloatArray | float) -> FloatArray | float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: FloatArray | float) -> FloatArray | float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_filters: int, fft_size: int, sample_rate: int) -> FloatArray:
    low_mel, high_mel = 0.0, float(_hz_to_mel(sample_rate / 2))
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2, dtype=np.float64)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, fft_size // 2 + 1))
    for i in range(1, num_filters + 1):
        left, center, right = bin_points[i - 1], bin_points[i], bin_points[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters


_filterbank_cache: dict[tuple[int, int, int], FloatArray] = {}


def _compute_mfcc(signal: FloatArray, sample_rate: int) -> FloatArray:
    emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frame_length = max(1, round(FRAME_LENGTH_MS / 1000 * sample_rate))
    frame_step = max(1, round(FRAME_STEP_MS / 1000 * sample_rate))
    frames, _padded_length = _frame_signal(emphasized, frame_length, frame_step)
    if frames.shape[0] == 0:
        return np.zeros((0, MFCC_COUNT))

    windowed = frames * np.hamming(frame_length)
    magnitude = np.abs(np.fft.rfft(windowed, FFT_SIZE))
    power = (magnitude**2) / FFT_SIZE

    cache_key = (MEL_FILTER_COUNT, FFT_SIZE, sample_rate)
    if cache_key not in _filterbank_cache:
        _filterbank_cache[cache_key] = _mel_filterbank(MEL_FILTER_COUNT, FFT_SIZE, sample_rate)
    filterbank_energies = power @ _filterbank_cache[cache_key].T
    filterbank_energies = np.where(
        filterbank_energies <= 0, np.finfo(float).eps, filterbank_energies
    )
    log_energies = np.log(filterbank_energies)

    mfcc = dct(log_energies, type=2, axis=1, norm="ortho")[:, :MFCC_COUNT]
    return cast(FloatArray, mfcc)


class _ClusterState(TypedDict):
    centroid: FloatArray
    count: int


class IncrementalVoiceClusterer:
    """Persistent, incrementally-updated centroid clusterer for voiceprints.

    Simpler than `enrichment.face.IncrementalFaceClusterer` (a running mean
    per cluster rather than a trained classifier), since voiceprints are
    already fixed-length real vectors that a plain distance metric handles
    directly. Persisted as JSON (not a binary model format) so it's
    trivially inspectable -- see `workspace/recognition/voices/model.json`.
    """

    def __init__(self, model_path: Path, distance_threshold: float = 0.05) -> None:
        self.model_path = model_path
        self.distance_threshold = distance_threshold
        self._clusters: dict[str, _ClusterState] = {}
        if model_path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.model_path.read_text(encoding="utf-8"))
        self._clusters = {
            cluster_id: _ClusterState(
                centroid=np.array(data["centroid"], dtype=np.float64), count=data["count"]
            )
            for cluster_id, data in payload.items()
        }

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            cluster_id: {"centroid": data["centroid"].tolist(), "count": data["count"]}
            for cluster_id, data in self._clusters.items()
        }
        self.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def match_or_create_cluster(self, voiceprint: FloatArray) -> tuple[str, float, bool]:
        """Return (cluster_id, distance, is_new_cluster) for a voiceprint vector.

        ``distance`` is cosine distance (0 = identical direction, 2 =
        opposite) to the nearest existing cluster centroid, or the
        (above-threshold) rejection distance that caused a new cluster to
        be created.
        """

        best_id: str | None = None
        best_distance = float("inf")
        for cluster_id, data in self._clusters.items():
            distance = _cosine_distance(voiceprint, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id

        if best_id is not None and best_distance <= self.distance_threshold:
            self._update_centroid(best_id, voiceprint)
            return best_id, float(best_distance), False

        new_cluster_id = str(_uuid.uuid4())
        self._clusters[new_cluster_id] = _ClusterState(centroid=voiceprint.copy(), count=1)
        return new_cluster_id, float(best_distance if best_id else 0.0), True

    def predict_nearest_cluster(
        self, voiceprint: FloatArray, exclude_cluster_id: str | None = None
    ) -> tuple[str | None, float]:
        """Read-only lookup, no mutation -- mirrors
        `IncrementalFaceClusterer.predict_nearest_cluster`, used for
        merge-suggestion generation.

        ``exclude_cluster_id`` matters when the query vector *is* a
        cluster's own centroid (exactly what
        `RecognitionService.suggest_voice_merges` passes): without
        excluding it from the search, a cluster's own centroid always
        matches itself at distance 0 and wins, so the real nearest
        *other* cluster -- the one actually worth suggesting -- would
        never be found. This was a real bug (caught via the vehicle
        appearance module's real test execution, then confirmed here by
        code symmetry): excluding the query's own id from the search
        itself, not just discarding a self-match after the fact, is what
        actually fixes it.
        """

        if not self._clusters:
            return None, 0.0
        best_id, best_distance = None, float("inf")
        for cluster_id, data in self._clusters.items():
            if cluster_id == exclude_cluster_id:
                continue
            distance = _cosine_distance(voiceprint, data["centroid"])
            if distance < best_distance:
                best_distance = distance
                best_id = cluster_id
        return best_id, float(best_distance)

    def get_cluster_centroid(self, cluster_id: str) -> FloatArray | None:
        """The current running-mean voiceprint for a cluster, or ``None`` if
        no such cluster exists in this model. Used for merge-suggestion
        generation (`core/recognition.py::RecognitionService.suggest_voice_merges`),
        which needs each cluster's own centroid to compare against every
        *other* cluster."""

        data = self._clusters.get(cluster_id)
        if data is None:
            return None
        return data["centroid"]

    def _update_centroid(self, cluster_id: str, voiceprint: FloatArray) -> None:
        data = self._clusters[cluster_id]
        count = data["count"]
        centroid = data["centroid"]
        new_centroid = (centroid * count + voiceprint) / (count + 1)
        self._clusters[cluster_id] = _ClusterState(centroid=new_centroid, count=count + 1)


def _cosine_distance(a: FloatArray, b: FloatArray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)
