"""Deterministic, explainable "rapid approach" (looming) detection via
dense optical flow.

**Why this, and not just another motion-detection variant.** Frame
differencing (`video_analysis.py`) answers "did something change here" --
it cannot distinguish lateral motion from something *approaching the
camera*, a structurally different, dashcam-relevant cue (tailgating, a
near-miss, a vehicle closing in fast). Dense optical flow can capture
that distinction; frame differencing cannot, by construction.

**The technique**: `cv2.calcOpticalFlowFarneback` between consecutive
sampled grayscale frames, then the flow field's *divergence*
(`du/dx + dv/dy`, a standard vector-calculus quantity computed here via
`numpy.gradient`) -- positive divergence means the flow is expanding
outward at that point ("looming"), negative means it's converging. Both
classical, deterministic, no learned model (invariant 9), same spirit as
`video_analysis.py`'s frame differencing and `telemetry_analysis.py`'s
haversine/bearing math.

**Ego-motion rejection is the crux of making this signal meaningful, not
just noisy.** A forward-driving dashcam's own motion produces strong
divergence essentially everywhere in frame (radial expansion from the
vanishing point) -- an absolute divergence threshold would flag ordinary
driving constantly. Instead, this module reports divergence two ways per
sample: `global_divergence` (mean over the whole frame, dominated by ego
motion) and `roi_divergence` (mean over a central ~60% region-of-interest).
`detect_rapid_approach_events` fires only when `roi_divergence` exceeds a
*rolling median of recent `global_divergence` samples* by a configurable
delta -- a comparative threshold, not an absolute one. Ordinary uniform
ego-motion raises both scalars together (small delta, no event); a real
object closing in raises `roi_divergence` well above the ambient baseline.

**The default `roi_divergence_delta_threshold` (0.015,
`core/config.py::OpticalFlowConfig`) was measured, not guessed**, the
same discipline `enrichment/voice.py`/`enrichment/vehicle_appearance.py`
already established: synthetic "true positive" scenes (a growing disc at
five different growth rates, simulating an approaching object) versus
"true negative" scenes (a whole-frame uniform zoom at five different
rates simulating ego-motion, plus a fully static scene). Measured
`roi_divergence - global_divergence` deltas: true positives ranged
0.030-0.091 (excluding one degenerate case below, at a growth rate too
subtle for Farneback to resolve at all -- max_delta exactly 0.0, not a
near-miss); true negatives (ego-motion + static) topped out at 0.011,
with most well under 0.008. 0.015 sits comfortably inside that gap,
above every measured true negative and below every measured true
positive strong enough to produce a real signal.

**Known limitations** (documented here, repeated in `docs/limitations.md`):
low-texture/night-driving frames starve Farneback of gradient information
(same failure mode frame differencing already has); very fast real-world
closures between samples can produce large frame-to-frame displacement
that breaks Farneback's local-window tracking assumption, showing up as
noisy rather than cleanly-signed divergence; a genuinely slow, gradual
approach can measure as weakly as the degenerate true-positive case
above and go undetected -- this is a corroborating signal only
(invariant 7), never sufficient alone for high severity, and never
claimed to catch every real approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.typing
import numpy as np

ImageArray = cv2.typing.MatLike

ANALYZER_VERSION = "1.0.0"

# Central region-of-interest as a fraction of width/height, symmetric
# around the frame center (0.6 -> the middle 60% each axis).
_ROI_FRACTION = 0.6

# Farneback parameters sized for this project's existing 320px analysis
# width (see video_analysis.py's own resize_width default) -- winsize=15
# balances noise-smoothing against small-object sensitivity at this
# resolution; going much smaller lets single-pixel noise dominate the
# divergence field.
_FARNEBACK_PYR_SCALE = 0.5
_FARNEBACK_LEVELS = 3
_FARNEBACK_WINSIZE = 15
_FARNEBACK_ITERATIONS = 3
_FARNEBACK_POLY_N = 5
_FARNEBACK_POLY_SIGMA = 1.2
_FARNEBACK_FLAGS = 0


class OpticalFlowAnalysisError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class DivergenceSample:
    """The flow-field divergence between two consecutive sampled frames,
    reported both globally (dominated by ego-motion) and over a central
    region-of-interest -- see the module docstring for why both matter."""

    offset_seconds: float
    roi_divergence: float
    global_divergence: float


@dataclass(frozen=True, slots=True)
class OpticalFlowResult:
    divergence_series: list[DivergenceSample]
    frames_sampled: int
    analyzer_version: str = ANALYZER_VERSION


@dataclass(frozen=True, slots=True)
class RapidApproachEvent:
    offset_seconds: float
    confidence: float  # 0-1
    roi_divergence: float
    global_divergence: float
    baseline_global_divergence: float


def analyze_optical_flow(
    path: Path,
    sample_rate_hz: float = 2.0,
    resize_width: int = 320,
) -> OpticalFlowResult:
    """Analyze ``path`` for looming (rapid-approach) flow patterns.

    Mirrors `video_analysis.py::analyze_video_motion`'s frame-sampling
    structure exactly (same stride computation, same resize behavior) so
    the two detectors stay easy to reason about side by side.
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OpticalFlowAnalysisError(f"could not open video file: {path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps <= 0:
            source_fps = 30.0
        frame_stride = max(1, round(source_fps / sample_rate_hz))

        divergence_series: list[DivergenceSample] = []
        previous_gray: ImageArray | None = None
        previous_offset = 0.0
        frame_index = 0
        frames_sampled = 0

        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue
            decoded, frame = capture.retrieve()
            frame_index += 1
            if not decoded or frame is None:
                continue
            offset_seconds = frame_index / source_fps
            scale = resize_width / frame.shape[1] if frame.shape[1] > resize_width else 1.0
            if scale != 1.0:
                frame = cv2.resize(frame, (resize_width, max(1, round(frame.shape[0] * scale))))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_sampled += 1

            if previous_gray is not None:
                # cv2's stubs type the `flow` positional as non-Optional,
                # but the real OpenCV API accepts (and expects) `None` here
                # to mean "allocate the output array" -- confirmed at
                # runtime, same class of stub gap as cv2.face.*/MSER_create
                # elsewhere in this project.
                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,  # type: ignore[call-overload]
                    _FARNEBACK_PYR_SCALE,
                    _FARNEBACK_LEVELS,
                    _FARNEBACK_WINSIZE,
                    _FARNEBACK_ITERATIONS,
                    _FARNEBACK_POLY_N,
                    _FARNEBACK_POLY_SIGMA,
                    _FARNEBACK_FLAGS,
                )
                global_divergence, roi_divergence = _flow_divergence(flow)
                divergence_series.append(
                    DivergenceSample(
                        offset_seconds=previous_offset,
                        roi_divergence=round(roi_divergence, 6),
                        global_divergence=round(global_divergence, 6),
                    )
                )

            previous_gray = gray
            previous_offset = offset_seconds

        return OpticalFlowResult(divergence_series=divergence_series, frames_sampled=frames_sampled)
    finally:
        capture.release()


def _flow_divergence(flow: ImageArray) -> tuple[float, float]:
    """Returns ``(global_divergence, roi_divergence)`` for one flow field."""

    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    divergence_field = np.gradient(flow_x, axis=1) + np.gradient(flow_y, axis=0)
    global_divergence = float(np.mean(divergence_field))

    height, width = divergence_field.shape
    margin_y = int(height * (1.0 - _ROI_FRACTION) / 2.0)
    margin_x = int(width * (1.0 - _ROI_FRACTION) / 2.0)
    roi = divergence_field[margin_y : height - margin_y, margin_x : width - margin_x]
    roi_divergence = float(np.mean(roi)) if roi.size else global_divergence

    return global_divergence, roi_divergence


def detect_rapid_approach_events(
    samples: list[DivergenceSample],
    roi_divergence_delta_threshold: float,
    baseline_window: int = 5,
) -> list[RapidApproachEvent]:
    """Flag samples where the ROI's divergence exceeds a rolling median of
    recent global (ego-motion) divergence by ``roi_divergence_delta_threshold``.

    A comparative threshold, not an absolute one -- see the module
    docstring for why. The first sample never fires (no baseline exists
    yet), exactly like `telemetry_analysis.py::detect_telemetry_events`'s
    pairwise structure needing at least one prior sample.
    """

    events: list[RapidApproachEvent] = []
    recent_global: list[float] = []
    for sample in samples:
        if recent_global:
            baseline = float(np.median(recent_global[-baseline_window:]))
            delta = sample.roi_divergence - baseline
            if delta >= roi_divergence_delta_threshold:
                confidence = min(1.0, delta / (roi_divergence_delta_threshold * 2))
                events.append(
                    RapidApproachEvent(
                        offset_seconds=sample.offset_seconds,
                        confidence=round(confidence, 6),
                        roi_divergence=sample.roi_divergence,
                        global_divergence=sample.global_divergence,
                        baseline_global_divergence=round(baseline, 6),
                    )
                )
        recent_global.append(sample.global_divergence)
    return events
