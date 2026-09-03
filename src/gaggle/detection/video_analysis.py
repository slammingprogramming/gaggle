"""Deterministic, explainable video motion analysis.

This module implements the "motion" and "object hint" detection primitives
using classic, inspectable computer-vision techniques: grayscale frame
differencing for motion scoring, and contour extraction on the same
difference mask for coarse moving-region ("blob") hints. Deliberately no
neural network or learned embedding is used here; per the project's
ML-first-avoidance directive, this is the built-in heuristic baseline, and
real object classifiers are expected to arrive later as
``gaggle.plugins.base.DetectorPlugin`` implementations that
produce the same ``Signal`` shape with clearer provenance.

Everything here is a pure function of the input file's bytes plus the
sampling parameters: decoding the same file with the same OpenCV build
always visits frames in the same order and produces the same numbers, so
results are reproducible across runs (though not necessarily bit-identical
across different OpenCV/ffmpeg builds, which is why the analyzer version and
the underlying tool versions are recorded in the caller's reasoning
metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import cv2.typing
import numpy as np

ImageArray = cv2.typing.MatLike

ANALYZER_VERSION = "1.0.0"


class VideoAnalysisError(RuntimeError):
    """Raised when a video file cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class MotionSample:
    offset_seconds: float
    value: float


@dataclass(frozen=True, slots=True)
class MotionRegion:
    """A contiguous moving region detected via contour analysis."""

    offset_seconds: float
    end_offset_seconds: float
    confidence: float
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 normalized to [0,1]
    area_ratio: float


@dataclass(frozen=True, slots=True)
class VideoAnalysisResult:
    motion_series: list[MotionSample]
    regions: list[MotionRegion]
    frames_sampled: int
    analyzer_version: str = ANALYZER_VERSION


def analyze_video_motion(
    path: Path,
    sample_rate_hz: float = 2.0,
    resize_width: int = 320,
    min_region_area_ratio: float = 0.01,
) -> VideoAnalysisResult:
    """Analyze ``path`` for frame-to-frame motion and moving regions.

    ``sample_rate_hz`` controls how many analysis samples are taken per
    second of video (frames are skipped to hit this rate rather than
    analyzing every decoded frame, keeping this correctness-first path fast
    enough for real dashcam clips). Each sample compares the current sampled
    frame to the previous one via absolute grayscale difference; the motion
    value is the fraction of pixels that changed more than a fixed noise
    threshold. Contours drawn from the same difference mask above
    ``min_region_area_ratio`` of frame area are reported as coarse moving
    regions with a normalized bounding box.
    """

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoAnalysisError(f"could not open video file: {path}")
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if source_fps <= 0:
            source_fps = 30.0
        frame_stride = max(1, round(source_fps / sample_rate_hz))

        motion_series: list[MotionSample] = []
        regions: list[MotionRegion] = []
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
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, round(frame.shape[0] * scale))),
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            frames_sampled += 1

            if previous_gray is not None:
                diff = cv2.absdiff(previous_gray, gray)
                _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_value = float(np.count_nonzero(mask)) / float(mask.size)
                motion_series.append(
                    MotionSample(offset_seconds=previous_offset, value=round(motion_value, 6))
                )
                regions.extend(
                    _extract_regions(
                        mask,
                        previous_offset,
                        offset_seconds,
                        min_region_area_ratio,
                    )
                )

            previous_gray = gray
            previous_offset = offset_seconds

        return VideoAnalysisResult(
            motion_series=motion_series,
            regions=regions,
            frames_sampled=frames_sampled,
        )
    finally:
        capture.release()


def _extract_regions(
    mask: ImageArray,
    start_offset: float,
    end_offset: float,
    min_region_area_ratio: float,
) -> list[MotionRegion]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(mask.shape[0] * mask.shape[1])
    height, width = mask.shape[0], mask.shape[1]
    found: list[MotionRegion] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / frame_area if frame_area else 0.0
        if area_ratio < min_region_area_ratio:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        found.append(
            MotionRegion(
                offset_seconds=start_offset,
                end_offset_seconds=end_offset,
                confidence=round(min(1.0, area_ratio * 5.0), 6),
                bbox=(
                    round(x / width, 6),
                    round(y / height, 6),
                    round((x + w) / width, 6),
                    round((y + h) / height, 6),
                ),
                area_ratio=round(area_ratio, 6),
            )
        )
    # Deterministic ordering: largest region first, ties broken by position.
    found.sort(key=lambda region: (-region.area_ratio, region.bbox))
    return found
