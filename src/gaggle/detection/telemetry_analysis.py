"""Deterministic vehicle telemetry (GPS track) analysis.

**Why GPX, and why this is the "real" (non-fixture) path.** No universal
dashcam telemetry format exists to target -- unlike audio/video, where
ffmpeg/OpenCV give this project a real, standard way to read almost any
source file. GPX is an open, widely-supported XML standard for GPS
tracks (exported by dedicated GPS loggers, many phone GPS-logging apps,
and some dashcam apps), and is parseable with the standard library alone
(`xml.etree.ElementTree`) -- no new dependency, consistent with this
project's existing "prefer stdlib or an external CLI tool over a heavy
pip dependency" pattern (ffmpeg, tesseract). This keeps the "real
ingestion format" claim honest: it isn't inventing a fake proprietary
format just to have *something* to parse, and it isn't silently only
supporting the sidecar-JSON-fixture path dressed up as "real" -- that
path stays purely for deterministic testing, exactly like
`detection/motion.py`/`detection/audio.py`'s sidecar pattern.

**Speed and heading are computed directly from consecutive
`(lat, lon, time)` points**, not read from a `<speed>` extension GPX
doesn't always include:

* **Speed**: the haversine great-circle distance between consecutive
  points, divided by the time delta between them. Haversine assumes a
  spherical Earth (radius 6,371,000m, the IUGG mean radius) -- accurate
  to within ~0.5% for distances at dashcam-trip scale, more than
  sufficient for a coarse, corroborating signal (see invariant 7: no
  single weak signal reaches high severity alone).
* **Heading**: the initial bearing formula (the compass bearing at the
  start of the great-circle path between two points), in degrees
  clockwise from true north.

Both are standard, well-documented, deterministic closed-form
calculations -- no learned model, no randomness (invariant 9) --
implemented by hand here the same way `enrichment/voice.py`'s MFCC
pipeline is: a classical algorithm this project can fully explain rather
than a black box.

**What counts as an "interesting" telemetry event** (each becomes one
weak `Signal(signal_type="telemetry")` in `detection/telemetry.py` --
never, by itself, sufficient for high severity, per invariant 7):

* **Hard braking**: deceleration between two consecutive samples at or
  above a configurable threshold (m/s²).
* **Speed spike**: absolute speed at or above a configurable threshold
  (m/s) -- deliberately an absolute-value check, not a rate-of-change
  one, to keep the first pass simple and explainable; a
  rate-of-acceleration variant is a reasonable future addition, not
  attempted here to avoid scope creep on this pass.
* **Sudden heading change**: the shortest angular distance between two
  consecutive headings, divided by the time delta, at or above a
  configurable degrees-per-second threshold (a hard swerve or turn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

ANALYZER_VERSION = "1.0.0"
_EARTH_RADIUS_METERS = 6_371_000.0


class TelemetryAnalysisError(RuntimeError):
    """Raised when a GPX file can't be parsed or has too few usable points."""


@dataclass(frozen=True, slots=True)
class TrackPoint:
    latitude: float
    longitude: float
    time: datetime  # timezone-aware


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """One consecutive-point-pair measurement: the speed/heading of the
    vehicle *during* the interval ending at `offset_seconds`."""

    offset_seconds: float  # relative to the first track point
    speed_mps: float
    heading_degrees: float  # compass bearing, 0-360, clockwise from north
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event_type: str  # "hard_braking" | "speed_spike" | "sudden_heading_change"
    offset_seconds: float
    confidence: float  # 0-1
    value: float  # the raw measurement: m/s^2, m/s, or degrees/second


def _local_tag(tag: str) -> str:
    """Strip any XML namespace prefix from an element tag (`{ns}trkpt` ->
    `trkpt`) so parsing works regardless of whether a GPX file declares
    the standard namespace, a versioned one, or none at all -- real-world
    GPX exporters are inconsistent about this."""

    return tag.rsplit("}", 1)[-1]


def parse_gpx(path: Path) -> list[TrackPoint]:
    """Parse a GPX track file's `<trkpt>` elements into time-ordered points.

    Points missing a `<time>` child (can't compute speed/heading without
    a timestamp) are skipped. Raises `TelemetryAnalysisError` if the file
    can't be parsed at all, or has fewer than 2 usable points.
    """

    try:
        tree = ElementTree.parse(path)
    except ElementTree.ParseError as error:
        raise TelemetryAnalysisError(f"could not parse GPX file {path}: {error}") from error

    points: list[TrackPoint] = []
    for element in tree.getroot().iter():
        if _local_tag(element.tag) != "trkpt":
            continue
        lat_str = element.get("lat")
        lon_str = element.get("lon")
        time_element = next((child for child in element if _local_tag(child.tag) == "time"), None)
        if lat_str is None or lon_str is None or time_element is None or not time_element.text:
            continue
        timestamp = datetime.fromisoformat(time_element.text.replace("Z", "+00:00"))
        points.append(TrackPoint(latitude=float(lat_str), longitude=float(lon_str), time=timestamp))

    if len(points) < 2:
        raise TelemetryAnalysisError(f"GPX file {path} has fewer than 2 usable track points")
    points.sort(key=lambda p: p.time)
    return points


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_METERS * c


def _initial_bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    x = math.sin(delta_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def compute_telemetry_samples(points: list[TrackPoint]) -> list[TelemetrySample]:
    """Compute speed/heading for every consecutive pair of track points.

    A non-positive time delta between two points (out-of-order or
    duplicate timestamps) is skipped rather than raising -- a single bad
    GPS fix shouldn't abort analysis of an otherwise-usable track.
    """

    if len(points) < 2:
        raise TelemetryAnalysisError("need at least 2 track points to compute telemetry")

    first_time = points[0].time
    samples: list[TelemetrySample] = []
    for previous, current in pairwise(points):
        delta_seconds = (current.time - previous.time).total_seconds()
        if delta_seconds <= 0:
            continue
        distance = _haversine_distance_meters(
            previous.latitude, previous.longitude, current.latitude, current.longitude
        )
        heading = _initial_bearing_degrees(
            previous.latitude, previous.longitude, current.latitude, current.longitude
        )
        samples.append(
            TelemetrySample(
                offset_seconds=(current.time - first_time).total_seconds(),
                speed_mps=distance / delta_seconds,
                heading_degrees=heading,
                timestamp=current.time,
            )
        )
    return samples


def detect_telemetry_events(
    samples: list[TelemetrySample],
    hard_braking_threshold_mps2: float,
    speed_spike_threshold_mps: float,
    heading_change_threshold_deg_per_sec: float,
) -> list[TelemetryEvent]:
    """Flag hard-braking, speed-spike, and sudden-heading-change moments.

    Each check is independent -- a single interval can produce more than
    one event (e.g. braking hard while also turning sharply).
    """

    events: list[TelemetryEvent] = []
    for previous, current in pairwise(samples):
        delta_seconds = current.offset_seconds - previous.offset_seconds
        if delta_seconds <= 0:
            continue

        acceleration = (current.speed_mps - previous.speed_mps) / delta_seconds
        deceleration = -acceleration
        if deceleration >= hard_braking_threshold_mps2:
            confidence = min(1.0, deceleration / (hard_braking_threshold_mps2 * 2))
            events.append(
                TelemetryEvent(
                    event_type="hard_braking",
                    offset_seconds=current.offset_seconds,
                    confidence=round(confidence, 6),
                    value=round(deceleration, 4),
                )
            )

        if current.speed_mps >= speed_spike_threshold_mps:
            confidence = min(1.0, current.speed_mps / (speed_spike_threshold_mps * 1.5))
            events.append(
                TelemetryEvent(
                    event_type="speed_spike",
                    offset_seconds=current.offset_seconds,
                    confidence=round(confidence, 6),
                    value=round(current.speed_mps, 4),
                )
            )

        raw_heading_delta = abs(current.heading_degrees - previous.heading_degrees)
        heading_delta = min(raw_heading_delta, 360.0 - raw_heading_delta)
        heading_rate = heading_delta / delta_seconds
        if heading_rate >= heading_change_threshold_deg_per_sec:
            confidence = min(1.0, heading_rate / (heading_change_threshold_deg_per_sec * 2))
            events.append(
                TelemetryEvent(
                    event_type="sudden_heading_change",
                    offset_seconds=current.offset_seconds,
                    confidence=round(confidence, 6),
                    value=round(heading_rate, 4),
                )
            )

    return events
