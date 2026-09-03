from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaggle.detection.telemetry_analysis import (
    TelemetryAnalysisError,
    TrackPoint,
    compute_telemetry_samples,
    detect_telemetry_events,
    parse_gpx,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_track.gpx"


def test_fixture_track_exists() -> None:
    assert FIXTURE_PATH.exists(), f"missing test fixture: {FIXTURE_PATH}"


def test_parse_gpx_reads_every_point_in_time_order() -> None:
    points = parse_gpx(FIXTURE_PATH)
    assert len(points) == 10
    assert points[0].latitude == pytest.approx(37.0)
    assert points[0].longitude == pytest.approx(-122.0)
    assert points == sorted(points, key=lambda p: p.time)


def test_parse_gpx_raises_for_a_malformed_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.gpx"
    bad_file.write_text("not xml at all <<<", encoding="utf-8")
    with pytest.raises(TelemetryAnalysisError):
        parse_gpx(bad_file)


def test_parse_gpx_raises_for_too_few_points(tmp_path: Path) -> None:
    single_point = tmp_path / "single.gpx"
    single_point.write_text(
        '<?xml version="1.0"?><gpx><trk><trkseg>'
        '<trkpt lat="1.0" lon="2.0"><time>2026-01-01T00:00:00Z</time></trkpt>'
        "</trkseg></trk></gpx>",
        encoding="utf-8",
    )
    with pytest.raises(TelemetryAnalysisError):
        parse_gpx(single_point)


def test_compute_telemetry_samples_recovers_the_intended_ground_truth() -> None:
    """The fixture's points were generated with the great-circle
    destination-point formula -- the exact inverse of the haversine
    distance / initial-bearing formulas this module implements -- so
    re-deriving speed/heading from them should recover the intended
    values to a tight tolerance, not just roughly."""

    points = parse_gpx(FIXTURE_PATH)
    samples = compute_telemetry_samples(points)
    assert len(samples) == 9

    # 5 cruise samples at ~15 m/s, heading due north (0 degrees)
    for sample in samples[:5]:
        assert sample.speed_mps == pytest.approx(15.0, abs=0.01)
        assert sample.heading_degrees == pytest.approx(0.0, abs=0.01)

    # braking: ~8 m/s then ~2 m/s, still heading north
    assert samples[5].speed_mps == pytest.approx(8.0, abs=0.01)
    assert samples[6].speed_mps == pytest.approx(2.0, abs=0.01)
    assert samples[5].heading_degrees == pytest.approx(0.0, abs=0.01)

    # sharp turn to due east at the same low speed
    assert samples[7].speed_mps == pytest.approx(2.0, abs=0.01)
    assert samples[7].heading_degrees == pytest.approx(90.0, abs=0.01)

    # speed spike, still heading east
    assert samples[8].speed_mps == pytest.approx(25.0, abs=0.01)
    assert samples[8].heading_degrees == pytest.approx(90.0, abs=0.01)


def test_compute_telemetry_samples_requires_at_least_two_points() -> None:
    with pytest.raises(TelemetryAnalysisError):
        compute_telemetry_samples([TrackPoint(latitude=1.0, longitude=2.0, time=datetime.now(UTC))])


def test_compute_telemetry_samples_skips_a_non_increasing_timestamp() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    points = [
        TrackPoint(latitude=1.0, longitude=1.0, time=t),
        TrackPoint(latitude=1.0001, longitude=1.0, time=t),  # duplicate timestamp -- skip
        TrackPoint(latitude=1.0002, longitude=1.0, time=t.replace(second=1)),
    ]
    samples = compute_telemetry_samples(points)
    assert len(samples) == 1  # only the third point produced a valid interval


def test_detect_telemetry_events_finds_exactly_the_constructed_events() -> None:
    """Reproduces the fixture's constructed ground truth end to end: 2
    hard-braking events (the two-step deceleration), 1 sudden heading
    change (the 90-degree turn), and 1 speed spike -- no more, no fewer."""

    points = parse_gpx(FIXTURE_PATH)
    samples = compute_telemetry_samples(points)
    events = detect_telemetry_events(
        samples,
        hard_braking_threshold_mps2=4.0,
        speed_spike_threshold_mps=20.0,
        heading_change_threshold_deg_per_sec=45.0,
    )

    event_types = [e.event_type for e in events]
    assert event_types.count("hard_braking") == 2
    assert event_types.count("sudden_heading_change") == 1
    assert event_types.count("speed_spike") == 1
    assert len(events) == 4

    for event in events:
        assert 0.0 <= event.confidence <= 1.0


def test_detect_telemetry_events_finds_nothing_during_steady_cruise() -> None:
    points = parse_gpx(FIXTURE_PATH)
    samples = compute_telemetry_samples(points)[:5]  # cruise-only samples
    events = detect_telemetry_events(
        samples,
        hard_braking_threshold_mps2=4.0,
        speed_spike_threshold_mps=20.0,
        heading_change_threshold_deg_per_sec=45.0,
    )
    assert events == []
