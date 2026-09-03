from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gaggle.core.pipeline import AnalysisPipeline
from gaggle.schemas.media import EventWindow
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _window(start_offset_seconds: float, duration_seconds: float = 10.0) -> EventWindow:
    start = BASE + timedelta(seconds=start_offset_seconds)
    return EventWindow(
        window_id=new_uuid(),
        start=start,
        end=start + timedelta(seconds=duration_seconds),
        involved_cameras=["front"],
        clip_ids=[new_uuid()],
        rationale="test",
    )


def _populated(windows: list[EventWindow]) -> list[tuple[EventWindow, list, list]]:
    return [(window, [], []) for window in windows]


def test_overlapping_windows_merge_into_one_cluster_with_no_cap() -> None:
    windows = [_window(0), _window(5), _window(10), _window(15)]  # each overlaps the next
    clusters = AnalysisPipeline._cluster_overlapping_windows(_populated(windows), None)
    assert len(clusters) == 1
    assert len(clusters[0]) == 4


def test_non_overlapping_windows_produce_separate_clusters() -> None:
    windows = [_window(0), _window(1000)]
    clusters = AnalysisPipeline._cluster_overlapping_windows(_populated(windows), None)
    assert len(clusters) == 2


def test_max_event_duration_forces_a_split_even_though_windows_still_overlap() -> None:
    # Windows every 5s, each 10s long, continuously overlapping for 300s --
    # simulates near-continuous real motion throughout a long recording.
    windows = [_window(i * 5) for i in range(60)]  # spans 0s to 300s
    clusters = AnalysisPipeline._cluster_overlapping_windows(
        _populated(windows), max_event_duration_seconds=120.0
    )
    assert len(clusters) > 1
    for cluster in clusters:
        span = (cluster[-1][0].end - cluster[0][0].start).total_seconds()
        assert span <= 120.0 + 10.0  # allow one window's worth of overshoot at the split point


def test_max_event_duration_none_matches_unbounded_behavior() -> None:
    windows = [_window(i * 5) for i in range(60)]
    uncapped = AnalysisPipeline._cluster_overlapping_windows(_populated(windows), None)
    assert len(uncapped) == 1


def test_a_short_real_incident_is_unaffected_by_a_generous_cap() -> None:
    windows = [_window(0), _window(5), _window(10)]  # spans ~20s
    clusters = AnalysisPipeline._cluster_overlapping_windows(
        _populated(windows), max_event_duration_seconds=120.0
    )
    assert len(clusters) == 1
    assert len(clusters[0]) == 3
