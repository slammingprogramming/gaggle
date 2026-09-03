from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gaggle.normalize.sync import ClipTimeInfo, compute_camera_sync

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def test_single_camera_is_its_own_reference() -> None:
    clips = [
        ClipTimeInfo("c1", "front", BASE, BASE + timedelta(seconds=60), 0.7),
    ]
    results = compute_camera_sync(clips)
    assert len(results) == 1
    assert results[0].is_reference is True
    assert results[0].offset_seconds == 0.0
    assert results[0].corrected_start == BASE


def test_higher_confidence_camera_becomes_reference() -> None:
    clips = [
        ClipTimeInfo("front-1", "front", BASE, BASE + timedelta(seconds=300), 0.7),
        ClipTimeInfo(
            "rear-1", "rear", BASE + timedelta(seconds=12), BASE + timedelta(seconds=305), 0.3
        ),
    ]
    results = compute_camera_sync(
        clips,
        session_gap_seconds=120.0,
        site_id_by_camera={"front": "site-a", "rear": "site-a"},
    )
    by_camera = {r.camera_id: r for r in results}
    assert by_camera["front"].is_reference is True
    assert by_camera["rear"].is_reference is False
    assert by_camera["rear"].reference_camera_id == "front"
    # rear started 12s after front; correcting it should align its start to front's.
    assert by_camera["rear"].corrected_start == BASE


def test_equal_confidence_ties_broken_alphabetically() -> None:
    clips = [
        ClipTimeInfo("z-1", "zebra", BASE, BASE + timedelta(seconds=60), 0.5),
        ClipTimeInfo("a-1", "alpha", BASE, BASE + timedelta(seconds=60), 0.5),
    ]
    results = compute_camera_sync(clips, site_id_by_camera={"zebra": "site-a", "alpha": "site-a"})
    reference = next(r for r in results if r.is_reference)
    assert reference.camera_id == "alpha"


def test_non_overlapping_sessions_do_not_share_a_reference() -> None:
    clips = [
        ClipTimeInfo("front-1", "front", BASE, BASE + timedelta(seconds=60), 0.7),
        ClipTimeInfo(
            "front-2",
            "front",
            BASE + timedelta(hours=5),
            BASE + timedelta(hours=5, seconds=60),
            0.7,
        ),
    ]
    results = compute_camera_sync(clips, session_gap_seconds=120.0)
    assert len(results) == 2
    assert all(r.is_reference for r in results)  # each session is alone in its group


def test_session_grouping_respects_gap_threshold() -> None:
    clips = [
        ClipTimeInfo("c1", "front", BASE, BASE + timedelta(seconds=60), 0.7),
        # 300s gap: exceeds a 120s session_gap_seconds threshold -> new session.
        ClipTimeInfo(
            "c2",
            "front",
            BASE + timedelta(seconds=360),
            BASE + timedelta(seconds=420),
            0.7,
        ),
    ]
    results = compute_camera_sync(clips, session_gap_seconds=120.0)
    assert len(results) == 2
    assert {r.session_id for r in results} == {"front#000", "front#001"}


def test_output_is_deterministic_across_runs() -> None:
    clips = [
        ClipTimeInfo("front-1", "front", BASE, BASE + timedelta(seconds=300), 0.7),
        ClipTimeInfo(
            "rear-1", "rear", BASE + timedelta(seconds=8), BASE + timedelta(seconds=301), 0.3
        ),
    ]
    site_map = {"front": "site-a", "rear": "site-a"}
    first = compute_camera_sync(clips, site_id_by_camera=site_map)
    second = compute_camera_sync(clips, site_id_by_camera=site_map)
    assert first == second


def test_default_site_scoping_isolates_different_cameras_with_no_site_info() -> None:
    """With no `site_id_by_camera` passed, every camera defaults to its own
    private site (keyed by its own camera_id) -- the safe default so two
    unrelated cameras never get spuriously cross-synced just because a
    caller didn't supply site metadata. See compute_camera_sync's
    docstring."""

    clips = [
        ClipTimeInfo("front-1", "front", BASE, BASE + timedelta(seconds=300), 0.7),
        ClipTimeInfo(
            "rear-1", "rear", BASE + timedelta(seconds=8), BASE + timedelta(seconds=301), 0.3
        ),
    ]
    results = compute_camera_sync(clips, session_gap_seconds=120.0)
    assert len(results) == 2
    assert all(r.is_reference for r in results)  # neither is aligned to the other


def test_cameras_sharing_a_site_still_cross_sync() -> None:
    clips = [
        ClipTimeInfo("front-1", "front", BASE, BASE + timedelta(seconds=300), 0.7),
        ClipTimeInfo(
            "rear-1", "rear", BASE + timedelta(seconds=12), BASE + timedelta(seconds=305), 0.3
        ),
    ]
    results = compute_camera_sync(
        clips,
        session_gap_seconds=120.0,
        site_id_by_camera={"front": "site-a", "rear": "site-a"},
    )
    by_camera = {r.camera_id: r for r in results}
    assert by_camera["rear"].is_reference is False
    assert by_camera["rear"].reference_camera_id == "front"


def test_cameras_in_different_sites_never_cross_sync_even_when_fully_overlapping() -> None:
    """Two independent cameras (e.g. a neighbor's security camera and your
    dashcam) with fully overlapping timestamps must never be aligned to
    each other just because they happened to record at the same time --
    unrelated clocks, unrelated recordings."""

    clips = [
        ClipTimeInfo("dashcam-1", "dashcam", BASE, BASE + timedelta(seconds=300), 0.9),
        ClipTimeInfo("porch-1", "porch", BASE, BASE + timedelta(seconds=300), 0.9),
    ]
    results = compute_camera_sync(
        clips,
        session_gap_seconds=120.0,
        site_id_by_camera={"dashcam": "site-home", "porch": "site-neighbor"},
    )
    assert len(results) == 2
    assert all(r.is_reference for r in results)
    assert all(r.reference_camera_id is None for r in results)
