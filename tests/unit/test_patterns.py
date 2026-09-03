from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from gaggle.patterns.service import EncounterIdentityPair, PatternAnalysisService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.signal import Signal

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_event(
    start_offset_seconds: float,
    cameras: list[str],
    object_labels: list[str] | None = None,
    vehicle_cluster_ids: list[str] | None = None,
) -> EventRecord:
    start = BASE + timedelta(seconds=start_offset_seconds)
    signals = [
        Signal(
            id=uuid4(),
            source="test",
            signal_type="object_hint",
            timestamp_start=start,
            timestamp_end=start + timedelta(seconds=1),
            confidence=0.5,
            camera_id=cameras[0] if cameras else None,
            reasoning_metadata={"label": label},
        )
        for label in (object_labels or [])
    ] + [
        Signal(
            id=uuid4(),
            source="test",
            signal_type="vehicle_appearance",
            timestamp_start=start,
            timestamp_end=start + timedelta(seconds=1),
            confidence=0.5,
            camera_id=cameras[0] if cameras else None,
            reasoning_metadata={"cluster_id": cluster_id},
        )
        for cluster_id in (vehicle_cluster_ids or [])
    ]
    return EventRecord(
        event_id=uuid4(),
        created_at=start,
        pipeline_version="test",
        event_start=start,
        event_end=start + timedelta(seconds=1),
        involved_cameras=cameras,
        signals=signals,
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def test_repeated_camera_activity_requires_minimum_count() -> None:
    events = [_make_event(i * 10, ["front"]) for i in range(3)]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    camera_patterns = [p for p in patterns if p["pattern_type"] == "repeated_camera_activity"]
    assert len(camera_patterns) == 1
    assert camera_patterns[0]["camera_id"] == "front"
    assert camera_patterns[0]["count"] == 3
    assert camera_patterns[0]["hypothesis_only"] is True


def test_single_event_camera_does_not_trigger_pattern() -> None:
    events = [_make_event(0, ["front"])]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    assert not [p for p in patterns if p["pattern_type"] == "repeated_camera_activity"]


def test_repeated_object_label_pattern() -> None:
    events = [
        _make_event(i * 10, ["front"], object_labels=["unclassified_moving_region"])
        for i in range(2)
    ]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    label_patterns = [p for p in patterns if p["pattern_type"] == "repeated_object_label"]
    assert len(label_patterns) == 1
    assert label_patterns[0]["label"] == "unclassified_moving_region"


def test_repeated_vehicle_appearance_pattern() -> None:
    events = [_make_event(i * 600, ["front"], vehicle_cluster_ids=["cluster-a"]) for i in range(3)]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    vehicle_patterns = [p for p in patterns if p["pattern_type"] == "repeated_vehicle_appearance"]
    assert len(vehicle_patterns) == 1
    assert vehicle_patterns[0]["cluster_id"] == "cluster-a"
    assert vehicle_patterns[0]["count"] == 3
    assert vehicle_patterns[0]["first_seen_at"] == events[0].event_start.isoformat()
    assert vehicle_patterns[0]["last_seen_at"] == events[-1].event_start.isoformat()
    assert vehicle_patterns[0]["hypothesis_only"] is True


def test_a_single_vehicle_appearance_does_not_trigger_pattern() -> None:
    events = [_make_event(0, ["front"], vehicle_cluster_ids=["cluster-a"])]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    assert not [p for p in patterns if p["pattern_type"] == "repeated_vehicle_appearance"]


def test_two_distinct_vehicle_clusters_are_tracked_independently() -> None:
    events = [
        _make_event(0, ["front"], vehicle_cluster_ids=["cluster-a"]),
        _make_event(10, ["front"], vehicle_cluster_ids=["cluster-a"]),
        _make_event(20, ["front"], vehicle_cluster_ids=["cluster-b"]),
    ]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    vehicle_patterns = [p for p in patterns if p["pattern_type"] == "repeated_vehicle_appearance"]
    assert len(vehicle_patterns) == 1  # cluster-b only appeared once, below min_repeat_count
    assert vehicle_patterns[0]["cluster_id"] == "cluster-a"


def test_temporal_clustering_groups_close_events() -> None:
    close_events = [_make_event(i * 60, ["front"]) for i in range(3)]  # within 1 hour
    far_event = _make_event(10 * 3600, ["front"])
    patterns = PatternAnalysisService().analyze(
        [*close_events, far_event], cluster_window_seconds=3600.0, min_repeat_count=2
    )
    clusters = [p for p in patterns if p["pattern_type"] == "temporal_clustering"]
    assert len(clusters) == 1
    assert clusters[0]["event_count"] == 3


def test_patterns_are_always_marked_hypothesis_only() -> None:
    events = [_make_event(i * 10, ["front", "rear"]) for i in range(2)]
    patterns = PatternAnalysisService().analyze(events, min_repeat_count=2)
    assert all(p["hypothesis_only"] is True for p in patterns)


def test_analyze_without_encounter_identities_behaves_identically_to_before() -> None:
    """Regression guard: omitting `encounter_identities` (the default)
    must not change any of the three pre-existing pattern methods."""

    events = [_make_event(i * 10, ["front"], vehicle_cluster_ids=["cluster-a"]) for i in range(3)]
    with_default = PatternAnalysisService().analyze(events, min_repeat_count=2)
    with_explicit_none = PatternAnalysisService().analyze(
        events, min_repeat_count=2, encounter_identities=None
    )
    assert with_default == with_explicit_none
    assert not [
        p for p in with_default if p["pattern_type"] == "recurring_face_vehicle_cooccurrence"
    ]


def test_recurring_face_vehicle_cooccurrence_requires_minimum_count() -> None:
    events = [_make_event(0, ["front"])]
    identities = [
        EncounterIdentityPair(face_cluster_id="face-a", vehicle_cluster_id="vehicle-a")
        for _ in range(3)
    ]
    patterns = PatternAnalysisService().analyze(
        events, min_repeat_count=2, encounter_identities=identities
    )
    cooccurrence = [
        p for p in patterns if p["pattern_type"] == "recurring_face_vehicle_cooccurrence"
    ]
    assert len(cooccurrence) == 1
    assert cooccurrence[0]["face_cluster_id"] == "face-a"
    assert cooccurrence[0]["vehicle_cluster_id"] == "vehicle-a"
    assert cooccurrence[0]["count"] == 3
    assert cooccurrence[0]["hypothesis_only"] is True


def test_a_single_cooccurrence_does_not_trigger_pattern() -> None:
    events = [_make_event(0, ["front"])]
    identities = [EncounterIdentityPair(face_cluster_id="face-a", vehicle_cluster_id="vehicle-a")]
    patterns = PatternAnalysisService().analyze(
        events, min_repeat_count=2, encounter_identities=identities
    )
    assert not [p for p in patterns if p["pattern_type"] == "recurring_face_vehicle_cooccurrence"]


def test_cooccurrence_ignores_pairs_missing_either_identity() -> None:
    events = [_make_event(0, ["front"])]
    identities = [
        EncounterIdentityPair(face_cluster_id="face-a", vehicle_cluster_id=None),
        EncounterIdentityPair(face_cluster_id=None, vehicle_cluster_id="vehicle-a"),
        EncounterIdentityPair(face_cluster_id=None, vehicle_cluster_id=None),
    ]
    patterns = PatternAnalysisService().analyze(
        events, min_repeat_count=2, encounter_identities=identities
    )
    assert not [p for p in patterns if p["pattern_type"] == "recurring_face_vehicle_cooccurrence"]


def test_distinct_face_vehicle_pairs_are_tracked_independently() -> None:
    events = [_make_event(0, ["front"])]
    identities = (
        [EncounterIdentityPair("face-a", "vehicle-a") for _ in range(2)]
        + [EncounterIdentityPair("face-a", "vehicle-b") for _ in range(2)]
        + [EncounterIdentityPair("face-b", "vehicle-a")]  # below min_repeat_count
    )
    patterns = PatternAnalysisService().analyze(
        events, min_repeat_count=2, encounter_identities=identities
    )
    cooccurrence = [
        p for p in patterns if p["pattern_type"] == "recurring_face_vehicle_cooccurrence"
    ]
    pairs = {(p["face_cluster_id"], p["vehicle_cluster_id"]) for p in cooccurrence}
    assert pairs == {("face-a", "vehicle-a"), ("face-a", "vehicle-b")}
