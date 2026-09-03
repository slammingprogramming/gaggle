from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gaggle.cli.app import (
    _covered_clip_ids,
    _merge_ingest_manifests,
    _partition_pending_manifests,
)
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.schemas.media import IngestManifest, MediaClip
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_clip(camera_id: str = "front") -> MediaClip:
    clip_id = new_uuid()
    return MediaClip(
        clip_id=clip_id,
        camera_id=camera_id,
        source_path=f"/source/{clip_id}.mp4",
        stored_path=f"/workspace/originals/{clip_id}.mp4",
        filename=f"{clip_id}.mp4",
        media_type="video",
        byte_size=1024,
        sha256=f"deadbeef{clip_id.hex}",
        observed_start=BASE,
        observed_end=BASE,
        original_timestamp_source="filename",
        timestamp_confidence=1.0,
        duration_seconds=10.0,
    )


def _make_manifest(*clips: MediaClip, source_root: str = "/source") -> IngestManifest:
    return IngestManifest(
        run_id=new_uuid(),
        created_at=BASE,
        source_root=source_root,
        copied_files=list(clips),
    )


def _make_event(covering_clip_ids: list[object] | None = None) -> EventRecord:
    derived_artifacts = [
        ArtifactReference(
            path=f"/workspace/events/x/clips/{clip_id}.mp4",
            artifact_type="derived_clip",
            created_at=BASE,
            metadata={"source_clip_id": str(clip_id)},
        )
        for clip_id in (covering_clip_ids or [])
    ]
    return EventRecord(
        event_id=new_uuid(),
        created_at=BASE,
        pipeline_version="test",
        event_start=BASE,
        event_end=BASE,
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
        derived_artifacts=derived_artifacts,
    )


# -- _covered_clip_ids --------------------------------------------------


def test_covered_clip_ids_collects_source_clip_ids_from_every_event(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    clip_a, clip_b, clip_c = new_uuid(), new_uuid(), new_uuid()
    repository.save_event(_make_event([clip_a]))
    repository.save_event(_make_event([clip_b, clip_c]))

    covered = _covered_clip_ids(repository)

    assert covered == {clip_a, clip_b, clip_c}


def test_covered_clip_ids_is_empty_with_no_events(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    assert _covered_clip_ids(repository) == set()


# -- _partition_pending_manifests ----------------------------------------


def test_partition_treats_a_manifest_as_pending_when_none_of_its_clips_are_covered() -> None:
    clip = _make_clip()
    manifest = _make_manifest(clip)

    pending, already_analyzed = _partition_pending_manifests([manifest], covered_clip_ids=set())

    assert pending == [manifest]
    assert already_analyzed == []


def test_partition_treats_a_manifest_as_already_analyzed_if_any_clip_is_covered() -> None:
    """A manifest is all-or-nothing by construction (`AnalysisPipeline.analyze()`
    never partially processes one), so a single covered clip is enough to
    prove the whole manifest already went through `analyze` -- even
    though its *other* clips (e.g. genuinely benign ones) never produced
    an event of their own."""

    covered_clip, benign_clip = _make_clip(), _make_clip()
    manifest = _make_manifest(covered_clip, benign_clip)

    pending, already_analyzed = _partition_pending_manifests(
        [manifest], covered_clip_ids={covered_clip.clip_id}
    )

    assert pending == []
    assert already_analyzed == [manifest]


def test_partition_treats_an_empty_manifest_as_already_analyzed() -> None:
    """Regression test: a manifest with zero copied_files (e.g. an ingest
    that deduplicated away to nothing) has nothing to analyze, ever --
    without this, `_covered_clip_ids` would find no overlap for it (empty
    set intersects nothing) and it would be reprocessed by every future
    `analyze` call forever, running the full pipeline for no reason."""

    empty_manifest = _make_manifest()

    pending, already_analyzed = _partition_pending_manifests(
        [empty_manifest], covered_clip_ids=set()
    )

    assert pending == []
    assert already_analyzed == [empty_manifest]


def test_partition_handles_a_mix_of_pending_and_already_analyzed_manifests() -> None:
    old_clip = _make_clip()
    old_manifest = _make_manifest(old_clip, source_root="/old")
    new_manifest = _make_manifest(_make_clip(), source_root="/new")

    pending, already_analyzed = _partition_pending_manifests(
        [old_manifest, new_manifest], covered_clip_ids={old_clip.clip_id}
    )

    assert pending == [new_manifest]
    assert already_analyzed == [old_manifest]


# -- _merge_ingest_manifests ----------------------------------------------


def test_merge_combines_every_pending_manifests_clips_into_one() -> None:
    clip_a, clip_b = _make_clip(), _make_clip()
    manifest_a = _make_manifest(clip_a, source_root="/a")
    manifest_b = _make_manifest(clip_b, source_root="/b")

    merged = _merge_ingest_manifests([manifest_a, manifest_b])

    assert {clip.clip_id for clip in merged.copied_files} == {clip_a.clip_id, clip_b.clip_id}
    assert merged.source_root == "/a; /b"
    # A fresh run_id, not reused from either source manifest -- this is a
    # synthesized manifest, not one of the real ingest records.
    assert merged.run_id not in (manifest_a.run_id, manifest_b.run_id)
