from __future__ import annotations

import json
import shutil
from itertools import pairwise
from pathlib import Path
from uuid import UUID

import pytest

from gaggle.core.config import load_config
from gaggle.core.pipeline import AnalysisPipeline
from gaggle.core.review import ReviewService
from gaggle.export.service import ExportService
from gaggle.ingest.service import IngestService
from gaggle.schemas.event import EVENT_SCHEMA_VERSION
from gaggle.storage.database import TimelineQuery
from gaggle.storage.repository import Repository
from gaggle.timeline.service import TimelineService

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _run_pipeline(tmp_path: Path) -> tuple[Repository, list]:
    source = tmp_path / "source"
    shutil.copytree(Path("examples/sample_media"), source)
    workspace = tmp_path / "workspace"
    config = load_config(Path("examples/config.yaml"))
    repository = Repository(workspace)
    repository.initialize()
    ingest_manifest = IngestService(repository.workspace, config).ingest_directory(source)
    repository.index_ingest_manifest(ingest_manifest)
    events = AnalysisPipeline(repository, config).analyze(ingest_manifest)
    return repository, events


def test_ingest_extracts_real_media_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(Path("examples/sample_media"), source)
    config = load_config(Path("examples/config.yaml"))
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    manifest = IngestService(repository.workspace, config).ingest_directory(source)

    assert len(manifest.copied_files) == 2
    for clip in manifest.copied_files:
        # A hardcoded 10.0s fallback would be a strong signal probing silently
        # failed; the real sample clips are ~12s.
        assert clip.duration_seconds == pytest.approx(12.0, abs=0.5)
        assert clip.fps == pytest.approx(15.0, abs=0.5)
        assert clip.metadata.get("probe_status") == "ok"


def test_pipeline_detects_real_motion_and_audio_signals(tmp_path: Path) -> None:
    _repository, events = _run_pipeline(tmp_path)
    assert events
    all_signal_types = {s.signal_type for event in events for s in event.signals}
    assert "motion" in all_signal_types
    # the front camera's synthetic horn spike must clear the default threshold
    assert "audio_spike" in all_signal_types


def test_events_have_timezone_aware_utc_timestamps(tmp_path: Path) -> None:
    _, events = _run_pipeline(tmp_path)
    for event in events:
        assert event.event_start.tzinfo is not None
        assert event.event_end.tzinfo is not None


def test_event_json_is_written_and_matches_schema_version(tmp_path: Path) -> None:
    repository, events = _run_pipeline(tmp_path)
    event = events[0]
    assert event.schema_version == EVENT_SCHEMA_VERSION
    event_path = repository.workspace.event_dir(event.event_id) / "event.json"
    assert event_path.exists()
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    assert payload["event_id"] == str(event.event_id)
    assert payload["revision"] == 0


def test_overlapping_windows_do_not_produce_duplicate_events(tmp_path: Path) -> None:
    """Regression test: sliding windows intentionally overlap (stride <
    duration); without cluster-merging this used to produce several
    near-duplicate events for one continuous span of activity."""

    _, events = _run_pipeline(tmp_path)
    intervals = sorted((event.event_start, event.event_end) for event in events)
    for (_, first_end), (second_start, _) in pairwise(intervals):
        assert second_start >= first_end


def test_derived_clips_are_generated_and_hashed(tmp_path: Path) -> None:
    _, events = _run_pipeline(tmp_path)
    events_with_clips = [e for e in events if e.derived_artifacts]
    assert events_with_clips, "expected at least one event to have a derived clip"
    for event in events_with_clips:
        for artifact in event.derived_artifacts:
            assert artifact.artifact_type == "derived_clip"
            assert Path(artifact.path).exists()
            assert artifact.sha256 is not None


def test_preservation_creates_immutable_bundle_and_updates_event_json(tmp_path: Path) -> None:
    repository, events = _run_pipeline(tmp_path)
    event = events[0]
    config = load_config(Path("examples/config.yaml"))
    preserved = AnalysisPipeline(repository, config).preserve_event(event.event_id)

    assert preserved.preservation_status.immutable is True
    assert preserved.revision == 1
    manifest = json.loads(
        Path(preserved.preservation_status.bundle_path, "bundle_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["event_id"] == str(event.event_id)

    # the live event.json must reflect the new state -- this is the fix for
    # the historical bug where preservation only updated SQLite.
    reloaded = repository.load_event(event.event_id)
    assert reloaded.preservation_status.state == "preserved"
    assert reloaded.revision == 1


def test_review_action_updates_event_and_index(tmp_path: Path) -> None:
    repository, events = _run_pipeline(tmp_path)
    event = events[0]
    service = ReviewService(repository)
    _, updated = service.append_action(event.event_id, "accept", "tester", notes="looks real")

    assert updated.review_summary.latest_decision == "accepted"
    indexed = TimelineService(repository.database).list_events(
        TimelineQuery(review_decision="accepted")
    )
    assert any(UUID(row["event_id"]) == event.event_id for row in indexed)


def test_export_bundle_includes_preserved_evidence(tmp_path: Path) -> None:
    repository, events = _run_pipeline(tmp_path)
    event = events[0]
    config = load_config(Path("examples/config.yaml"))
    AnalysisPipeline(repository, config).preserve_event(event.event_id)

    result = ExportService(repository).export_event_bundle(event.event_id)
    assert result.path.exists()
    assert result.file_count > 0
