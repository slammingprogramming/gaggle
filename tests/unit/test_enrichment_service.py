from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from gaggle.core.config import RuntimeConfig
from gaggle.enrichment import service as service_module
from gaggle.enrichment.service import EnrichmentService
from gaggle.enrichment.transcription import TranscribedSegment, TranscriptionResult
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC)


def _make_event() -> EventRecord:
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
    )


def test_enrichment_is_a_no_op_when_every_capability_is_disabled(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = False
    config.enrichment.transcription.enabled = False
    config.enrichment.cloud.enabled = False
    config.enrichment.encounters.enabled = False

    updated = EnrichmentService(repository, config).enrich_event(event.event_id)

    # Nothing enabled, nothing attempted -> no new revision at all.
    assert updated.revision == 0
    assert updated.signals == []
    assert updated.enrichment_completed == {}


def test_enrichment_skips_cleanly_when_event_has_no_derived_clips(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()  # derived_artifacts defaults to []
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = True  # enabled, but nothing to process
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = False
    config.enrichment.transcription.enabled = False
    config.enrichment.cloud.enabled = False
    config.enrichment.encounters.enabled = False

    updated = EnrichmentService(repository, config).enrich_event(event.event_id)

    # Face was attempted (marked complete) even though there was nothing
    # to actually process -- no clips means no future run could ever find
    # anything either, so it's correctly recorded as done, not left
    # pending forever.
    assert updated.signals == []
    assert "face" in updated.enrichment_completed

    # Calling it again does not attempt face a second time (already
    # marked complete) -> no new revision this time.
    again = EnrichmentService(repository, config).enrich_event(event.event_id)
    assert again.revision == updated.revision


def test_missing_tesseract_is_checked_and_warned_about_exactly_once(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression test: a real user reported plate OCR silently failing once
    per detected region (potentially hundreds of times per run) when
    tesseract wasn't installed, flooding the log with an identical warning
    and wastefully spawning a doomed subprocess every time.
    `_check_tesseract_once` must check availability exactly once per
    `EnrichmentService` instance and skip cleanly afterward."""

    import gaggle.enrichment.plate as plate_module

    monkeypatch.setattr(plate_module, "tesseract_available", lambda: False)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    service = EnrichmentService(repository, config)

    assert service._tesseract_checked is False

    first_result = service._check_tesseract_once()
    assert first_result is False
    assert service._tesseract_checked is True

    # A second call must not re-check availability or log another warning --
    # if it did, this would still return False (same monkeypatched value),
    # so the only way to prove it isn't re-checking is to change what the
    # (now-ignored) underlying check would return and confirm the cached
    # value wins.
    monkeypatch.setattr(plate_module, "tesseract_available", lambda: True)
    second_result = service._check_tesseract_once()
    assert second_result is False  # still the cached (stale) value, not re-checked


def test_available_tesseract_is_cached_as_available(tmp_path: Path, monkeypatch) -> None:
    import gaggle.enrichment.plate as plate_module

    monkeypatch.setattr(plate_module, "tesseract_available", lambda: True)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    config = RuntimeConfig()
    service = EnrichmentService(repository, config)

    assert service._check_tesseract_once() is True
    assert service._tesseract_available is True


def test_plate_recognition_returns_empty_immediately_without_tesseract(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end sanity check that `enrich_event` doesn't crash when plate
    recognition is enabled but tesseract is missing. The precise claim that
    `_check_tesseract_once` short-circuits *before* frame sampling/detection
    runs is proven directly by
    `test_missing_tesseract_is_checked_and_warned_about_exactly_once` above,
    independent of clip paths -- this test only confirms the whole pipeline
    still produces a clean, empty result rather than raising."""

    import gaggle.enrichment.plate as plate_module

    monkeypatch.setattr(plate_module, "tesseract_available", lambda: False)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = True

    updated = EnrichmentService(repository, config).enrich_event(event.event_id)
    assert updated.signals == []


def _make_synthetic_vehicle_clip(path: Path) -> None:
    """A short real video (readable by cv2.VideoCapture, unlike a fixture
    JSON) containing one colored rectangle standing in for a vehicle body
    -- same synthetic-scene construction as
    tests/unit/test_vehicle_appearance.py, just written to real video
    frames instead of a single in-memory array, so this test exercises
    the actual frame-sampling -> region-detection -> fingerprinting ->
    clustering -> signal/observation wiring end to end, not just the
    pieces in isolation."""

    canvas = np.full((300, 500, 3), (60, 60, 60), dtype=np.uint8)
    cv2.rectangle(canvas, (140, 90), (360, 190), (30, 30, 200), -1)
    cv2.rectangle(canvas, (150, 95), (350, 120), (20, 20, 20), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (500, 300))
    try:
        for _ in range(10):
            writer.write(canvas)
    finally:
        writer.release()


def test_vehicle_appearance_enrichment_produces_a_real_signal_and_cluster(
    tmp_path: Path,
) -> None:
    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_vehicle_clip(clip_path)

    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    source_clip_id = new_uuid()
    event = EventRecord(
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
        derived_artifacts=[
            ArtifactReference(
                path=str(clip_path),
                artifact_type="derived_clip",
                created_at=BASE,
                metadata={"source_clip_id": str(source_clip_id)},
            )
        ],
    )
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = True

    updated = EnrichmentService(repository, config).enrich_event(event.event_id)

    vehicle_signals = [s for s in updated.signals if s.signal_type == "vehicle_appearance"]
    assert vehicle_signals

    clusters = repository.database.list_vehicle_appearance_clusters()
    assert len(clusters) == 1
    assert clusters[0].observation_count >= 1

    observations = repository.database.list_all_vehicle_appearance_observations()
    assert observations
    assert Path(observations[0].crop_path).exists()
    assert observations[0].clip_id == str(source_clip_id)


def _make_event_with_derived_clip(clip_path: Path, source_clip_id: object) -> EventRecord:
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
        derived_artifacts=[
            ArtifactReference(
                path=str(clip_path),
                artifact_type="derived_clip",
                created_at=BASE,
                metadata={"source_clip_id": str(source_clip_id)},
            )
        ],
    )


def test_rerunning_enrich_with_unchanged_config_does_not_duplicate_observations(
    tmp_path: Path,
) -> None:
    """Regression test for the core idempotency requirement: calling
    `enrich_event` twice with nothing new to do must not retrain the
    incremental clusterer on the same crop twice or inflate
    `observation_count` -- both would silently corrupt real recognition
    data if enrich were ever called more than once on the same event,
    which the user must be free to do."""

    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_vehicle_clip(clip_path)
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event_with_derived_clip(clip_path, new_uuid())
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = True

    service = EnrichmentService(repository, config)
    first = service.enrich_event(event.event_id)
    first_observation_count = len(repository.database.list_all_vehicle_appearance_observations())
    first_cluster_count = repository.database.list_vehicle_appearance_clusters()[
        0
    ].observation_count
    assert first_observation_count > 0

    second = service.enrich_event(event.event_id)

    assert second.revision == first.revision  # no new revision -- nothing was attempted
    assert second.signals == first.signals
    assert (
        len(repository.database.list_all_vehicle_appearance_observations())
        == first_observation_count
    )
    assert (
        repository.database.list_vehicle_appearance_clusters()[0].observation_count
        == first_cluster_count
    )


def test_enabling_a_new_capability_later_only_runs_that_capability(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_vehicle_clip(clip_path)
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event_with_derived_clip(clip_path, new_uuid())
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = True

    first = EnrichmentService(repository, config).enrich_event(event.event_id)
    assert "vehicle_appearance" in first.enrichment_completed
    assert "face" not in first.enrichment_completed
    observation_count_after_first = len(
        repository.database.list_all_vehicle_appearance_observations()
    )

    # Enable a previously-disabled capability and enrich again -- only the
    # newly-enabled one should run; vehicle_appearance, already complete,
    # must not be retrained.
    config.enrichment.face.enabled = True
    second = EnrichmentService(repository, config).enrich_event(event.event_id)

    assert "face" in second.enrichment_completed
    assert (
        second.enrichment_completed["vehicle_appearance"]
        == first.enrichment_completed["vehicle_appearance"]
    )
    assert (
        len(repository.database.list_all_vehicle_appearance_observations())
        == observation_count_after_first
    )


def test_force_reruns_an_already_completed_capability(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_vehicle_clip(clip_path)
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event_with_derived_clip(clip_path, new_uuid())
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = True

    service = EnrichmentService(repository, config)
    service.enrich_event(event.event_id)
    observation_count_before = len(repository.database.list_all_vehicle_appearance_observations())
    assert observation_count_before > 0

    service.enrich_event(event.event_id, force=True)

    # force=True deliberately reprocesses -- this is the escape hatch, so
    # more observations (from the same clip, detected again) is the
    # expected/correct outcome here, not a bug.
    assert (
        len(repository.database.list_all_vehicle_appearance_observations())
        > observation_count_before
    )


def _fake_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        segments=[
            TranscribedSegment(
                start_offset_seconds=0.0, end_offset_seconds=1.0, text="hello", confidence=0.9
            )
        ],
        full_text="hello world",
        model_name="fake-model",
        device="cpu",
    )


def test_cloud_analysis_reuses_the_saved_transcript_without_rerunning_whisper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transcription and cloud/LLM analysis are independently toggleable,
    but cloud needs transcript *text* to do anything. If transcription
    already completed on an earlier `enrich_event` call and cloud is only
    now being enabled, re-running Whisper again just to get the same text
    a second time would defeat the entire point of skipping
    already-completed capabilities -- the saved sidecar file must be
    reused instead."""

    transcribe_calls: list[Path] = []

    class _FakeWhisperTranscriber:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def transcribe(self, audio_path: Path) -> TranscriptionResult:
            transcribe_calls.append(audio_path)
            return _fake_transcription_result()

    monkeypatch.setattr(service_module, "WhisperTranscriber", _FakeWhisperTranscriber)

    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_vehicle_clip(clip_path)
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event_with_derived_clip(clip_path, new_uuid())
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = False
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = False
    config.enrichment.transcription.enabled = True
    config.enrichment.cloud.enabled = False

    service = EnrichmentService(repository, config)
    first = service.enrich_event(event.event_id)
    assert len(transcribe_calls) == 1
    assert "transcription" in first.enrichment_completed
    assert "cloud" not in first.enrichment_completed

    # Cloud is enabled now; transcription is already complete and stays
    # off in terms of re-attempting -- Whisper must not be invoked again.
    config.enrichment.cloud.enabled = True
    second = service.enrich_event(event.event_id)

    assert len(transcribe_calls) == 1  # still just the one real call
    assert "cloud" in second.enrichment_completed


FIXTURE_FACE = Path(__file__).parent.parent / "fixtures" / "sample_face.jpg"


def _make_synthetic_face_clip(path: Path) -> None:
    """A short real video built from the real face fixture photo (not a
    fixture JSON) -- readable by `cv2.VideoCapture` and containing a
    genuinely detectable face, so a test can exercise real Haar detection
    end to end rather than mocking the detector itself."""

    image = cv2.imread(str(FIXTURE_FACE))
    assert image is not None, f"missing test fixture: {FIXTURE_FACE}"
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (width, height))
    try:
        for _ in range(5):
            writer.write(image)
    finally:
        writer.release()


def test_face_recognition_falls_back_to_lbph_when_auraface_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real bug: `_match_face_cluster`'s docstring
    always claimed a failed AuraFace load falls back to LBPH, but the
    code actually returned `None` and silently dropped the detection
    instead -- every face detection would vanish with zero signals
    whenever `embedding_model: auraface` was configured but the
    `face_recognition` extra wasn't installed (or CUDA/the model
    otherwise failed to load), which matters a great deal now that
    `auraface` is the *default* embedding model. A real detected face
    must still produce a real signal via the LBPH fallback."""

    class _AlwaysUnavailable:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise service_module.face_auraface_module.AuraFaceUnavailableError(
                "insightface not installed"
            )

    monkeypatch.setattr(service_module.face_auraface_module, "AuraFaceEmbedder", _AlwaysUnavailable)

    clip_path = tmp_path / "clip.mp4"
    _make_synthetic_face_clip(clip_path)
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event_with_derived_clip(clip_path, new_uuid())
    repository.save_event(event)

    config = RuntimeConfig()
    config.enrichment.face.enabled = True
    config.enrichment.face.detector = "haar"  # no network needed for detection
    config.enrichment.face.embedding_model = "auraface"  # will fail to load, must fall back
    config.enrichment.plate.enabled = False
    config.enrichment.voice.enabled = False
    config.enrichment.vision.enabled = False
    config.enrichment.vehicle_appearance.enabled = False

    updated = EnrichmentService(repository, config).enrich_event(event.event_id)

    face_signals = [s for s in updated.signals if s.signal_type == "face_detection"]
    assert face_signals, "a real detected face must not be silently dropped"
    lbph_version = service_module.face_module.RECOGNIZER_VERSION
    assert face_signals[0].reasoning_metadata["model_version"] == lbph_version
