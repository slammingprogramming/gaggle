"""Enrichment orchestration.

Runs *after* `core/pipeline.py::AnalysisPipeline.analyze()` has already
built events, and operates only on the derived clips of those events --
never on benign footage that produced zero signals, keeping enrichment
compute/storage proportional to what a human might actually review. This
is deliberately a separate pipeline stage (its own CLI command, `enrich`,
not folded into `analyze`) since face/plate/vehicle/transcription
processing is comparatively heavy and every one of its capabilities is
independently toggleable -- a user with a low-power machine can run the
core deterministic pipeline at full speed and skip enrichment entirely, or
enable only the parts they want.

New signals discovered during enrichment (face/plate/vehicle detections)
are added to the event via a new revision (see
`storage/repository.py::save_event_revision`) rather than by re-running
inference/scoring -- the original severity score stays exactly as
explainable and reproducible as it was, and enrichment findings are
additive, clearly time-stamped context a reviewer can see alongside it,
not a silent retroactive change to why an event was flagged.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import cv2
import cv2.typing

from gaggle.core.config import RuntimeConfig, VehicleAppearanceConfig
from gaggle.core.models import Device
from gaggle.detection import audio_analysis
from gaggle.enrichment import face as face_module
from gaggle.enrichment import face_auraface as face_auraface_module
from gaggle.enrichment import face_yunet as face_yunet_module
from gaggle.enrichment import person_appearance as person_appearance_module
from gaggle.enrichment import plate as plate_module
from gaggle.enrichment import plate_fast_alpr as plate_fast_alpr_module
from gaggle.enrichment import vehicle_appearance as vehicle_appearance_module
from gaggle.enrichment import voice as voice_module
from gaggle.enrichment.llm_analysis import (
    LlmEnrichmentError,
    LlmEnrichmentUnavailableError,
    analyze_transcript,
)
from gaggle.enrichment.transcription import (
    TranscriptionUnavailableError,
    WhisperTranscriber,
)
from gaggle.enrichment.vehicle_yolo import VisionModelUnavailableError, YoloOnnxDetector
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.encounter import Encounter
from gaggle.schemas.enrichment import AudioTranscript, LLMEnrichment, TranscriptSegment
from gaggle.schemas.event import EventRecord
from gaggle.schemas.recognition import (
    FaceCluster,
    FaceObservation,
    PersonAppearanceCluster,
    PersonAppearanceObservation,
    PlateObservation,
    PlateRecord,
    VehicleAppearanceCluster,
    VehicleAppearanceObservation,
    VoiceCluster,
    VoiceObservation,
)
from gaggle.schemas.signal import Signal
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

ImageArray = cv2.typing.MatLike

LOGGER = get_logger(__name__)
# The real, live-config value is `RuntimeConfig.enrichment.frame_sample_rate_hz`
# (same default); this module constant only remains as `_sample_frames`'s
# fallback default for callers that invoke it directly without a config
# (e.g. `recognize plates-debug`'s single-frame use elsewhere).
FRAME_SAMPLE_RATE_HZ = 1.0
# COCO class ids for actual vehicle body types (car/motorcycle/bus/truck)
# -- narrower than `vehicle_yolo.RELEVANT_CLASS_IDS`, which also includes
# person/bicycle (relevant to `_run_vehicle_detection`'s object-detection
# signal, but not to an appearance *fingerprint*, which only makes sense
# for a vehicle body).
_YOLO_VEHICLE_CLASS_IDS = frozenset({2, 3, 5, 7})
# COCO class id for "person" -- see enrichment/person_appearance.py's
# module docstring for why this detection path is YOLO-only.
_YOLO_PERSON_CLASS_IDS = frozenset({0})


class EnrichmentService:
    def __init__(self, repository: Repository, config: RuntimeConfig) -> None:
        self.repository = repository
        self.config = config
        self._face_clusterer: face_module.IncrementalFaceClusterer | None = None
        self._face_embedding_clusterer: (
            face_auraface_module.IncrementalFaceEmbeddingClusterer | None
        ) = None
        self._face_yunet_detector: face_yunet_module.YuNetDetector | None = None
        self._face_yunet_load_attempted = False
        self._face_embedder: face_auraface_module.AuraFaceEmbedder | None = None
        self._face_embedder_load_attempted = False
        self._fast_alpr_detector: plate_fast_alpr_module.FastAlprDetector | None = None
        self._fast_alpr_load_attempted = False
        self._voice_clusterer: voice_module.IncrementalVoiceClusterer | None = None
        self._vehicle_appearance_clusterer: (
            vehicle_appearance_module.IncrementalVehicleAppearanceClusterer | None
        ) = None
        self._person_appearance_clusterer: (
            person_appearance_module.IncrementalPersonAppearanceClusterer | None
        ) = None
        self._vehicle_detector: YoloOnnxDetector | None = None
        self._transcriber: WhisperTranscriber | None = None
        self._vehicle_load_attempted = False
        self._transcriber_load_attempted = False
        self._tesseract_checked = False
        self._tesseract_available = False

    def enrich_event(self, event_id: UUID, force: bool = False) -> EventRecord:
        """Runs every enabled enrichment capability that hasn't already
        completed on this event (tracked in `event.enrichment_completed`,
        see `schemas/event.py`) -- safe to call any number of times, in
        any order relative to `ingest`/`analyze`. A capability that
        already ran is skipped even if it found nothing that time
        (distinct from "never attempted"); a newly-enabled capability
        runs the next time this is called, regardless of what already
        ran. `force=True` reruns every enabled capability regardless of
        `enrichment_completed` -- the explicit escape hatch for "I
        changed a detector/model and want to redo this."
        """

        event = self.repository.load_event(event_id)
        new_signals: list[Signal] = []
        completed_updates: dict[str, datetime] = {}

        def should_run(capability: str) -> bool:
            return force or capability not in event.enrichment_completed

        clip_source_ids: dict[Path, UUID] = {
            Path(artifact.path): UUID(str(artifact.metadata["source_clip_id"]))
            for artifact in event.derived_artifacts
            if artifact.artifact_type == "derived_clip"
            and Path(artifact.path).exists()
            and artifact.metadata.get("source_clip_id")
        }
        clip_paths = list(clip_source_ids.keys())
        now = utc_now()

        if self.config.enrichment.face.enabled and should_run("face"):
            with self._timed_capability("face", event_id):
                new_signals.extend(self._run_face_recognition(event, clip_paths, clip_source_ids))
            completed_updates["face"] = now
        if self.config.enrichment.plate.enabled and should_run("plate"):
            with self._timed_capability("plate", event_id):
                new_signals.extend(self._run_plate_recognition(event, clip_paths, clip_source_ids))
            completed_updates["plate"] = now
        if self.config.enrichment.voice.enabled and should_run("voice"):
            with self._timed_capability("voice", event_id):
                new_signals.extend(self._run_voice_recognition(event, clip_paths, clip_source_ids))
            completed_updates["voice"] = now
        if self.config.enrichment.vision.enabled and should_run("vision"):
            with self._timed_capability("vision", event_id):
                new_signals.extend(self._run_vehicle_detection(event, clip_paths))
            completed_updates["vision"] = now
        if self.config.enrichment.vehicle_appearance.enabled and should_run("vehicle_appearance"):
            with self._timed_capability("vehicle_appearance", event_id):
                new_signals.extend(
                    self._run_vehicle_appearance_recognition(event, clip_paths, clip_source_ids)
                )
            completed_updates["vehicle_appearance"] = now
        if self.config.enrichment.person_appearance.enabled and should_run("person_appearance"):
            with self._timed_capability("person_appearance", event_id):
                new_signals.extend(
                    self._run_person_appearance_recognition(event, clip_paths, clip_source_ids)
                )
            completed_updates["person_appearance"] = now

        transcript: AudioTranscript | None = None
        if self.config.enrichment.transcription.enabled and should_run("transcription"):
            with self._timed_capability("transcription", event_id):
                transcript = self._run_transcription(event, clip_paths, clip_source_ids)
            completed_updates["transcription"] = now
        elif (
            self.config.enrichment.cloud.enabled
            and should_run("cloud")
            and "transcription" in event.enrichment_completed
        ):
            # Cloud analysis was just enabled (or is being forced) but
            # transcription already ran on an earlier pass and is being
            # skipped this time -- reuse its saved output instead of
            # re-running Whisper just to get the transcript text again.
            transcript = self._load_saved_transcript(event.event_id)

        if self._face_clusterer is not None:
            self._face_clusterer.save()
        if self._face_embedding_clusterer is not None:
            self._face_embedding_clusterer.save()
        if self._voice_clusterer is not None:
            self._voice_clusterer.save()
        if self._vehicle_appearance_clusterer is not None:
            self._vehicle_appearance_clusterer.save()

        # Encounters and cloud/LLM analysis are derived from data already
        # committed above (DB observation rows; the transcript object) --
        # neither needs the revision below to exist first, so both run
        # before it and fold their own completion markers into the same
        # single revision, rather than each writing their own.
        #
        # Encounters runs if it's never been derived for this event yet
        # (should_run), or if something new was added this pass that's
        # worth re-correlating (new_signals), or forced -- not gated by
        # should_run() alone like the other capabilities, since its job
        # is to stay in sync with whatever observations currently exist,
        # not to run exactly once ever.
        if self.config.enrichment.encounters.enabled and (
            should_run("encounters") or new_signals or force
        ):
            with self._timed_capability("encounters", event_id):
                self._derive_encounters(event)
            completed_updates["encounters"] = now

        if self.config.enrichment.cloud.enabled and should_run("cloud") and transcript is not None:
            with self._timed_capability("cloud", event_id):
                self._run_llm_analysis(event, transcript)
            completed_updates["cloud"] = now

        updated_event = event
        if new_signals or completed_updates:
            updated_event = self.repository.save_event_revision(
                event_id,
                reason="enrichment",
                update={
                    "signals": [*event.signals, *new_signals],
                    "enrichment_completed": {**event.enrichment_completed, **completed_updates},
                },
            )

        LOGGER.info("event_enriched", event_id=str(event_id), new_signal_count=len(new_signals))
        return updated_event

    @contextmanager
    def _timed_capability(self, capability: str, event_id: UUID) -> Iterator[None]:
        """Real per-capability wall-clock timing, logged unconditionally --
        added after a real user report that `enrich` was taking far longer
        than expected, where the answer turned out to require knowing
        *which* capability was actually slow rather than guessing (it
        wasn't transcription, which a real benchmark showed only took
        ~15-20s per 2-minute clip on GPU)."""

        start = time.perf_counter()
        try:
            yield
        finally:
            LOGGER.info(
                "enrichment_capability_timing",
                capability=capability,
                event_id=str(event_id),
                duration_seconds=round(time.perf_counter() - start, 3),
            )

    # -- face -----------------------------------------------------------------

    def _ensure_face_yunet_detector_loaded(self) -> face_yunet_module.YuNetDetector | None:
        """Load the optional YuNet face detector at most once per
        `EnrichmentService` instance, mirroring
        `_ensure_vehicle_detector_loaded`'s exact "load once, share the
        load-failed result too" convention. A load failure (e.g. no
        network on first use, so the model can't be downloaded) degrades
        to the caller falling back to the Haar cascade for this run
        rather than skipping face detection entirely."""

        if not self._face_yunet_load_attempted:
            self._face_yunet_load_attempted = True
            try:
                self._face_yunet_detector = face_yunet_module.YuNetDetector(
                    device=cast(Device, self.config.enrichment.face.device)
                )
            except face_yunet_module.YuNetUnavailableError as error:
                LOGGER.warning("face_yunet_detector_unavailable", reason=str(error))
        return self._face_yunet_detector

    def _detect_faces(self, frame: ImageArray) -> tuple[list[face_module.DetectedFace], str]:
        """Dispatches on `enrichment.face.detector`; returns the detections
        plus which detector actually produced them (the configured one, or
        `haar` if `yunet` was requested but couldn't be loaded)."""

        if self.config.enrichment.face.detector == "yunet":
            detector = self._ensure_face_yunet_detector_loaded()
            if detector is not None:
                return detector.detect(frame), face_yunet_module.DETECTOR_VERSION
        return (
            face_module.detect_faces(frame, min_size=self.config.enrichment.face.detector_min_size),
            face_module.DETECTOR_VERSION,
        )

    def _ensure_face_embedder_loaded(self) -> face_auraface_module.AuraFaceEmbedder | None:
        """Load the optional AuraFace embedder at most once per
        `EnrichmentService` instance, mirroring
        `_ensure_face_yunet_detector_loaded`'s exact convention. A load
        failure degrades to the caller falling back to LBPH for this run
        rather than skipping face recognition entirely."""

        if not self._face_embedder_load_attempted:
            self._face_embedder_load_attempted = True
            try:
                self._face_embedder = face_auraface_module.AuraFaceEmbedder(
                    device=cast(Device, self.config.enrichment.face.device)
                )
            except face_auraface_module.AuraFaceUnavailableError as error:
                LOGGER.warning("face_embedder_unavailable", reason=str(error))
        return self._face_embedder

    def _match_face_cluster(
        self, frame: ImageArray, bbox: tuple[int, int, int, int]
    ) -> tuple[str, float, str]:
        """Dispatches on `enrichment.face.embedding_model`; always returns
        (cluster_id, distance, model_version) -- a detection is never
        silently dropped just because the deep embedding path degraded.
        Falls through to LBPH if `auraface` was requested but the
        embedder/model couldn't be loaded, or a real embedding couldn't
        be extracted for this specific crop (a degenerate/empty region)."""

        if self.config.enrichment.face.embedding_model == "auraface":
            embedder = self._ensure_face_embedder_loaded()
            if embedder is not None:
                x, y, w, h = bbox
                color_crop = frame[max(0, y) : y + h, max(0, x) : x + w]
                embedding = embedder.get_embedding(color_crop)
                if embedding is not None:
                    if self._face_embedding_clusterer is None:
                        self._face_embedding_clusterer = (
                            face_auraface_module.IncrementalFaceEmbeddingClusterer(
                                self.repository.workspace.face_embedding_model_path,
                                distance_threshold=(
                                    self.config.enrichment.face.embedding_cluster_distance_threshold
                                ),
                            )
                        )
                    cluster_id, distance, _is_new = (
                        self._face_embedding_clusterer.match_or_create_cluster(embedding)
                    )
                    return cluster_id, distance, face_auraface_module.RECOGNIZER_VERSION
            # embedder unavailable, or couldn't extract an embedding for
            # this specific crop -- fall through to LBPH below rather
            # than dropping the detection.

        if self._face_clusterer is None:
            self._face_clusterer = face_module.IncrementalFaceClusterer(
                self.repository.workspace.face_model_path,
                distance_threshold=self.config.enrichment.face.cluster_distance_threshold,
            )
        crop = face_module.crop_and_normalize(frame, bbox)
        cluster_id, distance, _is_new = self._face_clusterer.match_or_create_cluster(crop)
        return cluster_id, distance, face_module.RECOGNIZER_VERSION

    def _run_face_recognition(
        self, event: EventRecord, clip_paths: list[Path], clip_source_ids: dict[Path, UUID]
    ) -> list[Signal]:
        signals: list[Signal] = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            observed_at = event.event_start + timedelta(seconds=offset_seconds)
            detections, detector_version = self._detect_faces(frame)
            for detected in detections:
                if detected.confidence < self.config.enrichment.face.min_detection_confidence:
                    continue
                cluster_id, distance, model_version = self._match_face_cluster(frame, detected.bbox)
                # The stored/displayed crop is always the LBPH-style
                # normalized grayscale image regardless of which
                # embedding model matched the cluster -- consistent
                # review_ui display, independent of the recognition
                # backend in use.
                crop = face_module.crop_and_normalize(frame, detected.bbox)
                crop_path = self._save_crop(self.repository.workspace.face_crops, "face", crop)
                crop_sha256 = hash_file(crop_path)
                signal = Signal(
                    id=new_uuid(),
                    source="enrichment.face",
                    signal_type="face_detection",
                    timestamp_start=observed_at,
                    timestamp_end=observed_at,
                    confidence=detected.confidence,
                    camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                    evidence_references=[
                        ArtifactReference(
                            path=str(crop_path),
                            artifact_type="face_crop",
                            created_at=utc_now(),
                            sha256=crop_sha256,
                        )
                    ],
                    reasoning_metadata={
                        "enrichment_stage": True,
                        "cluster_id": cluster_id,
                        "cluster_distance": distance,
                        "detector_version": detector_version,
                        "model_version": model_version,
                        "clip_path": str(clip_path),
                        "offset_seconds": offset_seconds,
                    },
                )
                signals.append(signal)
                self._record_face_observation(
                    event,
                    signal,
                    cluster_id,
                    distance,
                    crop_path,
                    crop_sha256,
                    clip_source_ids.get(clip_path, event.event_id),
                    observed_at,
                    detector_version,
                    model_version,
                )
        return signals

    def _record_face_observation(
        self,
        event: EventRecord,
        signal: Signal,
        cluster_id: str,
        distance: float,
        crop_path: Path,
        crop_sha256: str,
        clip_id: UUID,
        observed_at: datetime,
        detector_version: str,
        model_version: str,
    ) -> None:
        now = utc_now()
        prior = self.repository.database.get_face_cluster(UUID(cluster_id))
        prior_crops = (
            prior.representative_crops_csv.split(",")
            if prior and prior.representative_crops_csv
            else []
        )
        representative_crops = [*prior_crops, str(crop_path)][-4:]
        cluster = FaceCluster(
            cluster_id=UUID(cluster_id),
            created_at=prior.created_at if prior else now,
            updated_at=now,
            label=prior.label if prior else None,
            representative_crop_paths=representative_crops,
            observation_count=(prior.observation_count if prior else 0) + 1,
            first_seen_at=prior.first_seen_at if prior else observed_at,
            last_seen_at=observed_at,
            model_version=model_version,
            merged_into=UUID(prior.merged_into) if prior and prior.merged_into else None,
        )
        self.repository.database.upsert_face_cluster(cluster)
        observation = FaceObservation(
            observation_id=new_uuid(),
            signal_id=signal.id,
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id=signal.camera_id or "unknown",
            observed_at=observed_at,
            crop_path=str(crop_path),
            crop_sha256=crop_sha256,
            detector_confidence=signal.confidence,
            embedding_distance_to_cluster=distance,
            cluster_id=UUID(cluster_id),
            detector_version=detector_version,
        )
        self.repository.database.insert_face_observation(observation)

    # -- voice ------------------------------------------------------------------

    def _run_voice_recognition(
        self, event: EventRecord, clip_paths: list[Path], clip_source_ids: dict[Path, UUID]
    ) -> list[Signal]:
        if self._voice_clusterer is None:
            self._voice_clusterer = voice_module.IncrementalVoiceClusterer(
                self.repository.workspace.voice_model_path,
                distance_threshold=self.config.enrichment.voice.cluster_distance_threshold,
            )
        voice_config = self.config.enrichment.voice
        signals: list[Signal] = []
        for clip_path in clip_paths:
            try:
                extracted = audio_analysis.extract_normalized_waveform(
                    clip_path, timeout_seconds=voice_config.audio_extraction_timeout_seconds
                )
            except audio_analysis.AudioAnalysisError as error:
                LOGGER.warning("voice_audio_extraction_failed", reason=str(error))
                continue
            if extracted is None:
                continue
            waveform, sample_rate = extracted
            clip_hash = hash_file(clip_path)
            try:
                segments = voice_module.detect_voice_segments(
                    waveform,
                    sample_rate,
                    min_segment_seconds=voice_config.min_segment_seconds,
                    energy_percentile_threshold=voice_config.energy_percentile_threshold,
                    merge_gap_seconds=voice_config.merge_gap_seconds,
                )
            except voice_module.VoiceAnalysisError:
                continue

            for segment in segments:
                try:
                    observation_result = voice_module.compute_voiceprint(
                        waveform, sample_rate, segment
                    )
                except voice_module.VoiceAnalysisError:
                    continue
                cluster_id, distance, _is_new = self._voice_clusterer.match_or_create_cluster(
                    observation_result.voiceprint
                )
                observed_at = event.event_start + timedelta(seconds=segment.start_offset_seconds)
                signal = Signal(
                    id=new_uuid(),
                    source="enrichment.voice",
                    signal_type="voice_detection",
                    timestamp_start=observed_at,
                    timestamp_end=event.event_start + timedelta(seconds=segment.end_offset_seconds),
                    confidence=observation_result.energy_confidence,
                    camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                    evidence_references=[
                        ArtifactReference(
                            path=str(clip_path),
                            artifact_type="source_media",
                            created_at=utc_now(),
                            sha256=clip_hash,
                            metadata={
                                "segment_start_seconds": segment.start_offset_seconds,
                                "segment_end_seconds": segment.end_offset_seconds,
                            },
                        )
                    ],
                    reasoning_metadata={
                        "enrichment_stage": True,
                        "cluster_id": cluster_id,
                        "cluster_distance": distance,
                        "detector_version": voice_module.VOICEPRINT_VERSION,
                        "clip_path": str(clip_path),
                        "segment_start_seconds": segment.start_offset_seconds,
                        "segment_end_seconds": segment.end_offset_seconds,
                    },
                )
                signals.append(signal)
                self._record_voice_observation(
                    event,
                    signal,
                    cluster_id,
                    distance,
                    observation_result,
                    clip_source_ids.get(clip_path, event.event_id),
                    observed_at,
                )
        return signals

    def _record_voice_observation(
        self,
        event: EventRecord,
        signal: Signal,
        cluster_id: str,
        distance: float,
        observation_result: voice_module.VoicePrintResult,
        clip_id: UUID,
        observed_at: datetime,
    ) -> None:
        now = utc_now()
        prior = self.repository.database.get_voice_cluster(UUID(cluster_id))
        cluster = VoiceCluster(
            cluster_id=UUID(cluster_id),
            created_at=prior.created_at if prior else now,
            updated_at=now,
            label=prior.label if prior else None,
            observation_count=(prior.observation_count if prior else 0) + 1,
            first_seen_at=prior.first_seen_at if prior else observed_at,
            last_seen_at=observed_at,
            model_version=voice_module.VOICEPRINT_VERSION,
            merged_into=UUID(prior.merged_into) if prior and prior.merged_into else None,
        )
        self.repository.database.upsert_voice_cluster(cluster)
        observation = VoiceObservation(
            observation_id=new_uuid(),
            signal_id=signal.id,
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id=signal.camera_id or "unknown",
            observed_at=observed_at,
            segment_start_seconds=observation_result.segment.start_offset_seconds,
            segment_end_seconds=observation_result.segment.end_offset_seconds,
            voiceprint=cast(list[float], observation_result.voiceprint.tolist()),
            energy_confidence=observation_result.energy_confidence,
            cluster_id=UUID(cluster_id),
            detector_version=voice_module.VOICEPRINT_VERSION,
        )
        self.repository.database.insert_voice_observation(observation)

    def _check_tesseract_once(self) -> bool:
        """Check tesseract availability exactly once per EnrichmentService
        instance (i.e. once per `enrich` invocation, not once per event and
        not once per detected plate region).

        Plate *detection* (finding plate-shaped regions) needs no external
        dependency, but OCR does. Without this guard, a missing tesseract
        install means every single detected region across every sampled
        frame attempts and fails a subprocess spawn -- slow, and it floods
        the log with an identical warning hundreds of times. With it, the
        user sees one clear, actionable warning and plate recognition is
        skipped cleanly for the rest of the run.
        """

        if not self._tesseract_checked:
            self._tesseract_checked = True
            self._tesseract_available = plate_module.tesseract_available()
            if not self._tesseract_available:
                LOGGER.warning(
                    "tesseract_not_found",
                    message=(
                        "tesseract is not installed or not on PATH; license plate OCR will be "
                        "skipped for this entire run (plate detection produces no signals "
                        "without it). See docs/local-ai.md's 'License plate detection and OCR' "
                        "section for install instructions on your platform."
                    ),
                )
        return self._tesseract_available

    def _ensure_fast_alpr_detector_loaded(self) -> plate_fast_alpr_module.FastAlprDetector | None:
        """Load the optional fast-alpr detector+OCR at most once per
        `EnrichmentService` instance, mirroring
        `_ensure_face_yunet_detector_loaded`'s exact convention. A load
        failure (e.g. no network for fast-alpr's own first-use model
        download) degrades to the caller falling back to the classical
        cascade+Tesseract path for this run rather than skipping plate
        recognition entirely."""

        if not self._fast_alpr_load_attempted:
            self._fast_alpr_load_attempted = True
            try:
                self._fast_alpr_detector = plate_fast_alpr_module.FastAlprDetector(
                    device=cast(Device, self.config.enrichment.plate.device),
                    confidence_threshold=self.config.enrichment.plate.min_detection_confidence,
                )
            except plate_fast_alpr_module.FastAlprUnavailableError as error:
                LOGGER.warning("fast_alpr_detector_unavailable", reason=str(error))
        return self._fast_alpr_detector

    def _detect_and_ocr_plates(
        self, frame: ImageArray
    ) -> tuple[list[tuple[plate_module.PlateRegion, plate_module.OcrResult | None]], str]:
        """Dispatches on `enrichment.plate.detector`; returns detected
        plate/OCR pairs (already filtered to
        `min_detection_confidence`) plus which detector actually produced
        them (the configured one, or `cascade` if `fast_alpr` was
        requested but couldn't be loaded). A pair's `OcrResult` is `None`
        when OCR produced no usable reading for that detection."""

        plate_config = self.config.enrichment.plate
        if plate_config.detector == "fast_alpr":
            detector = self._ensure_fast_alpr_detector_loaded()
            if detector is not None:
                pairs = [
                    (region, ocr_result)
                    for region, ocr_result in detector.detect_and_ocr(frame)
                    if region.confidence >= plate_config.min_detection_confidence
                ]
                return pairs, plate_fast_alpr_module.DETECTOR_VERSION

        if not self._check_tesseract_once():
            return [], plate_module.DETECTOR_VERSION
        cascade_pairs: list[tuple[plate_module.PlateRegion, plate_module.OcrResult | None]] = []
        for region in plate_module.detect_plate_regions(
            frame, min_size=plate_config.detector_min_size
        ):
            if region.confidence < plate_config.min_detection_confidence:
                continue
            x, y, w, h = region.bbox
            crop = frame[max(0, y) : y + h, max(0, x) : x + w]
            if crop.size == 0:
                continue
            try:
                ocr_result = plate_module.ocr_plate_text(crop)
            except plate_module.PlateOcrError as error:
                LOGGER.warning("plate_ocr_failed", reason=str(error))
                continue
            cascade_pairs.append((region, ocr_result))
        return cascade_pairs, plate_module.DETECTOR_VERSION

    def _run_plate_recognition(
        self, event: EventRecord, clip_paths: list[Path], clip_source_ids: dict[Path, UUID]
    ) -> list[Signal]:
        signals: list[Signal] = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            observed_at = event.event_start + timedelta(seconds=offset_seconds)
            pairs, detector_version = self._detect_and_ocr_plates(frame)
            for region, ocr_result in pairs:
                if ocr_result is None or not ocr_result.normalized_text:
                    continue
                text_length = len(ocr_result.normalized_text)
                plate_config = self.config.enrichment.plate
                if not (
                    plate_config.min_plate_text_length
                    <= text_length
                    <= plate_config.max_plate_text_length
                ):
                    # Almost always noise (a stray character, or a run-on
                    # misread spanning unrelated background text) -- discard
                    # before it's ever stored, not just hidden from review.
                    # See docs/local-ai.md's false-positive automation section.
                    continue
                if ocr_result.confidence < self.config.enrichment.plate.min_ocr_confidence_to_keep:
                    continue
                x, y, w, h = region.bbox
                crop = frame[max(0, y) : y + h, max(0, x) : x + w]
                if crop.size == 0:
                    continue
                crop_path = self._save_crop(self.repository.workspace.plate_crops, "plate", crop)
                auto_accept = self.config.enrichment.plate.auto_accept_ocr_confidence
                review_status = (
                    "auto_accepted" if ocr_result.confidence >= auto_accept else "needs_review"
                )
                reasoning_metadata: dict[str, object] = {
                    "enrichment_stage": True,
                    "normalized_text": ocr_result.normalized_text,
                    "ocr_confidence": ocr_result.confidence,
                    "review_status": review_status,
                    "detector_source": region.source,
                    "detector_version": detector_version,
                    "clip_path": str(clip_path),
                    "offset_seconds": offset_seconds,
                }
                if ocr_result.region is not None:
                    reasoning_metadata["region_guess"] = ocr_result.region
                    reasoning_metadata["region_guess_confidence"] = ocr_result.region_confidence
                signal = Signal(
                    id=new_uuid(),
                    source="enrichment.plate",
                    signal_type="license_plate",
                    timestamp_start=observed_at,
                    timestamp_end=observed_at,
                    confidence=region.confidence,
                    camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                    evidence_references=[
                        ArtifactReference(
                            path=str(crop_path),
                            artifact_type="plate_crop",
                            created_at=utc_now(),
                            sha256=hash_file(crop_path),
                        )
                    ],
                    reasoning_metadata=reasoning_metadata,
                )
                signals.append(signal)
                self._record_plate_observation(
                    event,
                    signal,
                    region,
                    ocr_result,
                    review_status,
                    crop_path,
                    clip_source_ids.get(clip_path, event.event_id),
                    observed_at,
                    detector_version,
                )
        return signals

    def _record_plate_observation(
        self,
        event: EventRecord,
        signal: Signal,
        region: plate_module.PlateRegion,
        ocr_result: plate_module.OcrResult,
        review_status: str,
        crop_path: Path,
        clip_id: UUID,
        observed_at: datetime,
        detector_version: str,
    ) -> None:
        observation = PlateObservation(
            observation_id=new_uuid(),
            signal_id=signal.id,
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id=signal.camera_id or "unknown",
            observed_at=observed_at,
            crop_path=str(crop_path),
            crop_sha256=hash_file(crop_path),
            raw_ocr_text=ocr_result.raw_text,
            normalized_text=ocr_result.normalized_text,
            ocr_confidence=ocr_result.confidence,
            detector_confidence=region.confidence,
            review_status=review_status,  # type: ignore[arg-type]
            detector_version=detector_version,
        )
        self.repository.database.insert_plate_observation(observation)

        existing = self.repository.database.get_plate_record_by_text(ocr_result.normalized_text)
        prior_crop_paths = (
            existing.example_crops_csv.split(",") if existing and existing.example_crops_csv else []
        )
        example_crop_paths = [*prior_crop_paths, str(crop_path)][-4:]
        now = utc_now()
        record = PlateRecord(
            plate_id=UUID(existing.plate_id) if existing else new_uuid(),
            normalized_text=ocr_result.normalized_text,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            label=existing.label if existing else None,
            observation_count=(existing.observation_count if existing else 0) + 1,
            first_seen_at=existing.first_seen_at if existing else observed_at,
            last_seen_at=observed_at,
            example_crops=[
                ArtifactReference(path=p, artifact_type="plate_crop", created_at=now)
                for p in example_crop_paths
            ],
            merged_into=UUID(existing.merged_into) if existing and existing.merged_into else None,
        )
        self.repository.database.upsert_plate_record(record)

    # -- vehicle/object detection -------------------------------------------

    def _ensure_vehicle_detector_loaded(self) -> YoloOnnxDetector | None:
        """Load the optional YOLO vehicle detector at most once per
        `EnrichmentService` instance, and share the single loaded (or
        load-failed) result between `_run_vehicle_detection` and
        `_run_vehicle_appearance_recognition` -- both want the same YOLO
        boxes, and the "check availability exactly once" convention means
        this must not become two separate load attempts."""

        if not self._vehicle_load_attempted:
            self._vehicle_load_attempted = True
            model_path = self.config.enrichment.vision.model_path
            if model_path:
                try:
                    self._vehicle_detector = YoloOnnxDetector(
                        Path(model_path),
                        device=self.config.enrichment.vision.device,
                        confidence_threshold=self.config.enrichment.vision.confidence_threshold,
                    )
                except VisionModelUnavailableError as error:
                    LOGGER.warning("vehicle_detector_unavailable", reason=str(error))
        return self._vehicle_detector

    def _run_vehicle_detection(self, event: EventRecord, clip_paths: list[Path]) -> list[Signal]:
        detector = self._ensure_vehicle_detector_loaded()
        if detector is None:
            return []

        signals = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            observed_at = event.event_start + timedelta(seconds=offset_seconds)
            for detection in detector.detect(frame):
                signals.append(
                    Signal(
                        id=new_uuid(),
                        source="enrichment.vehicle_yolo",
                        signal_type="vehicle_detection",
                        timestamp_start=observed_at,
                        timestamp_end=observed_at,
                        confidence=detection.confidence,
                        camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                        reasoning_metadata={
                            "enrichment_stage": True,
                            "class_name": detection.class_name,
                            "class_id": detection.class_id,
                            "clip_path": str(clip_path),
                            "offset_seconds": offset_seconds,
                        },
                    )
                )
        return signals

    # -- vehicle appearance re-identification --------------------------------

    def _vehicle_appearance_regions(
        self, frame: ImageArray, vehicle_config: VehicleAppearanceConfig
    ) -> list[vehicle_appearance_module.VehicleRegion]:
        """YOLO boxes (filtered to actual vehicle body classes) when the
        `vision` extra is enabled and a model loaded; otherwise the
        always-available classical heuristic. Never requires YOLO -- see
        `enrichment/vehicle_appearance.py`'s module docstring."""

        if self.config.enrichment.vision.enabled:
            detector = self._ensure_vehicle_detector_loaded()
            if detector is not None:
                yolo_regions = [
                    vehicle_appearance_module.VehicleRegion(
                        bbox=(
                            detection.bbox[0],
                            detection.bbox[1],
                            detection.bbox[2] - detection.bbox[0],
                            detection.bbox[3] - detection.bbox[1],
                        ),
                        confidence=detection.confidence,
                        source="yolo",
                    )
                    for detection in detector.detect(frame)
                    if detection.class_id in _YOLO_VEHICLE_CLASS_IDS
                ]
                if yolo_regions:
                    return yolo_regions
        return vehicle_appearance_module.detect_vehicle_regions(
            frame, min_size=vehicle_config.detector_min_size
        )

    def _run_vehicle_appearance_recognition(
        self, event: EventRecord, clip_paths: list[Path], clip_source_ids: dict[Path, UUID]
    ) -> list[Signal]:
        vehicle_config = self.config.enrichment.vehicle_appearance
        if self._vehicle_appearance_clusterer is None:
            clusterer_cls = vehicle_appearance_module.IncrementalVehicleAppearanceClusterer
            self._vehicle_appearance_clusterer = clusterer_cls(
                self.repository.workspace.vehicle_appearance_model_path,
                distance_threshold=vehicle_config.cluster_distance_threshold,
            )
        signals: list[Signal] = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            observed_at = event.event_start + timedelta(seconds=offset_seconds)
            for region in self._vehicle_appearance_regions(frame, vehicle_config):
                if region.confidence < vehicle_config.min_detection_confidence:
                    continue
                try:
                    fingerprint_result = vehicle_appearance_module.compute_fingerprint(
                        frame, region
                    )
                except vehicle_appearance_module.VehicleAppearanceError:
                    continue
                cluster_id, distance, _is_new = (
                    self._vehicle_appearance_clusterer.match_or_create_cluster(
                        fingerprint_result.fingerprint
                    )
                )
                x, y, w, h = region.bbox
                crop = frame[max(0, y) : y + h, max(0, x) : x + w]
                crop_path = self._save_crop(
                    self.repository.workspace.vehicle_appearance_crops, "vehicle", crop
                )
                crop_sha256 = hash_file(crop_path)
                signal = Signal(
                    id=new_uuid(),
                    source="enrichment.vehicle_appearance",
                    signal_type="vehicle_appearance",
                    timestamp_start=observed_at,
                    timestamp_end=observed_at,
                    confidence=region.confidence,
                    camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                    evidence_references=[
                        ArtifactReference(
                            path=str(crop_path),
                            artifact_type="vehicle_crop",
                            created_at=utc_now(),
                            sha256=crop_sha256,
                        )
                    ],
                    reasoning_metadata={
                        "enrichment_stage": True,
                        "cluster_id": cluster_id,
                        "cluster_distance": distance,
                        "detector_version": vehicle_appearance_module.DETECTOR_VERSION,
                        "detector_source": region.source,
                        "clip_path": str(clip_path),
                        "offset_seconds": offset_seconds,
                    },
                )
                signals.append(signal)
                self._record_vehicle_appearance_observation(
                    event,
                    signal,
                    cluster_id,
                    distance,
                    fingerprint_result,
                    crop_path,
                    crop_sha256,
                    clip_source_ids.get(clip_path, event.event_id),
                    observed_at,
                )
        return signals

    def _record_vehicle_appearance_observation(
        self,
        event: EventRecord,
        signal: Signal,
        cluster_id: str,
        distance: float,
        fingerprint_result: vehicle_appearance_module.VehicleAppearanceFingerprint,
        crop_path: Path,
        crop_sha256: str,
        clip_id: UUID,
        observed_at: datetime,
    ) -> None:
        now = utc_now()
        prior = self.repository.database.get_vehicle_appearance_cluster(UUID(cluster_id))
        prior_crops = (
            prior.representative_crops_csv.split(",")
            if prior and prior.representative_crops_csv
            else []
        )
        representative_crops = [*prior_crops, str(crop_path)][-4:]
        cluster = VehicleAppearanceCluster(
            cluster_id=UUID(cluster_id),
            created_at=prior.created_at if prior else now,
            updated_at=now,
            label=prior.label if prior else None,
            representative_crop_paths=representative_crops,
            observation_count=(prior.observation_count if prior else 0) + 1,
            first_seen_at=prior.first_seen_at if prior else observed_at,
            last_seen_at=observed_at,
            model_version=vehicle_appearance_module.DETECTOR_VERSION,
            merged_into=UUID(prior.merged_into) if prior and prior.merged_into else None,
        )
        self.repository.database.upsert_vehicle_appearance_cluster(cluster)
        observation = VehicleAppearanceObservation(
            observation_id=new_uuid(),
            signal_id=signal.id,
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id=signal.camera_id or "unknown",
            observed_at=observed_at,
            crop_path=str(crop_path),
            crop_sha256=crop_sha256,
            fingerprint=cast(list[float], fingerprint_result.fingerprint.tolist()),
            detector_confidence=signal.confidence,
            embedding_distance_to_cluster=distance,
            cluster_id=UUID(cluster_id),
            detector_version=vehicle_appearance_module.DETECTOR_VERSION,
        )
        self.repository.database.insert_vehicle_appearance_observation(observation)

    # -- person appearance ------------------------------------------------

    def _run_person_appearance_recognition(
        self, event: EventRecord, clip_paths: list[Path], clip_source_ids: dict[Path, UUID]
    ) -> list[Signal]:
        """YOLO-only -- see `enrichment/person_appearance.py`'s module
        docstring for why there's no classical fallback here, unlike
        `_run_vehicle_appearance_recognition`. Reuses the same shared
        YOLO detector instance `_run_vehicle_detection`/
        `_run_vehicle_appearance_recognition` already load, filtered to
        the "person" class."""

        person_config = self.config.enrichment.person_appearance
        if not self.config.enrichment.vision.enabled:
            return []
        detector = self._ensure_vehicle_detector_loaded()
        if detector is None:
            return []
        if self._person_appearance_clusterer is None:
            clusterer_cls = person_appearance_module.IncrementalPersonAppearanceClusterer
            self._person_appearance_clusterer = clusterer_cls(
                self.repository.workspace.person_appearance_model_path,
                distance_threshold=person_config.cluster_distance_threshold,
            )
        signals: list[Signal] = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            observed_at = event.event_start + timedelta(seconds=offset_seconds)
            for detection in detector.detect(frame):
                if detection.class_id not in _YOLO_PERSON_CLASS_IDS:
                    continue
                if detection.confidence < person_config.min_detection_confidence:
                    continue
                region = person_appearance_module.PersonRegion(
                    bbox=(
                        detection.bbox[0],
                        detection.bbox[1],
                        detection.bbox[2] - detection.bbox[0],
                        detection.bbox[3] - detection.bbox[1],
                    ),
                    confidence=detection.confidence,
                    source="yolo",
                )
                try:
                    fingerprint_result = person_appearance_module.compute_fingerprint(frame, region)
                except person_appearance_module.PersonAppearanceError:
                    continue
                cluster_id, distance, _is_new = (
                    self._person_appearance_clusterer.match_or_create_cluster(
                        fingerprint_result.fingerprint
                    )
                )
                x, y, w, h = region.bbox
                crop = frame[max(0, y) : y + h, max(0, x) : x + w]
                crop_path = self._save_crop(
                    self.repository.workspace.person_appearance_crops, "person", crop
                )
                crop_sha256 = hash_file(crop_path)
                signal = Signal(
                    id=new_uuid(),
                    source="enrichment.person_appearance",
                    signal_type="person_appearance",
                    timestamp_start=observed_at,
                    timestamp_end=observed_at,
                    confidence=region.confidence,
                    camera_id=event.involved_cameras[0] if event.involved_cameras else None,
                    evidence_references=[
                        ArtifactReference(
                            path=str(crop_path),
                            artifact_type="person_crop",
                            created_at=utc_now(),
                            sha256=crop_sha256,
                        )
                    ],
                    reasoning_metadata={
                        "enrichment_stage": True,
                        "cluster_id": cluster_id,
                        "cluster_distance": distance,
                        "detector_version": person_appearance_module.DETECTOR_VERSION,
                        "detector_source": region.source,
                        "clip_path": str(clip_path),
                        "offset_seconds": offset_seconds,
                        # Structured attributes -- never a free-text
                        # description, see person_appearance.py's module
                        # docstring.
                        "dominant_hue_bin": fingerprint_result.dominant_hue_bin,
                        "height_to_frame_ratio": fingerprint_result.height_to_frame_ratio,
                    },
                )
                signals.append(signal)
                self._record_person_appearance_observation(
                    event,
                    signal,
                    cluster_id,
                    distance,
                    fingerprint_result,
                    crop_path,
                    crop_sha256,
                    clip_source_ids.get(clip_path, event.event_id),
                    observed_at,
                )
        return signals

    def _record_person_appearance_observation(
        self,
        event: EventRecord,
        signal: Signal,
        cluster_id: str,
        distance: float,
        fingerprint_result: person_appearance_module.PersonAppearanceFingerprint,
        crop_path: Path,
        crop_sha256: str,
        clip_id: UUID,
        observed_at: datetime,
    ) -> None:
        now = utc_now()
        prior = self.repository.database.get_person_appearance_cluster(UUID(cluster_id))
        prior_crops = (
            prior.representative_crops_csv.split(",")
            if prior and prior.representative_crops_csv
            else []
        )
        representative_crops = [*prior_crops, str(crop_path)][-4:]
        cluster = PersonAppearanceCluster(
            cluster_id=UUID(cluster_id),
            created_at=prior.created_at if prior else now,
            updated_at=now,
            label=prior.label if prior else None,
            representative_crop_paths=representative_crops,
            observation_count=(prior.observation_count if prior else 0) + 1,
            first_seen_at=prior.first_seen_at if prior else observed_at,
            last_seen_at=observed_at,
            model_version=person_appearance_module.DETECTOR_VERSION,
            merged_into=UUID(prior.merged_into) if prior and prior.merged_into else None,
        )
        self.repository.database.upsert_person_appearance_cluster(cluster)
        observation = PersonAppearanceObservation(
            observation_id=new_uuid(),
            signal_id=signal.id,
            event_id=event.event_id,
            clip_id=clip_id,
            camera_id=signal.camera_id or "unknown",
            observed_at=observed_at,
            crop_path=str(crop_path),
            crop_sha256=crop_sha256,
            fingerprint=cast(list[float], fingerprint_result.fingerprint.tolist()),
            detector_confidence=signal.confidence,
            embedding_distance_to_cluster=distance,
            cluster_id=UUID(cluster_id),
            detector_version=person_appearance_module.DETECTOR_VERSION,
        )
        self.repository.database.insert_person_appearance_observation(observation)

    # -- encounters -----------------------------------------------------------

    def _derive_encounters(self, event: EventRecord) -> None:
        """Group `event`'s already-persisted face/plate/voice/vehicle-
        appearance observations into `Encounter` records -- a pure
        post-processing pass over data every other enrichment pass already
        wrote, with no detection logic of its own. Clears this event's
        existing Encounters first (`database.delete_encounters_for_event`)
        so re-running is a clean replace against whatever observations
        currently exist, not an unbounded accumulate -- Encounters are a
        freely-rebuildable derived index, never primary evidence
        themselves (the observations they reference are never touched).

        Grouping algorithm, per clip: sort every observation (any
        modality) by `observed_at`; start a new time window whenever the
        gap to the previous observation exceeds twice the configured
        frame sampling interval (`enrichment.frame_sample_rate_hz`) --
        i.e. two sampled frames apart.
        Within one window, observations of each modality are paired up
        index-wise (1st face with 1st plate/voice/vehicle observation in
        that window, 2nd with 2nd, ...) rather than combined combinatorially
        -- every observation ends up in exactly one Encounter, and no
        Encounter is fabricated from entities that were never actually
        close in time. See `schemas/encounter.py`'s module docstring: this
        pairing is a bookkeeping convenience, never a claim that the paired
        entities are actually related to each other.
        """

        tagged: list[tuple[str, datetime, str, UUID, str]] = [
            (row.clip_id, row.observed_at, "face", UUID(row.observation_id), row.camera_id)
            for row in self.repository.database.list_face_observations_for_event(event.event_id)
        ]
        tagged += [
            (row.clip_id, row.observed_at, "plate", UUID(row.observation_id), row.camera_id)
            for row in self.repository.database.list_plate_observations_for_event(event.event_id)
        ]
        tagged += [
            (row.clip_id, row.observed_at, "voice", UUID(row.observation_id), row.camera_id)
            for row in self.repository.database.list_voice_observations_for_event(event.event_id)
        ]
        tagged += [
            (
                row.clip_id,
                row.observed_at,
                "vehicle_appearance",
                UUID(row.observation_id),
                row.camera_id,
            )
            for row in self.repository.database.list_vehicle_appearance_observations_for_event(
                event.event_id
            )
        ]
        tagged += [
            (
                row.clip_id,
                row.observed_at,
                "person_appearance",
                UUID(row.observation_id),
                row.camera_id,
            )
            for row in self.repository.database.list_person_appearance_observations_for_event(
                event.event_id
            )
        ]
        self.repository.database.delete_encounters_for_event(event.event_id)
        if not tagged:
            return

        by_clip: dict[str, list[tuple[str, datetime, str, UUID, str]]] = {}
        for item in tagged:
            by_clip.setdefault(item[0], []).append(item)

        tolerance_seconds = 2.0 / self.config.enrichment.frame_sample_rate_hz
        for clip_id, items in by_clip.items():
            items.sort(key=lambda item: item[1])
            window: list[tuple[str, datetime, str, UUID, str]] = []
            for item in items:
                if window and (item[1] - window[-1][1]).total_seconds() > tolerance_seconds:
                    self._save_encounters_for_window(event.event_id, UUID(clip_id), window)
                    window = []
                window.append(item)
            if window:
                self._save_encounters_for_window(event.event_id, UUID(clip_id), window)

    def _save_encounters_for_window(
        self,
        event_id: UUID,
        clip_id: UUID,
        items: list[tuple[str, datetime, str, UUID, str]],
    ) -> None:
        by_modality: dict[str, list[tuple[datetime, UUID]]] = {}
        for _clip_id, observed_at, modality, observation_id, _camera_id in items:
            by_modality.setdefault(modality, []).append((observed_at, observation_id))
        camera_id = items[0][4]
        face = by_modality.get("face", [])
        plate = by_modality.get("plate", [])
        voice = by_modality.get("voice", [])
        vehicle = by_modality.get("vehicle_appearance", [])
        person = by_modality.get("person_appearance", [])
        count = max(len(face), len(plate), len(voice), len(vehicle), len(person))
        for index in range(count):
            picked = [
                entries[index][0]
                for entries in (face, plate, voice, vehicle, person)
                if index < len(entries)
            ]
            observed_at = min(picked) if picked else items[0][1]
            encounter = Encounter(
                encounter_id=new_uuid(),
                event_id=event_id,
                clip_id=clip_id,
                camera_id=camera_id,
                observed_at=observed_at,
                face_observation_id=face[index][1] if index < len(face) else None,
                plate_observation_id=plate[index][1] if index < len(plate) else None,
                voice_observation_id=voice[index][1] if index < len(voice) else None,
                vehicle_appearance_observation_id=(
                    vehicle[index][1] if index < len(vehicle) else None
                ),
                person_appearance_observation_id=(
                    person[index][1] if index < len(person) else None
                ),
            )
            self.repository.database.insert_encounter(encounter)

    # -- transcription -----------------------------------------------------

    def _run_transcription(
        self,
        event: EventRecord,
        clip_paths: list[Path],
        clip_source_ids: dict[Path, UUID],
    ) -> AudioTranscript | None:
        if not self._transcriber_load_attempted:
            self._transcriber_load_attempted = True
            try:
                self._transcriber = WhisperTranscriber(
                    model_name=self.config.enrichment.transcription.model_name,
                    device=self.config.enrichment.transcription.device,
                    compute_type=self.config.enrichment.transcription.compute_type,
                )
            except TranscriptionUnavailableError as error:
                LOGGER.warning("transcription_unavailable", reason=str(error))
        if self._transcriber is None or not clip_paths:
            return None

        source_clip_path = clip_paths[0]
        result = self._transcriber.transcribe(source_clip_path)
        transcript = AudioTranscript(
            transcript_id=new_uuid(),
            clip_id=clip_source_ids.get(source_clip_path, event.event_id),
            created_at=utc_now(),
            language=result.language,
            model_name=result.model_name,
            model_version=result.model_name,
            device=result.device,
            segments=[
                TranscriptSegment(
                    start_offset_seconds=s.start_offset_seconds,
                    end_offset_seconds=s.end_offset_seconds,
                    text=s.text,
                    confidence=s.confidence,
                )
                for s in result.segments
            ],
            full_text=result.full_text,
        )
        output_path = self.repository.workspace.transcripts / f"{event.event_id}.json"
        self.repository.workspace.write_json(output_path, transcript.model_dump(mode="json"))
        return transcript

    def _load_saved_transcript(self, event_id: UUID) -> AudioTranscript | None:
        """Reads back a transcript `_run_transcription` already wrote on a
        prior `enrich_event` call, for when transcription is already
        marked complete but cloud/LLM analysis is only just now enabled
        (or being forced) -- re-running Whisper again just to get the
        same text a second time would defeat the point of skipping
        already-completed capabilities. Returns `None` if the sidecar is
        missing (e.g. transcription found nothing) or unreadable, the
        same graceful-degradation shape every other optional read in this
        module uses."""

        path = self.repository.workspace.transcripts / f"{event_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return AudioTranscript.model_validate(payload)
        except (OSError, ValueError) as error:
            LOGGER.warning("saved_transcript_unreadable", event_id=str(event_id), reason=str(error))
            return None

    # -- optional cloud LLM analysis -----------------------------------------

    def _run_llm_analysis(self, event: EventRecord, transcript: AudioTranscript) -> None:
        cloud_config = self.config.enrichment.cloud
        if not cloud_config.endpoint:
            LOGGER.warning("llm_analysis_skipped_no_endpoint")
            return
        api_key = os.environ.get(cloud_config.api_key_env_var)
        if not api_key:
            LOGGER.warning("llm_analysis_skipped_no_api_key", env_var=cloud_config.api_key_env_var)
            return
        try:
            result = analyze_transcript(
                transcript.full_text,
                endpoint=cloud_config.endpoint,
                api_key=api_key,
                model=cloud_config.model,
                timeout_seconds=cloud_config.timeout_seconds,
            )
        except (LlmEnrichmentUnavailableError, LlmEnrichmentError) as error:
            LOGGER.warning("llm_analysis_failed", reason=str(error))
            return

        enrichment = LLMEnrichment(
            enrichment_id=new_uuid(),
            event_id=event.event_id,
            created_at=utc_now(),
            provider="openai-compatible",
            model=cloud_config.model,
            endpoint=cloud_config.endpoint,
            summary=result.summary,
            extracted_events=result.extracted_events,
            extracted_entities=result.extracted_entities,
            importance_score=result.importance_score,
            prompt_version="1.0.0",
        )
        output_path = self.repository.workspace.transcripts / f"{event.event_id}.llm.json"
        self.repository.workspace.write_json(output_path, enrichment.model_dump(mode="json"))
        LOGGER.info("llm_analysis_completed", event_id=str(event.event_id))

    # -- debug / accuracy inspection -----------------------------------------

    def debug_render_plate_detections(self, event_id: UUID, output_dir: Path) -> list[Path]:
        """Re-run plate detection on an event's derived clips and save one
        annotated frame per sampled frame that had at least one candidate,
        with every region drawn and labeled by source and confidence.

        Built specifically to answer "how do I check whether this is
        actually working" -- run this on an event, look at the images, and
        you can see exactly what the detector found (and, just as
        importantly, what it *didn't* find) on real footage, not a
        synthetic test scene. See `docs/local-ai.md`'s plate-accuracy
        section and `recognize plates-debug`.
        """

        event = self.repository.load_event(event_id)
        clip_paths = [
            Path(artifact.path)
            for artifact in event.derived_artifacts
            if artifact.artifact_type == "derived_clip" and Path(artifact.path).exists()
        ]
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for clip_path, offset_seconds, frame in _sample_frames(
            clip_paths, self.config.enrichment.frame_sample_rate_hz
        ):
            regions = plate_module.detect_plate_regions(
                frame, min_size=self.config.enrichment.plate.detector_min_size
            )
            if not regions:
                continue
            annotated = plate_module.render_debug_frame(frame, regions)
            out_path = output_dir / f"{clip_path.stem}_{offset_seconds:07.2f}s.jpg"
            cv2.imwrite(str(out_path), annotated)
            written.append(out_path)
        LOGGER.info(
            "plate_debug_render_completed",
            event_id=str(event_id),
            frame_count=len(written),
            output_dir=str(output_dir),
        )
        return written

    # -- shared helpers -------------------------------------------------------

    def _save_crop(self, directory: Path, prefix: str, crop: ImageArray) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{prefix}_{new_uuid()}.jpg"
        cv2.imwrite(str(path), crop)
        return path


def _sample_frames(
    clip_paths: list[Path],
    sample_rate_hz: float = FRAME_SAMPLE_RATE_HZ,
) -> Iterator[tuple[Path, float, ImageArray]]:
    for clip_path in clip_paths:
        capture = cv2.VideoCapture(str(clip_path))
        if not capture.isOpened():
            continue
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            stride = max(1, round(fps / sample_rate_hz))
            frame_index = 0
            while True:
                grabbed = capture.grab()
                if not grabbed:
                    break
                if frame_index % stride == 0:
                    decoded, frame = capture.retrieve()
                    if decoded and frame is not None:
                        yield clip_path, frame_index / fps, frame
                frame_index += 1
        finally:
            capture.release()
