from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import UUID

from gaggle.core.config import RuntimeConfig
from gaggle.core.derived_clips import ClipExtractionError, extract_clip_segment
from gaggle.detection.audio import AudioSpikeDetector
from gaggle.detection.base import DetectionInputs
from gaggle.detection.gunshot import GunshotDetector
from gaggle.detection.motion import MotionDetector
from gaggle.detection.object_detection import MetadataObjectDetector
from gaggle.detection.optical_flow import OpticalFlowDetector
from gaggle.detection.telemetry import TelemetryDetector
from gaggle.inference.service import InferenceService
from gaggle.normalize.service import NormalizationService
from gaggle.plugins.registry import DETECTOR_PLUGIN_GROUP, load_plugins
from gaggle.preservation.service import PreservationOrchestrator
from gaggle.schemas.common import ArtifactReference, ChainOfCustodyEntry, HashDigest
from gaggle.schemas.event import EventRecord, Hypothesis, PreservationStatus
from gaggle.schemas.media import (
    EventWindow,
    IngestManifest,
    NormalizationManifest,
)
from gaggle.schemas.signal import Signal
from gaggle.scoring.service import ScoringService
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now
from gaggle.windowing.service import WindowingService

LOGGER = get_logger(__name__)

_ClusterItem = tuple[EventWindow, list[Signal], list[Hypothesis]]


class AnalysisPipeline:
    """Orchestrates ingest -> normalize -> window -> detect -> infer -> score -> preserve.

    Each stage's output is a distinct, inspectable artifact written to the
    workspace before the next stage runs (see ``docs/architecture.md``), and
    each stage can be replayed independently against its own persisted
    manifest. This class only sequences the stages and builds the resulting
    ``EventRecord`` objects; it contains no detection/inference/scoring
    logic itself.
    """

    version = "1.1.0"

    def __init__(self, repository: Repository, config: RuntimeConfig) -> None:
        self.repository = repository
        self.config = config
        self.normalization = NormalizationService(
            repository.workspace, config, database=repository.database
        )
        self.windowing = WindowingService(repository.workspace, config)
        self.inference = InferenceService()
        self.scoring = ScoringService(config)
        self.detectors = [
            MotionDetector(config),
            AudioSpikeDetector(config),
            MetadataObjectDetector(config),
            TelemetryDetector(config),
            OpticalFlowDetector(config),
            GunshotDetector(config),
            *load_plugins(DETECTOR_PLUGIN_GROUP),
        ]
        self.preservation = PreservationOrchestrator(repository)

    def analyze(self, ingest_manifest: IngestManifest) -> list[EventRecord]:
        normalization_manifest = self.normalization.normalize(ingest_manifest)
        window_manifest = self.windowing.generate(normalization_manifest)
        detection_inputs = DetectionInputs(
            workspace_root=self.repository.workspace.root,
            windows=window_manifest.windows,
            clips=normalization_manifest.clips,
            config=self.config,
        )
        signals: list[Signal] = []
        for detector in self.detectors:
            detector_signals = detector.detect(detection_inputs)
            LOGGER.info(
                "detector_completed",
                detector=getattr(detector, "name", repr(detector)),
                signal_count=len(detector_signals),
            )
            signals.extend(detector_signals)
        hypotheses = self.inference.infer(signals)
        events = self._build_events(
            ingest_manifest, window_manifest.windows, signals, hypotheses, normalization_manifest
        )
        for event in events:
            self.repository.save_event(event)
        self.repository.index_ingest_manifest(ingest_manifest)
        LOGGER.info("analysis_completed", event_count=len(events))
        return events

    def preserve_event(self, event_id: UUID) -> EventRecord:
        return self.preservation.preserve_event(event_id)

    # -- event assembly -----------------------------------------------------

    def _build_events(
        self,
        ingest_manifest: IngestManifest,
        windows: list[EventWindow],
        signals: list[Signal],
        hypotheses: list[Hypothesis],
        normalization_manifest: NormalizationManifest,
    ) -> list[EventRecord]:
        windows_by_id = {str(window.window_id): window for window in windows}

        signals_by_window: dict[str, list[Signal]] = defaultdict(list)
        for signal in signals:
            if signal.window_id is None:
                continue
            signals_by_window[str(signal.window_id)].append(signal)

        hypotheses_by_window: dict[str, list[Hypothesis]] = defaultdict(list)
        for hypothesis in hypotheses:
            window_id = hypothesis.metadata.get("window_id")
            if window_id is None:
                continue
            hypotheses_by_window[str(window_id)].append(hypothesis)

        populated: list[_ClusterItem] = [
            (windows_by_id[window_id], window_signals, hypotheses_by_window.get(window_id, []))
            for window_id, window_signals in signals_by_window.items()
            if window_id in windows_by_id and window_signals
        ]
        populated.sort(key=lambda item: (item[0].start, str(item[0].window_id)))

        clusters = self._cluster_overlapping_windows(
            populated, self.config.pipeline.max_event_duration_seconds
        )
        return [
            self._build_event_from_cluster(ingest_manifest, cluster, normalization_manifest)
            for cluster in clusters
        ]

    @staticmethod
    def _cluster_overlapping_windows(
        populated: list[_ClusterItem], max_event_duration_seconds: float | None
    ) -> list[list[_ClusterItem]]:
        """Merge temporally-overlapping windows-with-signals into single clusters.

        The sliding window stage intentionally overlaps windows (stride <
        duration) so that a signal near a boundary is never split awkwardly
        across two windows. Left un-merged, that overlap would otherwise
        produce multiple near-duplicate events for what is really one
        continuous span of activity; this merge step is what keeps each
        real-world incident as a single event.

        ``max_event_duration_seconds`` (``None`` disables the cap) forces a
        split once the current cluster's span would exceed it, even though
        the next window still overlaps -- without this, real footage with
        near-continuous activity throughout (e.g. a long stretch of actual
        driving, where motion is present in nearly every window) merges
        into one arbitrarily long event, with a derived clip effectively
        the length of the entire source recording. See
        ``docs/limitations.md`` for the tradeoff this introduces: a forced
        split at the cap boundary can in principle separate two halves of
        one real continuous incident into two events.
        """

        clusters: list[list[_ClusterItem]] = []
        current_cluster: list[_ClusterItem] = []
        current_start = None
        current_end = None
        for item in populated:
            window = item[0]
            within_cap = (
                max_event_duration_seconds is None
                or current_start is None
                or (window.end - current_start).total_seconds() <= max_event_duration_seconds
            )
            if (
                current_cluster
                and current_end is not None
                and window.start < current_end
                and within_cap
            ):
                current_cluster.append(item)
                current_end = max(current_end, window.end)
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [item]
                current_start = window.start
                current_end = window.end
        if current_cluster:
            clusters.append(current_cluster)
        return clusters

    def _build_event_from_cluster(
        self,
        ingest_manifest: IngestManifest,
        cluster: list[_ClusterItem],
        normalization_manifest: NormalizationManifest,
    ) -> EventRecord:
        signals_by_id: dict[UUID, Signal] = {}
        hypotheses_by_id: dict[UUID, Hypothesis] = {}
        window_ids: list[str] = []
        for window, window_signals, window_hypotheses in cluster:
            window_ids.append(str(window.window_id))
            for signal in window_signals:
                signals_by_id[signal.id] = signal
            for hypothesis in window_hypotheses:
                hypotheses_by_id[hypothesis.hypothesis_id] = hypothesis

        sorted_signals = sorted(
            signals_by_id.values(), key=lambda s: (s.timestamp_start, str(s.id))
        )
        sorted_hypotheses = sorted(hypotheses_by_id.values(), key=lambda h: str(h.hypothesis_id))

        scoring = self.scoring.score(sorted_hypotheses, sorted_signals)
        involved_cameras = sorted({s.camera_id for s in sorted_signals if s.camera_id is not None})
        event_id = new_uuid()
        event_start = min(s.timestamp_start for s in sorted_signals)
        event_end = max(s.timestamp_end for s in sorted_signals)

        derived_artifacts = self._extract_derived_clips(
            event_id, event_start, event_end, normalization_manifest
        )

        return EventRecord(
            event_id=event_id,
            created_at=utc_now(),
            pipeline_version=self.version,
            config_snapshot=self.config.model_dump(mode="json"),
            event_start=event_start,
            event_end=event_end,
            involved_cameras=involved_cameras,
            signals=sorted_signals,
            hypotheses=sorted_hypotheses,
            scoring=scoring,
            preservation_status=PreservationStatus(state="pending", immutable=False),
            chain_of_custody=[
                ChainOfCustodyEntry(
                    entry_id=new_uuid(),
                    action="event_generated",
                    actor="gaggle",
                    timestamp=utc_now(),
                    details={
                        "ingest_run_id": str(ingest_manifest.run_id),
                        "signal_count": len(sorted_signals),
                        "hypothesis_count": len(sorted_hypotheses),
                        "source_window_count": len(cluster),
                    },
                    input_hashes=[
                        HashDigest(value=clip.sha256) for clip in ingest_manifest.copied_files
                    ],
                    output_hashes=[],
                )
            ],
            hashes=[
                signal.evidence_references[0].sha256
                for signal in sorted_signals
                if signal.evidence_references and signal.evidence_references[0].sha256 is not None
            ],
            derived_artifacts=derived_artifacts,
            evidence_summary=self._summarize(sorted_signals, sorted_hypotheses),
            metadata={"window_ids": window_ids},
        )

    def _extract_derived_clips(
        self,
        event_id: UUID,
        event_start: datetime,
        event_end: datetime,
        normalization_manifest: NormalizationManifest,
    ) -> list[ArtifactReference]:
        artifacts: list[ArtifactReference] = []
        clips_dir = self.repository.workspace.event_clips_dir(event_id)
        for clip in normalization_manifest.clips:
            overlap_start = max(event_start, clip.corrected_start)
            overlap_end = min(event_end, clip.corrected_end)
            if overlap_end <= overlap_start:
                continue
            offset_start = (overlap_start - clip.corrected_start).total_seconds()
            offset_end = (overlap_end - clip.corrected_start).total_seconds()
            destination = clips_dir / f"{clip.camera_id}__{str(clip.clip_id)[:8]}.mp4"
            try:
                extract_clip_segment(Path(clip.stored_path), destination, offset_start, offset_end)
            except ClipExtractionError as error:
                LOGGER.warning(
                    "derived_clip_extraction_failed",
                    clip_id=str(clip.clip_id),
                    reason=str(error),
                )
                continue
            artifacts.append(
                ArtifactReference(
                    path=str(destination.resolve()),
                    artifact_type="derived_clip",
                    created_at=utc_now(),
                    sha256=hash_file(destination),
                    metadata={
                        "camera_id": clip.camera_id,
                        "source_clip_id": str(clip.clip_id),
                        "source_sha256": clip.sha256,
                        "extraction_note": (
                            "stream-copy cut; may extend slightly past the requested "
                            "window to the nearest preceding keyframe"
                        ),
                    },
                )
            )
        return artifacts

    @staticmethod
    def _summarize(signals: list[Signal], hypotheses: list[Hypothesis]) -> str:
        signal_types = sorted({signal.signal_type for signal in signals})
        labels = [hypothesis.label for hypothesis in hypotheses]
        return (
            f"Signals: {', '.join(signal_types)}. "
            f"Hypotheses: {', '.join(labels) if labels else 'none'}."
        )
