from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from gaggle.core.config import RuntimeConfig
from gaggle.normalize.sync import ClipTimeInfo, SessionSyncResult, compute_camera_sync
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.media import (
    CameraSync,
    IngestManifest,
    MediaClip,
    NormalizationManifest,
    NormalizedClip,
)
from gaggle.storage.database import TimelineDatabase
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class NormalizationService:
    """Applies cross-camera time synchronization to ingested clips.

    See ``gaggle.normalize.sync`` for the synchronization algorithm
    itself. This service is a thin, pydantic-aware wrapper: it converts
    ``MediaClip`` records into the plain ``ClipTimeInfo`` the sync algorithm
    operates on, runs the algorithm, and re-assembles the typed
    ``NormalizationManifest`` -- including one ``CameraSync`` record per
    recording session and one ``NormalizedClip`` per input clip that carries
    both its original and corrected timestamps. The underlying ``MediaClip``
    from the ingest stage is never mutated; correction is purely additive.
    """

    def __init__(
        self,
        workspace: WorkspacePaths,
        config: RuntimeConfig,
        database: TimelineDatabase | None = None,
    ) -> None:
        self.workspace = workspace
        self.config = config
        # Optional, same reasoning as IngestService.database: sources the
        # camera_id -> site_id map live (never snapshotted into the ingest
        # manifest, since camera metadata can change between ingest and
        # normalize) for site-scoped sync. Absent entirely, every camera
        # falls back to its own private site -- see compute_camera_sync's
        # docstring.
        self.database = database

    def normalize(self, ingest_manifest: IngestManifest) -> NormalizationManifest:
        clips = sorted(
            ingest_manifest.copied_files, key=lambda clip: (clip.camera_id, clip.observed_start)
        )
        site_id_by_camera = self.database.site_id_by_camera() if self.database else {}

        sync_results = compute_camera_sync(
            [
                ClipTimeInfo(
                    clip_id=str(clip.clip_id),
                    camera_id=clip.camera_id,
                    observed_start=clip.observed_start,
                    observed_end=clip.observed_end,
                    timestamp_confidence=clip.timestamp_confidence,
                )
                for clip in clips
            ],
            session_gap_seconds=self.config.sync.session_gap_seconds,
            site_id_by_camera=site_id_by_camera,
        )

        sync_results = [self._apply_manual_offset(result) for result in sync_results]
        camera_sync = [self._to_camera_sync(clips, result) for result in sync_results]
        session_by_clip_id: dict[str, SessionSyncResult] = {
            clip_id: result for result in sync_results for clip_id in result.clip_ids
        }
        normalized_clips = [
            self._normalize_clip(clip, session_by_clip_id[str(clip.clip_id)]) for clip in clips
        ]

        manifest = NormalizationManifest(
            run_id=new_uuid(),
            created_at=utc_now(),
            clips=normalized_clips,
            camera_sync=camera_sync,
            derived_artifacts=[
                ArtifactReference(
                    path=str((self.workspace.normalized / "latest.json").resolve()),
                    artifact_type="normalization_manifest",
                    created_at=utc_now(),
                )
            ],
        )
        output_path = self.workspace.normalized / f"{manifest.run_id}.json"
        self.workspace.write_json(output_path, manifest.model_dump(mode="json"))
        LOGGER.info(
            "normalization_completed",
            clip_count=len(clips),
            session_count=len(sync_results),
            run_id=str(manifest.run_id),
        )
        return manifest

    def _apply_manual_offset(self, result: SessionSyncResult) -> SessionSyncResult:
        """Applies `sync.manual_offset_overrides[result.camera_id]` (in
        seconds, default 0.0 -- no correction) on top of whatever
        `compute_camera_sync` already computed. This is a correction to
        the *algorithm's output*, not a change to the algorithm itself --
        see `SyncConfig.manual_offset_overrides`'s docstring in
        `core/config.py` for why. The reference session for its own group
        (`is_reference=True`) is exempt: every other session in the group
        already expresses its own offset *relative to* the reference, so
        shifting only the reference's own corrected timestamps would
        misalign it from the very sessions that were aligned to it --
        correcting a reference camera specifically would require shifting
        its whole group together, out of scope for a per-camera
        correction. Logged (not silently ignored) so a configured
        override that happens to target that run's reference camera is
        visible, not a silent no-op.
        """

        override_seconds = self.config.sync.manual_offset_overrides.get(result.camera_id, 0.0)
        if override_seconds == 0.0:
            return result
        if result.is_reference:
            LOGGER.warning(
                "manual_sync_offset_ignored_for_reference_camera",
                camera_id=result.camera_id,
                override_seconds=override_seconds,
            )
            return result
        offset = timedelta(seconds=override_seconds)
        return replace(
            result,
            corrected_start=result.corrected_start + offset,
            corrected_end=result.corrected_end + offset,
            offset_seconds=result.offset_seconds + override_seconds,
            rationale=(
                f"{result.rationale} Manual correction of {override_seconds:+.2f}s applied "
                f"via sync.manual_offset_overrides['{result.camera_id}']."
            ),
        )

    @staticmethod
    def _to_camera_sync(clips: list[MediaClip], result: SessionSyncResult) -> CameraSync:
        clip_by_id = {str(clip.clip_id): clip.clip_id for clip in clips}
        return CameraSync(
            camera_id=result.camera_id,
            session_id=result.session_id,
            clip_ids=[clip_by_id[clip_id] for clip_id in result.clip_ids],
            original_start=result.original_start,
            original_end=result.original_end,
            corrected_start=result.corrected_start,
            corrected_end=result.corrected_end,
            offset_seconds=result.offset_seconds,
            drift_seconds_per_hour=result.drift_seconds_per_hour,
            confidence=result.confidence,
            is_reference=result.is_reference,
            reference_camera_id=result.reference_camera_id,
            rationale=result.rationale,
        )

    @staticmethod
    def _normalize_clip(clip: MediaClip, session: SessionSyncResult) -> NormalizedClip:
        offset_seconds = (session.corrected_start - session.original_start).total_seconds()
        offset = timedelta(seconds=offset_seconds)
        return NormalizedClip(
            clip=clip,
            session_id=session.session_id,
            corrected_start=clip.observed_start + offset,
            corrected_end=clip.observed_end + offset,
            sync_confidence=session.confidence,
            sync_rationale=session.rationale,
        )
