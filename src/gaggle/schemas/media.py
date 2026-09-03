from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import (
    ArtifactReference,
    HashDigest,
    JsonDict,
    StrictModel,
    UTCDateTime,
)


class MediaClip(StrictModel):
    clip_id: UUID
    camera_id: str
    source_path: str
    stored_path: str
    filename: str
    media_type: Literal["video", "audio", "image", "unknown"]
    byte_size: int = Field(ge=0)
    sha256: str
    observed_start: UTCDateTime
    observed_end: UTCDateTime
    original_timestamp_source: Literal["filename", "mtime", "sidecar", "manual"]
    timestamp_confidence: float = Field(ge=0.0, le=1.0)
    fps: float | None = Field(default=None, ge=0.0)
    duration_seconds: float = Field(ge=0.0)
    sidecar_artifacts: list[ArtifactReference] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    # How this clip's bytes ended up at `stored_path` -- "copy"/"move" mean
    # the workspace owns that file (originals/...); "reference" means
    # `stored_path` still points at wherever the source media lived at
    # ingest time (never copied or moved). This distinction matters for
    # deletion safety: deleting a "reference" clip's file deletes something
    # outside the workspace's own storage, not a workspace-owned copy -- see
    # `core/triage.py::TriageService.confirm_deletion`.
    ingest_mode: Literal["copy", "move", "reference"] = "copy"


class IngestManifest(StrictModel):
    run_id: UUID
    created_at: UTCDateTime
    source_root: str
    copied_files: list[MediaClip]
    config_snapshot: JsonDict = Field(default_factory=dict)
    hashes: list[HashDigest] = Field(default_factory=list)


class CameraSync(StrictModel):
    """Explainable synchronization correction for one recording session.

    A "session" is a contiguous run of clips from a single camera with no
    gap larger than the configured session-gap tolerance (see
    ``gaggle.core.config.SyncConfig``), modelling a single
    power-on/record cycle of that camera. Sessions from different cameras
    that overlap in time are grouped and aligned against a deterministically
    chosen reference camera (highest declared timestamp confidence, ties
    broken alphabetically by camera id). Cameras are never assumed to be
    correctly synchronized; when there is nothing to correlate against, the
    session is its own reference and receives no correction, which is
    reflected in the rationale and a lower confidence.
    """

    camera_id: str
    session_id: str
    clip_ids: list[UUID]
    original_start: UTCDateTime
    original_end: UTCDateTime
    corrected_start: UTCDateTime
    corrected_end: UTCDateTime
    offset_seconds: float
    drift_seconds_per_hour: float
    confidence: float = Field(ge=0.0, le=1.0)
    is_reference: bool
    reference_camera_id: str | None
    rationale: str


class NormalizedClip(StrictModel):
    """A clip annotated with synchronization-corrected timestamps.

    The normalize stage never mutates the ingest-stage ``MediaClip``; it
    wraps it with a corrected view instead, so both the original and
    corrected timestamps remain permanently inspectable side by side.
    """

    clip: MediaClip
    session_id: str
    corrected_start: UTCDateTime
    corrected_end: UTCDateTime
    sync_confidence: float = Field(ge=0.0, le=1.0)
    sync_rationale: str

    @property
    def camera_id(self) -> str:
        return self.clip.camera_id

    @property
    def clip_id(self) -> UUID:
        return self.clip.clip_id

    @property
    def stored_path(self) -> str:
        return self.clip.stored_path

    @property
    def sha256(self) -> str:
        return self.clip.sha256


class NormalizationManifest(StrictModel):
    run_id: UUID
    created_at: UTCDateTime
    clips: list[NormalizedClip]
    camera_sync: list[CameraSync]
    derived_artifacts: list[ArtifactReference] = Field(default_factory=list)


class EventWindow(StrictModel):
    window_id: UUID
    start: UTCDateTime
    end: UTCDateTime
    involved_cameras: list[str]
    clip_ids: list[UUID]
    rationale: str
    metadata: JsonDict = Field(default_factory=dict)


class WindowManifest(StrictModel):
    run_id: UUID
    created_at: UTCDateTime
    windows: list[EventWindow]
    source_normalization_run_id: UUID
