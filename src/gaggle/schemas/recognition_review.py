"""Append-only audit records for human review and storage-reclamation of
recognition data (faces, plates, vehicle-appearance).

Two distinct, separately-logged actions, mirroring the storage-lifecycle
module's `DeletionRecord`/`EventVideoPurgeRecord` split exactly:

* **Reviewing** (`RecognitionReviewRecord`) -- a human confirms a cluster
  is a real, consistent identity (optionally picking representative
  crop(s) and a label) or rejects it/an individual observation as a false
  positive. Never touches a file; only changes `review_status`/
  `representative_observation_ids` on the underlying schema rows (see
  `schemas/recognition.py`).
* **Purging** (`RecognitionCropPurgeRecord`) -- a separate, later step
  that actually deletes already-reviewed, non-representative/rejected
  crop image files from disk, written *before* any file is unlinked
  (same ordering discipline as `DeletionRecord`). The observation row
  itself, including `crop_path` as a historical pointer, is never
  deleted or rewritten -- only `crop_purged_at` is set. See AGENTS.md
  invariant 22.

Written to `workspace/recognition_review_log.jsonl` and
`workspace/recognition_crop_purge_log.jsonl` respectively via
`storage/filesystem.py::WorkspacePaths.append_recognition_review_record`/
`append_recognition_crop_purge_record`.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import StrictModel, UTCDateTime

RECOGNITION_REVIEW_SCHEMA_VERSION = "1.0.0"


class RecognitionReviewRecord(StrictModel):
    schema_version: str = RECOGNITION_REVIEW_SCHEMA_VERSION
    review_id: UUID
    # "detached"/"moved" added for RecognitionService.detach_observation/
    # move_observation -- purely additive, old log lines with the
    # original three values still validate unchanged.
    action: Literal["confirmed", "rejected", "representative_selected", "detached", "moved"]
    # "person_appearance" added alongside enrichment/person_appearance.py --
    # purely additive, old log lines with the original three values still
    # validate unchanged.
    entity_type: Literal["face", "plate", "vehicle_appearance", "person_appearance"]
    # None for a plate, which has no cluster concept -- each observation
    # is reviewed individually.
    cluster_id: UUID | None = None
    # One or many -- a bulk cluster-confirm/reject touches every
    # observation in the cluster in a single record, not one per
    # observation.
    observation_ids: list[UUID] = Field(default_factory=list)
    label: str | None = None
    actor: str
    timestamp: UTCDateTime
    notes: str = ""


class RecognitionCropPurgeRecord(StrictModel):
    schema_version: str = RECOGNITION_REVIEW_SCHEMA_VERSION
    purge_id: UUID
    entity_type: Literal["face", "plate", "vehicle_appearance", "person_appearance"]
    purged_observation_ids: list[UUID] = Field(default_factory=list)
    purged_crop_paths: list[str] = Field(default_factory=list)
    purged_crop_hashes: list[str] = Field(default_factory=list)
    confirmed_by: str
    confirmed_at: UTCDateTime
    notes: str = ""
