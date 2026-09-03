"""Storage-lifecycle schemas: triage classification and deletion records.

The core tension this module exists to resolve: dashcam ingests are large
(a 256GB card at a time) and most of it is uneventful driving, but this is
still a forensic system -- nothing gets deleted silently or automatically.
See `docs/architecture.md`'s storage-lifecycle section for the full
design. In short:

* A clip that contributed zero signals during analysis is classified
  ``benign`` and becomes eligible for human-confirmed deletion.
* A clip that contributed to at least one event is classified
  ``reviewable`` and is never a deletion candidate through this workflow.
* Deleting a clip's original bytes is always a deliberate, actor-attributed,
  logged action (``DeletionRecord``), appended to a durable, append-only
  log (``workspace/deletion_log.jsonl``) -- the same append-only pattern
  used for review actions -- so there is a permanent record that the file
  existed, what its hash was, why it was judged benign, and who confirmed
  its deletion, even though the bytes themselves are gone.
* Separately, an event that has already been reviewed and (usually)
  preserved can have its *video* purged -- its own derived clips, and
  cascading to contributing originals only where safe -- while its
  ``event.json``, signals, hypotheses, scoring, and full history stay
  exactly as they were forever. See ``EventVideoPurgeRecord``.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import JsonDict, StrictModel, UTCDateTime

LIFECYCLE_SCHEMA_VERSION = "1.0.0"

TriageState = Literal["unclassified", "reviewable", "benign_pending_deletion", "deleted"]


class TriageRecord(StrictModel):
    """Current triage classification for one ingested clip.

    This is operational metadata, not primary forensic evidence -- it is
    fully re-derivable from the set of events referencing (or not
    referencing) a clip, exactly like the SQLite index (see
    ``docs/architecture.md``'s hybrid storage model). It is stored in the
    database, not as a frozen filesystem artifact.
    """

    schema_version: str = LIFECYCLE_SCHEMA_VERSION
    clip_id: UUID
    camera_id: str
    state: TriageState
    signal_count: int
    event_ids: list[UUID] = Field(default_factory=list)
    classified_at: UTCDateTime
    reason: str


class DeletionRecord(StrictModel):
    """Permanent, append-only record that an original clip's bytes were deleted.

    Written to ``workspace/deletion_log.jsonl`` *before* the file is
    unlinked, never edited afterward. This is the durable evidence that a
    review pass happened and a human explicitly confirmed deletion --
    intentionally kept independent of the SQLite index so it survives even
    if the index is deleted or corrupted.
    """

    schema_version: str = LIFECYCLE_SCHEMA_VERSION
    deletion_id: UUID
    clip_id: UUID
    camera_id: str
    original_stored_path: str
    sha256: str
    byte_size: int
    ingest_run_id: UUID | None = None
    triage_reason: str
    confirmed_by: str
    confirmed_at: UTCDateTime
    notes: str = ""
    metadata: JsonDict = Field(default_factory=dict)


class EventVideoPurgeRecord(StrictModel):
    """Permanent, append-only record that an event's video evidence was purged.

    Written to ``workspace/event_video_purge_log.jsonl`` *before* any file
    is deleted, mirroring ``DeletionRecord``'s pattern. Purging an event's
    video is a coarser, event-scoped operation than deleting a single
    clip: it removes the event's own derived clips
    (``events/<id>/clips/``) and, only where no other unpurged event still
    needs them, cascades to the contributing original clips too (each
    cascaded original deletion gets its own ``DeletionRecord`` in the usual
    deletion log -- this record is the event-level summary of that whole
    operation, not a replacement for it).

    ``event.json`` and its full revision history are never touched by a
    purge -- signals, hypotheses, scoring, chain of custody, and every
    review decision remain exactly as they were. Only the video bytes are
    removed; see ``EventRecord.video_purged_at``.
    """

    schema_version: str = LIFECYCLE_SCHEMA_VERSION
    purge_id: UUID
    event_id: UUID
    deleted_derived_clip_paths: list[str] = Field(default_factory=list)
    deleted_derived_clip_hashes: list[str] = Field(default_factory=list)
    cascaded_original_clip_ids: list[UUID] = Field(default_factory=list)
    retained_original_clip_ids: list[UUID] = Field(default_factory=list)
    retained_reason: str = ""
    required_preservation: bool
    was_preserved_at_time_of_purge: bool
    confirmed_by: str
    confirmed_at: UTCDateTime
    notes: str = ""
