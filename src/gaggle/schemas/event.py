from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import (
    ArtifactReference,
    ChainOfCustodyEntry,
    JsonDict,
    StrictModel,
    UTCDateTime,
)
from gaggle.schemas.signal import Signal

EVENT_SCHEMA_VERSION = "1.2.0"


class Hypothesis(StrictModel):
    hypothesis_id: UUID
    rule_name: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_signal_ids: list[UUID] = Field(default_factory=list)
    escalation_reasons: list[str] = Field(default_factory=list)
    confidence_math: str
    metadata: JsonDict = Field(default_factory=dict)


class SeverityAssessment(StrictModel):
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high"]
    reasons: list[str]
    version: str


class PreservationStatus(StrictModel):
    state: Literal["pending", "preserved", "exported"]
    immutable: bool
    preserved_at: UTCDateTime | None = None
    bundle_path: str | None = None
    bundle_hash: str | None = None


class ReviewSummary(StrictModel):
    latest_decision: Literal["pending", "accepted", "rejected"] = "pending"
    action_count: int = 0
    last_reviewed_at: UTCDateTime | None = None
    last_action_id: UUID | None = None


class EventRecord(StrictModel):
    """The canonical forensic artifact for a single detected incident window.

    ``EventRecord`` is versioned and revisioned rather than mutated in place.
    Every change to an already-written event (a preservation action, a review
    decision, a retag) MUST be persisted as a new revision via
    ``gaggle.storage.repository.Repository.save_event_revision``.
    The filesystem keeps the full, append-only revision history; the
    top-level ``event.json`` always mirrors the latest revision so it stays
    convenient to read, while ``revisions/000N_<reason>.json`` files are
    frozen (read-only) at write time and are never edited or deleted.
    ``previous_revision_hash`` links each revision to the canonical JSON
    hash of the one before it, forming an inspectable hash chain that can
    later support cryptographic signing without a schema change.
    """

    schema_version: str = EVENT_SCHEMA_VERSION
    event_id: UUID
    created_at: UTCDateTime
    pipeline_version: str
    config_snapshot: JsonDict = Field(default_factory=dict)
    event_start: UTCDateTime
    event_end: UTCDateTime
    involved_cameras: list[str]
    signals: list[Signal]
    hypotheses: list[Hypothesis]
    scoring: SeverityAssessment
    preservation_status: PreservationStatus
    review_summary: ReviewSummary = Field(default_factory=ReviewSummary)
    chain_of_custody: list[ChainOfCustodyEntry] = Field(default_factory=list)
    hashes: list[str] = Field(default_factory=list)
    derived_artifacts: list[ArtifactReference] = Field(default_factory=list)
    evidence_summary: str
    metadata: JsonDict = Field(default_factory=dict)
    revision: int = 0
    revision_reason: str = "initial_generation"
    revised_at: UTCDateTime | None = None
    previous_revision_hash: str | None = None
    # Set only when the workspace has signing enabled and an Ed25519 key
    # (see `core/signing.py`, `workspace signing-init`) -- a hex-encoded
    # signature over this revision's own canonical JSON payload with this
    # field itself excluded, the same self-referential-hash pattern
    # `export/service.py`'s `manifest_hash` uses. None means either signing
    # was never enabled, or this revision predates it being turned on.
    revision_signature: str | None = None
    # Set when this event's video evidence (its own derived clips, and any
    # contributing originals that were safe to cascade) has been purged via
    # `core/triage.py::TriageService.purge_event_video`. None means the
    # video is still present (in `events/<id>/clips/`, and/or the
    # contributing originals); everything else about the event --
    # signals, hypotheses, scoring, chain of custody, review history -- is
    # never affected by a purge. See `schemas/lifecycle.py::EventVideoPurgeRecord`.
    video_purged_at: UTCDateTime | None = None
    # Per-capability completion tracking for `enrichment/service.py::EnrichmentService.enrich_event`
    # -- keys are "face"/"plate"/"voice"/"vision"/"vehicle_appearance"/
    # "person_appearance"/"transcription"/"cloud"/"encounters", values are
    # when that capability
    # last ran to completion on this event (set even if it found nothing --
    # "ran and found nothing" must be distinguishable from "never
    # attempted", which a Signal.source scan alone cannot tell apart).
    # Absent from event.json files written before this field existed;
    # pydantic defaults those to `{}`, meaning every capability is treated
    # as not-yet-run, which is the safe direction (re-confirms rather than
    # silently skips real work on old data). Enables `enrich` to be called
    # any number of times, in any order relative to `ingest`/`analyze`,
    # without redoing already-completed work or corrupting the incremental
    # clustering models that a duplicate run would otherwise retrain twice.
    enrichment_completed: dict[str, UTCDateTime] = Field(default_factory=dict)
    # Set via a final revision by `core/events.py::EventSplitService.split_event`
    # when a human determines this event incorrectly bundled clips from
    # separate recording sessions (normalize/sync.py's pure time-overlap
    # heuristic has no camera/session-boundary or duration-similarity
    # check -- see its module docstring) and splits it into these
    # independent replacement events. The original is never deleted or
    # edited beyond this field -- every signal/derived_artifact/revision
    # stays exactly as it was, fully readable and auditable; review_ui
    # shows a "split into" banner instead of normal content when this is
    # non-empty. Absent/empty for every event that was never split, which
    # is every event written before this field existed -- pydantic
    # defaults those to `[]`.
    superseded_by_event_ids: list[UUID] = Field(default_factory=list)
