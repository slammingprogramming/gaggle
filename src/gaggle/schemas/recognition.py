"""Local re-identification schemas for recurring faces and license plates.

**Scope and intent, read this before extending this module:** everything
here does *pattern re-identification within the user's own footage* --
"have I seen this face/plate before, and when" -- never *identification*.
There is no name resolution, no linking to public records, no reverse
lookup, and no networking with other cameras or users. A face becomes a
`FaceCluster` (an anonymous, locally-generated group of similar-looking
detections) that the user may optionally label with their own private
nickname; nothing here ever attempts to determine who someone actually is.
See `docs/local-ai.md` and `docs/forensic-considerations.md` for the full
rationale, including jurisdiction-specific legal considerations around
biometric data (e.g. BIPA, GDPR) that a deployer should review before
enabling face recognition.

Storage design: only small, cheap artifacts are kept forever (a handful of
JPEG crops per cluster, a numeric embedding/histogram, OCR text) -- never
the full source video -- so recognition data survives long after the
originating raw footage has been deleted (see `docs/architecture.md`'s
storage-lifecycle section).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import ArtifactReference, JsonDict, StrictModel, UTCDateTime

RECOGNITION_SCHEMA_VERSION = "1.1.0"


class FaceObservation(StrictModel):
    """One detected face in one frame of one clip."""

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    observation_id: UUID
    signal_id: UUID
    event_id: UUID | None = None
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    crop_path: str
    crop_sha256: str
    detector_confidence: float = Field(ge=0.0, le=1.0)
    embedding_distance_to_cluster: float | None = None
    cluster_id: UUID | None = None
    detector_version: str
    # Set only by automated cleanup (`recognize faces-cleanup`,
    # `core/recognition.py::RecognitionService.cleanup_duplicate_face_observations`)
    # when this observation was judged a redundant repeat of another
    # observation of the same cluster within the same event, sampled
    # within seconds of it. Never set by a human action. The observation
    # itself is never deleted -- this only affects whether
    # `faces-sightings` surfaces it by default.
    duplicate_of_observation_id: UUID | None = None
    # Human review outcome (`recognize faces-confirm`/`faces-reject-*`) --
    # never set automatically. `needs_review` until a human looks at it.
    review_status: Literal["needs_review", "user_confirmed", "user_rejected"] = "needs_review"
    # Set once this observation's crop image has actually been deleted
    # from disk (`recognize faces-purge-reviewed`) -- the observation row
    # itself, including `crop_path` as a historical pointer, is never
    # deleted. See AGENTS.md invariant 22.
    crop_purged_at: UTCDateTime | None = None


class FaceCluster(StrictModel):
    """A locally-generated group of similar-looking face observations.

    Deliberately anonymous by default: `label` is a free-text field the
    user may set for their own private reference (e.g. "neighbor," "mail
    carrier") -- never populated automatically, never a real name lookup.

    Because the underlying LBPH clusterer is an *incremental* algorithm
    (see `enrichment/face.py`), the same real face can sometimes spawn more
    than one `FaceCluster` (e.g. a different angle or lighting condition
    that fell outside the match threshold). `merged_into`, set via
    `recognize faces-merge`, lets a human declare "these are the same" --
    this cluster is never deleted or rewritten, it's just marked as an
    alias of the target cluster. Resolving the *canonical* identity for a
    cluster means following `merged_into` until it's `None`; see
    `core/recognition.py::RecognitionService.resolve_face_identity`.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    cluster_id: UUID
    created_at: UTCDateTime
    updated_at: UTCDateTime
    label: str | None = None
    representative_crop_paths: list[str] = Field(default_factory=list)
    observation_count: int = 0
    first_seen_at: UTCDateTime | None = None
    last_seen_at: UTCDateTime | None = None
    model_version: str
    merged_into: UUID | None = None
    # Human-curated override, set only by `recognize faces-confirm`. Empty
    # means "not yet reviewed" -- `representative_crop_paths` above keeps
    # being auto-maintained (rolling last-4) as it always has. Once a
    # human confirms this cluster, this becomes authoritative and
    # `representative_crop_paths` is recomputed to match exactly these
    # observations' crops -- every other observation in the cluster
    # becomes eligible for crop purging.
    representative_observation_ids: list[UUID] = Field(default_factory=list)


class PlateObservation(StrictModel):
    """One detected + OCR'd license plate in one frame of one clip."""

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    observation_id: UUID
    signal_id: UUID
    event_id: UUID | None = None
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    crop_path: str
    crop_sha256: str
    raw_ocr_text: str
    normalized_text: str
    ocr_confidence: float = Field(ge=0.0, le=1.0)
    detector_confidence: float = Field(ge=0.0, le=1.0)
    review_status: Literal[
        "auto_accepted",
        "needs_review",
        "user_confirmed",
        "user_rejected",
        "duplicate_suppressed",
    ]
    user_corrected_text: str | None = None
    detector_version: str
    metadata: JsonDict = Field(default_factory=dict)
    # Set only when review_status == "duplicate_suppressed": which other
    # observation (of the same plate, same event, sampled within seconds of
    # this one) was judged the representative sighting instead. This is an
    # automated bookkeeping decision, not a human one -- see
    # `core/recognition.py::RecognitionService.cleanup_duplicate_plate_observations`
    # and never overrides a `user_confirmed`/`user_rejected` observation.
    duplicate_of_observation_id: UUID | None = None
    # Set once this observation's crop image has actually been deleted
    # from disk (`recognize plates-purge-reviewed`, or `--purge` on
    # `plates-confirm`/`plates-reject`) -- the observation row itself,
    # including `crop_path` as a historical pointer, is never deleted.
    # See AGENTS.md invariant 22.
    crop_purged_at: UTCDateTime | None = None


class PlateRecord(StrictModel):
    """Aggregated view of all observations that normalize to the same plate text.

    `merged_into` mirrors `FaceCluster.merged_into`: if OCR ever reads the
    same real plate slightly differently across sightings (e.g. "1" vs
    "I"), two `PlateRecord`s can end up representing one real vehicle.
    `recognize plates-merge` lets a human declare that explicitly without
    editing or deleting either record.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    plate_id: UUID
    normalized_text: str
    created_at: UTCDateTime
    updated_at: UTCDateTime
    label: str | None = None
    observation_count: int = 0
    first_seen_at: UTCDateTime | None = None
    last_seen_at: UTCDateTime | None = None
    example_crops: list[ArtifactReference] = Field(default_factory=list)
    merged_into: UUID | None = None


class VoiceObservation(StrictModel):
    """One detected voice segment in one clip's audio track.

    Unlike face/plate observations, there's no crop image -- the
    identifying artifact is `voiceprint`, a fixed-length numeric vector
    (see `enrichment/voice.py` for what it actually is and how it's
    computed), stored directly rather than derived from a file that could
    later be deleted. This is deliberate: it's what lets voice
    re-identification keep working even if the source audio/clip is later
    purged or deleted, the same durability property LBPH's trained model
    gives face re-identification independent of crop images.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    observation_id: UUID
    signal_id: UUID
    event_id: UUID | None = None
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    segment_start_seconds: float = Field(ge=0.0)
    segment_end_seconds: float = Field(ge=0.0)
    voiceprint: list[float]
    energy_confidence: float = Field(ge=0.0, le=1.0)
    cluster_id: UUID | None = None
    detector_version: str
    duplicate_of_observation_id: UUID | None = None


class VoiceCluster(StrictModel):
    """A locally-generated group of similar-sounding voice observations.

    Mirrors `FaceCluster` in every respect that matters: anonymous by
    default, `label` is free-text and user-set only, `merged_into` lets a
    human declare two clusters the same speaker without editing or
    deleting either one. See `enrichment/voice.py`'s module docstring for
    why this is meaningfully less reliable than face re-identification and
    should be treated as a heuristic aid, not a confident match.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    cluster_id: UUID
    created_at: UTCDateTime
    updated_at: UTCDateTime
    label: str | None = None
    observation_count: int = 0
    first_seen_at: UTCDateTime | None = None
    last_seen_at: UTCDateTime | None = None
    model_version: str
    merged_into: UUID | None = None


class VehicleAppearanceObservation(StrictModel):
    """One detected vehicle-appearance fingerprint in one frame of one clip.

    Vehicle re-identification here is a coarse, classical appearance
    fingerprint (dominant hue/saturation histogram + aspect ratio -- see
    `enrichment/vehicle_appearance.py`'s module docstring), used to
    recognize a vehicle seen without a legible plate. Meaningfully weaker
    than plate-based identity: it cannot distinguish two different
    vehicles of the same color and body shape, and is far more sensitive
    to lighting/angle than face or plate re-identification. Treat every
    match as a considerably weaker signal than a face or plate match --
    see `docs/limitations.md`.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    observation_id: UUID
    signal_id: UUID
    event_id: UUID | None = None
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    crop_path: str
    crop_sha256: str
    fingerprint: list[float]
    detector_confidence: float = Field(ge=0.0, le=1.0)
    embedding_distance_to_cluster: float | None = None
    cluster_id: UUID | None = None
    detector_version: str
    # Set only by automated cleanup
    # (`recognize vehicles-cleanup`,
    # `core/recognition.py::RecognitionService.cleanup_duplicate_vehicle_appearance_observations`)
    # -- mirrors `FaceObservation.duplicate_of_observation_id` exactly.
    duplicate_of_observation_id: UUID | None = None
    # Human review outcome (`recognize vehicles-confirm`/`vehicles-reject-*`)
    # -- mirrors `FaceObservation.review_status` exactly.
    review_status: Literal["needs_review", "user_confirmed", "user_rejected"] = "needs_review"
    # Mirrors `FaceObservation.crop_purged_at` exactly. See AGENTS.md
    # invariant 22.
    crop_purged_at: UTCDateTime | None = None


class VehicleAppearanceCluster(StrictModel):
    """A locally-generated group of similar-looking vehicle-appearance
    observations. Mirrors `FaceCluster` in every respect that matters:
    anonymous by default, `label` is free-text and user-set only,
    `merged_into` lets a human declare two clusters the same vehicle
    without editing or deleting either one. See
    `docs/forensic-considerations.md`'s "Recognition data: scope and
    intent" -- no plate/name link, no external lookup, ever.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    cluster_id: UUID
    created_at: UTCDateTime
    updated_at: UTCDateTime
    label: str | None = None
    representative_crop_paths: list[str] = Field(default_factory=list)
    observation_count: int = 0
    first_seen_at: UTCDateTime | None = None
    last_seen_at: UTCDateTime | None = None
    model_version: str
    merged_into: UUID | None = None
    # Mirrors `FaceCluster.representative_observation_ids` exactly.
    representative_observation_ids: list[UUID] = Field(default_factory=list)


class PersonAppearanceObservation(StrictModel):
    """One detected person-appearance fingerprint in one frame of one clip.

    Structurally identical to `VehicleAppearanceObservation` -- same
    coarse, classical appearance fingerprint technique (dominant
    hue/saturation histogram + aspect ratio, see
    `enrichment/person_appearance.py`'s module docstring), used to
    recognize a person seen again without face recognition necessarily
    having matched (e.g. facing away from the camera). Meaningfully
    weaker than face-based identity: it cannot distinguish two different
    people wearing similarly-colored clothing, and is far more sensitive
    to lighting/angle/clothing changes than face re-identification. Treat
    every match as a considerably weaker signal than a face match -- see
    `docs/limitations.md`.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    observation_id: UUID
    signal_id: UUID
    event_id: UUID | None = None
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    crop_path: str
    crop_sha256: str
    fingerprint: list[float]
    detector_confidence: float = Field(ge=0.0, le=1.0)
    embedding_distance_to_cluster: float | None = None
    cluster_id: UUID | None = None
    detector_version: str
    # Mirrors `VehicleAppearanceObservation.duplicate_of_observation_id` exactly.
    duplicate_of_observation_id: UUID | None = None
    # Mirrors `VehicleAppearanceObservation.review_status` exactly.
    review_status: Literal["needs_review", "user_confirmed", "user_rejected"] = "needs_review"
    # Mirrors `VehicleAppearanceObservation.crop_purged_at` exactly. See
    # AGENTS.md invariant 22.
    crop_purged_at: UTCDateTime | None = None


class PersonAppearanceCluster(StrictModel):
    """A locally-generated group of similar-looking person-appearance
    observations. Mirrors `VehicleAppearanceCluster` in every respect
    that matters: anonymous by default, `label` is free-text and
    user-set only, `merged_into` lets a human declare two clusters the
    same person without editing or deleting either one. See
    `docs/forensic-considerations.md`'s "Recognition data: scope and
    intent" -- no name link, no external lookup, ever.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    cluster_id: UUID
    created_at: UTCDateTime
    updated_at: UTCDateTime
    label: str | None = None
    representative_crop_paths: list[str] = Field(default_factory=list)
    observation_count: int = 0
    first_seen_at: UTCDateTime | None = None
    last_seen_at: UTCDateTime | None = None
    model_version: str
    merged_into: UUID | None = None
    representative_observation_ids: list[UUID] = Field(default_factory=list)


class IdentityMergeRecord(StrictModel):
    """Permanent, append-only record that a human declared two clusters/plate
    records the same identity. Written to `workspace/identity_merge_log.jsonl`,
    never edited afterward -- the same append-only pattern as `ReviewAction`
    and `DeletionRecord`. This is what makes a merge traceable: not just
    "these are now linked" but "who declared that, and when."
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    merge_id: UUID
    # "person_appearance" added alongside enrichment/person_appearance.py --
    # purely additive, old log lines with the original four values still
    # validate unchanged.
    entity_type: Literal["face", "plate", "voice", "vehicle_appearance", "person_appearance"]
    source_id: UUID
    target_id: UUID
    actor: str
    timestamp: UTCDateTime
    notes: str = ""


class MergeSuggestion(StrictModel):
    """An automated "these might be the same identity" suggestion, awaiting
    human confirmation or rejection -- never merged automatically.

    Generated by `core/recognition.py::RecognitionService.suggest_face_merges`/
    `suggest_plate_merges`/`suggest_voice_merges`. `similarity_score` is
    normalized to 0-1 (higher = more similar) so it reads consistently
    across entity types even though the underlying distance metrics differ
    (LBPH distance for faces, text-edit-distance ratio for plates, cosine
    distance for voiceprints). `basis` is a short, human-readable
    explanation of why the suggestion was made -- always something a
    reviewer can sanity-check, never an opaque score alone.

    Confirming a suggestion performs the actual merge (via `merge_faces`/
    `merge_plates`/`merge_voices`, which still writes its own
    `IdentityMergeRecord`) and marks the suggestion `confirmed`. Rejecting
    one performs no merge and marks it `rejected`. Either way the
    suggestion itself is retained (not deleted) as a record of what was
    proposed and how it was resolved.
    """

    schema_version: str = RECOGNITION_SCHEMA_VERSION
    suggestion_id: UUID
    entity_type: Literal["face", "plate", "voice", "vehicle_appearance", "person_appearance"]
    source_id: UUID
    target_id: UUID
    similarity_score: float = Field(ge=0.0, le=1.0)
    basis: str
    status: Literal["pending", "confirmed", "rejected"] = "pending"
    created_at: UTCDateTime
    resolved_at: UTCDateTime | None = None
    resolved_by: str | None = None
