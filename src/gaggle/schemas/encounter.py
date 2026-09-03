"""Cross-modality encounter records -- grouping face/plate/voice/vehicle-
appearance observations that occurred close together in time within the
same clip.

**Scope and intent, read this before extending this module:** an
`Encounter` does *not* claim any spatial correspondence between the
observations it groups. If one references both a face and a
vehicle-appearance observation, that means only "these were observed in
the same clip within a few seconds of each other" -- never "this face
belongs to this vehicle" or "this person was near/driving this vehicle."
None of the four observation schemas (`schemas/recognition.py`) currently
store bounding-box coordinates on the record itself (only `crop_path`), so
per-entity spatial disambiguation isn't available data yet -- this is a
documented first-pass limitation, not an oversight. See
`docs/limitations.md`.

Derived automatically by
`enrichment/service.py::EnrichmentService._derive_encounters` as a final
post-processing step over already-persisted observations (see that
method's docstring for the grouping algorithm), gated by
`RuntimeConfig.enrichment.encounters.enabled`. Never user-edited or
human-reviewed the way an event/review action is -- purely a derived query
convenience for "what was observed together," consistent with the
non-accusatory framing used throughout `patterns/service.py`.
"""

from __future__ import annotations

from uuid import UUID

from gaggle.schemas.common import StrictModel, UTCDateTime

ENCOUNTER_SCHEMA_VERSION = "1.1.0"


class Encounter(StrictModel):
    schema_version: str = ENCOUNTER_SCHEMA_VERSION
    encounter_id: UUID
    event_id: UUID
    clip_id: UUID
    camera_id: str
    observed_at: UTCDateTime
    face_observation_id: UUID | None = None
    plate_observation_id: UUID | None = None
    voice_observation_id: UUID | None = None
    vehicle_appearance_observation_id: UUID | None = None
    # A 5th modality, added alongside enrichment/person_appearance.py --
    # absent from every Encounter written before this field existed;
    # pydantic defaults those to `None`, same as every other optional
    # observation-id field here.
    person_appearance_observation_id: UUID | None = None
