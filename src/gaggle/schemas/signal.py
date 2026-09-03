from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import ArtifactReference, JsonDict, StrictModel, UTCDateTime

SIGNAL_SCHEMA_VERSION = "1.2.0"


class Signal(StrictModel):
    schema_version: str = SIGNAL_SCHEMA_VERSION
    id: UUID
    source: str
    signal_type: Literal[
        "motion",
        "audio_spike",
        "object_hint",
        "telemetry",
        "coverage",
        "face_detection",
        "license_plate",
        "vehicle_detection",
        "vehicle_appearance",
        "transcript_keyword",
        "voice_detection",
        "rapid_approach",
        # Added alongside enrichment/person_appearance.py -- purely
        # additive; an old Signal already on disk with one of the values
        # above still validates unchanged.
        "person_appearance",
        # Added alongside detection/gunshot_analysis.py -- purely
        # additive, same as above.
        "gunshot",
    ]
    timestamp_start: UTCDateTime
    timestamp_end: UTCDateTime
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str | None = None
    window_id: UUID | None = None
    evidence_references: list[ArtifactReference] = Field(default_factory=list)
    spatial_metadata: JsonDict = Field(default_factory=dict)
    reasoning_metadata: JsonDict = Field(default_factory=dict)
