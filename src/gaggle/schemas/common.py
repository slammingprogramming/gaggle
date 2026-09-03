from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from gaggle.utils.time import ensure_utc

UTCDateTime = Annotated[datetime, AfterValidator(ensure_utc)]
JsonDict = dict[str, Any]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HashDigest(StrictModel):
    algorithm: str = Field(default="sha256")
    value: str


class ArtifactReference(StrictModel):
    path: str
    artifact_type: str
    created_at: UTCDateTime
    sha256: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


class ChainOfCustodyEntry(StrictModel):
    entry_id: UUID
    action: str
    actor: str
    timestamp: UTCDateTime
    details: JsonDict = Field(default_factory=dict)
    input_hashes: list[HashDigest] = Field(default_factory=list)
    output_hashes: list[HashDigest] = Field(default_factory=list)
