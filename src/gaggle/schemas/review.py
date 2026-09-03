from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import JsonDict, StrictModel, UTCDateTime

REVIEW_ACTION_SCHEMA_VERSION = "1.0.0"


class ReviewAction(StrictModel):
    schema_version: str = REVIEW_ACTION_SCHEMA_VERSION
    action_id: UUID
    event_id: UUID
    action: Literal["accept", "reject", "annotate", "retag", "preserve", "export"]
    actor: str
    timestamp: UTCDateTime
    notes: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
