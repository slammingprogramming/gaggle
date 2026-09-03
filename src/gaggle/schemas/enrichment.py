"""Transcription and optional cloud-LLM enrichment schemas.

Transcription (`AudioTranscript`) is a local, offline capability (see
`enrichment/transcription.py`) once a Whisper model is downloaded --
consistent with the project's offline-first requirement, the one-time model
download is the only network dependency, not runtime operation.

`LLMEnrichment` is different in kind: it is the *only* schema in this
project whose data originates from an external network service, and it is
always optional (disabled unless explicitly configured with an endpoint and
API key -- see `core/config.py::CloudEnrichmentConfig`). Its output is
treated exactly like any other detector's output per the project's
ML-avoidance/explainability directives: a labeled, versioned, non-authoritative
annotation that never overwrites or supersedes signals, hypotheses, or
scoring, and is always clearly attributable to "this came from an external
model," never presented as if it were a local deterministic finding.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from gaggle.schemas.common import JsonDict, StrictModel, UTCDateTime

ENRICHMENT_SCHEMA_VERSION = "1.0.0"


class TranscriptSegment(StrictModel):
    start_offset_seconds: float
    end_offset_seconds: float
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class AudioTranscript(StrictModel):
    schema_version: str = ENRICHMENT_SCHEMA_VERSION
    transcript_id: UUID
    clip_id: UUID
    created_at: UTCDateTime
    language: str | None = None
    model_name: str
    model_version: str
    device: str
    segments: list[TranscriptSegment] = Field(default_factory=list)
    full_text: str = ""


class LLMEnrichment(StrictModel):
    """An external model's read on a transcript. Always a hypothesis, never a fact.

    ``provider``/``model`` identify exactly which external service and
    model produced this, so it's never ambiguous which findings are local
    and deterministic versus remote and probabilistic.
    """

    schema_version: str = ENRICHMENT_SCHEMA_VERSION
    enrichment_id: UUID
    event_id: UUID
    created_at: UTCDateTime
    provider: str
    model: str
    endpoint: str
    summary: str
    extracted_events: list[str] = Field(default_factory=list)
    extracted_entities: list[str] = Field(default_factory=list)
    importance_score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_response_hash: str | None = None
    prompt_version: str
    metadata: JsonDict = Field(default_factory=dict)
