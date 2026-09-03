"""Optional transcript analysis via an OpenAI-compatible endpoint (e.g. OpenRouter,
a self-hosted vLLM/Ollama server, or any other OpenAI-chat-compatible API).

This is the **only** module in the entire project that sends data over the
network by default when enabled, and it is disabled by default everywhere
(`core/config.py::CloudEnrichmentConfig.enabled` defaults to `False`). It
sends transcript *text only* -- never video, audio, images, or file paths
-- to a user-configured endpoint, and only when both an endpoint and an API
key are explicitly configured. See `docs/local-ai.md` for the full
data-flow explanation and `docs/threat-model.md` for what this changes
about the system's trust boundary.

The result is stored as an `LLMEnrichment` record: a labeled, versioned,
non-authoritative annotation, exactly like any other detector's output.
It never modifies `signals`, `hypotheses`, or `scoring` -- an LLM
"importance score" is not a substitute for the deterministic scoring
pipeline, only an additional data point a human reviewer can see.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PROMPT_VERSION = "1.0.0"

_SYSTEM_PROMPT = (
    "You are analyzing a short transcript of audio captured by a vehicle dashcam "
    "during a flagged incident window. Extract only what is explicitly said or "
    "clearly implied. Respond with strict JSON matching this shape: "
    '{"summary": str, "extracted_events": [str], "extracted_entities": [str], '
    '"importance_score": float between 0 and 1}. '
    "Do not invent details not present in the transcript. If the transcript is "
    "empty, uninformative, or just background noise/silence, say so plainly in "
    "the summary and set importance_score low."
)


class LlmEnrichmentUnavailableError(RuntimeError):
    """Raised when the `requests` dependency is missing."""


class LlmEnrichmentError(RuntimeError):
    """Raised when the remote call fails or returns an unparseable response."""


@dataclass(frozen=True, slots=True)
class LlmAnalysisResult:
    summary: str
    extracted_events: list[str]
    extracted_entities: list[str]
    importance_score: float | None
    raw_response_text: str


def requests_available() -> bool:
    try:
        import requests  # noqa: F401
    except ImportError:
        return False
    return True


def analyze_transcript(
    transcript_text: str,
    endpoint: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 30.0,
) -> LlmAnalysisResult:
    """POST ``transcript_text`` to an OpenAI-chat-compatible endpoint and parse the result.

    Never called unless the caller has already confirmed
    ``CloudEnrichmentConfig.enabled`` is true and both ``endpoint``/``api_key``
    are set -- this function itself doesn't consult config, so it can't
    accidentally fire from a code path that forgot to check.
    """

    if not requests_available():
        raise LlmEnrichmentUnavailableError(
            "the 'requests' package is not installed; install the 'cloud' extra "
            "(pip install gaggle[cloud]) to enable LLM transcript analysis"
        )
    import requests

    if not transcript_text.strip():
        return LlmAnalysisResult(
            summary="Empty transcript; nothing to analyze.",
            extracted_events=[],
            extracted_entities=[],
            importance_score=0.0,
            raw_response_text="",
        )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript_text},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
    except requests.RequestException as error:
        raise LlmEnrichmentError(f"request to {endpoint} failed: {error}") from error

    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as error:
        raise LlmEnrichmentError(
            f"could not parse a structured response from {endpoint}: {error}"
        ) from error

    importance = parsed.get("importance_score")
    return LlmAnalysisResult(
        summary=str(parsed.get("summary", "")),
        extracted_events=[str(item) for item in parsed.get("extracted_events", [])],
        extracted_entities=[str(item) for item in parsed.get("extracted_entities", [])],
        importance_score=(
            max(0.0, min(1.0, float(importance))) if isinstance(importance, int | float) else None
        ),
        raw_response_text=content,
    )
