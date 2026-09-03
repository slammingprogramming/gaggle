from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from gaggle.schemas.event import EventRecord, Hypothesis
from gaggle.schemas.media import EventWindow, NormalizedClip
from gaggle.schemas.review import ReviewAction
from gaggle.schemas.signal import Signal


class DetectorPlugin(Protocol):
    """Extension point for additional signal sources (e.g. a real ML object
    detector, vehicle telemetry, a different audio front-end). A plugin
    detector must produce the same ``Signal`` shape as the built-in
    detectors -- evidence, never a conclusion -- and should be a pure
    function of its inputs for reproducibility.
    """

    name: str
    version: str

    def detect(
        self, windows: Iterable[EventWindow], clips: Iterable[NormalizedClip]
    ) -> list[Signal]: ...


class InferenceRulePlugin(Protocol):
    """Extension point for additional rule-based inference logic. Plugin
    rules run after the built-in rules and contribute additional
    hypotheses; they never see or alter hypotheses produced by other rules,
    keeping each rule independently auditable.
    """

    name: str
    version: str

    def apply(self, signals: Iterable[Signal]) -> list[Hypothesis]: ...


class ExporterPlugin(Protocol):
    """Extension point for additional export formats (e.g. an
    insurance-specific bundle format, a law-enforcement handoff format).
    """

    name: str
    version: str
    format_id: str

    def export(self, event_path: str, destination: str) -> str: ...


class ReviewExtensionPlugin(Protocol):
    """Extension point that observes review actions after they've already
    been durably persisted (e.g. a custom notification, an external
    ticketing hand-off, a site-specific audit mirror). Called strictly
    *after* ``Repository.append_review_action`` returns -- a review
    extension can never block, delay, or alter a review action, only react
    to one that has already happened. This keeps the append-only review
    log's guarantees (invariant: human review actions are never lost or
    silently altered) entirely independent of whatever third-party plugins
    are installed.
    """

    name: str
    version: str

    def on_review_action(self, action: ReviewAction, event: EventRecord) -> None: ...
