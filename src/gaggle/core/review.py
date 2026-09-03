from __future__ import annotations

from typing import Literal
from uuid import UUID

from gaggle.plugins.registry import REVIEW_EXTENSION_PLUGIN_GROUP, load_plugins
from gaggle.schemas.event import EventRecord
from gaggle.schemas.review import ReviewAction
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class ReviewService:
    """Human-in-the-loop review actions.

    Every action is appended to the append-only review log
    (``review/<event_id>.jsonl``, never rewritten) and its effect on the
    event's review status is folded into a new event revision, so
    ``event.json`` always reflects the latest human decision. Human review
    never overwrites automated outputs (signals, hypotheses, scoring) -- it
    only ever adds a review summary and audit trail alongside them.

    Actions with a real-world side effect beyond bookkeeping ("preserve",
    "export") are recorded here as an audit trail entry, but the side
    effect itself (invoking ``PreservationService`` / an exporter) is
    triggered explicitly by the caller (CLI or review UI), never implicitly
    by this service, so it is always visible which code path performed the
    irreversible action.
    """

    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        self.review_extension_plugins = load_plugins(REVIEW_EXTENSION_PLUGIN_GROUP)

    def append_action(
        self,
        event_id: UUID,
        action: Literal["accept", "reject", "annotate", "retag", "preserve", "export"],
        actor: str,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> tuple[ReviewAction, EventRecord]:
        review_action = ReviewAction(
            action_id=new_uuid(),
            event_id=event_id,
            action=action,
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
            tags=tags or [],
        )
        updated_event = self.repository.append_review_action(review_action)
        self._notify_review_extensions(review_action, updated_event)
        return review_action, updated_event

    def _notify_review_extensions(self, action: ReviewAction, event: EventRecord) -> None:
        """Call every loaded review-extension plugin, strictly after the
        action is already durably persisted. Plugin isolation (invariant
        8's explicitly-permitted broad except): one broken third-party
        extension must never undo, block, or appear to invalidate a review
        action a human already took -- it can only fail to observe it."""

        for plugin in self.review_extension_plugins:
            plugin_name = getattr(plugin, "name", repr(plugin))
            try:
                plugin.on_review_action(action, event)
            except Exception as error:
                LOGGER.error(
                    "review_extension_plugin_failed",
                    plugin=plugin_name,
                    action_id=str(action.action_id),
                    reason=str(error),
                )
