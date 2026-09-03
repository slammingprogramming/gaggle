from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from gaggle.core.signing import EventSigner, SigningUnavailableError
from gaggle.schemas.event import EventRecord
from gaggle.schemas.media import IngestManifest, MediaClip
from gaggle.schemas.review import ReviewAction
from gaggle.storage.database import TimelineDatabase, TimelineQuery
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.utils.time import utc_now

REVIEW_DECISION_BY_ACTION: dict[str, str] = {
    "accept": "accepted",
    "reject": "rejected",
}


class Repository:
    """The single seam between the pydantic schema layer and durable storage.

    All reads and writes of events, media, and review actions go through
    here so that the filesystem (source of truth) and SQLite (index) stay
    consistent with each other. No other module should touch
    ``WorkspacePaths`` or ``TimelineDatabase`` directly.
    """

    def __init__(
        self,
        workspace_root: Path,
        signer: EventSigner | None = None,
        signing_enabled: bool = False,
    ) -> None:
        self.workspace = WorkspacePaths(workspace_root)
        self.database = TimelineDatabase(self.workspace.database)
        self.signer = signer
        # A signer implies signing is enabled; kept as a separate flag too
        # so "enabled but no key yet" can raise a clear error at write time
        # (see `_maybe_sign`) instead of silently skipping signing.
        self.signing_enabled = signing_enabled or signer is not None

    def initialize(self) -> None:
        self.workspace.ensure_layout()
        self.database.initialize()

    def close(self) -> None:
        """Release the SQLite connection pool. Only needed before deleting
        the underlying database file (see `workspace reindex --rebuild`)."""

        self.database.close()

    # -- ingest / media -----------------------------------------------------

    def index_ingest_manifest(self, manifest: IngestManifest) -> None:
        for clip in manifest.copied_files:
            self.index_media_clip(clip)

    def index_media_clip(self, clip: MediaClip) -> None:
        self.database.upsert_media(clip)

    # -- events ---------------------------------------------------------

    def _maybe_sign(self, event: EventRecord) -> EventRecord:
        """Attach `revision_signature` if signing is enabled for this
        workspace. Raises `SigningUnavailableError` if enabled but no key
        has been generated yet -- see `workspace signing-init`."""

        if not self.signing_enabled:
            return event
        if self.signer is None:
            raise SigningUnavailableError(
                "signing.enabled is true but no signing key exists for this "
                "workspace; run 'gaggle workspace signing-init "
                "--workspace <path>' first"
            )
        payload = event.model_dump(mode="json", exclude={"revision_signature"})
        signature = self.signer.sign_payload(payload)
        return event.model_copy(update={"revision_signature": signature})

    def save_event(self, event: EventRecord) -> Path:
        """Persist a brand-new event as revision 0."""

        if event.revision != 0:
            raise ValueError("save_event is only for the initial revision (revision=0)")
        event = self._maybe_sign(event)
        path = self.workspace.write_event_revision(event)
        self.database.upsert_event(event, path)
        return path

    def save_event_revision(
        self,
        event_id: UUID,
        reason: str,
        update: dict[str, Any],
    ) -> EventRecord:
        """Apply ``update`` to the latest revision of an event as a new, append-only revision.

        This is the *only* way an already-written event may change. It never
        edits an existing revision file; it always creates
        ``revisions/000{N+1}_<reason>.json``, chains it to the previous
        revision's hash, and refreshes the ``event.json`` pointer.
        """

        current = self.load_event(event_id)
        previous_hash = self.workspace.latest_revision_hash(event_id)
        next_revision = current.revision + 1
        updated = current.model_copy(
            update={
                **update,
                "revision": next_revision,
                "revision_reason": reason,
                "revised_at": utc_now(),
                "previous_revision_hash": previous_hash,
                "revision_signature": None,
            }
        )
        updated = self._maybe_sign(updated)
        path = self.workspace.write_event_revision(updated)
        self.database.upsert_event(updated, path)
        return updated

    def load_event(self, event_id: UUID) -> EventRecord:
        event_path = self.workspace.event_dir(event_id) / "event.json"
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        return EventRecord.model_validate(payload)

    def load_event_revision(self, event_id: UUID, revision_path: Path) -> EventRecord:
        payload = json.loads(revision_path.read_text(encoding="utf-8"))
        return EventRecord.model_validate(payload)

    def list_event_revisions(self, event_id: UUID) -> list[EventRecord]:
        return [
            self.load_event_revision(event_id, path)
            for path in self.workspace.list_event_revisions(event_id)
        ]

    def list_events(self) -> list[EventRecord]:
        return [
            self.load_event(UUID(path.parent.name)) for path in self.workspace.list_event_files()
        ]

    def query_events(self, query: TimelineQuery) -> list[EventRecord]:
        rows = self.database.query_events(query)
        return [self.load_event(UUID(row.event_id)) for row in rows]

    def reindex(self) -> int:
        """Rebuild the SQLite index from the filesystem (the source of truth).

        Because SQLite is only a query accelerator, this is always safe to
        run: it does not touch any event.json or revision file, only the
        derived index rows.
        """

        count = 0
        for event in self.list_events():
            path = self.workspace.event_dir(event.event_id) / "event.json"
            self.database.upsert_event(event, path)
            count += 1
        return count

    # -- review -----------------------------------------------------------

    def append_review_action(self, action: ReviewAction) -> EventRecord:
        """Append ``action`` to the append-only review log and fold its effect
        into a new event revision.

        The review log itself (``review/<event_id>.jsonl``) is never
        rewritten -- only appended to. The event's ``review_summary`` is
        recomputed and persisted as a new revision so ``event.json`` never
        goes stale relative to the true review history.
        """

        self.workspace.append_review_action(action)
        self.database.append_review_action(action)
        current = self.load_event(action.event_id)
        fallback_decision = current.review_summary.latest_decision
        decision = REVIEW_DECISION_BY_ACTION.get(action.action, fallback_decision)
        updated_summary = current.review_summary.model_copy(
            update={
                "latest_decision": decision,
                "action_count": current.review_summary.action_count + 1,
                "last_reviewed_at": action.timestamp,
                "last_action_id": action.action_id,
            }
        )
        return self.save_event_revision(
            action.event_id,
            reason=f"review_{action.action}",
            update={"review_summary": updated_summary},
        )

    def list_review_actions(self, event_id: UUID) -> list[ReviewAction]:
        log_path = self.workspace.review_log_path(event_id)
        if not log_path.exists():
            return []
        return [
            ReviewAction.model_validate_json(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
