"""Storage lifecycle: triage classification and human-confirmed deletion.

See `docs/architecture.md`'s storage-lifecycle section for the full design.
Summary of the two states a clip lands in after `analyze()`:

* **reviewable** -- contributed to at least one `Signal` in at least one
  `Event`. The original is *never moved* (moving it would break the
  `evidence_references` paths already embedded in that event's revision
  history); instead a best-effort, non-authoritative symlink appears under
  `for_review/` purely so it's easy to browse from a file manager or shell,
  and `triage list --state reviewable` gives the authoritative listing.
* **benign_pending_deletion** -- contributed to zero signals across the
  full analysis. Nothing in any event.json can reference such a clip (by
  construction -- events are only ever built from clips that produced
  signals), so it is safe to physically move it out of `originals/` and
  into `pending_deletion/`, in preparation for an explicit,
  actor-attributed, logged deletion.

Deletion itself never happens implicitly. `confirm_deletion` requires an
explicit actor name, writes a `DeletionRecord` to the append-only
`deletion_log.jsonl` *before* unlinking the file, and only then deletes the
bytes -- so there is always a durable record that the file existed, what
its hash was, and who confirmed removing it, even after the bytes are
gone.

A separate, event-scoped operation lives here too: `purge_event_video`.
Once you're done reviewing an event, its video (the event's own derived
clips, and the original clip(s) that contributed to it) is usually the
single biggest thing about it taking up disk space -- but `event.json`,
its signals/hypotheses/scoring, its full revision history, and every
review decision are tiny by comparison and worth keeping forever. Purging
removes the video and leaves everything else untouched. See
`purge_event_video`'s docstring for the safety rules (preservation-gated
by default, cascade-to-originals logic).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID

from gaggle.schemas.event import EventRecord
from gaggle.schemas.lifecycle import (
    DeletionRecord,
    EventVideoPurgeRecord,
    TriageRecord,
    TriageState,
)
from gaggle.storage.database import MediaIndexRow, TimelineQuery
from gaggle.storage.repository import Repository
from gaggle.utils.filesystem import delete_even_if_read_only
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class TriageService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    def classify_all(self) -> list[TriageRecord]:
        """Classify every indexed clip as reviewable or benign-pending-deletion.

        Safe and idempotent to rerun: a clip already moved to
        `pending_deletion/` on a prior run is detected via its current
        indexed location and left alone rather than being "moved" again.
        """

        media_rows = self.repository.database.list_media()
        events = self.repository.list_events()

        signal_counts_by_hash: dict[str, int] = {}
        event_ids_by_hash: dict[str, list[UUID]] = {}
        for event in events:
            for signal in event.signals:
                for ref in signal.evidence_references:
                    if not ref.sha256:
                        continue
                    signal_counts_by_hash[ref.sha256] = signal_counts_by_hash.get(ref.sha256, 0) + 1
                    ids = event_ids_by_hash.setdefault(ref.sha256, [])
                    if event.event_id not in ids:
                        ids.append(event.event_id)

        records: list[TriageRecord] = []
        for row in media_rows:
            record = self._classify_one(row, signal_counts_by_hash, event_ids_by_hash)
            records.append(record)
        return records

    def _classify_one(
        self,
        row: MediaIndexRow,
        signal_counts_by_hash: dict[str, int],
        event_ids_by_hash: dict[str, list[UUID]],
    ) -> TriageRecord:
        clip_id = UUID(row.clip_id)
        signal_count = signal_counts_by_hash.get(row.sha256, 0)
        event_ids = event_ids_by_hash.get(row.sha256, [])
        is_reviewable = signal_count > 0
        state: TriageState = "reviewable" if is_reviewable else "benign_pending_deletion"
        reason = (
            f"{signal_count} signal(s) across {len(event_ids)} event(s)"
            if is_reviewable
            else "zero signals detected across full analysis"
        )
        classified_at = utc_now()

        existing = self.repository.database.get_triage(clip_id)
        already_deleted = existing is not None and existing.state == "deleted"
        if already_deleted:
            state = "deleted"
            reason = "previously confirmed deleted"
        elif not is_reviewable:
            if row.ingest_mode == "reference":
                # Never copy a reference-mode file into the workspace just to
                # mark it for deletion -- that would defeat the entire point
                # of reference mode (avoiding extra disk use). It's left at
                # its external location; confirm_deletion operates on it
                # there directly, gated behind acknowledge_external_deletion.
                LOGGER.info(
                    "reference_clip_classified_benign",
                    clip_id=str(clip_id),
                    stored_path=row.stored_path,
                )
            else:
                self._move_to_pending_deletion(clip_id, row)
        else:
            self.repository.workspace.create_for_review_symlink(clip_id, Path(row.stored_path))

        self.repository.database.upsert_triage(
            clip_id=clip_id,
            camera_id=row.camera_id,
            state=state,
            signal_count=signal_count,
            event_ids=event_ids,
            classified_at=classified_at,
            reason=reason,
        )
        return TriageRecord(
            clip_id=clip_id,
            camera_id=row.camera_id,
            state=state,
            signal_count=signal_count,
            event_ids=event_ids,
            classified_at=classified_at,
            reason=reason,
        )

    def _move_to_pending_deletion(self, clip_id: UUID, row: MediaIndexRow) -> None:
        source = Path(row.stored_path)
        pending_dir = self.repository.workspace.pending_deletion
        if source.parent.resolve() == pending_dir.resolve():
            return  # already moved on a prior triage run
        if not source.exists():
            LOGGER.warning("triage_move_source_missing", clip_id=str(clip_id), source=str(source))
            return
        destination = pending_dir / source.name
        pending_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        self.repository.database.update_media_location(clip_id, str(destination))
        self.repository.workspace.remove_for_review_symlink(clip_id, source.name)
        LOGGER.info(
            "clip_moved_to_pending_deletion", clip_id=str(clip_id), destination=str(destination)
        )

    def list_state(self, state: TriageState | None = None) -> list[TriageRecord]:
        rows = self.repository.database.list_triage(state)
        return [
            TriageRecord(
                clip_id=UUID(row.clip_id),
                camera_id=row.camera_id,
                state=row.state,  # type: ignore[arg-type]
                signal_count=row.signal_count,
                event_ids=[UUID(e) for e in row.event_ids_csv.split(",") if e],
                classified_at=row.classified_at,
                reason=row.reason,
            )
            for row in rows
        ]

    def confirm_deletion(
        self,
        clip_id: UUID,
        actor: str,
        notes: str = "",
        acknowledge_external_deletion: bool = False,
    ) -> DeletionRecord:
        """Permanently delete one clip's original bytes. Requires an explicit actor.

        Writes the `DeletionRecord` to the append-only deletion log *before*
        unlinking the file -- if the process is interrupted between those
        two steps, the log still shows deletion was confirmed even though
        the unlink didn't complete, which is the safer failure mode for a
        forensic system (an over-cautious log entry, never a silent
        deletion with no record).

        If this clip was ingested in "reference" mode (see
        `core/config.py::StorageConfig.ingest_mode`), `stored_path` points
        outside the workspace -- typically back at the SD card or drive the
        footage was ingested from. Deleting it deletes the user's actual
        source file, not a workspace-owned copy, so this requires
        `acknowledge_external_deletion=True` in addition to everything
        `confirm_deletion` already requires; without it, this raises rather
        than silently deleting something the workspace never made a copy
        of.
        """

        triage = self.repository.database.get_triage(clip_id)
        if triage is None or triage.state != "benign_pending_deletion":
            raise ValueError(
                f"clip {clip_id} is not in benign_pending_deletion state "
                "(only clips classified benign by triage can be deleted through this workflow)"
            )
        media_row = self.repository.database.get_media(clip_id)
        if media_row is None:
            raise ValueError(f"no indexed media row for clip {clip_id}")
        if media_row.ingest_mode == "reference" and not acknowledge_external_deletion:
            raise ValueError(
                f"clip {clip_id} was ingested in 'reference' mode -- its stored_path "
                f"({media_row.stored_path}) points outside the workspace, at the original "
                "source location. Deleting it deletes that external file, not a workspace-owned "
                "copy. Pass acknowledge_external_deletion=True (CLI: --acknowledge-external) "
                "to proceed."
            )

        source = Path(media_row.stored_path)
        actual_hash = hash_file(source) if source.exists() else media_row.sha256
        if source.exists() and actual_hash != media_row.sha256:
            raise RuntimeError(
                f"refusing to delete {source}: current hash {actual_hash} does not match "
                f"indexed hash {media_row.sha256} -- the file may have been modified"
            )

        record = DeletionRecord(
            deletion_id=new_uuid(),
            clip_id=clip_id,
            camera_id=media_row.camera_id,
            original_stored_path=str(source),
            sha256=media_row.sha256,
            byte_size=source.stat().st_size if source.exists() else 0,
            triage_reason=triage.reason,
            confirmed_by=actor,
            confirmed_at=utc_now(),
            notes=notes,
            metadata={"ingest_mode": media_row.ingest_mode},
        )
        self.repository.workspace.append_deletion_record(record)

        if source.exists():
            delete_even_if_read_only(source)
        self.repository.database.upsert_triage(
            clip_id=clip_id,
            camera_id=media_row.camera_id,
            state="deleted",
            signal_count=triage.signal_count,
            event_ids=[UUID(e) for e in triage.event_ids_csv.split(",") if e],
            classified_at=utc_now(),
            reason=f"deleted by {actor}",
        )
        LOGGER.info("clip_deleted", clip_id=str(clip_id), actor=actor)
        return record

    def confirm_deletion_all(
        self, actor: str, notes: str = "", acknowledge_external_deletion: bool = False
    ) -> list[DeletionRecord]:
        """Delete every clip currently classified benign-pending-deletion.

        A clip ingested in "reference" mode is skipped (with a logged
        warning, not silently) unless `acknowledge_external_deletion=True`,
        since deleting it deletes something outside the workspace -- see
        `confirm_deletion`. One clip failing for any reason never aborts
        the rest of the batch.
        """

        pending = self.list_state("benign_pending_deletion")
        records: list[DeletionRecord] = []
        for record in pending:
            try:
                records.append(
                    self.confirm_deletion(
                        record.clip_id,
                        actor,
                        notes,
                        acknowledge_external_deletion=acknowledge_external_deletion,
                    )
                )
            except (ValueError, RuntimeError) as error:
                LOGGER.warning(
                    "bulk_deletion_skipped", clip_id=str(record.clip_id), reason=str(error)
                )
        return records

    # -- ingest mode conversion -------------------------------------------

    def convert_ingest_mode(
        self, clip_id: UUID, to_mode: Literal["copy", "move"], actor: str, notes: str = ""
    ) -> MediaIndexRow:
        """Convert a `reference`-mode clip into a durable, workspace-owned
        `copy` or `move`-mode one.

        Only `reference -> copy`/`reference -> move` are supported.
        Converting `copy`/`move -> reference` is refused outright, with no
        override flag -- that direction would mean deleting the
        workspace's one and only owned copy of a file that may already be
        the sole surviving copy (the original source could be long gone,
        e.g. if it was ingested via `move`, or simply deleted since), with
        no way to verify a *new* external dependency actually has matching
        bytes the way `reference` mode's original design guaranteed at
        ingest time. See `docs/local-ai.md`'s "Choosing an ingest storage
        mode" section.

        Re-hashes the file at its current (external) `stored_path` and
        compares against the indexed hash before doing anything, mirroring
        `confirm_deletion`'s hash-verification-before-mutation pattern --
        refuses if the file is missing or has changed.

        Important, non-obvious caveat: any *already-existing* event whose
        `Signal.evidence_references` point at the old external path keep
        pointing there after conversion -- that's intentional (append-only
        provenance, never retroactively rewritten -- invariants 2/4), not a
        bug to fix. Conversion only changes where *future* reads (further
        enrichment, preservation) find the clip; it doesn't repair
        historical evidence references. This is also a real, pre-existing
        tension with invariant 5 ("SQLite is an index, re-derivable from
        the filesystem"), not one this feature introduces: neither
        `ingest_mode` nor `stored_path` conversions are reflected back into
        the original ingest manifest JSON, which stays historically
        accurate to what happened *at ingest time* (a point-in-time
        record, not a live index) -- so a full `reindex()` from manifests
        alone would not reproduce a post-conversion state. Documented here
        rather than left ambiguous.
        """

        media_row = self.repository.database.get_media(clip_id)
        if media_row is None:
            raise ValueError(f"no indexed media row for clip {clip_id}")
        if media_row.ingest_mode != "reference":
            raise ValueError(
                f"clip {clip_id} is in '{media_row.ingest_mode}' mode -- only converting from "
                "'reference' to 'copy' or 'move' is supported. Converting a 'copy'/'move'-mode "
                "clip to 'reference' is refused outright: it would mean deleting the "
                "workspace's only owned copy of a file that may already be the sole "
                "surviving copy, with no way to verify a new external dependency actually "
                "has matching bytes."
            )

        source = Path(media_row.stored_path)
        if not source.exists():
            raise RuntimeError(
                f"refusing to convert clip {clip_id}: its external source {source} no longer exists"
            )
        actual_hash = hash_file(source)
        if actual_hash != media_row.sha256:
            raise RuntimeError(
                f"refusing to convert clip {clip_id}: current hash {actual_hash} of {source} "
                f"does not match indexed hash {media_row.sha256} -- the file may have changed"
            )

        relative_destination = Path(media_row.camera_id) / f"{media_row.sha256[:12]}_{source.name}"
        if to_mode == "move":
            new_path = self.repository.workspace.move_original(source, relative_destination)
        else:
            new_path = self.repository.workspace.copy_original(source, relative_destination)

        self.repository.database.update_media_ingest_mode(clip_id, str(new_path), to_mode)
        LOGGER.info(
            "ingest_mode_converted",
            clip_id=str(clip_id),
            from_mode="reference",
            to_mode=to_mode,
            actor=actor,
            new_path=str(new_path),
            notes=notes,
        )
        updated = self.repository.database.get_media(clip_id)
        assert updated is not None  # just written above
        return updated

    # -- event video purge ----------------------------------------------------

    def purge_event_video(
        self,
        event_id: UUID,
        actor: str,
        notes: str = "",
        force: bool = False,
    ) -> EventVideoPurgeRecord:
        """Delete an event's video evidence while keeping its metadata forever.

        Removes the event's own derived clips (`events/<id>/clips/`), then
        cascades to each contributing original clip -- but only for
        originals that no other *unpurged* event still references. An
        original still needed elsewhere is left alone and listed in
        `retained_original_clip_ids`, not silently skipped.

        Refuses to run unless the event has already been preserved (a
        frozen copy of its derived clips already exists under
        `preserved/<id>/`), since otherwise this would be the only copy of
        that video ever destroyed. Pass `force=True` to purge anyway and
        explicitly accept losing that evidence -- there is no confirmation
        prompt at this layer; the caller (CLI) is expected to have already
        gotten one from the human.

        `event.json` itself -- signals, hypotheses, scoring, chain of
        custody, review history, revision history -- is never touched
        except to record `video_purged_at` as a new revision.
        """

        event = self.repository.load_event(event_id)
        if event.video_purged_at is not None:
            raise ValueError(f"event {event_id}'s video has already been purged")
        was_preserved = event.preservation_status.state == "preserved"
        if not was_preserved and not force:
            raise ValueError(
                f"event {event_id} has not been preserved yet -- its derived clips are "
                "currently the only copy of this video evidence. Preserve it first "
                "(`preserve <event-id>`), or pass force=True (CLI: --force) to purge anyway "
                "and accept losing that evidence entirely."
            )

        deleted_paths, deleted_hashes = self._purge_derived_clips(event_id)
        cascaded_clip_ids, retained_clip_ids, retained_reason = self._cascade_original_deletion(
            event, actor
        )

        record = EventVideoPurgeRecord(
            purge_id=new_uuid(),
            event_id=event_id,
            deleted_derived_clip_paths=deleted_paths,
            deleted_derived_clip_hashes=deleted_hashes,
            cascaded_original_clip_ids=cascaded_clip_ids,
            retained_original_clip_ids=retained_clip_ids,
            retained_reason=retained_reason,
            required_preservation=not force,
            was_preserved_at_time_of_purge=was_preserved,
            confirmed_by=actor,
            confirmed_at=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_event_video_purge_record(record)
        self.repository.save_event_revision(
            event_id, reason="video_purged", update={"video_purged_at": record.confirmed_at}
        )
        LOGGER.info(
            "event_video_purged",
            event_id=str(event_id),
            deleted_clip_count=len(deleted_paths),
            cascaded_original_count=len(cascaded_clip_ids),
            retained_original_count=len(retained_clip_ids),
        )
        return record

    def purge_event_video_bulk(
        self,
        query: TimelineQuery,
        actor: str,
        notes: str = "",
        force: bool = False,
    ) -> list[EventVideoPurgeRecord]:
        """Purge video for every event matching `query` that isn't already purged.

        Events that fail to purge (not yet preserved and `force=False`, or
        already purged) are skipped with a logged warning rather than
        aborting the whole batch -- see each skip's log entry for why.
        """

        records: list[EventVideoPurgeRecord] = []
        for event in self.repository.query_events(query):
            if event.video_purged_at is not None:
                continue
            try:
                records.append(self.purge_event_video(event.event_id, actor, notes, force))
            except ValueError as error:
                LOGGER.warning(
                    "bulk_purge_skipped", event_id=str(event.event_id), reason=str(error)
                )
        return records

    def _purge_derived_clips(self, event_id: UUID) -> tuple[list[str], list[str]]:
        clips_dir = self.repository.workspace.event_clips_dir(event_id)
        deleted_paths: list[str] = []
        deleted_hashes: list[str] = []
        if not clips_dir.exists():
            return deleted_paths, deleted_hashes
        for path in sorted(clips_dir.rglob("*")):
            if path.is_file():
                deleted_paths.append(str(path))
                deleted_hashes.append(hash_file(path))
        shutil.rmtree(clips_dir)
        return deleted_paths, deleted_hashes

    def _cascade_original_deletion(
        self, event: EventRecord, actor: str
    ) -> tuple[list[UUID], list[UUID], str]:
        referenced_hashes = {
            ref.sha256
            for signal in event.signals
            for ref in signal.evidence_references
            if ref.sha256
        }
        if not referenced_hashes:
            return [], [], ""

        all_media = self.repository.database.list_media()
        hash_to_clip_id = {
            row.sha256: UUID(row.clip_id) for row in all_media if row.sha256 in referenced_hashes
        }

        other_events = [
            other
            for other in self.repository.list_events()
            if other.event_id != event.event_id and other.video_purged_at is None
        ]
        still_needed_hashes = {
            ref.sha256
            for other in other_events
            for signal in other.signals
            for ref in signal.evidence_references
            if ref.sha256
        }

        cascaded: list[UUID] = []
        retained: list[UUID] = []
        for sha256_value, clip_id in hash_to_clip_id.items():
            if sha256_value in still_needed_hashes:
                retained.append(clip_id)
                continue
            media_row = self.repository.database.get_media(clip_id)
            if media_row is None:
                continue
            self.repository.database.upsert_triage(
                clip_id=clip_id,
                camera_id=media_row.camera_id,
                state="benign_pending_deletion",
                signal_count=0,
                event_ids=[],
                classified_at=utc_now(),
                reason=f"no longer needed after purging event {event.event_id}'s video",
            )
            try:
                self.confirm_deletion(
                    clip_id,
                    actor=actor,
                    notes=f"cascaded from purging event {event.event_id}'s video",
                    acknowledge_external_deletion=(media_row.ingest_mode == "reference"),
                )
                cascaded.append(clip_id)
            except (ValueError, RuntimeError) as error:
                LOGGER.warning("purge_cascade_failed", clip_id=str(clip_id), reason=str(error))
                retained.append(clip_id)

        retained_reason = "still referenced by at least one un-purged event" if retained else ""
        return cascaded, retained, retained_reason
