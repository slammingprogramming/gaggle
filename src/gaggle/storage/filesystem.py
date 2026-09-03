from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from gaggle.schemas.event import EventRecord
from gaggle.schemas.lifecycle import DeletionRecord, EventVideoPurgeRecord
from gaggle.schemas.recognition import IdentityMergeRecord
from gaggle.schemas.recognition_review import RecognitionCropPurgeRecord, RecognitionReviewRecord
from gaggle.schemas.review import ReviewAction
from gaggle.utils.filesystem import (
    append_line,
    copy_file_preserve_metadata,
    delete_even_if_read_only,
    list_files_sorted,
    set_read_only,
)
from gaggle.utils.hashing import hash_file
from gaggle.utils.json import canonical_json_bytes, write_canonical_json

_REASON_SANITIZER = re.compile(r"[^a-z0-9_]+")


@dataclass(frozen=True)
class WorkspacePaths:
    """The forensic storage layout.

    See ``docs/architecture.md`` for the full rationale. In short: the
    filesystem is the source of truth for evidence; SQLite (see
    ``gaggle.storage.database``) is only an index/query
    accelerator layered on top of it.

    Within each event directory, ``event.json`` is a convenience pointer to
    the *latest* revision and is the only file in the tree that is ever
    rewritten in place. The true append-only history lives in
    ``revisions/000N_<reason>.json``, and every file in that directory is
    made read-only at write time and is never modified or deleted again.
    """

    root: Path

    @property
    def ingest(self) -> Path:
        return self.root / "ingest"

    @property
    def originals(self) -> Path:
        return self.root / "originals"

    @property
    def normalized(self) -> Path:
        return self.root / "normalized"

    @property
    def windows(self) -> Path:
        return self.root / "windows"

    @property
    def events(self) -> Path:
        return self.root / "events"

    @property
    def preserved(self) -> Path:
        return self.root / "preserved"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def patterns(self) -> Path:
        return self.root / "patterns"

    @property
    def database(self) -> Path:
        return self.root / "timeline" / "index.sqlite3"

    @property
    def timeline(self) -> Path:
        return self.root / "timeline"

    @property
    def recognition(self) -> Path:
        return self.root / "recognition"

    @property
    def face_crops(self) -> Path:
        return self.recognition / "faces" / "crops"

    @property
    def face_model_path(self) -> Path:
        return self.recognition / "faces" / "model.yml"

    @property
    def face_embedding_model_path(self) -> Path:
        """Separate from `face_model_path` -- an embedding-space centroid
        model (`IncrementalFaceEmbeddingClusterer`, JSON) is not
        compatible with an LBPH model (`IncrementalFaceClusterer`,
        OpenCV's binary `.yml` format), and a workspace may switch
        `enrichment.face.embedding_model` between runs, so both are kept
        side by side rather than one overwriting the other."""

        return self.recognition / "faces" / "embedding_model.json"

    @property
    def plate_crops(self) -> Path:
        return self.recognition / "plates" / "crops"

    @property
    def voice_model_path(self) -> Path:
        return self.recognition / "voices" / "model.json"

    @property
    def vehicle_appearance_crops(self) -> Path:
        return self.recognition / "vehicle_appearance" / "crops"

    @property
    def vehicle_appearance_model_path(self) -> Path:
        return self.recognition / "vehicle_appearance" / "model.json"

    @property
    def person_appearance_crops(self) -> Path:
        return self.recognition / "person_appearance" / "crops"

    @property
    def person_appearance_model_path(self) -> Path:
        return self.recognition / "person_appearance" / "model.json"

    @property
    def transcripts(self) -> Path:
        return self.root / "transcripts"

    @property
    def signing(self) -> Path:
        return self.root / "signing"

    @property
    def signing_private_key_path(self) -> Path:
        """Workspace-internal Ed25519 private key (see `core/signing.py`).

        Deliberately never created by `ensure_layout()` -- key generation
        only ever happens via the explicit `workspace signing-init`
        command, never as an implicit side effect. Also deliberately
        outside `events/`, so `export/service.py::export_event_bundle`'s
        file walk (which only touches `events/<id>/`) can never
        accidentally include it in an exported bundle.
        """

        return self.signing / "private_key.pem"

    @property
    def for_review(self) -> Path:
        """Human-browsable symlinks to originals still awaiting review.

        Never authoritative -- purely a convenience view. Symlinks here can
        be freely recreated, removed, or (if the underlying filesystem
        doesn't support symlinks) skipped without affecting any event's
        evidence references, which always point directly at ``originals/``.
        """

        return self.root / "for_review"

    @property
    def pending_deletion(self) -> Path:
        """Originals classified as benign (zero signals), physically moved
        here awaiting an explicit, human-confirmed deletion. Safe to move
        (unlike reviewable originals) because nothing in any event.json
        references a clip with zero signals -- see
        ``gaggle.core.triage``.
        """

        return self.root / "pending_deletion"

    @property
    def deletion_log_path(self) -> Path:
        return self.root / "deletion_log.jsonl"

    @property
    def identity_merge_log_path(self) -> Path:
        return self.root / "identity_merge_log.jsonl"

    @property
    def event_video_purge_log_path(self) -> Path:
        return self.root / "event_video_purge_log.jsonl"

    @property
    def recognition_review_log_path(self) -> Path:
        return self.root / "recognition_review_log.jsonl"

    @property
    def recognition_crop_purge_log_path(self) -> Path:
        return self.root / "recognition_crop_purge_log.jsonl"

    def ensure_layout(self) -> None:
        for path in (
            self.root,
            self.ingest,
            self.originals,
            self.normalized,
            self.windows,
            self.events,
            self.preserved,
            self.review,
            self.exports,
            self.patterns,
            self.timeline,
            self.face_crops,
            self.plate_crops,
            self.vehicle_appearance_crops,
            self.person_appearance_crops,
            self.transcripts,
            self.for_review,
            self.pending_deletion,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def event_dir(self, event_id: UUID) -> Path:
        return self.events / str(event_id)

    def event_revisions_dir(self, event_id: UUID) -> Path:
        return self.event_dir(event_id) / "revisions"

    def event_clips_dir(self, event_id: UUID) -> Path:
        return self.event_dir(event_id) / "clips"

    def preserved_event_dir(self, event_id: UUID) -> Path:
        return self.preserved / str(event_id)

    def review_log_path(self, event_id: UUID) -> Path:
        return self.review / f"{event_id}.jsonl"

    def write_json(self, path: Path, payload: dict[str, Any], read_only: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(path, payload)
        if read_only:
            set_read_only(path)

    def append_review_action(self, action: ReviewAction) -> None:
        append_line(self.review_log_path(action.event_id), action.model_dump_json())

    def append_deletion_record(self, record: DeletionRecord) -> None:
        append_line(self.deletion_log_path, record.model_dump_json())

    def append_identity_merge_record(self, record: IdentityMergeRecord) -> None:
        append_line(self.identity_merge_log_path, record.model_dump_json())

    def append_event_video_purge_record(self, record: EventVideoPurgeRecord) -> None:
        append_line(self.event_video_purge_log_path, record.model_dump_json())

    def append_recognition_review_record(self, record: RecognitionReviewRecord) -> None:
        append_line(self.recognition_review_log_path, record.model_dump_json())

    def append_recognition_crop_purge_record(self, record: RecognitionCropPurgeRecord) -> None:
        append_line(self.recognition_crop_purge_log_path, record.model_dump_json())

    def create_for_review_symlink(self, clip_id: UUID, original_path: Path) -> Path | None:
        """Best-effort convenience symlink into ``for_review/``. Never authoritative;
        returns None (and logs nothing -- callers decide whether to log) if
        the filesystem doesn't support symlinks rather than raising."""

        link_path = self.for_review / f"{clip_id}_{original_path.name}"
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(original_path.resolve())
        except OSError:
            return None
        return link_path

    def remove_for_review_symlink(self, clip_id: UUID, original_name: str) -> None:
        link_path = self.for_review / f"{clip_id}_{original_name}"
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink(missing_ok=True)

    def copy_original(
        self, source: Path, relative_destination: Path, read_only: bool = True
    ) -> Path:
        """Copy ``source`` into ``originals/`` at ``relative_destination``.

        ``relative_destination`` is always content-addressed (a hash
        prefix + the original filename -- see ``IngestService``), so a
        destination that already exists here means this exact content was
        already copied in, not a name collision between different bytes.
        Idempotent in that case: skip the copy (the existing file may
        already be read-only, so re-copying over it would fail on some
        platforms anyway) and return the existing path unchanged, rather
        than crashing on what is actually a harmless re-ingest.
        """

        destination = self.originals / relative_destination
        if not destination.exists():
            copy_file_preserve_metadata(source, destination)
            if read_only:
                set_read_only(destination)
        return destination

    def move_original(
        self, source: Path, relative_destination: Path, read_only: bool = True
    ) -> Path:
        """Relocate ``source`` into ``originals/`` instead of copying it.

        Uses ``shutil.move``, which transparently falls back to a copy+
        delete when the source and destination are on different volumes
        (e.g. an SD card and the workspace's drive) -- this is slower than
        a same-volume rename but still correct; there is no risk of ending
        up with neither a source nor a destination file from an
        interrupted cross-volume move, since the source is only removed
        after the copy completes successfully.

        ``relative_destination`` is content-addressed (see
        ``copy_original``'s docstring), so an already-existing destination
        means this exact content was already relocated in previously --
        idempotent in that case: just remove the now-redundant source
        (consistent with "move" promising the source goes away) rather
        than attempting a same-name move, which raises on some platforms
        (``os.rename`` fails with an existing destination on Windows,
        unlike POSIX's silent overwrite) instead of failing gracefully.
        """

        destination = self.originals / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            delete_even_if_read_only(source)
        else:
            shutil.move(str(source), str(destination))
            if read_only:
                set_read_only(destination)
        return destination

    def latest_revision_number(self, event_id: UUID) -> int | None:
        revisions_dir = self.event_revisions_dir(event_id)
        if not revisions_dir.exists():
            return None
        numbers = [
            int(path.name.split("_", 1)[0])
            for path in revisions_dir.glob("[0-9][0-9][0-9][0-9]_*.json")
        ]
        return max(numbers) if numbers else None

    def latest_revision_hash(self, event_id: UUID) -> str | None:
        latest = self.latest_revision_path(event_id)
        if latest is None:
            return None
        return hash_file(latest)

    def latest_revision_path(self, event_id: UUID) -> Path | None:
        revisions_dir = self.event_revisions_dir(event_id)
        if not revisions_dir.exists():
            return None
        candidates = sorted(revisions_dir.glob("[0-9][0-9][0-9][0-9]_*.json"))
        return candidates[-1] if candidates else None

    def write_event_revision(self, event: EventRecord) -> Path:
        """Write a new frozen revision and refresh the ``event.json`` pointer.

        ``event.revision`` must already be set correctly by the caller
        (typically ``gaggle.storage.repository.Repository``,
        which owns revision-numbering and hash-chaining logic). This method
        is only responsible for the filesystem side effects: an immutable,
        read-only revision file that is never touched again, and the
        always-current ``event.json`` pointer that mirrors it.
        """

        event_dir = self.event_dir(event.event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        reason_slug = _REASON_SANITIZER.sub("_", event.revision_reason.lower()).strip("_")
        revision_path = (
            self.event_revisions_dir(event.event_id)
            / f"{event.revision:04d}_{reason_slug or 'revision'}.json"
        )
        if revision_path.exists():
            raise FileExistsError(
                f"revision {event.revision} already exists for event {event.event_id}"
            )
        payload = event.model_dump(mode="json")
        self.write_json(revision_path, payload, read_only=True)
        pointer_path = event_dir / "event.json"
        write_canonical_json(pointer_path, payload)  # pointer is intentionally not frozen
        return pointer_path

    def list_event_revisions(self, event_id: UUID) -> list[Path]:
        revisions_dir = self.event_revisions_dir(event_id)
        if not revisions_dir.exists():
            return []
        return sorted(revisions_dir.glob("[0-9][0-9][0-9][0-9]_*.json"))

    def freeze_directory(self, root: Path) -> None:
        for item in list_files_sorted(root):
            set_read_only(item)

    def write_preservation_bundle_manifest(
        self, bundle_root: Path, payload: dict[str, Any]
    ) -> Path:
        manifest_path = bundle_root / "bundle_manifest.json"
        self.write_json(manifest_path, payload, read_only=True)
        return manifest_path

    def list_event_files(self) -> list[Path]:
        return sorted(self.events.glob("*/event.json"))


def hash_canonical_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
