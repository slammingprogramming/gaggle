from __future__ import annotations

import shutil
from uuid import UUID

from gaggle.schemas.common import ChainOfCustodyEntry, HashDigest
from gaggle.schemas.event import EventRecord, PreservationStatus
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.storage.repository import Repository
from gaggle.utils.filesystem import list_files_sorted
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class PreservationService:
    """Copies an event's evidence into an immutable, self-contained bundle.

    Preservation is deliberately cheap and biased towards over-preserving:
    per the project's false-positive philosophy, retaining a bundle for an
    event that turns out not to matter is far less costly than failing to
    preserve one that did. The bundle directory is a full copy of the
    event's history (every revision, review log, and derived clip) plus a
    hash manifest, frozen read-only immediately after it is written.
    """

    def __init__(self, workspace: WorkspacePaths) -> None:
        self.workspace = workspace

    def preserve(self, event: EventRecord) -> PreservationStatus:
        source_dir = self.workspace.event_dir(event.event_id)
        destination_dir = self.workspace.preserved_event_dir(event.event_id)
        if destination_dir.exists():
            raise FileExistsError(f"preservation bundle already exists for {event.event_id}")
        shutil.copytree(source_dir, destination_dir)

        review_log = self.workspace.review_log_path(event.event_id)
        if review_log.exists():
            shutil.copy2(review_log, destination_dir / review_log.name)

        # Include a preservation-confirmation record inside the bundle itself
        # so it is self-describing even if separated from the live workspace.
        confirmation = {
            "event_id": str(event.event_id),
            "preserved_at": utc_now().isoformat(),
            "source_revision": event.revision,
        }
        self.workspace.write_json(
            destination_dir / "preservation_confirmation.json", confirmation, read_only=False
        )

        hashes = [
            HashDigest(value=hash_file(path), algorithm="sha256").model_dump(mode="json")
            for path in list_files_sorted(destination_dir)
        ]
        manifest_path = self.workspace.write_preservation_bundle_manifest(
            destination_dir,
            {
                "event_id": str(event.event_id),
                "preserved_at": confirmation["preserved_at"],
                "file_count": len(hashes),
                "hashes": hashes,
            },
        )
        self.workspace.freeze_directory(destination_dir)
        LOGGER.info("event_preserved", event_id=str(event.event_id), bundle=str(destination_dir))
        return PreservationStatus(
            state="preserved",
            immutable=True,
            preserved_at=utc_now(),
            bundle_path=str(destination_dir.resolve()),
            bundle_hash=hash_file(manifest_path),
        )


class PreservationOrchestrator:
    """Preserves an event's evidence and folds the result into a new event revision.

    This is the seam between the pure filesystem-copying ``PreservationService``
    and the repository's revisioning logic: it is what keeps ``event.json``
    from going stale after preservation.
    """

    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        self.service = PreservationService(repository.workspace)

    def preserve_event(self, event_id: UUID) -> EventRecord:
        event = self.repository.load_event(event_id)
        if event.preservation_status.state == "preserved":
            raise ValueError(f"event {event_id} is already preserved")
        status = self.service.preserve(event)
        chain_entry = ChainOfCustodyEntry(
            entry_id=new_uuid(),
            action="event_preserved",
            actor="gaggle",
            timestamp=utc_now(),
            details={"bundle_path": status.bundle_path, "bundle_hash": status.bundle_hash},
            input_hashes=[HashDigest(value=h) for h in event.hashes],
            output_hashes=[HashDigest(value=status.bundle_hash)] if status.bundle_hash else [],
        )
        return self.repository.save_event_revision(
            event_id,
            reason="preserved",
            update={
                "preservation_status": status,
                "chain_of_custody": [*event.chain_of_custody, chain_entry],
            },
        )
