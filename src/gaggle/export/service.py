"""Structured forensic metadata and evidence bundle export.

Two export shapes are supported:

* ``export_event_bundle`` -- a single self-contained, hash-verified zip
  archive for one event: its full revision history, review log, derived
  clips, and (if preserved) the frozen preservation bundle. This is the
  "hand this to someone outside the system" artifact.
* ``export_timeline`` -- a flat CSV or JSON export of the indexed timeline
  (optionally filtered) for reporting/spreadsheet use. This is metadata
  only; it never includes media.

Both are additive, read-only operations: nothing about the source event or
workspace is modified by exporting it (beyond an audit trail entry noting
that an export occurred). Third-party export formats can be added via
``ExporterPlugin`` (see ``gaggle.plugins.base``) without touching
this module.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from gaggle.core.signing import EventSigner, SigningUnavailableError
from gaggle.plugins.registry import EXPORTER_PLUGIN_GROUP, load_plugins
from gaggle.schemas.common import ChainOfCustodyEntry, HashDigest
from gaggle.storage.database import TimelineQuery
from gaggle.storage.filesystem import hash_canonical_payload
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class ExportError(RuntimeError):
    """Raised when an export cannot be completed -- including a matched
    exporter plugin failing or misbehaving (see invariant 8's
    plugin-isolation carve-out: one broken third-party exporter must
    produce a clear error here, not a raw traceback from its own code)."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    path: Path
    manifest_hash: str
    file_count: int


class ExportService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository
        self.exporter_plugins = load_plugins(EXPORTER_PLUGIN_GROUP)

    def export_event_bundle(self, event_id: UUID, export_format: str | None = None) -> ExportResult:
        """Export one event as a self-contained bundle.

        ``export_format=None`` (default) produces the built-in
        hash-verified zip bundle. Any other value is looked up against
        loaded ``ExporterPlugin``s by ``format_id`` -- see
        ``docs/plugin-authoring.md``'s "Exporter plugins" section for the
        plugin contract. Raises ``ExportError`` if no plugin matches, or
        if a matched plugin raises or misbehaves (isolated so a broken
        third-party plugin can't crash the CLI with a raw traceback).
        """

        if export_format is not None:
            return self._export_event_via_plugin(event_id, export_format)
        event = self.repository.load_event(event_id)
        event_dir = self.repository.workspace.event_dir(event_id)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        destination = self.repository.workspace.exports / f"event_{event_id}_{timestamp}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)

        files_to_include: dict[str, Path] = {}
        for path in sorted(event_dir.rglob("*")):
            if path.is_file():
                files_to_include[f"event/{path.relative_to(event_dir).as_posix()}"] = path

        review_log = self.repository.workspace.review_log_path(event_id)
        if review_log.exists():
            files_to_include[f"review/{review_log.name}"] = review_log

        preserved_state = event.preservation_status.state == "preserved"
        if preserved_state and event.preservation_status.bundle_path:
            bundle_dir = Path(event.preservation_status.bundle_path)
            if bundle_dir.exists():
                for path in sorted(bundle_dir.rglob("*")):
                    if path.is_file():
                        arcname = f"preserved/{path.relative_to(bundle_dir).as_posix()}"
                        files_to_include[arcname] = path

        manifest: dict[str, Any] = {
            "event_id": str(event_id),
            "exported_at": utc_now().isoformat(),
            "pipeline_version": event.pipeline_version,
            "event_revision": event.revision,
            "files": [
                {"name": name, "sha256": hash_file(path)}
                for name, path in sorted(files_to_include.items())
            ],
        }
        signing_public_key_hex = self._signing_public_key_hex()
        if signing_public_key_hex is not None:
            # Included inline so a recipient can verify revision signatures
            # (see scripts/verify_export_bundle.py) without needing a
            # second file from this workspace. Added before manifest_hash
            # is computed so tampering with the key is itself detectable.
            manifest["signing_public_key_hex"] = signing_public_key_hex
        manifest_hash = hash_canonical_payload(manifest)
        manifest["manifest_hash"] = manifest_hash

        with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(files_to_include.items()):
                archive.write(path, arcname=name)
            archive.writestr("export_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

        self._record_export(event_id, destination, manifest_hash, len(files_to_include))
        LOGGER.info(
            "event_exported",
            event_id=str(event_id),
            destination=str(destination),
            file_count=len(files_to_include),
        )
        return ExportResult(
            path=destination, manifest_hash=manifest_hash, file_count=len(files_to_include)
        )

    def _signing_public_key_hex(self) -> str | None:
        """The workspace's signing public key, if a key has ever been
        generated -- independent of whether *this* process's config
        currently has signing enabled, since export should always be able
        to publish a key that exists on disk."""

        key_path = self.repository.workspace.signing_private_key_path
        if not key_path.exists():
            return None
        try:
            return EventSigner.load(key_path).public_key_hex
        except SigningUnavailableError:
            return None

    def _find_exporter(self, format_id: str) -> Any | None:
        for plugin in self.exporter_plugins:
            if getattr(plugin, "format_id", None) == format_id:
                return plugin
        return None

    def _export_event_via_plugin(self, event_id: UUID, export_format: str) -> ExportResult:
        plugin = self._find_exporter(export_format)
        if plugin is None:
            raise ExportError(f"no exporter plugin registered for format '{export_format}'")
        plugin_name = getattr(plugin, "name", export_format)

        event_dir = self.repository.workspace.event_dir(event_id)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        destination = (
            self.repository.workspace.exports / f"event_{event_id}_{timestamp}.{export_format}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Plugin isolation (invariant 8's explicitly-permitted broad
        # except): one broken third-party exporter must never crash the
        # CLI with a raw traceback from code this project doesn't own.
        try:
            output_path = Path(plugin.export(str(event_dir), str(destination)))
        except Exception as error:
            LOGGER.error(
                "exporter_plugin_failed",
                plugin=plugin_name,
                format_id=export_format,
                reason=str(error),
            )
            raise ExportError(f"exporter plugin '{plugin_name}' failed: {error}") from error
        if not output_path.exists():
            raise ExportError(
                f"exporter plugin '{plugin_name}' reported output at {output_path}, "
                "but that file does not exist"
            )

        manifest_hash = hash_file(output_path)
        self._record_export(event_id, output_path, manifest_hash, 1)
        LOGGER.info(
            "event_exported_via_plugin",
            event_id=str(event_id),
            plugin=plugin_name,
            format_id=export_format,
            destination=str(output_path),
        )
        return ExportResult(path=output_path, manifest_hash=manifest_hash, file_count=1)

    def export_timeline(
        self, query: TimelineQuery, export_format: Literal["csv", "json"] = "csv"
    ) -> Path:
        events = self.repository.query_events(query)
        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        destination = self.repository.workspace.exports / f"timeline_{timestamp}.{export_format}"
        destination.parent.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "event_id": str(event.event_id),
                "event_start": event.event_start.isoformat(),
                "event_end": event.event_end.isoformat(),
                "severity": event.scoring.severity,
                "confidence": event.scoring.confidence,
                "involved_cameras": ",".join(event.involved_cameras),
                "preservation_state": event.preservation_status.state,
                "review_decision": event.review_summary.latest_decision,
                "evidence_summary": event.evidence_summary,
            }
            for event in events
        ]

        if export_format == "json":
            destination.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        else:
            buffer = io.StringIO()
            fieldnames = (
                list(rows[0].keys())
                if rows
                else [
                    "event_id",
                    "event_start",
                    "event_end",
                    "severity",
                    "confidence",
                    "involved_cameras",
                    "preservation_state",
                    "review_decision",
                    "evidence_summary",
                ]
            )
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            destination.write_text(buffer.getvalue(), encoding="utf-8")

        LOGGER.info("timeline_exported", destination=str(destination), row_count=len(rows))
        return destination

    def _record_export(
        self, event_id: UUID, destination: Path, manifest_hash: str, file_count: int
    ) -> None:
        event = self.repository.load_event(event_id)
        entry = ChainOfCustodyEntry(
            entry_id=new_uuid(),
            action="event_exported",
            actor="gaggle",
            timestamp=utc_now(),
            details={
                "export_path": str(destination.resolve()),
                "manifest_hash": manifest_hash,
                "file_count": file_count,
            },
            input_hashes=[HashDigest(value=h) for h in event.hashes],
            output_hashes=[HashDigest(value=manifest_hash)],
        )
        self.repository.save_event_revision(
            event_id,
            reason="exported",
            update={"chain_of_custody": [*event.chain_of_custody, entry]},
        )
