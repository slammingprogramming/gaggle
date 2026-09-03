from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaggle.core.signing import (
    EventSigner,
    cryptography_available,
    generate_signing_key,
    save_private_key_pem,
)
from gaggle.export.service import ExportError, ExportService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.database import TimelineQuery
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid


def _make_event(camera: str = "front") -> EventRecord:
    start = datetime(2026, 5, 12, 12, 0, tzinfo=UTC)
    return EventRecord(
        event_id=new_uuid(),
        created_at=start,
        pipeline_version="test",
        event_start=start,
        event_end=start,
        involved_cameras=[camera],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def test_export_event_bundle_produces_verifiable_zip(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    result = ExportService(repository).export_event_bundle(event.event_id)

    assert result.path.exists()
    with zipfile.ZipFile(result.path) as archive:
        names = archive.namelist()
        assert "export_manifest.json" in names
        assert any(name.startswith("event/") for name in names)
        manifest = json.loads(archive.read("export_manifest.json"))
        assert manifest["event_id"] == str(event.event_id)
        assert manifest["manifest_hash"] == result.manifest_hash
        # every file's declared hash must match the archived bytes
        for entry in manifest["files"]:
            if entry["name"] == "export_manifest.json":
                continue
            data = archive.read(entry["name"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]


def test_export_appends_chain_of_custody_entry_without_mutating_prior_revision(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    ExportService(repository).export_event_bundle(event.event_id)

    revisions = repository.list_event_revisions(event.event_id)
    assert len(revisions) == 2
    assert revisions[0].chain_of_custody == event.chain_of_custody
    assert any(entry.action == "event_exported" for entry in revisions[1].chain_of_custody)


def test_export_timeline_csv_contains_event_row(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event(camera="rear")
    repository.save_event(event)

    path = ExportService(repository).export_timeline(TimelineQuery(), export_format="csv")
    content = path.read_text(encoding="utf-8")
    assert str(event.event_id) in content
    assert "rear" in content


def test_export_timeline_json_round_trips(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    path = ExportService(repository).export_timeline(TimelineQuery(), export_format="json")
    rows = json.loads(path.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["event_id"] == str(event.event_id)


class _FakeEntry:
    def __init__(self, name: str, plugin: object) -> None:
        self.name = name
        self._plugin = plugin

    def load(self):
        return self._plugin


class _FakeEntryPoints:
    def __init__(self, entries: list[_FakeEntry]) -> None:
        self._entries = entries

    def select(self, group: str) -> list[_FakeEntry]:
        return self._entries


def _register_fake_exporters(monkeypatch: pytest.MonkeyPatch, *plugins: object) -> None:
    entries = [_FakeEntry(getattr(p, "name", "plugin"), p) for p in plugins]
    monkeypatch.setattr("gaggle.plugins.registry.entry_points", lambda: _FakeEntryPoints(entries))


class _MarkerExporter:
    """A minimal, real ExporterPlugin: reads the event directory and writes
    a trivial marker file -- this doubles as the reference exporter plugin
    example referenced from docs/plugin-authoring.md."""

    name = "marker-exporter"
    version = "1.0.0"
    format_id = "marker"

    def export(self, event_path: str, destination: str) -> str:
        content = f"exported from {event_path}\n"
        Path(destination).write_text(content, encoding="utf-8")
        return destination


class _BrokenExporter:
    name = "broken-exporter"
    version = "1.0.0"
    format_id = "broken"

    def export(self, event_path: str, destination: str) -> str:
        raise RuntimeError("boom")


class _LyingExporter:
    name = "lying-exporter"
    version = "1.0.0"
    format_id = "lying"

    def export(self, event_path: str, destination: str) -> str:
        return str(Path(destination).with_name("this-file-does-not-exist.out"))


def test_export_event_dispatches_to_a_matching_exporter_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_fake_exporters(monkeypatch, _MarkerExporter())
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    result = ExportService(repository).export_event_bundle(event.event_id, export_format="marker")

    assert result.path.exists()
    assert result.path.suffix == ".marker"
    assert str(event.event_id) in result.path.read_text(encoding="utf-8")
    assert result.file_count == 1

    revisions = repository.list_event_revisions(event.event_id)
    assert any(entry.action == "event_exported" for entry in revisions[-1].chain_of_custody)


def test_export_event_raises_for_an_unregistered_format(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    with pytest.raises(ExportError, match="no exporter plugin"):
        ExportService(repository).export_event_bundle(event.event_id, export_format="nonexistent")


def test_export_event_isolates_a_broken_exporter_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_fake_exporters(monkeypatch, _BrokenExporter())
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    with pytest.raises(ExportError, match="broken-exporter"):
        ExportService(repository).export_event_bundle(event.event_id, export_format="broken")


def test_export_event_rejects_a_plugin_that_reports_a_nonexistent_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _register_fake_exporters(monkeypatch, _LyingExporter())
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    with pytest.raises(ExportError, match="does not exist"):
        ExportService(repository).export_event_bundle(event.event_id, export_format="lying")


@pytest.mark.skipif(not cryptography_available(), reason="cryptography not installed")
def test_export_manifest_includes_the_public_key_when_a_signing_key_exists(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "workspace" / "signing" / "private_key.pem"
    save_private_key_pem(key_path, generate_signing_key())
    public_key_hex = EventSigner.load(key_path).public_key_hex

    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    result = ExportService(repository).export_event_bundle(event.event_id)

    with zipfile.ZipFile(result.path) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
    assert manifest["signing_public_key_hex"] == public_key_hex
    # the public key is itself covered by manifest_hash, not tamperable
    # without invalidating it
    manifest_without_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    manifest_without_hash["signing_public_key_hex"] = "tampered"
    recomputed = hashlib.sha256(
        json.dumps(manifest_without_hash, indent=2, sort_keys=True, ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()
    assert recomputed != manifest["manifest_hash"]


def test_export_manifest_has_no_public_key_when_signing_was_never_set_up(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)

    result = ExportService(repository).export_event_bundle(event.event_id)

    with zipfile.ZipFile(result.path) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
    assert "signing_public_key_hex" not in manifest
