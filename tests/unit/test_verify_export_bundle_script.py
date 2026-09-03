"""Exercises scripts/verify_export_bundle.py -- the dependency-free,
standalone verifier a recipient runs *without* installing
gaggle -- against a real signed bundle produced by
`ExportService`, confirming the two halves (signing production, signing
verification) actually agree with each other end to end, not just each
in isolation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from gaggle.core.signing import (
    EventSigner,
    cryptography_available,
    generate_signing_key,
    public_key_hex,
    save_private_key_pem,
)
from gaggle.export.service import ExportService
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_export_bundle.py"

pytestmark = pytest.mark.skipif(not cryptography_available(), reason="cryptography not installed")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_export_bundle", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_event() -> EventRecord:
    start = datetime(2026, 5, 12, 12, 0, tzinfo=UTC)
    return EventRecord(
        event_id=new_uuid(),
        created_at=start,
        pipeline_version="test",
        event_start=start,
        event_end=start,
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.3, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def _signed_bundle(tmp_path: Path) -> Path:
    key_path = tmp_path / "workspace" / "signing" / "private_key.pem"
    save_private_key_pem(key_path, generate_signing_key())
    signer = EventSigner.load(key_path)
    repository = Repository(tmp_path / "workspace", signer=signer, signing_enabled=True)
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    result = ExportService(repository).export_event_bundle(event.event_id)
    return result.path


def test_verify_bundle_confirms_a_valid_signature(tmp_path: Path) -> None:
    module = _load_script()
    bundle_path = _signed_bundle(tmp_path)

    ok, problems, notices = module.verify_bundle(bundle_path)

    assert ok is True
    assert problems == []
    assert any("valid Ed25519 signature" in notice for notice in notices)


def test_verify_bundle_detects_a_tampered_revision(tmp_path: Path) -> None:
    module = _load_script()
    bundle_path = _signed_bundle(tmp_path)

    revision_name = next(
        name
        for name in zipfile.ZipFile(bundle_path).namelist()
        if name.startswith("event/revisions/") and name.endswith(".json")
    )
    _tamper_with_archive_member(bundle_path, revision_name)

    ok, problems, _ = module.verify_bundle(bundle_path)

    assert ok is False
    assert any("FAILED" in problem for problem in problems)


def test_verify_bundle_detects_a_wrong_public_key(tmp_path: Path) -> None:
    """Swap in an unrelated (but syntactically valid) Ed25519 public key,
    keeping manifest_hash internally consistent with the swap -- isolates
    the signature check from the separate hash-consistency check, which
    would otherwise also (correctly) fail for an unrelated reason."""

    module = _load_script()
    bundle_path = _signed_bundle(tmp_path)

    with zipfile.ZipFile(bundle_path) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
    manifest["signing_public_key_hex"] = public_key_hex(generate_signing_key())
    manifest_without_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    manifest["manifest_hash"] = hashlib.sha256(
        module.canonical_json_bytes(manifest_without_hash)
    ).hexdigest()
    _replace_archive_member(bundle_path, "export_manifest.json", json.dumps(manifest).encode())

    ok, problems, _ = module.verify_bundle(bundle_path)

    assert ok is False
    assert any("signature verification FAILED" in problem for problem in problems)


def test_verify_bundle_has_no_signature_notices_without_a_signing_key(tmp_path: Path) -> None:
    module = _load_script()
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _make_event()
    repository.save_event(event)
    result = ExportService(repository).export_event_bundle(event.event_id)

    ok, problems, notices = module.verify_bundle(result.path)

    assert ok is True
    assert problems == []
    assert notices == []


def _tamper_with_archive_member(bundle_path: Path, member_name: str) -> None:
    with zipfile.ZipFile(bundle_path) as archive:
        payload = json.loads(archive.read(member_name))
    payload["evidence_summary"] = "tampered after signing"
    _replace_archive_member(bundle_path, member_name, json.dumps(payload).encode())


def _replace_archive_member(bundle_path: Path, member_name: str, new_content: bytes) -> None:
    with zipfile.ZipFile(bundle_path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[member_name] = new_content
    with zipfile.ZipFile(bundle_path, mode="w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
