from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaggle.core.signing import (
    EventSigner,
    SigningUnavailableError,
    cryptography_available,
    generate_signing_key,
    public_key_hex,
    save_private_key_pem,
    verify_signature,
)
from gaggle.schemas.event import EventRecord, PreservationStatus, SeverityAssessment
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

pytestmark = pytest.mark.skipif(not cryptography_available(), reason="cryptography not installed")


def _signer(tmp_path: Path, name: str = "key.pem") -> EventSigner:
    key_path = tmp_path / name
    save_private_key_pem(key_path, generate_signing_key())
    return EventSigner.load(key_path)


def _event() -> EventRecord:
    return EventRecord(
        event_id=new_uuid(),
        created_at=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        pipeline_version="test",
        event_start=datetime(2026, 5, 12, 12, 0, tzinfo=UTC),
        event_end=datetime(2026, 5, 12, 12, 0, 1, tzinfo=UTC),
        involved_cameras=["front"],
        signals=[],
        hypotheses=[],
        scoring=SeverityAssessment(confidence=0.2, severity="low", reasons=["test"], version="1"),
        preservation_status=PreservationStatus(state="pending", immutable=False),
        evidence_summary="test",
    )


def test_generate_and_load_round_trip(tmp_path: Path) -> None:
    key_path = tmp_path / "private_key.pem"
    private_key = generate_signing_key()
    save_private_key_pem(key_path, private_key)
    assert key_path.exists()

    loaded = EventSigner.load(key_path)
    assert loaded.public_key_hex == public_key_hex(private_key)
    assert len(loaded.public_key_hex) == 64  # 32 raw bytes, hex-encoded


def test_load_raises_when_key_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SigningUnavailableError):
        EventSigner.load(tmp_path / "does_not_exist.pem")


def test_load_raises_for_a_non_ed25519_key(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key_path = tmp_path / "rsa_key.pem"
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = rsa_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path.write_bytes(pem)

    with pytest.raises(SigningUnavailableError):
        EventSigner.load(key_path)


def test_sign_and_verify_round_trip(tmp_path: Path) -> None:
    signer = _signer(tmp_path)
    payload = {"event_id": "abc", "revision": 0, "value": 3.14}

    signature = signer.sign_payload(payload)

    assert verify_signature(payload, signature, signer.public_key_hex) is True


def test_verify_fails_when_payload_is_tampered_with(tmp_path: Path) -> None:
    signer = _signer(tmp_path)
    payload = {"event_id": "abc", "revision": 0}
    signature = signer.sign_payload(payload)

    tampered = {"event_id": "abc", "revision": 1}

    assert verify_signature(tampered, signature, signer.public_key_hex) is False


def test_verify_fails_with_the_wrong_public_key(tmp_path: Path) -> None:
    signer_a = _signer(tmp_path, "a.pem")
    signer_b = _signer(tmp_path, "b.pem")
    payload = {"event_id": "abc", "revision": 0}

    signature = signer_a.sign_payload(payload)

    assert verify_signature(payload, signature, signer_b.public_key_hex) is False


def test_verify_fails_for_a_malformed_signature(tmp_path: Path) -> None:
    signer = _signer(tmp_path)
    payload = {"event_id": "abc"}

    assert verify_signature(payload, "not-hex", signer.public_key_hex) is False


def test_two_generated_keys_are_different(tmp_path: Path) -> None:
    signer_a = _signer(tmp_path, "a.pem")
    signer_b = _signer(tmp_path, "b.pem")
    assert signer_a.public_key_hex != signer_b.public_key_hex


def test_repository_signs_the_initial_revision_when_enabled(tmp_path: Path) -> None:
    signer = _signer(tmp_path / "keys")
    repository = Repository(tmp_path / "workspace", signer=signer, signing_enabled=True)
    repository.initialize()
    event = _event()

    repository.save_event(event)

    reloaded = repository.load_event(event.event_id)
    assert reloaded.revision_signature is not None
    payload = reloaded.model_dump(mode="json", exclude={"revision_signature"})
    assert verify_signature(payload, reloaded.revision_signature, signer.public_key_hex)


def test_repository_signs_a_later_revision_independently(tmp_path: Path) -> None:
    signer = _signer(tmp_path / "keys")
    repository = Repository(tmp_path / "workspace", signer=signer, signing_enabled=True)
    repository.initialize()
    event = _event()
    repository.save_event(event)

    updated = repository.save_event_revision(
        event.event_id, reason="annotate", update={"evidence_summary": "changed"}
    )

    assert updated.revision_signature is not None
    revisions = repository.list_event_revisions(event.event_id)
    assert revisions[0].revision_signature != revisions[1].revision_signature
    for revision in revisions:
        payload = revision.model_dump(mode="json", exclude={"revision_signature"})
        assert verify_signature(payload, revision.revision_signature, signer.public_key_hex)


def test_repository_does_not_sign_when_disabled(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()
    event = _event()

    repository.save_event(event)

    reloaded = repository.load_event(event.event_id)
    assert reloaded.revision_signature is None


def test_repository_raises_when_enabled_but_no_key_is_loaded(tmp_path: Path) -> None:
    repository = Repository(tmp_path / "workspace", signer=None, signing_enabled=True)
    repository.initialize()
    event = _event()

    with pytest.raises(SigningUnavailableError):
        repository.save_event(event)


def test_a_revision_written_after_signing_is_turned_off_has_no_stale_signature(
    tmp_path: Path,
) -> None:
    """`save_event_revision` builds the new revision via
    `current.model_copy(update=...)` -- without an explicit reset, a
    revision written while signing is off would otherwise inherit the
    *previous* revision's signature (computed over different content),
    which would look like a valid-but-wrong signature sitting on disk."""

    signer = _signer(tmp_path / "keys")
    signing_repository = Repository(tmp_path / "workspace", signer=signer, signing_enabled=True)
    signing_repository.initialize()
    event = _event()
    signing_repository.save_event(event)
    signed = signing_repository.load_event(event.event_id)
    assert signed.revision_signature is not None

    unsigned_repository = Repository(tmp_path / "workspace")
    updated = unsigned_repository.save_event_revision(
        event.event_id, reason="annotate", update={"evidence_summary": "changed"}
    )

    assert updated.revision_signature is None
    reloaded = unsigned_repository.load_event(event.event_id)
    assert reloaded.revision_signature is None


def test_a_tampered_revision_file_fails_verification(tmp_path: Path) -> None:
    """Not just a signing-module unit check -- confirms the actual
    file-on-disk written by `write_event_revision` fails signature
    verification if edited after the fact, the real threat this feature
    defends against."""

    signer = _signer(tmp_path / "keys")
    repository = Repository(tmp_path / "workspace", signer=signer, signing_enabled=True)
    repository.initialize()
    event = _event()
    repository.save_event(event)

    revision_path = repository.workspace.list_event_revisions(event.event_id)[0]
    payload = json.loads(revision_path.read_text(encoding="utf-8"))
    payload["evidence_summary"] = "tampered"
    signature = payload["revision_signature"]
    payload_without_sig = {k: v for k, v in payload.items() if k != "revision_signature"}

    assert verify_signature(payload_without_sig, signature, signer.public_key_hex) is False
