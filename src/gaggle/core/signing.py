"""Ed25519 cryptographic signing of the event revision hash chain.

Each `EventRecord` revision already links to the canonical JSON hash of
the revision before it (`previous_revision_hash`), forming an internally
consistent chain -- but internal consistency alone can't prove a bundle
wasn't produced (or altered) by someone with the ability to also
recompute consistent hashes. Signing a revision's canonical payload with
a private key held only by this workspace closes that gap: a third party
holding just the public key (exported inline in every bundle's
`export_manifest.json`, see `export/service.py`) can verify a revision
was written by whoever holds the private key, without trusting the
exporting process itself.

Requires the `signing` extra (`pip install gaggle[signing]`).
Fully offline -- key generation, signing, and verification never touch
the network. Optional and off by default (`signing.enabled: false`),
following the same `*_available()` / `*Unavailable` / best-effort-loader
convention as every other optional dependency in this project (see
`enrichment/transcription.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gaggle.utils.json import canonical_json_bytes


class SigningUnavailableError(RuntimeError):
    """Raised when signing is requested but `cryptography` isn't installed,
    or no signing key exists yet for this workspace."""


def cryptography_available() -> bool:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def generate_signing_key() -> Any:
    """Generate a new Ed25519 private key. Raises `SigningUnavailableError`
    if `cryptography` isn't installed."""

    if not cryptography_available():
        raise SigningUnavailableError(
            "the 'cryptography' package is not installed; install the 'signing' "
            "extra (pip install gaggle[signing]) to generate a signing key"
        )
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.generate()


def save_private_key_pem(path: Path, private_key: Any) -> None:
    from cryptography.hazmat.primitives import serialization

    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pem)


def public_key_hex(private_key: Any) -> str:
    from cryptography.hazmat.primitives import serialization

    raw: bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return raw.hex()


class EventSigner:
    """Wraps a workspace's Ed25519 private key for signing revision payloads.

    Only ever holds the private key in memory for the lifetime of one
    `Repository`/CLI-command invocation -- never persisted anywhere except
    the PEM file it was loaded from.
    """

    def __init__(self, private_key: Any) -> None:
        self._private_key = private_key
        self.public_key_hex = public_key_hex(private_key)

    @classmethod
    def load(cls, private_key_path: Path) -> EventSigner:
        if not cryptography_available():
            raise SigningUnavailableError(
                "the 'cryptography' package is not installed; install the 'signing' "
                "extra (pip install gaggle[signing]) to load a signing key"
            )
        if not private_key_path.exists():
            raise SigningUnavailableError(
                f"no signing key found at {private_key_path}; run "
                "'gaggle workspace signing-init --workspace <path>' first"
            )
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = serialization.load_pem_private_key(
            private_key_path.read_bytes(), password=None
        )
        if not isinstance(private_key, Ed25519PrivateKey):
            raise SigningUnavailableError(f"key at {private_key_path} is not an Ed25519 key")
        return cls(private_key)

    def sign_payload(self, payload: dict[str, Any]) -> str:
        """Sign `payload`'s canonical JSON bytes, returning a hex-encoded signature."""

        signature: bytes = self._private_key.sign(canonical_json_bytes(payload))
        return signature.hex()


def verify_signature(
    payload: dict[str, Any], signature_hex: str, public_key_hex_value: str
) -> bool:
    """Verify `signature_hex` over `payload`'s canonical JSON bytes against a
    hex-encoded Ed25519 public key. Returns False for a bad signature or a
    malformed hex value; raises `SigningUnavailableError` if `cryptography`
    isn't installed."""

    if not cryptography_available():
        raise SigningUnavailableError(
            "the 'cryptography' package is not installed; install the 'signing' "
            "extra (pip install gaggle[signing]) to verify a signature"
        )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex_value))
        public_key.verify(bytes.fromhex(signature_hex), canonical_json_bytes(payload))
    except (InvalidSignature, ValueError):
        return False
    return True
