#!/usr/bin/env python3
"""Verify the integrity of a gaggle exported evidence bundle.

Deliberately dependency-free (standard library only: zipfile, json,
hashlib, argparse) so a recipient can verify a bundle's integrity without
installing gaggle or any of its dependencies -- this closes the
"no standalone bundle verifier" gap noted in docs/limitations.md.

What this verifies:
  1. Every file listed in export_manifest.json is present in the archive.
  2. Every file's actual SHA-256 matches its declared hash in the manifest.
  3. The manifest's own declared `manifest_hash` matches a fresh SHA-256 of
     the canonical (sorted-key, indent=2) JSON of the manifest contents
     minus the manifest_hash field itself -- the same self-referential-hash
     scheme used when the bundle was created (see
     gaggle/export/service.py and
     gaggle/storage/filesystem.py::hash_canonical_payload).

What this does NOT verify when the bundle has no signing_public_key_hex
(see docs/threat-model.md):
  - That the manifest itself wasn't fabricated by someone with the ability
    to also recompute consistent hashes. Without a signature, there is
    only hash consistency. Chain-of-custody trust beyond that is a
    process/organizational control, not a technical guarantee this script
    can make.

If the bundle DOES include a `signing_public_key_hex` (workspace had
Ed25519 signing enabled -- see gaggle/core/signing.py), this
script also verifies every included revision file's `revision_signature`
against that public key using the `cryptography` package, when it's
installed in this verifying environment. If `cryptography` isn't
installed, signature verification is skipped with a clear message --
hash-consistency checks above are unaffected either way. A verified
signature proves the revision was written by whoever holds the
corresponding private key; it does not by itself prove that key belongs
to who you think it does (key custody/distribution is out of scope here,
same as any public-key scheme).

Usage:
    python3 scripts/verify_export_bundle.py path/to/event_<id>_<ts>.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


def canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8")


def _cryptography_available() -> bool:
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def verify_signatures(
    archive: zipfile.ZipFile, manifest: dict, names: set[str]
) -> tuple[list[str], list[str]]:
    """Returns (problems, notices). Problems fail verification overall;
    notices are informational only (e.g. an unsigned revision, or
    `cryptography` not being installed here)."""

    public_key_hex = manifest.get("signing_public_key_hex")
    if not public_key_hex:
        return [], []

    if not _cryptography_available():
        return [], [
            "bundle declares a signing_public_key_hex but the 'cryptography' package "
            "is not installed in this verifying environment -- install it "
            "(pip install cryptography) to verify revision signatures; the "
            "hash-consistency checks above are unaffected"
        ]

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    except ValueError:
        return [f"signing_public_key_hex '{public_key_hex}' is not a valid Ed25519 public key"], []

    problems: list[str] = []
    notices: list[str] = []
    verified_count = 0
    revision_names = sorted(
        name for name in names if name.startswith("event/revisions/") and name.endswith(".json")
    )
    for name in revision_names:
        payload = json.loads(archive.read(name))
        signature_hex = payload.get("revision_signature")
        if signature_hex is None:
            notices.append(
                f"'{name}' has no revision_signature (signing may have been "
                "enabled after this revision was written)"
            )
            continue
        payload_without_sig = {k: v for k, v in payload.items() if k != "revision_signature"}
        signed_bytes = canonical_json_bytes(payload_without_sig)
        try:
            public_key.verify(bytes.fromhex(signature_hex), signed_bytes)
            verified_count += 1
        except (InvalidSignature, ValueError):
            problems.append(
                f"signature verification FAILED for '{name}' -- this revision may "
                "have been tampered with, or signed by a different key"
            )

    if verified_count:
        notices.append(f"{verified_count} revision(s) had a valid Ed25519 signature")
    return problems, notices


def verify_bundle(bundle_path: Path) -> tuple[bool, list[str], list[str]]:
    problems: list[str] = []
    notices: list[str] = []

    if not zipfile.is_zipfile(bundle_path):
        return False, [f"{bundle_path} is not a valid zip archive"], []

    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        if "export_manifest.json" not in names:
            return False, ["archive has no export_manifest.json"], []

        manifest = json.loads(archive.read("export_manifest.json"))
        declared_hash = manifest.get("manifest_hash")
        files = manifest.get("files", [])

        if not declared_hash:
            problems.append("manifest has no manifest_hash field")
        else:
            manifest_without_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
            recomputed = hashlib.sha256(canonical_json_bytes(manifest_without_hash)).hexdigest()
            if recomputed != declared_hash:
                problems.append(
                    f"manifest_hash mismatch: declared={declared_hash} recomputed={recomputed}"
                )

        for entry in files:
            name = entry.get("name")
            expected_hash = entry.get("sha256")
            if name is None or expected_hash is None:
                problems.append(f"malformed manifest entry: {entry!r}")
                continue
            if name not in names:
                problems.append(f"manifest lists '{name}' but it is missing from the archive")
                continue
            actual_hash = hashlib.sha256(archive.read(name)).hexdigest()
            if actual_hash != expected_hash:
                problems.append(
                    f"hash mismatch for '{name}': expected={expected_hash} actual={actual_hash}"
                )

        archived_only = names - {entry.get("name") for entry in files} - {"export_manifest.json"}
        for extra in sorted(archived_only):
            problems.append(f"file present in archive but not listed in manifest: '{extra}'")

        signature_problems, signature_notices = verify_signatures(archive, manifest, names)
        problems.extend(signature_problems)
        notices.extend(signature_notices)

    return (len(problems) == 0), problems, notices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to an exported .zip evidence bundle")
    args = parser.parse_args()

    if not args.bundle.exists():
        print(f"error: {args.bundle} does not exist", file=sys.stderr)
        return 2

    ok, problems, notices = verify_bundle(args.bundle)
    for notice in notices:
        print(f"note: {notice}")

    if ok:
        print(f"OK: {args.bundle} is internally consistent (all hashes verified)")
        return 0

    print(f"FAILED: {args.bundle} has integrity problems:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
