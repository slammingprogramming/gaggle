# Threat model

This document is a plain accounting of what `gaggle` protects
against, what it explicitly does not, and where the boundaries are. It is
written for two audiences: contributors deciding whether a change preserves
these properties, and users deciding how much to trust the system's output
in a context where that trust matters (insurance disputes, legal
proceedings, personal safety records).

## Assumptions

* The machine running `gaggle` is not compromised. The system has
  no defense against an attacker with root/administrator access to the host
  -- they can edit any file, including "frozen" ones (see below).
* The person running ingest has physical access to the source media (SD
  card, USB drive) and is not adversarial to the evidence they're ingesting.
  `gaggle` cannot detect if the SD card itself was tampered with
  *before* ingest.
* Python, ffmpeg, and their dependencies are installed from trusted sources.
  Supply-chain compromise of a dependency is out of scope for this document
  (see your organization's normal dependency-vetting process).

## What "immutable" and "frozen" actually mean

Read-only permissions (`chmod`) are the mechanism used throughout the
codebase for "originals," "preserved bundles," and individual revision
files. This is **tamper-evidence and accident-prevention, not
tamper-proofing**:

* It reliably prevents `gaggle`'s own code from accidentally
  overwriting evidence -- this is the primary goal, and it's a hard
  guarantee within the pipeline's own code paths (enforced by review in
  every PR that touches storage).
* It does **not** prevent a user with filesystem access from running
  `chmod +w` and editing a "frozen" file. Optional Ed25519 signing
  (`core/signing.py`, `signing.enabled`, off by default) makes such
  tampering *detectable* against the signing key -- see below -- but the
  private key itself is only as protected as `chmod` makes it; a user
  with filesystem access who also finds the key file could re-sign a
  tampered revision.
* Every hash (`sha256` on originals, revision hash chains, bundle
  manifests) exists so tampering is *detectable after the fact* by
  recomputing hashes and comparing -- not so it's *impossible*. The same
  is true of a signature: it's detection, not prevention.

**Guarantee:** if a file's declared hash matches its current on-disk
content, the content has not changed since that hash was recorded.
**Non-guarantee:** the system cannot prove no one had the opportunity to
tamper with it; it can only prove whether they did, if you check.

## Chain-of-custody / hash-chain limitations

`EventRecord.previous_revision_hash` chains each revision to the canonical
JSON hash of the one before it. This means:

* **Guarantee:** given the full revision history for an event, you can
  verify the chain is internally consistent (each `previous_revision_hash`
  matches the actual hash of the prior file) and detect if any revision file
  was altered or if one was deleted and the chain re-spliced around it
  incorrectly.
* **Non-guarantee:** there is no external timestamping authority tying the
  chain to a specific real-world moment beyond the `revised_at` field,
  which itself comes from the host system clock and is only as
  trustworthy as that clock. With `signing.enabled` (see
  `docs/local-ai.md`), each revision's canonical payload is signed with
  an Ed25519 key generated via `workspace signing-init` and held only in
  `workspace/signing/private_key.pem` -- this proves a revision was
  written by whoever holds that key, closing the "internally-consistent
  but fabricated history" gap **only for an attacker who does not also
  have that private key**. A sufficiently privileged attacker who
  controls both the workspace filesystem *and* the private key (e.g. read
  access to `workspace/signing/`) can still edit and re-sign a fabricated
  history; signing does not raise the bar against that specific attacker,
  only against one who can edit revision files but doesn't hold the
  signing key (e.g. someone editing an exported bundle after the fact,
  without workspace access). Recipients of an exported bundle verify
  signatures via `scripts/verify_export_bundle.py`, which reads the
  public key inline from `export_manifest.json` -- see
  `docs/local-ai.md`'s signing section for the full design and its
  caveats.

## Specific attack surfaces

| Surface | Mitigation | Residual risk |
|---|---|---|
| Original media modified after ingest | Copied to `originals/`, `chmod` read-only, `sha256` recorded at ingest | Root/owner can `chmod +w` and edit; always verify hash before treating media as trustworthy |
| `event.json` silently rewritten to hide a review decision | Revision history in `revisions/`, each frozen, hash-chained | Same as above -- filesystem permissions, not cryptography |
| Review log (`review/<id>.jsonl`) edited to remove or alter a past decision | Append-only write pattern (`utils/filesystem.py::append_line`), never opened for truncation/rewrite by any code path | The file itself is not `chmod`-frozen (it needs to remain appendable); a privileged user could edit it directly. Consider `chattr +a` (append-only attribute) at the OS level in high-assurance deployments -- not automated by `gaggle` itself |
| Malicious/broken third-party plugin | `plugins/registry.py::load_plugins` catches and logs any exception from `.load()` or instantiation, isolating one broken plugin from crashing the run | A plugin that loads successfully and then produces bad `Signal`/`Hypothesis` data is only caught by normal schema validation (`extra="forbid"`, field constraints) -- it can still produce *wrong but well-typed* evidence. Review third-party plugins before installing them, the same as any code you run locally |
| Corrupted SD card / partial file | `ffprobe`/`ffmpeg` calls are wrapped and failures are logged, not silently swallowed; ingest records `probe_status` in clip metadata so a degraded probe is visible in the data itself | A corrupted file that still probes "successfully" but has garbage frames is not separately detected; frame-differencing on garbage data will just produce garbage (but typed, bounded 0-1) motion values |
| Partial ingest failure mid-run | Each file is hashed and copied independently; a failure on one file does not corrupt already-copied files. The ingest manifest only lists files that completed | If ingest is killed mid-run, already-copied files remain (correctly, as partial evidence) but no manifest is written for that run; rerunning ingest is safe (idempotent per-file, since destination paths are content-hash-prefixed) |
| Timestamp spoofing (malicious filename/mtime) | `timestamp_confidence` scores filename-derived timestamps (0.7) higher than mtime-derived ones (0.3) but never claims certainty; sync corrections and their rationale are always inspectable | The system has no way to independently verify a camera's clock was accurate at capture time. A dashcam with a maliciously pre-set clock will produce a plausible-looking but wrong timestamp; nothing here defends against that specific case beyond flagging low confidence |
| Replay attack (re-ingesting old/already-processed media as if new) | Every file is content-hashed; nothing currently rejects a duplicate `sha256` at ingest | Re-ingesting the same file produces a second, independent copy and a second set of events. This is intentionally permissive (see false-positive philosophy) but means deduplication is a manual review step, not automatic |
| Manipulated export bundle | Each export has a `manifest_hash` (hash of the file manifest) and per-file hashes inside `export_manifest.json`; recipients can verify without installing the package via `scripts/verify_export_bundle.py` | Verification is hash-consistency only, not cryptographic proof of authenticity -- see the hash-chain limitations above |
| Malicious/oversized ingest source (zip bomb, symlink attacks) | `ingest_directory` only follows `Path.rglob("*")` on the given source root and only copies recognized media extensions | No explicit symlink-loop protection or size cap; a maliciously crafted source directory could in principle cause excessive disk use. Treat ingest sources as at least as trusted as any other local file operation |
| Optional cloud LLM enrichment sends data off-machine | Disabled by default everywhere; requires explicit `enrichment.cloud.enabled: true` plus a configured endpoint and an API key from an environment variable (never stored in a config file); sends transcript *text only*, never media | Once enabled, the configured endpoint operator can see transcript content; this is an inherent, disclosed property of the feature, not a bug -- see `docs/local-ai.md`. Anyone with access to the environment variable has the API key |
| Recognition database (face/plate) misused for surveillance beyond personal review | Local-only, single-user, no identity resolution, no networking with other cameras/users -- see `docs/forensic-considerations.md`'s "Recognition data: scope and intent" | The underlying capability (detect + locally cluster faces/plates) is present in the codebase; nothing technically prevents a deployer from pointing it at a fixed public-facing camera to build a standing log of passersby. This is explicitly out of the intended scope and likely to carry separate legal obligations in many jurisdictions -- the software does not enforce a usage boundary here, the documentation does |
| Accidental permanent loss of benign footage via triage/deletion workflow | `confirm_deletion` requires an explicit actor name, verifies the file's hash against the indexed hash before deleting (refuses if they don't match), and writes a `DeletionRecord` to the append-only deletion log *before* unlinking | A clip is only eligible for this workflow if it contributed zero signals to any event -- if detection missed something genuinely relevant (see the false-positive philosophy in `docs/forensic-considerations.md`), that footage could still be deleted. There is no "undo": once confirmed, the original bytes are gone; only the `DeletionRecord` (hash, timestamp, actor, reason) remains |

## Explicit non-goals (repeated from the project's design directives)

This system does not attempt to, and should never be extended to:

* Autonomously determine fault, guilt, or legal conclusions.
* Identify individuals (no face/plate recognition in the built-in pipeline).
* Operate as predictive policing or mass-surveillance infrastructure.
* Replace human review -- every automated output here is a hypothesis with
  attached confidence and reasoning, and severity scoring is deliberately
  biased toward over-retention (see the false-positive philosophy in
  `docs/forensic-considerations.md`) precisely so a human makes the final
  call.

## Reporting a security issue

If you find a way to make `gaggle` silently corrupt, lose, or
misattribute evidence, please open an issue describing the scenario. Given
the project's forensic use case, silent-corruption bugs are treated as the
highest-severity class of bug regardless of exploitability.
