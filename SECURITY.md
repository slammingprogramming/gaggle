# Security Policy

Gaggle handles forensic evidence and personal recognition data (faces,
plates, voices, locations). Security reports are taken seriously and
handled privately, with a verification step designed to protect both the
reporter and the maintainer from impersonation.

## Supported versions

This project does not yet maintain parallel release branches. Security
fixes are made against the `master` branch and the most recent tagged
release. If you're running an older version, please update before
reporting to confirm the issue still applies.

## Reporting a vulnerability

**Do not open a public GitHub issue, discussion, or pull request that
describes the vulnerability or includes a proof of concept.** Public
disclosure before a fix is available puts every user of this project at
risk.

Instead, reports are handled through a two-step private-contact process:

### 1. Make initial contact

Do both of the following:

- **Open a GitHub issue** using the "Security Report (initial contact
  only)" issue template. This issue must **not** contain any technical
  detail about the vulnerability -- it exists only to establish a public,
  timestamped record that a report is in progress and to anchor the
  identity-verification step below. State only that you have found a
  security-relevant issue and intend to report it privately.
- **Connect over SimpleX Chat** at:
  <https://smp14.simplex.im/a#3gZ-zeHs4QrFZKLAN0o3SC_XQJXhj1eYBVTO_c0FAtg>

### 2. Mutual identity verification

Before any technical detail is discussed, we verify each other:

- You sign a message referencing your GitHub issue (by number/URL) with a
  key you control, and share the signature and public key over SimpleX.
- The maintainer does the same in return, so you can confirm you're
  actually talking to the maintainer of this repository and not an
  impersonator on SimpleX.

This step exists so that a report can later be attributed to a real,
verifiable GitHub identity (useful for credit, coordinated disclosure
timelines, and CVE requests), while keeping the substance of every report
off any public channel until a fix ships. It also means you only have to
do this once: after your first verified report, you can reach the same
maintainer directly over the already-verified SimpleX connection for any
future report, with no need to repeat the GitHub-issue/signature exchange.

### 3. Discuss and resolve

Once both sides are verified, the technical details, severity assessment,
and fix timeline are discussed entirely over the verified SimpleX
connection. The GitHub issue from step 1 stays free of technical detail
throughout and is only updated (or closed, or replaced by a published
advisory) once a fix is available and public disclosure is appropriate.

## Scope

In scope: the `gaggle` package itself (ingest, detection, inference,
scoring, storage, recognition, review UI, CLI, export/import,
plugin-loading) and its official Docker image/devcontainer.

Generally out of scope: vulnerabilities in third-party dependencies
(report those upstream; feel free to also let us know so we can track an
update), and issues that require an attacker to already have write access
to a workspace directory or the machine running Gaggle (this project's
threat model assumes the operator's own machine and workspace are
trusted -- see `docs/threat-model.md`).

## What to expect

You'll get an acknowledgment on SimpleX once mutual verification
completes. From there, expect ongoing communication as the issue is
triaged, reproduced, and fixed -- timelines depend on severity and
complexity, and will be discussed directly with you.
