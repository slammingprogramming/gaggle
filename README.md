# Gaggle

[![CI](https://github.com/slammingprogramming/gaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/slammingprogramming/gaggle/actions/workflows/ci.yml)
[![CodeQL](https://github.com/slammingprogramming/gaggle/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/slammingprogramming/gaggle/security/code-scanning)
[![License: AGPL v3 or later](https://img.shields.io/badge/License-AGPLv3--or--later-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Security Policy](https://img.shields.io/badge/security-policy-blue.svg)](SECURITY.md)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

Gaggle is an offline-first, forensic-oriented, **camera-agnostic** encounter
and incident analysis system. It started as a dashcam-only tool and is
growing into something broader: a local, personal encounter-intelligence
fabric that works the same way whether the footage comes from a dashcam,
an outdoor security camera, or an indoor one -- dashcams remain a
first-class, fully-supported source, not a legacy special case. It ingests
multi-camera footage from any of these sources, runs deterministic
motion/audio/object-region detection, correlates signals across cameras
with explainable rule-based inference, and produces versioned, append-only,
hash-chained event metadata (`event.json`) as its primary artifact -- video
is evidence to preserve, not the point of the system. The CLI command is
still `gaggle`.

**Contents:** [Principles](#principles) &middot; [Quick start](#quick-start)
&middot; [Repository layout](#repository-layout) &middot;
[What it does](#what-it-does) &middot; [License](#license) &middot;
[Contributing](CONTRIBUTING.md) &middot; [Security](SECURITY.md)

## Principles

- Deterministic pipelines over opaque automation
- Explainable signals, hypotheses, and scoring (every confidence number
  comes with a plain-language formula, never a bare float)
- Immutable evidence preservation, append-only review and revision history
- Human-in-the-loop review -- automated output is always a hypothesis,
  never a conclusion
- Hybrid storage: the filesystem is the source of truth for evidence;
  SQLite is a query accelerator that could be deleted and rebuilt
- Fully offline operation after installation; optional online enhancements
  (cloud LLM transcript analysis) are opt-in and off by default everywhere

## Quick start

**Not sure which setup fits you?** See `docs/getting-started.md` for a
tiered guide (minimal / recommended / full) before you install anything.

```bash
pip install -e .[dev,vision,cloud]   # requires ffmpeg/ffprobe/tesseract on PATH -- see docs/developer-setup.md
gaggle workspace init --workspace ./workspace
gaggle ingest examples/sample_media --workspace ./workspace
gaggle analyze --workspace ./workspace     # also triages benign vs. reviewable
gaggle enrich --workspace ./workspace      # local face/plate recognition, offline by default
gaggle review start --actor "you" --workspace ./workspace
gaggle review-ui --workspace ./workspace   # http://127.0.0.1:8000
```

`examples/sample_media` contains two short, real, generated video clips (not
fixtures) -- running the commands above exercises real ffprobe metadata
extraction, real OpenCV motion analysis, and real scipy audio-spike
detection, end to end. **For the complete guided workflow** -- choosing an
ingest storage mode, reviewing and cleaning up false positives, preserving
what matters, and reclaiming disk space from footage you're done with
while keeping every bit of metadata -- see
`docs/pipeline-walkthrough.md`. See `docs/cli-examples.md` for the full
command reference, `docs/local-ai.md` for the face/plate/vehicle/
transcription/LLM enrichment and storage-lifecycle workflow, and
`docs/developer-setup.md` for setup details including Docker.

## Repository layout

```
src/gaggle/
  schemas/      typed data model for everything (Pydantic v2)
  ingest/       immutable copy-in + real ffprobe metadata extraction
  normalize/    cross-camera time synchronization
  windowing/    deterministic sliding-window generation
  detection/    motion / audio / object-hint signal producers
  inference/    rule-based signal -> hypothesis engine
  scoring/      hypothesis -> severity assessment
  core/         pipeline orchestration, config, review actions, storage-
                lifecycle triage + event-video purge, identity recognition
  enrichment/   local face/plate recognition, optional local vehicle
                detection + transcription, optional cloud LLM analysis
  preservation/ immutable evidence bundle creation
  export/       hash-manifested evidence bundle + timeline export
  timeline/     filtered querying over the SQLite index
  patterns/     metadata-only pattern hypotheses
  plugins/      DetectorPlugin / InferenceRulePlugin / ExporterPlugin
  storage/      filesystem layout + revisioning, SQLite index (incl.
                recognition + triage tables), repository
  cli/          Typer CLI (the primary interface)
  review_ui/    FastAPI review UI: JSON API + synchronized playback page
tests/          unit + integration tests, mirroring src/ structure
docs/           getting-started guide, pipeline walkthrough, architecture,
                schema, threat model, chain of custody, forensic
                considerations, limitations, plugin authoring, local AI
                capabilities, developer setup, CLI examples, sequence
                diagrams
examples/       real sample media + example config + example event.json
scripts/        standalone operational tooling (e.g. verify_export_bundle.py)
AGENTS.md       working reference for anyone (human or agent) developing here
```

See `docs/architecture.md` for the full module map, the hybrid storage
model, the event-revisioning design, and why this is one package with
strict submodule boundaries rather than a literal multi-package monorepo.

## What it does

**Core pipeline (fully offline, no setup beyond ffmpeg):**

- Immutable ingest with content hashing and real `ffprobe`-derived
  duration/fps/codec metadata (not a hardcoded guess), with a choice of
  storage mode (`copy`/`move`/`reference`) so you control disk usage
  during ingest, not just afterward
- Cross-camera time synchronization: session grouping, deterministic
  reference-camera selection, offset/drift estimation, always alongside the
  original uncorrected timestamps
- Deterministic sliding-window generation over sync-corrected time, with
  window-overlap merging so one real incident produces one event, not
  several near-duplicates
- Real motion detection (OpenCV frame differencing), audio-spike detection
  (ffmpeg + scipy RMS envelope), and moving-region hints (contour analysis)
  -- with an optional sidecar-fixture override for deterministic testing
- Rule-based, explainable inference with multi-camera and multi-signal-type
  corroboration bonuses; no single weak signal reaches high severity alone
- Versioned, append-only `event.json` with a hash-chained revision history
  -- preservation and review actions are folded into new revisions, never
  silent in-place edits
- Immutable, self-contained preservation bundles and hash-manifested export
  bundles (zip) plus flat timeline CSV/JSON export, with a dependency-free
  standalone verifier script for recipients (`scripts/verify_export_bundle.py`)
- Filtered timeline querying and metadata-only pattern detection (repeated
  camera activity, repeated object labels, temporal clustering)
- A plugin system (detectors, inference rules, exporters) discovered via
  standard Python entry points, with failure isolation
- A review UI with real synchronized multi-camera playback, plus an
  interactive `review start` CLI walkthrough

**Local AI enrichment (`enrichment/`, see `docs/local-ai.md`):**

- Local face detection + on-device re-identification -- "have I seen this
  face before," never identity lookup; on by default with real
  deep-learning models (YuNet detection, AuraFace embeddings, GPU-capable),
  falling back automatically to a classical zero-dependency path (Haar
  cascade + LBPH clustering) if those extras aren't installed; a
  duplicate-cleanup pass (`recognize faces-cleanup`) mirrors the plate one
- Local license plate detection + OCR -- on by default via fast-alpr
  (real deep-learning detection+OCR, international plate formats,
  GPU-capable), falling back automatically to a classical path combining
  OpenCV cascades, MSER blob detection, and rotation-aware contour
  analysis (added after real feedback that an earlier detector missed
  real plates and picked up junk -- verified fixed against a reproduced
  regression scene) if the extra isn't installed, with a confidence-gated
  review queue, a dedicated reject action, automated duplicate-cleanup
  (`recognize plates-cleanup`), and a debug tool (`recognize
  plates-debug`) that renders every candidate region on real footage so
  you can see exactly what it found
- Local voice activity detection + classical MFCC-based voiceprinting for
  recurring-speaker re-identification -- built entirely from numpy/scipy
  signal processing since no pretrained speaker-embedding model is
  reachable offline; zero setup, on by default, but explicitly documented
  as a meaningfully weaker fingerprint than face/plate recognition and
  validated only against synthetic test signals, not real speech
- Optional local vehicle/object detection via a user-supplied YOLO ONNX
  model (`vision` extra)
- Optional local audio transcription via Whisper (`transcription` extra)
- Optional cloud LLM transcript analysis via any OpenAI-compatible endpoint
  (OpenRouter, self-hosted, etc.) -- the only network-calling feature in
  the project, off by default everywhere (`cloud` extra)
- Identity linking (`core/recognition.py`): merge fragmented face/plate/
  voice clusters into one traceable identity UUID, search by text/id/label
  with fuzzy OCR-typo-tolerant fallback, and pull every sighting for a
  person, vehicle, or voice across the whole workspace -- local-only, no
  name resolution, every merge permanently logged
- Automated merge suggestions: a scan flags likely-fragmented identities
  (the same face/plate/voice split across multiple clusters) into a
  review queue -- never merges automatically, a human confirms or rejects
  each one (`recognize suggest-merges`)

**Storage lifecycle (`core/triage.py`, see `docs/local-ai.md` and
`docs/pipeline-walkthrough.md`):**

- Automatic classification of every clip as reviewable (contributed to a
  signal) or benign (contributed to nothing) after `analyze`
- Benign originals move to a dedicated `pending_deletion/` folder (or, for
  `reference`-mode clips, stay put and are classified in place); nothing
  is ever deleted without an explicit, actor-attributed
  `triage confirm-deletion`, which writes a permanent append-only
  `DeletionRecord` before removing any bytes
- **Event-video purge**: once you've reviewed and (usually) preserved an
  event, `triage purge-event-video`/`purge-reviewed` deletes its derived
  clips and any contributing original clips no other event still needs --
  safely cascading across events that share footage -- while
  `event.json`'s signals, hypotheses, scoring, chain of custody, and full
  history stay exactly as they were, forever
- Face crops, plate observations, transcripts, and event metadata all
  survive deletion or purging of the raw footage they came from -- so a
  256GB import doesn't have to become 256GB of permanent storage

See `docs/limitations.md` for an honest list of what's a deliberate design
choice versus a known gap for a future release, and
`docs/forensic-considerations.md` for the scope and intent boundaries
around face/plate recognition specifically.

## License

Copyright (C) 2026 SlammingProgramming and contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

AGPL-3.0-or-later. See [`LICENSE`](LICENSE) for the full text,
[`EULA.md`](EULA.md) for a plain-language addendum on intended use, and
[`GENERATED_CONTENT_LICENSE.md`](GENERATED_CONTENT_LICENSE.md) for the
license covering *output* the tool generates (as distinct from the
AGPL-licensed source code itself).
