# Architecture

`gaggle` is a single Python package (`gaggle`, laid out under
`src/`) organized into focused submodules, each responsible for one stage of a
linear, replayable pipeline. There is no microservice split and no message
queue: the project runs entirely on one machine, offline, and the "modularity"
requirement is satisfied by strict module boundaries and typed interfaces
rather than process boundaries.

```
ingest -> normalize -> window -> detect -> infer -> score -> (build events) -> triage -> enrich -> preserve / review / export
```

Each arrow is a real stage boundary: every stage reads the previous stage's
persisted manifest from the workspace, does its work, and writes its own
manifest before the next stage runs. Nothing is held only in memory between
stages, which is what makes replay possible -- you can, for example, rerun
`normalize` and `window` with different config against an existing `ingest`
manifest without re-copying source media.

## Module map

| Module | Responsibility |
|---|---|
| `schemas/` | Every typed data shape in the system (Pydantic v2, `extra="forbid"`). The one place field names are allowed to be defined. |
| `utils/` | Dependency-free helpers: hashing, UTC time handling, canonical JSON, filesystem primitives, structured logging setup. |
| `core/config.py` | `RuntimeConfig` and YAML/env-based config loading. |
| `ingest/` | Immutable copy-in of source media, real `ffprobe` metadata extraction (`probe.py`), timestamp inference. |
| `normalize/` | Cross-camera time synchronization (`sync.py` -- pure logic, no I/O) and the service that wraps it with typed I/O. |
| `windowing/` | Deterministic sliding-window generation over sync-corrected clip intervals. |
| `detection/` | Signal producers: `motion.py`, `audio.py`, `object_detection.py`, plus the underlying real analysis in `video_analysis.py` (OpenCV) and `audio_analysis.py` (ffmpeg + scipy). |
| `inference/` | Rule-based `Signal -> Hypothesis` engine, plus third-party rule plugin loading. |
| `scoring/` | `Hypothesis -> SeverityAssessment`. |
| `core/pipeline.py` | Orchestrates the above and assembles `EventRecord`s, including the window-overlap-merge step and derived-clip extraction. |
| `preservation/` | Copies an event's evidence into a frozen, self-contained bundle. |
| `core/review.py` | Human review actions, folded into event revisions. |
| `core/triage.py` | Storage-lifecycle classification (benign vs. reviewable) and human-confirmed deletion. |
| `enrichment/` | Optional/local-first face, plate, vehicle, transcription, and cloud-LLM enrichment -- see `docs/local-ai.md`. |
| `export/` | Self-contained evidence-bundle export and timeline CSV/JSON export. |
| `timeline/`, `patterns/` | Read-only querying and metadata-only pattern hypotheses over already-generated events. |
| `plugins/` | The `DetectorPlugin` / `InferenceRulePlugin` / `ExporterPlugin` / `ReviewExtensionPlugin` protocols and the `entry_points()`-based loader. |
| `storage/` | `filesystem.py` (the forensic layout + revisioning), `database.py` (SQLite index, incl. the `Camera`/`Encounter` tables), `repository.py` (the single seam between them), `migrate.py` + `migrations/` (Alembic schema-upgrade driver -- see below). |
| `core/cameras.py` | `CameraRepository` -- thin lookups over the optional camera registry, used by ingest auto-registration and site-scoped sync. |
| `cli/app.py` | Typer CLI -- the primary interface. |
| `review_ui/app.py` | FastAPI review UI: JSON API plus a server-rendered synchronized multi-camera playback page. |

## Hybrid storage model

This is the most important structural rule in the codebase:

* **The filesystem is the source of truth for evidence.** Original media,
  normalized/derived media, `event.json` and its revision history, review
  logs, and preservation/export bundles all live under the workspace as
  plain files.
* **SQLite (`storage/database.py`) is a query accelerator only.** It exists
  so `timeline query` and the review queue don't have to scan every
  `event.json` on every request. `Repository.reindex()` can rebuild it
  entirely from the filesystem at any time -- if `index.sqlite3` were
  deleted, nothing would be lost. `gaggle workspace reindex
  --rebuild` is the CLI command that makes this promise actionable: it
  deletes and recreates `index.sqlite3` from scratch, then re-syncs it
  from `events/`, never touching anything else.
* **Schema upgrades happen automatically, via Alembic** (`storage/migrate.py`
  + `storage/migrations/`), on `TimelineDatabase.initialize()` -- which runs
  on essentially every CLI invocation. A brand-new workspace gets
  `create_all()` + stamped straight to head; a workspace from before this
  mechanism shipped is detected (no `alembic_version` table, but app tables
  present) and stamped to a `0001_baseline` migration capturing that exact
  pre-Alembic shape, then upgraded forward; a normal workspace just runs
  `alembic upgrade head`, with a fast-path skip once already current so
  the common case stays cheap. No manual migration step is ever required.
  `workspace reindex` still runs `check_schema_drift()` (an independent,
  older sanity check) as a fallback -- with Alembic authoritative, drift
  detected there means a migration was somehow skipped or the database was
  hand-edited, not the routine upgrade path; `--rebuild` is the fix for
  that or for index corruption unrelated to schema.

Every write to storage that matters forensically goes through
`storage/repository.py::Repository`, which is the only module allowed to
touch both `WorkspacePaths` and `TimelineDatabase` directly. No other module
should reach into either one on its own.

### Workspace layout

```
workspace/
  ingest/        raw ingest manifests (one per ingest run)
  originals/     byte-for-byte copies of source media, read-only
  normalized/    normalization manifests (sync-corrected timestamps)
  windows/       window manifests
  events/
    <event_id>/
      event.json           <- always mirrors the latest revision (not frozen)
      revisions/
        0000_initial_generation.json   <- frozen at write time, never touched again
        0001_review_accept.json
        0002_enrichment.json
        0003_preserved.json
      clips/
        front__a1b2c3d4.mp4            <- ffmpeg-cut derived clips (source-hashed)
  preserved/
    <event_id>/            <- full frozen copy of the event dir + review log + confirmation
  review/
    <event_id>.jsonl       <- append-only human review actions
  exports/
    event_<id>_<ts>.zip
    timeline_<ts>.csv
  patterns/
    <timestamp>.json       <- pattern-analysis snapshots
  recognition/
    faces/
      model.yml             <- persistent LBPH clusterer state
      model.labels.txt       <- cluster-id <-> internal-label mapping
      crops/                 <- small face crop JPEGs
    plates/
      crops/                 <- small plate crop JPEGs
  transcripts/
    <event_id>.json          <- AudioTranscript
    <event_id>.llm.json      <- LLMEnrichment, only if enrichment.cloud is enabled
  for_review/
    <clip_id>_<name>.mp4     <- convenience symlinks to reviewable originals, never authoritative
  pending_deletion/
    <name>.mp4                <- benign originals awaiting confirm-deletion (physically moved here)
  deletion_log.jsonl          <- append-only, written before any original is actually deleted
  identity_merge_log.jsonl    <- append-only, human-confirmed face/plate identity merges
  event_video_purge_log.jsonl <- append-only, event-scoped video-evidence purge records
  timeline/
    index.sqlite3          <- the index, not a source of truth
```

The `recognition/` tier (face/plate crops and clusters) is stored
authoritatively in SQLite + small crop files rather than following the
filesystem-JSON-with-revisions pattern events use. This is a deliberate
exception, not an inconsistency: individual face/plate observations are
too numerous for a revision-per-observation file to be practical, and this
data is explicitly *non-forensic-primary* -- a re-identification
convenience layer for the user, not evidence in the same sense as an
event's `signals`/`hypotheses`/`scoring`, which remain fully covered by the
revisioning guarantees described above. See `docs/local-ai.md` for the full
design.

## Event revisioning (why `event.json` never goes stale)

`EventRecord` is the canonical forensic artifact, and it is **versioned and
revisioned, never mutated in place**. Concretely:

* `events/<id>/event.json` is a convenience pointer that always mirrors the
  *latest* revision. It is the only file in an event's directory that is
  ever rewritten.
* `events/<id>/revisions/000N_<reason>.json` is the true append-only
  history. Each file is made read-only at write time (`chmod`) and is never
  edited or deleted again. Each revision's `previous_revision_hash` field
  is the SHA-256 of the previous revision's canonical JSON, forming an
  inspectable hash chain.
* `storage/repository.py::Repository.save_event_revision()` is the *only*
  code path allowed to change an already-written event. It loads the
  current latest revision, applies a partial update, bumps `revision`,
  stamps `revised_at`/`previous_revision_hash`, and writes a new frozen
  revision plus refreshes the pointer.

This exists to fix a real bug found during the v1.0 audit: preservation and
review actions used to update *only* the SQLite index, leaving `event.json`
permanently showing `preservation_status.state = "pending"` even after an
event had been preserved. `save_event_revision` is what both
`PreservationOrchestrator` and `ReviewService`/`Repository.append_review_action`
use now, so the filesystem and the index can never disagree about an event's
current state.

## Time synchronization (`normalize/sync.py`)

See the module's docstring for the full algorithm; in short:

0. Partition sessions by `site_id` before anything else (see
   `docs/local-ai.md`'s "Security camera support" section) -- only cameras
   sharing a site are ever candidates for the grouping below; independent
   cameras with unrelated clocks (e.g. a neighbor's security camera and
   your dashcam) are never cross-synced just because their recording times
   happened to overlap.
1. Group each camera's clips into **sessions** (gaps larger than
   `sync.session_gap_seconds` start a new session -- this models power-on
   cycles, since a dashcam's clock is only as good as its last boot).
2. Group sessions from different cameras into a **sync group** when their
   time ranges overlap.
3. Within a group, deterministically pick a **reference session** (highest
   average timestamp confidence, ties broken alphabetically by camera id).
4. Align every other session to the reference by shifting its start
   (`offset_seconds`) and estimating drift from the proportional difference
   in session span (`drift_seconds_per_hour`).

This is a heuristic, not ground-truth clock recovery -- there is no
audio/video cross-correlation. That is an intentional, documented
limitation (see `docs/limitations.md`) and a natural extension point for a
future `DetectorPlugin`-style signal-correlation approach. Every corrected
timestamp is stored alongside its original (`NormalizedClip.clip.observed_start`
vs. `NormalizedClip.corrected_start`), and every correction carries a
plain-language rationale.

## Detection: real analysis with a fixture escape hatch

Motion (`detection/motion.py`), audio (`detection/audio.py`), and object-hint
(`detection/object_detection.py`) detectors all follow the same pattern:

1. If a `<file>.samples.json` sidecar exists next to the source media *and*
   `detection.use_fixture_signals_when_available` is true (the default),
   use it verbatim. This exists for deterministic test fixtures and manual
   calibration.
2. Otherwise, run real analysis:
   * Motion: OpenCV grayscale frame differencing
     (`detection/video_analysis.py`), sampled at `motion_sample_rate_hz`.
   * Audio: ffmpeg extracts the audio track to mono 16 kHz WAV, then scipy
     computes a rolling RMS envelope (`detection/audio_analysis.py`).
   * Object hints: contour extraction on the same frame-difference mask
     used for motion, reported as `unclassified_moving_region` bounding
     boxes -- an explainable heuristic, not a classifier. Real ML object
     detection is an intentional extension point via `DetectorPlugin`, kept
     out of the built-in path per the project's ML-avoidance-by-default
     directive.

All three are deterministic: given the same file bytes and the same code,
they produce the same output every time (verified against generated fixture
media during development -- see `tests/unit/test_video_analysis.py` and
`tests/unit/test_audio_analysis.py`).

## Event assembly and window-overlap merging

`WindowingService` intentionally generates *overlapping* sliding windows
(`window_stride_seconds < window_duration_seconds`) so a signal near a
window boundary is never awkwardly split. Left alone, that overlap would
produce several near-duplicate events for one continuous span of activity.
`AnalysisPipeline._cluster_overlapping_windows` fixes this: after inference
runs per-window, windows-with-signals are merged into clusters wherever
their time ranges overlap, and each cluster becomes exactly one
`EventRecord`. See `tests/unit/test_event_clustering.py` for a focused
regression test and
`tests/integration/test_pipeline_e2e.py::test_overlapping_windows_do_not_produce_duplicate_events`
for an end-to-end one.

## Storage lifecycle: ingest modes, triage, event-video purge, and deletion

Solves a specific, real problem: a 256GB SD card import is mostly
uneventful driving, and nobody wants to keep that at full resolution
forever.

### Ingest storage modes

`core/config.py::StorageConfig.ingest_mode` (`copy | move | reference`,
overridable per-run via `ingest --mode`) controls what `ingest/service.py`
does with each source file: duplicate it into `originals/` (`copy`,
safest, default), relocate it there (`move`, frees the source
immediately), or leave it exactly where it was and just index that
location (`reference`, zero extra disk use, but the workspace now depends
on that location staying available). `MediaClip.ingest_mode` records which
one produced a given clip, since it changes what "deleting" that clip
means later -- see below.

`ingest/service.py`'s traversal is symlink-loop-safe (it tracks every
resolved real directory path it has already walked and refuses to
descend into one again, rather than `rglob`'s unguarded traversal) and,
by default, content-addressed-deduplicating: a file whose SHA-256 already
matches a previously-indexed, still-present clip is skipped (logged, not
re-copied) rather than silently producing a redundant second copy. Never
touches the source file or the earlier copy either way -- see invariant 1.
Controlled by `StorageConfig.dedupe_on_ingest` (default `true`).

### Triage classification

`core/triage.py::TriageService` classifies every ingested clip after
`analyze` (automatically, unless `lifecycle.auto_triage_after_analyze`
is disabled):

* **reviewable** -- contributed to at least one `Signal` in at least one
  `Event`. The original is never moved (moving it would break the
  `evidence_references` paths already embedded in that event's revision
  history); a best-effort, non-authoritative symlink appears under
  `for_review/` for convenient browsing, and `triage list --state
  reviewable` is the authoritative listing.
* **benign_pending_deletion** -- contributed to zero signals across the
  full analysis. A `copy`/`move`-mode clip is safe to physically move
  (nothing in any `event.json` can reference a clip that produced no
  signals, by construction), so it moves to `pending_deletion/` awaiting
  an explicit `triage confirm-deletion`. A `reference`-mode clip is left
  exactly where it was -- moving it would mean copying it into the
  workspace, defeating the entire point of that mode -- and is classified
  in place instead.

Deletion itself is never implicit. `confirm_deletion` requires an actor
name, re-verifies the file's current hash against the indexed hash
(refusing if they don't match, in case something touched the file), writes
a `DeletionRecord` to the append-only `deletion_log.jsonl` *before*
unlinking the file, and only then deletes the bytes. If the process dies
between those two steps, the log still shows deletion was confirmed even
though the unlink didn't finish -- the safer failure mode for a forensic
system is an over-cautious log entry, never a silent deletion with no
record. Deleting a `reference`-mode clip deletes something outside the
workspace's own storage (the original source location, e.g. an SD card),
not a workspace-owned copy, so `confirm_deletion` additionally requires
`acknowledge_external_deletion=True` (CLI: `--acknowledge-external`) for
those specifically. See `docs/local-ai.md` for the CLI workflow and
`docs/chain-of-custody.md` for how `DeletionRecord` fits the broader
provenance model.

### Event-video purge

A coarser, event-scoped sibling to clip-level deletion:
`TriageService.purge_event_video` deletes an event's own derived clips
(`events/<id>/clips/`) and cascades to contributing original clips --
*only* for originals no other still-unpurged event references, computed
fresh by matching `Signal.evidence_references[].sha256` across every
event, not just the one being purged -- while leaving `event.json`,
signals, hypotheses, scoring, chain of custody, and full revision history
untouched (only `EventRecord.video_purged_at` is set, via a new revision).
Refuses to run unless the event has already been preserved, unless
`force=True` is passed, since otherwise the purge would be destroying the
only copy of that video that ever existed. Logged to a dedicated
append-only `event_video_purge_log.jsonl`
(`schemas/lifecycle.py::EventVideoPurgeRecord`), separate from but
cross-referencing the per-clip `DeletionRecord`s any cascaded original
deletions still produce in the usual deletion log.

## Enrichment: local face/plate/voice/vehicle detection, transcription, optional cloud LLM

A separate, optional pipeline stage (`enrichment/service.py::EnrichmentService`,
run via the `enrich` CLI command) that operates only on the derived clips
of events that `analyze` already built -- never on benign footage. Six
independently-toggleable capabilities: face detection + local
re-identification (real, on by default, zero setup), license plate
detection + OCR (real, on by default, zero setup -- as of this pass,
detection combines OpenCV cascades, MSER blob detection, and a
rotation-aware contour heuristic, see `docs/local-ai.md` for the
before/after evidence), voice activity detection + classical MFCC
voiceprinting (on by default, zero setup, but a meaningfully weaker
fingerprint than face/plate -- see `docs/local-ai.md`), local YOLO-ONNX
vehicle detection (optional, needs a user-supplied model), local Whisper
transcription (optional, needs a one-time model download), and cloud LLM
transcript analysis (optional, off by default, the only network-calling
feature in the project). Full design, config, and CLI reference in
`docs/local-ai.md`; ethical/legal scope boundaries in
`docs/forensic-considerations.md`'s "Recognition data: scope and intent."

New signals enrichment discovers are appended to the event via a new
revision, never by re-running scoring -- see `docs/local-ai.md` for why.

### Identity linking (`core/recognition.py`)

`FaceCluster`, `PlateRecord`, and `VoiceCluster` each have an optional
`merged_into` pointer. A human uses
`recognize faces-merge`/`plates-merge`/`voices-merge` to declare two
clusters/records the same person, vehicle, or speaker; the source is never
edited or deleted, only aliased. `RecognitionService.resolve_face_identity`/
`resolve_plate_identity`/`resolve_voice_identity` follow that pointer to a
canonical root UUID (cycle-safe, both by construction -- a merge that
would create a cycle is rejected at merge time -- and defensively at
resolution time), and `get_face_identity`/`get_plate_identity`/
`get_voice_identity` aggregate stats and sightings across every
cluster/record resolving to that root, computed fresh on every read rather
than rewritten into stored counters. Every merge is permanently logged to
`workspace/identity_merge_log.jsonl`. Full CLI reference and the privacy
framing in `docs/local-ai.md`'s "Linking sightings to the same person or
vehicle."

### Automated merge suggestions

`RecognitionService.suggest_face_merges`/`suggest_plate_merges`/
`suggest_voice_merges` scan for pairs of un-merged clusters/records that
look like fragments of the same identity and write a `MergeSuggestion`
row (`pending` by default) rather than merging anything -- human-in-the-
loop by construction, not by convention. Confirming a suggestion routes
through the exact same `merge_faces`/`merge_plates`/`merge_voices` calls a
manual merge would use, so it produces the identical `IdentityMergeRecord`
audit trail either way. See `docs/local-ai.md`'s "Automated merge
suggestions" section for how each entity type's candidates are generated.

## Plugins

Three extension points, all `typing.Protocol`s in `plugins/base.py`, all
loaded via `importlib.metadata.entry_points()` in `plugins/registry.py`:

* `DETECTOR_PLUGIN_GROUP` (`gaggle.plugins.detectors`)
* `INFERENCE_RULE_PLUGIN_GROUP` (`gaggle.plugins.inference_rules`)
* `EXPORTER_PLUGIN_GROUP` (`gaggle.plugins.exporters`)

`load_plugins()` isolates failures: a plugin that raises on load or
instantiation is logged and skipped, never allowed to crash the built-in
pipeline. See `docs/plugin-authoring.md`.

## Why not a literal multi-package monorepo?

The original project brief called for a `/packages/*` monorepo with one
installable package per subsystem. During the v1.0 pass that structure was
found to exist only as an unused, out-of-sync duplicate of the real `src/`
tree (the actual installed package, per `pyproject.toml`'s
`[tool.setuptools.packages.find] where = ["src"]`) and was removed. A single
package with strict submodule boundaries gets the same modularity and
independent-testability benefits without the packaging overhead
(inter-package version pinning, N separate `pyproject.toml` files) that a
project this size doesn't need. If a subsystem ever needs to ship and
version independently (e.g. a heavy ML detector plugin), the plugin system
is the intended seam for that, not a package split.
