# AGENTS.md

Working reference for anyone -- human or AI agent -- developing
`gaggle`. Read this before making structural changes. It covers
what the project is, the invariants that must never be violated, how to
build/test/run it, what's actually implemented as of this pass, and what's
still open.

## What this project is

An offline-first, forensic-oriented, **camera-agnostic** encounter and
incident analysis system, named Gaggle (CLI command `gaggle`). It began as
a dashcam-only tool; as of the 1.7 pass it's broadening into a general
personal encounter-intelligence system covering any camera a user owns --
dashcams, indoor security cameras, outdoor security cameras, doorbells,
NVR exports -- ingested and analyzed through the same pipeline. Dashcams
remain a first-class, fully-supported source among several, not a legacy
special case being phased out. Full philosophy in
`docs/forensic-considerations.md` and `docs/architecture.md`; the one-line
version: **deterministic pipelines, explainable outputs, immutable
evidence, human-in-the-loop review, no cloud dependency, ever.**

It is explicitly *not*: a cloud/SaaS product, an autonomous
guilt-determination system, a networked/mass surveillance tool, or an
ML-first black box. Automated output is always a hypothesis with an
attached confidence formula, never a conclusion. Broadening to more camera
sources does not change this -- it's still local-only, user-controlled,
and scoped to a person's own footage (see invariant 11 and
`docs/forensic-considerations.md`'s "Recognition data: scope and intent"),
never identity resolution or networked lookup.

## Repo history context (read this once, then ignore it)

This repo was originally scaffolded by a prior agent from a detailed spec.
That first pass produced a real schema layer and a real pipeline skeleton,
but a substantial amount of it was facade: the "detection" subsystem only
ever read hand-authored JSON fixtures and never touched real video/audio
despite declaring opencv/scipy as dependencies; the shipped "sample media"
was a placeholder text file; time synchronization was a hardcoded no-op;
the plugin system existed but was never invoked; there was no exporter at
all despite it being a stated success criterion; and a real bug meant
`event.json` went stale after preservation or review (updates only landed
in SQLite, inverting the hybrid-storage design). A dead, out-of-sync
duplicate `packages/`/`apps/` tree also existed alongside the real `src/`
layout the build actually used, and the Dockerfile copied the dead tree
instead of `src/`.

A full v1.0 pass (this one) fixed all of the above: real ffprobe/OpenCV/
scipy-backed ingestion and detection (with a sidecar-fixture override kept
for deterministic testing), a real time-sync algorithm, event revisioning
that keeps `event.json` from ever going stale, plugin wiring, a full
exporter subsystem, richer timeline/pattern querying, a review UI with real
synchronized playback, an expanded test suite, and this document. The dead
`packages/`/`apps/` trees were removed; `src/` is the one canonical layout.
**If you ever see a reference to `packages/*` or `apps/*` anywhere
(old docs, old issues, muscle memory from the spec) it is stale -- the
canonical package is `src/gaggle/`.**

A subsequent pass added the `enrichment/` package (local face/plate
re-identification, optional local vehicle detection and transcription,
optional cloud LLM transcript analysis -- all off-by-default except
face/plate, which are local and zero-setup), the recognition database, and
`core/triage.py` (the storage-lifecycle workflow: classify footage as
benign vs. reviewable, and only ever delete originals through an explicit,
logged, actor-attributed confirmation). This was scoped deliberately around
personal re-identification within the user's own footage, not identity
resolution or networked surveillance -- see invariant 11 below and
`docs/forensic-considerations.md`'s "Recognition data: scope and intent"
before extending anything in `enrichment/`.

A 1.1 pass added `core/recognition.py` (`RecognitionService`): a
`merged_into` pointer on `FaceCluster`/`PlateRecord` so a human can declare
two clusters/records the same person or vehicle (aliasing, never editing
or deleting either), cycle-safe resolution to a canonical identity UUID,
aggregated sightings/search across a merge group, and a permanent
append-only `identity_merge_log.jsonl`. This pass also fixed two real
bugs found via actual user testing rather than review here, both worth
knowing about if you're touching `storage/database.py`:

1. `DetachedInstanceError` in `TriageService.classify_all()` (and latent in
   every other `TimelineDatabase` `list_*`/`get_*` caller) -- SQLAlchemy
   expires loaded row attributes on commit by default, and every one of
   these methods returns rows for the caller to read *after* the
   short-lived query session has already closed. Fixed via
   `expire_on_commit=False` on the session factory; see the docstring on
   `TimelineDatabase.session()` and don't remove that setting.
2. A prior edit to `storage/database.py` had silently merged
   `FaceClusterRow` and `FaceObservationRow` into one broken class
   (duplicate `__tablename__`, a duplicate `cluster_id` column that
   clobbered the primary key definition, and `FaceObservationRow` not
   existing as its own class at all) -- caught by an AST-based
   "does every imported name actually exist where it's imported from"
   sweep before it shipped, not by execution. If you're ever unsure
   whether a large `str_replace` edit to this file landed cleanly, rerun
   that kind of check rather than trusting a diff by eye; see the
   "Development environment note" below for the actual check used.

A 1.1.1 pass fixed a third real bug, again found via actual user testing:
`enrich` crashed with `ValidationError: timestamp must be timezone-aware`
the first time it re-read a previously-stored `FaceCluster`/`PlateRecord`
from SQLite (exactly the read-then-reconstruct pattern
`EnrichmentService` uses to preserve prior stats across observations).
Root cause: SQLite has no native timestamp-with-timezone type, and
SQLAlchemy's SQLite dialect silently returns *naive* datetimes on read
regardless of what was written with `DateTime(timezone=True)` -- a
separate, well-known limitation from the one fixed in 1.1's bug #1 above,
affecting all 16 datetime columns in the schema, not just the two that
happened to crash first. Fixed with a custom `UTCDateTimeColumn` type
(`storage/database.py`) that re-attaches UTC on every read and normalizes
to UTC on every write, applied to every datetime column in the schema --
a systemic fix, not a per-call-site patch. See
`docs/limitations.md`'s bug case study for the full writeup and why this
project's stated confidence levels distinguish "executed against real
data" from "reviewed but not run."

A 1.1.2 pass fixed a usability/efficiency issue, again found via real
usage: with `enrichment.plate.enabled` (the default) but no `tesseract`
binary on `PATH`, `_run_plate_recognition` attempted and failed OCR
separately for *every detected plate-shaped region in every sampled
frame* -- potentially hundreds of doomed subprocess spawns per `enrich`
run, each logging an identical `plate_ocr_failed` warning. Fixed by
checking `tesseract_available()` exactly once per `EnrichmentService`
instance (`_check_tesseract_once`, mirroring the `_vehicle_load_attempted`/
`_transcriber_load_attempted` pattern already used for the other optional
capabilities) and skipping plate recognition cleanly for the rest of the
run with one clear, actionable warning if it's missing. Also added
platform-specific tesseract install instructions (Windows via the
UB-Mannheim build, since `apt-get install tesseract-ocr` -- the only
instruction previously given -- is meaningless on Windows) to
`docs/local-ai.md` and `docs/developer-setup.md`.

A 1.2 pass added three real, requested capabilities on top of real user
feedback about storage flexibility and review workload:

1. **Ingest storage modes** (`copy | move | reference`,
   `core/config.py::StorageConfig.ingest_mode`, CLI `ingest --mode`).
   While building this, a real design gap was caught before shipping:
   `TriageService`'s benign-clip classification would have silently
   *copied* a `reference`-mode file into the workspace the moment it was
   classified benign, defeating the entire point of the mode. Fixed by
   leaving `reference`-mode clips at their external location and gating
   `confirm_deletion` on those specifically behind
   `acknowledge_external_deletion` (CLI: `--acknowledge-external`), since
   deleting one deletes the user's actual source file, not a
   workspace-owned copy.
2. **Event-video purge** (`core/triage.py::TriageService.purge_event_video`/
   `purge_event_video_bulk`, CLI `triage purge-event-video`/
   `purge-reviewed`). Deletes an event's own derived clips and cascades to
   contributing originals -- only when no other still-*unpurged* event
   references them, recomputed fresh against every event's signals, not
   cached -- while leaving `event.json`'s signals/hypotheses/scoring/
   chain-of-custody/revision-history completely untouched. Gated behind
   the event already being preserved, unless `force=True`. New append-only
   `EventVideoPurgeRecord`/`event_video_purge_log.jsonl`, mirroring the
   `DeletionRecord` pattern.
3. **Plate false-positive cleanup automation**
   (`core/recognition.py::RecognitionService.cleanup_duplicate_plate_observations`,
   CLI `recognize plates-cleanup`), plus a config-driven garbage-OCR-text
   pre-filter (`enrichment.plate.min_plate_text_length`/
   `max_plate_text_length`) that discards obviously-invalid OCR results
   before they're ever stored. The cleanup pass groups observations by
   (event, plate text), clusters by time proximity, and marks all but the
   highest-confidence observation per cluster `duplicate_suppressed` -- a
   status distinct from `user_rejected` specifically so it's always clear
   whether a machine or a human made a given call. Never touches an
   observation a human has already confirmed or rejected.

New docs `docs/getting-started.md` (tiered minimal/recommended/full setup
guide) and `docs/pipeline-walkthrough.md` (the complete narrated workflow
from SD card to a storage-optimized final state) were added as the
primary deliverable of this pass -- if you're looking for "how does a user
actually use this end to end," those two are the canonical answer, more
so than this file.

A 1.3 pass responded to real accuracy feedback and a large capability
request, expanding this toward the project's stated "Flock-alike, but for
dashcam footage" scope:

1. **Plate detector accuracy.** Real user feedback: the detector grabbed
   "a lot of junk" and missed real plates, especially at angles, with no
   way to check what it was actually doing. Fixed with real, reproduced
   evidence, not just parameter tuning: added MSER (Maximally Stable
   Extremal Regions) blob detection as the primary strategy after
   confirming the old contour-only heuristic produced a 0.00 IoU complete
   miss on a synthetic cluttered scene versus 0.97 IoU from MSER on the
   same scene; made the contour heuristic's aspect-ratio check
   rotation-aware (`cv2.minAreaRect` instead of an axis-aligned bounding
   box) for angled plates, verified against a 20-degree-rotated synthetic
   plate. Also added `recognize plates-debug`, which renders every
   candidate region on real footage frames so the "how do I check this is
   working" problem has a direct answer.
2. **Automated merge suggestions** (`core/recognition.py`,
   `suggest_face_merges`/`suggest_plate_merges`/`suggest_voice_merges`,
   CLI `suggest-merges`/`merge-suggestions*`). Closes a gap explicitly
   flagged in the 1.1 pass's "Known gaps" list: fragmented identities (the
   same person/plate/voice split across multiple clusters) previously
   required the user to notice and manually merge. Now a scan flags
   likely candidates into a `pending` queue; nothing merges automatically,
   confirming a suggestion routes through the same `merge_faces`/
   `merge_plates`/`merge_voices` calls a manual merge uses, so the audit
   trail is identical either way.
3. **Voiceprinting** (`enrichment/voice.py`), built from scratch using
   only numpy/scipy since no pretrained speaker-embedding model is
   reachable without network access: classical voice-activity-detection
   (energy + spectral-flatness gating, verified to correctly reject a
   pure tone standing in for horn/engine noise) and MFCC-based
   voiceprinting (mel filterbank, DCT, the full classical pipeline,
   implemented by hand). A real mistake was caught and fixed during this
   same pass, not glossed over: an initial default clustering threshold
   of 0.15 was chosen from a single clean-signal test and turned out to
   be too loose once noise was added, producing an actual false merge
   between two different synthetic voices in testing. Caught, re-measured
   across 8 noise seeds for a real distribution (same-voice ~0.0002-0.0003,
   different-voice ~0.14), and re-set to a conservative 0.05 before
   shipping. Explicitly and repeatedly documented as a meaningfully
   weaker fingerprint than face/plate recognition, validated only against
   synthetic signals, never real recorded speech -- see
   `enrichment/voice.py`'s module docstring and `docs/limitations.md`.
4. **Gap closures**: `recognize faces-cleanup` (duplicate-observation
   automation, previously plates-only) and `recognize plates-reject` (a
   dedicated reject action, previously missing) were both added, closing
   two items from the 1.2 pass's own "Known gaps" list.
5. **Explicitly not implemented, and explicitly documented as such**
   rather than left ambiguous: vehicle re-identification by visual
   description (color/body-shape) without a legible plate. The user's
   framing name-checked this; building it responsibly would mean a new
   appearance-fingerprint subsystem, which wasn't attempted this pass in
   favor of finishing (and properly testing) everything above. See
   `docs/limitations.md`.

Two real implementation bugs were caught and fixed *during* this pass,
before they shipped, both by the same discipline described in the
"Development environment note" below: a `Signal.signal_type` Literal that
didn't yet include `"voice_detection"` (would have crashed the first real
`enrich` run with voice enabled), and a naming collision --
`enrichment/voice.py`'s result dataclass was originally named
`VoiceObservation`, identical to the SQLAlchemy/pydantic schema class of
the same name, breaking the convention face/plate deliberately follow
(`DetectedFace`, `PlateRegion`, never reusing a schema class's name for an
internal dataclass) -- renamed to `VoicePrintResult`.

Two of this pass's newest modules -- `enrichment/voice.py` and the MSER
additions to `enrichment/plate.py` -- have **zero pydantic/SQLAlchemy
dependency**, unlike almost everything else in this codebase. That made
it possible to actually *execute* their test suites for real in this
sandbox (via a small `pytest.raises`/`pytest.mark.skipif` shim, since real
`pytest` itself still isn't installable) rather than only hand-tracing
them -- see `tests/unit/test_voice_recognition.py` and the new tests
appended to `tests/unit/test_plate_recognition.py`. If you're adding a new
pure-computation module, keeping it free of pydantic/SQLAlchemy imports
the way these are is worth doing deliberately: it's the difference between
"verified by execution" and "verified by careful reading" for anyone
picking this repo up in an environment like this one.

A 1.5 pass closed the remaining `docs/limitations.md` gaps, including the
flagship one: vehicle re-identification by visual description
(`enrichment/vehicle_appearance.py`, classical hue/saturation histogram +
aspect-ratio fingerprint, mirroring `enrichment/voice.py`'s pattern end to
end -- pure module, incremental clusterer, database rows, `RecognitionService`
methods, `recognize vehicles-*` CLI, merge-suggestion support). Building
its real end-to-end test (a 2-cluster merge-suggestion scenario) surfaced
a genuine, previously-undiscovered bug affecting **already-shipped** 1.1/1.3
functionality, not just the new module: `suggest_face_merges` and
`suggest_voice_merges` could never actually produce a suggestion once 2+
clusters existed. Both queried `predict_nearest_cluster` with a cluster's
*own* representative crop/centroid, which -- confirmed empirically, not
assumed -- always matches itself at distance 0 (an LBPH model trained on
an exact image reliably re-predicts that same image back to its own
label; a centroid compared to itself is trivially distance 0 by
definition). The existing code then discarded exactly that self-match
via `if matched_cluster_id == cluster.cluster_id: continue`, which is
correct as a safety check but, since `predict()`/a plain nearest-centroid
search only ever return the single closest match, threw away the search
entirely -- the real nearest *other* cluster was never found, silently
producing zero suggestions no matter how close two real clusters were.
Fixed by having `predict_nearest_cluster` accept an `exclude_cluster_id`
and exclude it *during* the search rather than discarding a match found
after the fact: for the centroid-based clusterers (voice, vehicle
appearance) this is a plain loop skip; for face's OpenCV LBPH model,
which only exposes a single best match via `.predict()`, this needed
`predict_collect()` with a `cv2.face.StandardCollector` (confirmed
present at runtime, another opencv-contrib stub gap) to see every trained
label's distance in one pass and pick the best non-excluded one. Caught
and fixed via real test execution -- see
`tests/unit/test_face_recognition.py::test_predict_nearest_cluster_excludes_the_queried_clusters_own_match`
and `tests/unit/test_merge_suggestions.py::test_suggest_face_merges_flags_a_second_close_cluster`
for the reproduction -- exactly the kind of thing hand-tracing reliably
misses, consistent with every other bug documented in this section.

The same 1.5 pass also closed the vehicle telemetry detector
(`detection/telemetry_analysis.py` + `detection/telemetry.py`),
mirroring `motion.py`/`audio.py`'s sidecar-then-real-analysis structure
and hand-implementing haversine speed / initial-bearing heading from GPX
track points -- validated against a synthetic fixture track with
precisely-computed coordinates (`tests/fixtures/sample_track.gpx`) and
confirmed end to end with a real `ingest` -> `analyze` run against
`examples/sample_media` plus a colocated GPX file
(`DASHCAM_SENTINEL__detection__use_fixture_signals_when_available=false`
to force the real GPX path rather than the existing `sample_metrics`
sidecar, which -- worth knowing if you extend this further --
short-circuits *all* fixture-backed detectors including telemetry the
moment it's present, even for series keys it doesn't itself define; this
is existing, intentional behavior shared with `motion.py`/`audio.py`,
not something new to telemetry). Ingest-side wiring
(`ingest/service.py::_find_gpx_track`) matches a `.gpx` file by presence
in the same camera directory as each source clip, not by filename or
per-clip time-range correlation -- see `docs/limitations.md` for that
scope boundary.

The 1.5 pass's last gap was Ed25519 signing of the revision hash chain
(`core/signing.py`, `workspace signing-init`/`signing-status`,
`EventRecord.revision_signature`). One design detail worth flagging for
future work: `Repository.save_event_revision` explicitly resets
`revision_signature` to `None` on the `model_copy(update=...)` call
*before* re-signing -- without that, a new revision built from
`current.model_copy(...)` would otherwise inherit the *previous*
revision's signature (computed over different content) into the new one
whenever signing is off or unavailable for that particular write, which
would look like a valid-but-actually-wrong signature sitting on disk.
Caught before shipping by reasoning through the `model_copy` semantics,
not by a failing test -- confirmed with
`tests/unit/test_signing.py::test_a_revision_written_after_signing_is_turned_off_has_no_stale_signature`.
Verified end to end with a
real `workspace signing-init` -> `ingest` -> `analyze` (signed revision
on disk) -> `export event` -> `scripts/verify_export_bundle.py` run
against `examples/sample_media`, confirming a real Ed25519 signature
round-trips through the whole pipeline, not just through the signing
module's own unit tests.

A 1.6 pass audited the codebase against its own founding specification
(a much older design brief) to catch any good ideas from it that had
been missed as the project evolved, rather than to roll anything back.
Closed: a fourth pattern-detection type (repeated vehicle appearances,
reusing the vehicle-appearance clustering infrastructure the 1.5 pass
built), a fourth plugin type (`ReviewExtensionPlugin`, completing the
originally-specified detector/rule/exporter/review-extension set),
actual use of the previously-unused `hypothesis` dev dependency for
property-based tests on this project's pure math, GPU/CUDA setup
documentation, a real gap in SQLite upgrade safety (`create_all()` never
alters an existing table -- `workspace reindex --rebuild` and
`TimelineDatabase.check_schema_drift()` close it), and a genuinely new
detection capability: `OpticalFlowDetector`, using dense Farneback flow
and flow-field divergence to detect "something is rapidly approaching
the camera" -- a cue frame-differencing motion detection cannot capture
by construction. One idea from the original spec was deliberately left
out: distributed/parallel processing across clips/detectors, since it
directly conflicts with this project's own "avoid nondeterministic
ordering" principle without a carefully designed deterministic-merge
strategy first -- noted as considered-and-deferred, not silently
dropped, in case a future pass wants to revisit it properly.

The 1.6 pass's real CLI smoke test for the new `repeated_vehicle_appearance`
pattern surfaced an unrelated, pre-existing bug: `patterns analyze`
(`cli/app.py::patterns_analyze`) built its output filename from
`datetime.now(tz=UTC).isoformat()`, which contains `:` characters --
invalid in a Windows filename, so the command crashed with `OSError:
[Errno 22] Invalid argument` on every real invocation on Windows.
Untouched by any existing test because `test_patterns.py` (correctly)
tests `PatternAnalysisService.analyze()` directly, never through the CLI
layer, so nothing had exercised this exact line for real before. Fixed
by switching to the colon-free `%Y%m%dT%H%M%SZ` format
`export/service.py` already uses for its own output filenames -- a good
reminder that a component being unit-tested doesn't mean the thin CLI
glue around it has ever actually run.

**The 1.7 pass** renamed the project from dashcam-sentinel to Gaggle and
broadened it from a dashcam-only tool to a camera-agnostic
encounter-intelligence system covering any camera source (security
cameras, doorbells, NVR exports) alongside dashcams, plus added real
Alembic-based SQL migrations so the schema can keep growing without
breaking existing workspaces. Check `git log`/the status tracker below for
exactly how far it's gotten before trusting this as complete -- only the
final full-suite verification pass is expected to remain by the time you
read this.

The rename: `src/dashcam_sentinel/` → `src/gaggle/`, package/CLI name
`dashcam-sentinel` → `gaggle` in `pyproject.toml`, a mechanical find-replace
of exactly the four project-name forms (`dashcam_sentinel`,
`dashcam-sentinel`, `Dashcam Sentinel`, `dashcam sentinel`) across every
tracked *and* untracked file -- deliberately **not** a blind replace of the
word "dashcam" itself, which still correctly describes dashcam footage/the
dashcam detector/etc. throughout, since dashcams remain a real, supported
source type. Verified via a real `gaggle workspace init` -> `ingest` ->
`analyze` -> `enrich` run against `examples/sample_media` under the new
command name, full `ruff`/`mypy`/`pytest` green. The stale `dashcam-sentinel`
pip distribution (a leftover editable install pointing at a module that no
longer exists on disk) was explicitly uninstalled, not just shadowed by the
new one -- if you ever see both `gaggle` and `dashcam-sentinel` registered
via `pip list` in a dev environment picking this back up, that's exactly
this same leftover-registration issue recurring, fix it the same way.

The Alembic migration infrastructure (`storage/migrate.py` +
`storage/migrations/`): `TimelineDatabase.initialize()` now branches three
ways depending on what the sqlite file looks like -- brand-new (fast-path
`create_all()` + stamp straight to head), legacy pre-Alembic (stamp to a
hand-verified `0001_baseline` migration capturing the exact 14-table shape
that existed the moment Alembic was introduced, then upgrade forward), or
already-tracked (`alembic upgrade head`, with a fast-path skip once already
current so the common case -- this runs on nearly every CLI invocation --
stays cheap). `workspace reindex --rebuild` stays as a manual fallback for
drift/corruption, not the routine upgrade path anymore. Every migration
after the baseline is generated for real via `alembic revision
--autogenerate` against a scratch database, then hand-fixed (revision id,
missing imports, `ruff format`) rather than trusted blindly -- see
`0002_camera.py` and `0003_encounter.py` for the pattern to follow.

The Camera entity (`schemas/camera.py`, `core/cameras.py::CameraRepository`,
`camera list/register/update`): optional metadata (source type,
indoor/outdoor, `site_id`) layered on the pre-existing free-form
`camera_id` string, never required. `site_id` is what scopes
`normalize/sync.py`'s cross-camera time-sync grouping -- cameras sharing a
site are candidates for alignment, cameras with no shared site never are,
which matters once footage from unrelated cameras (a neighbor's security
camera, your dashcam) can end up in the same workspace.
`ingest/service.py::IngestService` auto-derives a deterministic
`default_site_id` (a hash of the source root) once per ingest run and
auto-registers every first-seen `camera_id` under it, so an existing
dashcam SD-card workflow (front/rear/interior in one directory) keeps
cross-syncing with zero configuration, while a security camera ingested in
a separate run is naturally isolated. Two example config profiles
(`examples/config/security-outdoor.yaml`/`security-indoor.yaml`) reuse the
pre-existing profile-loading mechanism, no new config plumbing needed.

The Encounter model (`schemas/encounter.py`,
`enrichment/service.py::EnrichmentService._derive_encounters`,
`recognize encounters`): a derived, non-accusatory record grouping
face/plate/voice/vehicle-appearance observations that happened within a
couple of seconds of each other in the same clip -- explicitly **not** a
claim of spatial correspondence (see the module docstring and
`docs/limitations.md`), since none of the four observation schemas store a
bounding box, only a crop path. Runs as a final post-processing step in
`enrich_event()`, after every other enrichment pass, gated by
`enrichment.encounters.enabled` (on by default, zero extra cost since it
only reads what already ran). `patterns/service.py` gained a fourth
pattern method, `_recurring_face_vehicle_cooccurrence`, fed by a
caller-side (CLI) lookup pass that resolves Encounter observation ids to
canonical cluster ids -- deliberately keeping `PatternAnalysisService`
itself free of any `Repository`/`TimelineDatabase` import, exactly like the
other three pattern methods.

If you're an agent picking this up fresh: don't assume anything is done
just because a docstring or an old summary says so. Read the code. This
document tries to be an accurate snapshot as of this pass, but it will
drift -- treat "grep the actual code" as the ultimate source of truth over
this file.

## Non-negotiable invariants

These are checked in code review and several are covered by regression
tests. Do not weaken them for convenience.

1. **Original media is never mutated.** Ingest copies into `originals/` and
   chmods read-only. No code path should ever open an original for writing.
2. **`event.json` is versioned and append-only-revisioned, never edited in
   place.** Any change to an already-written event MUST go through
   `Repository.save_event_revision()`. See `docs/architecture.md`'s
   revisioning section. Do not add a code path that calls
   `workspace.write_event_revision()` directly from outside `Repository`.
3. **The review log (`review/<id>.jsonl`) is append-only.** Use
   `utils/filesystem.py::append_line`; never open it for truncation/rewrite.
4. **Human review never overwrites automated output.** A `ReviewAction`'s
   effect on an `EventRecord` is scoped to `review_summary` (and, for
   preserve/export, `preservation_status`/`chain_of_custody`) -- never
   `signals`, `hypotheses`, or `scoring`.
5. **SQLite is an index, not a source of truth.** Every field in
   `EventIndexRow` must be re-derivable from the filesystem via
   `Repository.reindex()`. If you add a new indexed field, make sure
   reindexing still populates it correctly from `event.json` alone.
6. **All timestamps are timezone-aware UTC.** Use `UTCDateTime` from
   `schemas/common.py` for any new datetime field; it enforces this via an
   `AfterValidator`. Never use naive `datetime` objects.
7. **No single weak signal reaches high severity alone.** Any new inference
   rule or scoring change must preserve the corroboration requirement (see
   `docs/forensic-considerations.md`'s false-positive philosophy).
8. **No silent failures.** Every `except` block either re-raises, or logs
   via `structlog` (`utils/logging.py::get_logger`) with enough context to
   debug, or both. Plugin isolation (`plugins/registry.py`) is the one
   place a broad `except Exception` is intentional and documented --
   anywhere else, prefer catching specific exceptions.
9. **Detectors and pure algorithms are deterministic.** Same input bytes +
   same code + same config => same output, every time. No `random`, no
   wall-clock-dependent branching in detection/inference/scoring logic.
10. **Strict typing, `extra="forbid"` schemas.** Every Pydantic model uses
    `StrictModel`. Don't add `Any`-typed fields where a real type exists.
    `mypy src` runs in strict mode in CI.
11. **No identity resolution in `enrichment/`.** Face and plate
    re-identification are local pattern-matching aids ("have I seen this
    before"), never identity lookup. No name fields, no external database
    queries, no networking with other cameras/users. See
    `docs/forensic-considerations.md`'s "Recognition data: scope and
    intent" before touching `enrichment/face.py`, `enrichment/plate.py`, or
    `schemas/recognition.py` -- this boundary is deliberate, not an
    oversight to "improve."
12. **Enrichment signals never trigger re-scoring.** New signals found
    during `enrich` are appended to an event via a new revision, but
    `scoring.severity`/`scoring.confidence` are never recomputed from them.
    If you need enrichment to influence severity, that's a deliberate,
    separately-designed change, not an incidental side effect.
13. **Nothing is ever deleted without an explicit, actor-attributed,
    logged confirmation.** `core/triage.py::TriageService.confirm_deletion`
    is the only code path allowed to unlink an original clip's bytes, and
    it must write a `DeletionRecord` to the append-only
    `deletion_log.jsonl` *before* deleting. A clip is only ever eligible if
    it contributed zero signals to any event (see invariant 1 for why this
    doesn't conflict with immutability -- nothing references a benign
    clip's path, so moving/deleting it can't break an existing evidence
    reference).
14. **The only network-calling code path is `enrichment/llm_analysis.py`,
    and only when explicitly enabled.** Every other capability in this
    project, including all the new local-AI ones, must work with network
    access fully disabled after any one-time optional model download. If
    you add a feature that calls out to a remote service, it must default
    to off and be gated behind its own explicit config flag, matching
    `CloudEnrichmentConfig`'s pattern.
15. **Identity merges are aliases, never edits.** `RecognitionService.merge_faces`/
    `merge_plates` are the only code paths allowed to set `merged_into` on
    a `FaceCluster`/`PlateRecord`, and doing so must never modify, delete,
    or rename the source record -- and must always append an
    `IdentityMergeRecord` to `identity_merge_log.jsonl` first. Any code
    that rebuilds a `FaceCluster`/`PlateRecord` (e.g. `EnrichmentService`
    recording a new observation) must carry the prior row's `merged_into`
    value through unchanged, the same way it already preserves `label` --
    forgetting this silently undoes every merge the next time that
    cluster/record gets a new observation. A merge that would create a
    cycle (directly or transitively) must be rejected at merge time, not
    just handled defensively at resolution time.
16. **Every SQLAlchemy datetime column must use `UTCDateTimeColumn`
    (`storage/database.py`), never `DateTime(timezone=True)` directly.**
    SQLite silently returns naive datetimes on read regardless of what was
    written with the latter -- this caused a real production crash (see
    "Repo history context" above and `docs/limitations.md`). If you add a
    new datetime-typed column to any `Row` class, use `UTCDateTimeColumn()`
    or you will reintroduce this bug.
17. **Purging an event's video must never touch `event.json`'s
    forensic content.** `TriageService.purge_event_video` may only update
    `EventRecord.video_purged_at` (via a new revision). It must never
    modify `signals`, `hypotheses`, `scoring`, `chain_of_custody`, or any
    review data. If a future change wants purge to affect scoring or
    review state, that's a deliberate, separately-designed change, not an
    incidental side effect of reclaiming disk space.
18. **An original clip is only ever cascade-deleted during a video purge
    if *every* event referencing it (by content hash, not clip id) has
    also been purged.** This check must be recomputed against current
    state on every call, not cached -- an event purged five minutes ago
    changes what's safe to delete now. Never assume a clip referenced by
    multiple events is safe to delete just because the *current* purge
    target no longer needs it.
19. **A `reference`-mode clip's bytes live outside the workspace.** Any
    code path that deletes, moves, or otherwise mutates a clip's stored
    file must check `MediaClip.ingest_mode`/`MediaIndexRow.ingest_mode`
    first. Silently treating a `reference`-mode clip like a `copy`/`move`-mode
    one risks either destroying a user's only copy of source footage (on
    delete) or defeating the entire purpose of the mode (on an unnecessary
    copy, e.g. the benign-classification bug described in "Repo history
    context" above).
20. **A `MergeSuggestion` is never a merge.** `suggest_face_merges`/
    `suggest_plate_merges`/`suggest_voice_merges` may only ever write rows
    with `status="pending"` -- they must never call
    `merge_faces`/`merge_plates`/`merge_voices` themselves. Only
    `confirm_merge_suggestion`, which a human explicitly invoked, is
    allowed to perform the actual merge. If you add a new entity type's
    suggestion generator, keep this separation; a suggestion engine that
    sometimes auto-merges defeats the entire point of a review queue.
21. **Internal pure-computation dataclasses never reuse a schema class's
    name.** `enrichment/face.py` uses `DetectedFace`, `enrichment/plate.py`
    uses `PlateRegion`/`OcrResult`, `enrichment/voice.py` uses
    `VoicePrintResult` -- none of them are named `FaceObservation`/
    `PlateObservation`/`VoiceObservation` even though those are the
    natural-sounding names, because those exact names are already taken by
    the pydantic/SQLAlchemy schema classes in `schemas/recognition.py` and
    `storage/database.py`. This was violated once during the 1.3 pass
    (`enrichment/voice.py` was briefly named `VoiceObservation`) and
    caught before shipping; don't reintroduce it in a new enrichment
    module.
22. **Reviewing a recognition observation/cluster and purging its crop are
    two distinct, both-logged actions -- neither ever deletes or rewrites
    an observation row.** `RecognitionService.confirm_identity`/
    `reject_cluster`/`reject_observation` may only change
    `review_status`/`representative_observation_ids` (via a new
    `RecognitionReviewRecord` appended to `recognition_review_log.jsonl`
    *before* the change) and must never touch a crop file.
    `RecognitionService.purge_reviewed_crops` may only unlink an
    already-reviewed, purge-eligible crop file and set
    `crop_purged_at` (via a new `RecognitionCropPurgeRecord` appended to
    `recognition_crop_purge_log.jsonl` *before* the unlink, hash-verified
    first, same defensive check `TriageService.confirm_deletion` already
    does) -- `crop_path` and every other observation field stay exactly as
    they were, a historical pointer even after the file is gone. If you
    add a fourth entity type's review/purge support, keep this same
    two-record, log-before-mutate split; don't collapse review and purge
    into one silent step.

## Module map / where things live

Full detail in `docs/architecture.md`. Quick orientation:

```
schemas/     -- all typed data shapes (the only place field names are defined)
ingest/      -- copy-in + probe.py (real ffprobe wrapper, no pydantic dep);
                also copies colocated `*.samples.json`/`.gpx` sidecars
normalize/   -- sync.py (pure time-sync algorithm, no pydantic/I-O dep) + service.py wrapper
windowing/   -- deterministic sliding windows over corrected time
detection/   -- motion.py / audio.py / object_detection.py / telemetry.py /
                optical_flow.py (pydantic-aware)
             -- video_analysis.py / audio_analysis.py / telemetry_analysis.py /
                optical_flow_analysis.py (pure, no pydantic dep --
                telemetry_analysis.py is stdlib-only: GPX parsing +
                hand-implemented haversine/bearing math;
                optical_flow_analysis.py: dense Farneback flow + a
                comparative divergence threshold for "rapid approach"
                detection, see its module docstring)
inference/   -- rule-based Signal -> Hypothesis, + plugin rule loading
scoring/     -- Hypothesis -> SeverityAssessment
core/        -- pipeline.py (orchestration + event assembly + clustering),
                config.py (RuntimeConfig), review.py (ReviewService),
                derived_clips.py (pure ffmpeg clip-cutting wrapper),
                triage.py (storage-lifecycle: benign/reviewable
                classification, event-video purge, deletion),
                recognition.py (identity merge/resolve/search across face/
                plate/voice clusters, duplicate-cleanup automation,
                automated merge-suggestion generation),
                cameras.py (CameraRepository -- thin lookups over the
                optional camera registry, used by ingest auto-registration
                and normalize's site-scoped sync),
                signing.py (optional Ed25519 keygen/sign/verify, `signing`
                extra, no pydantic/SQLAlchemy dep -- same *_available()/
                *Unavailable convention as enrichment/'s optional deps),
                models.py (ModelRegistry -- on-demand per-machine ML model
                cache/download/precision-derivation shared by
                face_yunet.py/face_auraface.py, `gaggle models` CLI;
                fast-alpr manages its own separate download, see its
                module docstring for why)
preservation/-- PreservationService (pure copy+freeze) + PreservationOrchestrator (revisioning glue)
enrichment/  -- face.py / plate.py / voice.py / vehicle_appearance.py
                (real, on by default, zero-download -- voice and
                vehicle_appearance are classical, hand-built signal/image
                processing rather than a pretrained model, see each
                module's docstring for accuracy caveats); face_yunet.py /
                face_auraface.py / plate_fast_alpr.py (optional real
                deep-learning upgrades for face detection/embedding and
                plate detection+OCR respectively, `face_recognition`/
                `plate_recognition` extras, graceful degradation to the
                classical default without them -- see
                `enrichment.face.detector`/`embedding_model` and
                `enrichment.plate.detector` in `core/config.py`);
                vehicle_yolo.py / transcription.py (optional, graceful
                degradation without extras+model, vehicle_yolo.py's boxes
                are also used opportunistically by vehicle_appearance.py
                when loaded); llm_analysis.py (optional, off by default,
                the ONLY network-calling module in the project); service.py
                (EnrichmentService orchestrator)
export/      -- ExportService (bundle zip + timeline CSV/JSON)
timeline/    -- TimelineService (thin façade over TimelineQuery)
patterns/    -- PatternAnalysisService (metadata-only pattern hypotheses)
plugins/     -- base.py (Protocols) + registry.py (entry_points loader)
storage/     -- filesystem.py (WorkspacePaths + revisioning), database.py
                (SQLite models + TimelineQuery, incl. recognition/triage/
                camera/encounter tables; close()/check_schema_drift() for
                `workspace reindex --rebuild`), repository.py (the one
                seam), migrate.py + migrations/ (Alembic schema-upgrade
                driver, called from TimelineDatabase.initialize() -- see
                migrate.py's module docstring for the three-way branch;
                migrations/versions/ holds one file per schema change,
                each hand-verified after `alembic revision --autogenerate`,
                never trusted blindly)
cli/app.py       -- Typer CLI, the primary interface
review_ui/app.py -- FastAPI: JSON API + synchronized-playback HTML page
```

**Modules with zero pydantic/pytest/typer/fastapi/sqlalchemy/structlog
dependency** (useful to know because they can be imported and exercised
even in an environment without those installed, e.g. for quick standalone
verification): `ingest/probe.py`, `detection/video_analysis.py`,
`detection/audio_analysis.py`, `normalize/sync.py`, `core/derived_clips.py`,
`enrichment/face.py`, `enrichment/plate.py`, `enrichment/voice.py`,
`enrichment/vehicle_appearance.py`, `enrichment/vehicle_yolo.py`,
`enrichment/transcription.py`, `enrichment/llm_analysis.py`. Keep it that
way if you touch them -- their
value is partly that they're independently testable pure logic, which is
exactly how the face/plate/voice/vehicle/transcription/LLM modules were
actually verified during development without a full pydantic environment
(see below).

**Real `pytest` isn't installable in this sandbox, but a minimal shim is
enough to actually run a pure module's test file** (as opposed to only
hand-tracing it), discovered during the 1.3 pass:

```python
import types, sys
pytest_shim = types.ModuleType('pytest')
class RaisesContext:
    def __init__(self, exc_type): self.exc_type = exc_type
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        assert exc_type is not None and issubclass(exc_type, self.exc_type)
        return True
pytest_shim.raises = RaisesContext
pytest_shim.mark = type('MarkModule', (), {
    'skipif': staticmethod(lambda cond, reason='': (lambda x: x))
})
sys.modules['pytest'] = pytest_shim
# then import the test module and call every test_* function directly
# (substitute a real tempfile.TemporaryDirectory() for any tmp_path fixture)
```

This is how `tests/unit/test_voice_recognition.py` (all 12 tests) and
`tests/unit/test_plate_recognition.py` (all 11 tests, tesseract permitting)
were actually *executed*, not just reviewed, during the 1.3 pass -- a
meaningfully stronger verification bar than everything in this codebase
that touches pydantic/SQLAlchemy, which still can't be run this way. If
you're extending a zero-dependency module, use this to get real test
execution before treating your change as verified.

## Development environment note

This pass (and the subsequent local-AI/lifecycle expansion) was authored in
a sandbox with **no network access**, meaning
`pydantic`/`sqlalchemy`/`fastapi`/`typer`/`structlog`/`pytest`/`ruff`/`mypy`
could not be installed or executed, and no ML model weights (YOLO ONNX,
Whisper) could be downloaded. What *was* verified, by actually running it:

- `ingest/probe.py` -- confirmed correct duration/fps/codec/resolution
  extraction against a real ffmpeg-generated clip.
- `detection/video_analysis.py` -- confirmed zero motion during a static
  segment, correct spike detection and bounding-box regions during an
  animated segment, using a real concatenated static+animated test video.
- `detection/audio_analysis.py` -- confirmed correct RMS spike detection
  against a real quiet-loud-quiet synthetic audio clip, and correct
  graceful handling of a video with no audio stream at all.
- `normalize/sync.py` -- confirmed session grouping, reference-camera
  selection, and offset/drift math against a hand-constructed 3-camera
  scenario (verified the sign and magnitude of every computed correction
  by hand).
- `core/derived_clips.py` -- confirmed real ffmpeg segment extraction.
- `enrichment/face.py` -- confirmed real Haar-cascade detection against a
  real photographic test fixture (`tests/fixtures/sample_face.jpg`, a
  public-domain U.S. Navy photo bundled for exactly this purpose), and
  confirmed LBPH incremental clustering: same face re-matches its cluster,
  a dissimilar input spawns a new one, and cluster state survives a
  save/reload cycle.
- `enrichment/plate.py` -- confirmed the heuristic contour-based plate
  localizer against a synthetic car scene (0.94 IoU against the true
  region) and confirmed real Tesseract OCR read-back (92% confidence,
  correct text), plus the negative case (blank scene -> no confident
  regions) and IoU-based deduplication.
- `enrichment/vehicle_yolo.py` -- no real ONNX model weights were
  available to test end-to-end inference, but the box-decode/NMS/class-
  mapping logic was verified directly against a synthetic raw-output
  tensor (correctly recovers one confident detection, filters a
  below-threshold one, maps the right COCO class), and graceful
  degradation (missing package or missing model file) was verified for
  real.
- `enrichment/transcription.py` -- `faster-whisper` isn't installed in this
  sandbox and couldn't be; graceful degradation was verified for real
  (clean `TranscriptionUnavailableError`, no crash). The integration itself was
  written carefully against the library's documented API but not executed.
- `enrichment/llm_analysis.py` -- the empty-transcript short-circuit and
  real network-failure handling were both verified for real (the sandbox
  has no outbound network access, so the failure path was exercised
  genuinely, not simulated).
- The plugin-isolation logic, window-overlap-merge logic, and
  triage-classification logic were each verified by extracting the exact
  algorithm into a standalone script (or, for triage, by careful hand trace
  against the actual `TriageService` implementation) and running/checking
  it against constructed inputs, since the real modules pull in
  `structlog`/pydantic transitively.
- `core/recognition.py`'s cycle-rejection logic was hand-traced against a
  3-way merge chain (A->B->C, then reject C->A). The fuzzy-search
  assumption (`difflib.get_close_matches` catching a 1-character OCR-style
  plate typo) was verified numerically: `"ABC1Z34"` vs `"ABC1234"` scores
  0.857 similarity, comfortably above the 0.7 cutoff used.
- `storage/database.py::UTCDateTimeColumn`'s fix logic was verified by
  extracting its exact `process_bind_param`/`process_result_value` bodies
  into a standalone script and replaying the precise failure from the
  traceback (a naive `datetime(2025, 6, 27, 15, 6, 43, 33333)`, byte-for-
  byte the value SQLite actually returned) through it, confirming the
  recovered value is timezone-aware, round-trips to the original, and
  passes the same style of tz-aware check pydantic's validator performs.
  The class itself (real SQLAlchemy `TypeDecorator` usage) was not
  instantiated or run, since SQLAlchemy isn't installed in this sandbox --
  and for that reason the riskier parts of the fix (a `Dialect` type import
  location, `TypeDecorator` generic subscripting) were deliberately
  written in their most conservative, version-safe form rather than the
  more precisely-typed form, since a wrong import there would break every
  database operation, not just the one that crashed.
- **A real user running the CLI against real footage found two crashes
  this process's own review had missed**: `DetachedInstanceError` in
  `TriageService.classify_all()`, and (in a follow-up report against
  `enrich`) `ValidationError: timestamp must be timezone-aware`. Root
  causes and fixes for both are described in the "Repo history context"
  section above. This is the clearest evidence in this document that
  hand-tracing and static review, however careful, cannot substitute for
  actually running the code -- treat every claim in this file about
  something being merely "hand-traced" or "cross-checked" with
  appropriately less confidence than something marked as executed.
- The 1.2 pass's cross-event cascade-safety logic (an original clip
  referenced by two events is only deleted once *both* have been purged,
  never just the one currently being purged) was verified by hand-tracing
  a full two-event scenario line-by-line against the actual
  `_cascade_original_deletion` implementation: purge event A while event B
  still references the shared clip (must retain), then purge event B too
  (must now cascade-delete it) -- see
  `tests/unit/test_event_video_purge.py::test_purge_retains_original_still_needed_by_another_unpurged_event`.
  The plate-observation time-clustering algorithm was verified standalone
  with a constructed multi-cluster input before being trusted in
  `core/recognition.py`.
- **A markdown-editing mistake was caught and fixed during this same
  pass**: a `str_replace` edit to `docs/limitations.md` accidentally
  deleted a section header (the replacement text didn't include the
  heading line the match was anchored on). Caught by a manual header-count
  and duplicate-header sweep across every doc touched that session, not by
  any tooling -- there is no AST-equivalent check for markdown the way
  there is for Python in this repo. If you make a large edit to any `.md`
  file, grep for `^## ` before and after and compare, the same discipline
  used for the Python duplicate-class-attribute checks described above.
- **The 1.3 pass's plate-detector claims are the strongest-evidenced
  claims in this document**, because they were reproduced, not just
  asserted: the exact synthetic scene that exposed the old detector
  missing a real plate (0.00 IoU) was kept as a regression test
  (`test_mser_finds_the_plate_in_a_junk_heavy_scene_the_old_detector_missed`)
  and actually re-run (via the pytest-shim technique above) to confirm
  the fix, not just reasoned about.
- **The voice module's default clustering threshold went through a real
  correction cycle, not a single confident guess**: an initial choice
  (0.15) based on one clean-signal test was re-examined with a proper
  8-seed distribution once realistic noise was added, revealed to be too
  loose (produced an actual false merge), and replaced with a
  conservative value (0.05) comfortably inside the measured gap between
  same-voice (~0.0002-0.0003) and different-voice (~0.14) distances. All
  of this -- including the initial mistake -- is preserved in
  `enrichment/voice.py`'s module docstring rather than quietly corrected
  and forgotten, because knowing a threshold was empirically corrected
  once is useful signal for whether to trust it going forward.

At the time of the 1.3 pass, the following had **not** been executed, for
the reasons above: the full pydantic-based schema layer, the
SQLAlchemy-backed repository/database layer (including the recognition and
triage tables), the Typer CLI (including all the `enrich`/`triage`/
`recognize`/`review start` commands), the FastAPI review UI, and the pytest
suite's pydantic-dependent files -- including `test_triage.py`,
`test_enrichment_service.py`, `test_recognition_identity.py`,
`test_event_video_purge.py`, `test_plate_cleanup.py`, `test_ingest_modes.py`,
and `test_merge_suggestions.py` (`test_voice_recognition.py` and the
plate-detector additions to `test_plate_recognition.py` were the exception
-- see the pytest-shim technique above). All of it had only been
cross-checked by hand (field names, types, method signatures, import graph,
AST-based duplicate-attribute sweeps) rather than run.

### 1.4 pass: first real execution (2026-08-09)

A later pass ran in an environment with real network access and every
dependency already installed (`pydantic`/`sqlalchemy`/`fastapi`/`typer`/
`structlog`/`pytest`/`ruff`/`mypy`, plus the `vision`/`cloud` extras,
`ffmpeg`/`ffprobe`, and a real `tesseract` binary -- `transcription`'s
`faster-whisper` was also present, so even the tests that only exist to
verify graceful degradation *without* it were skipped as designed). This
is the first time this codebase's full pydantic/SQLAlchemy-dependent
surface -- the schema layer, the database layer, the CLI, the FastAPI
review UI, and the previously-never-run test files listed above -- was
actually executed rather than hand-traced.

Result: **144 passed, 2 skipped (the two `faster-whisper`-not-installed
degradation tests, correctly skipped since it *is* installed here), 0
failed.** `ruff check .` and `ruff format --check .` are clean.
`mypy src` is clean (strict mode, 69 source files).

None of what running these tools for the first time turned up was a
runtime crash on the order of the three bugs in "Repo history context"
above -- this pass found no equivalent of `DetachedInstanceError` or the
SQLite timezone bug. What it did find were real, previously-undetected
static-analysis gaps, exactly the class of thing hand-tracing reliably
misses because nothing was ever run through the actual tools:

- **Every one of the 20 SQLAlchemy datetime columns in `storage/database.py`
  was declared `Mapped[object]` instead of `Mapped[datetime]` /
  `Mapped[datetime | None]`**, and `UTCDateTimeColumn` itself subclassed
  bare `TypeDecorator` instead of `TypeDecorator[datetime]` with a properly
  typed `dialect: Dialect` parameter -- exactly the "riskier parts of the
  fix" the 1.1.1 pass's writeup flagged as written in a conservative,
  unverified form because SQLAlchemy couldn't be imported at the time.
  Fixing the generic subscripting resolved the bulk of a 102-error mypy
  run in one change; the `Mapped[object]` columns accounted for most of
  the rest. Runtime behavior was unaffected either way (Python doesn't
  enforce `Mapped[]` annotations), but it meant every caller reading a
  cluster/observation/event timestamp back out of the database was doing
  so through a type checker's blind spot.
- **Two cases of a shared helper's type signature silently falling behind
  when voice support was added on top of it**, both in
  `core/recognition.py`: `_cluster_by_time`'s type parameter was
  constrained to `(FaceObservationRow, PlateObservationRow)` only, despite
  `cleanup_duplicate_voice_observations` (added in the 1.3 pass) calling it
  with `VoiceObservationRow`; and `_resolve`/`_merge`'s `get_row` callback
  type was `Callable[[UUID], FaceClusterRow | PlateRecordRow | None]`,
  omitting `VoiceClusterRow` despite `resolve_voice_identity`/
  `merge_voices` passing exactly that. Both fixed by adding the missing
  type to the union/constraint -- functionally harmless (Python doesn't
  enforce this at runtime either), but exactly the kind of drift a real
  mypy run catches immediately and hand-tracing can miss indefinitely.
- **A first real `ruff check .`/`ruff format .` run** (these had never
  executed before) found 29 files with formatting drift and ~20 real lint
  violations: unsorted imports, three exception classes missing the
  `Error` suffix convention (`LlmEnrichmentUnavailable` ->
  `LlmEnrichmentUnavailableError`, etc., with all call sites updated),
  unnecessary `int(round(...))` casts, list-concatenation that should have
  been unpacking, and a few unused variables in tests. All fixed; none
  were behavior bugs.
- **`opencv-contrib-python-headless`'s bundled type stubs don't declare
  `cv2.data`, `cv2.face`, or `cv2.MSER_create`**, even though all three are
  real, working attributes at runtime (confirmed directly) -- a genuine gap
  in the third-party stub package, not a bug in this codebase. Addressed
  with narrow, commented `# type: ignore[attr-defined]` at each call site
  rather than suppressing the whole module.
- `scipy`, `faster_whisper`, and `onnxruntime` ship no type stubs at all;
  added to `[[tool.mypy.overrides]] ignore_missing_imports`. `requests`
  does have a real stub package (`types-requests`); installed it and added
  it to the `dev` extra instead of suppressing it.

Separately, worth recording as an environment-specific footnote rather
than a code bug: this pass's `tesseract` binary was correctly installed
(`C:\Program Files\Tesseract-OCR`, registered in the Windows Machine
`PATH`) but wasn't visible in the shell sessions used to run these tools,
which had started before that `PATH` change propagated -- prepending it to
`PATH` for those sessions was enough to get the tesseract-dependent plate
OCR tests running for real instead of skipping. Not a project issue, but
worth knowing if a future session reports tesseract-gated tests skipping
on a machine where tesseract is actually installed.

**Before making further changes, re-run the full suite** (now expected to
stay green):

```bash
pip install -e .[dev,vision,cloud]   # add ,transcription if you want those tests to run for real too
pytest
ruff check . && ruff format --check .
mypy src
```

Treat any failures you find as real bugs, not as expected behavior --
that's still the standing instruction from every prior pass, and it held
up: the fixes above were all found this way, not anticipated in advance.

## Build / test / run

See `docs/developer-setup.md` and `docs/local-ai.md` for full details.
Summary:

```bash
pip install -e .[dev,vision,cloud]    # requires ffmpeg/ffprobe/tesseract on PATH
gaggle workspace init --workspace ./workspace
gaggle ingest examples/sample_media --workspace ./workspace
gaggle analyze --workspace ./workspace    # also triages automatically
gaggle enrich --workspace ./workspace     # face/plate recognition, offline by default

ruff check .
ruff format --check .
mypy src                              # strict mode, scoped to src/ not tests/
pytest                                # some tests skip automatically without ffmpeg/tesseract
pytest --cov=gaggle --cov-report=term-missing
```

CI (`.github/workflows/ci.yml`) runs all of the above (installing
`ffmpeg`+`tesseract-ocr` and the `vision`/`cloud` extras so those tests run
for real rather than skipping; `transcription` is left out of CI by design
-- it's heavy and gracefully degrades, so CI verifies the fallback path
instead) plus a Docker build + smoke test. Pre-commit
(`.pre-commit-config.yaml`) runs ruff (lint + format) and mypy on `src/` on
every commit if installed (`pre-commit install`).

## Status tracker

A honest checklist against the original project brief's headline
deliverables, as of this pass.

| Area | Status |
|---|---|
| Repo scaffolding, schemas, CLI skeleton, storage layout, logging, config | Done |
| Immutable ingest, content hashing | Done |
| Real media metadata extraction (ffprobe) | Done |
| Media normalization / timestamp alignment / camera sync | Done (heuristic start-alignment + proportional drift; no audio/video cross-correlation -- see `docs/limitations.md`) |
| Event window generation | Done, including overlap-merge fix |
| Motion detection | Done (real OpenCV frame differencing + sidecar override) |
| Audio spike detection | Done (real ffmpeg+scipy RMS + sidecar override) |
| Object detection abstraction | Done as a heuristic (contour bounding boxes); real classification is an intentional plugin extension point, not built in |
| Vehicle telemetry | Done (1.5) -- `detection/telemetry_analysis.py` (hand-implemented haversine speed / initial-bearing heading from GPX track points) + `detection/telemetry.py` (`TelemetryDetector`, sidecar-fixture override same as motion/audio); GPX is the real ingestion format (open standard, stdlib-parseable, no new dependency) -- see `docs/local-ai.md` |
| Rule-based inference engine | Done, 5 built-in rules + plugin rule loading |
| Scoring + severity | Done |
| Preservation subsystem | Done, including the event-revision fix so `event.json` no longer goes stale |
| Review interface | Done -- real synchronized multi-camera playback, not just a JSON API; (1.5) event-detail page now also surfaces face/plate/voice/vehicle-appearance observations (with crop thumbnails) and transcripts inline, not CLI-only |
| Timeline querying | Done, with real filters |
| Pattern detection | Done -- 4 pattern types (repeated camera, repeated object label, repeated vehicle appearance (1.6), temporal clustering) |
| Plugin architecture | Done -- wired into the pipeline, not just scaffolding; entry-point based, failure-isolated; 4 plugin types (detectors, inference rules, exporters, review extensions (1.6)) |
| Export / structured metadata output | Done -- was entirely missing before this pass; now a full subsystem (event bundle zip + timeline CSV/JSON), plus a dependency-free standalone verifier script (`scripts/verify_export_bundle.py`) |
| Comprehensive tests | **Executed for real as of the 1.4 pass** (see "Development environment note" above): 144 passed, 2 skipped (expected), 0 failed. `ruff check .`/`ruff format --check .`/`mypy src` all clean |
| Documentation | Substantially expanded this pass (was 7-38 line stubs; now real docs for every listed topic) |
| Docker/devcontainer support | Fixed this pass -- the prior Dockerfile copied a dead directory tree and didn't install ffmpeg; both fixed |
| Reproducible builds | Config/pipeline/rule versions are recorded in every `EventRecord`; deterministic detector/sync algorithms verified by hand this pass |
| AGPL-compliant licensing | Present (`LICENSE`, `EULA.md`, `GENERATED_CONTENT_LICENSE.md`), unchanged this pass |
| Local face detection + re-identification | Done -- real, on by default, zero setup (Haar cascade + incremental LBPH clustering). No identity resolution by design -- see `docs/forensic-considerations.md`. **Superseded (1.9)**: `yunet`/`auraface` are now the config defaults, Haar/LBPH kept as automatic fallback -- see below |
| Local license plate detection + OCR | Done -- real, on by default, zero setup (cascade + heuristic contour detector, real Tesseract OCR). Confidence-gated human review queue for low-confidence readings. **Superseded (1.9)**: `fast_alpr` is now the config default, the classical cascade+contour detector kept as automatic fallback -- see below |
| Local vehicle/object detection (YOLO) | Architecture done, optional (`vision` extra + user-supplied ONNX model required); decode/NMS logic verified against a synthetic tensor, not a real model -- see the environment note above |
| Local audio transcription (Whisper) | Architecture done, optional (`transcription` extra + one-time model download required); graceful degradation verified for real, transcription itself not executed |
| Optional cloud LLM transcript analysis | Done -- off by default everywhere, the only network-calling code path in the project, clearly labeled non-authoritative output |
| Recognition database (faces/plates) | Done -- new SQLite tables + CLI (`recognize faces/plates/*`), crops+metadata survive raw-footage deletion |
| Storage lifecycle (triage + human-confirmed deletion) | Done -- `core/triage.py`, append-only `DeletionRecord` log written before any unlink, hash-verified before deletion |
| Interactive review walkthrough | Done -- `review start` command |
| Identity linking (same person/vehicle across sightings) | Done (1.1) -- `merged_into` alias pointer on `FaceCluster`/`PlateRecord`, cycle-safe resolution, aggregated identity summaries, `identity_merge_log.jsonl` audit trail |
| Search by plate text or identity UUID | Done (1.1) -- `recognize faces-search`/`plates-search`, exact-match-first with difflib fuzzy fallback for likely OCR misreads |
| Ingest storage flexibility (copy/move/reference) | Done (1.2) -- `ingest --mode`; reference-mode deletion gated behind `--acknowledge-external` since it deletes files outside the workspace |
| Event-video purge (keep metadata, drop the video) | Done (1.2) -- `triage purge-event-video`/`purge-reviewed`, preservation-gated, safe cross-event cascade to originals, `event.json` untouched except `video_purged_at` |
| Plate false-positive cleanup automation | Done (1.2) -- config-driven garbage-OCR pre-filter + `recognize plates-cleanup` duplicate-collapsing pass; face-observation cleanup parity added in 1.3, see below |
| Tiered setup / full pipeline walkthrough docs | Done (1.2) -- `docs/getting-started.md`, `docs/pipeline-walkthrough.md` |
| Plate detector accuracy (MSER + rotation-aware detection) | Done (1.3) -- reproduced the reported miss (0.00 IoU) and confirmed the fix (0.97 IoU) with a kept regression test |
| Plate detection debug/audit tooling | Done (1.3) -- `recognize plates-debug` renders every candidate region on real footage |
| Automated merge suggestions (face/plate/voice) | Done (1.3) -- `suggest-merges`/`merge-suggestions*`, human-in-the-loop, never auto-merges |
| Voice detection + local voiceprinting | Done (1.3) -- classical VAD + MFCC, built from scratch (no pretrained model reachable); explicitly documented as weaker/unvalidated against real speech |
| Face duplicate-cleanup automation | Done (1.3) -- parity with the existing plate cleanup |
| Dedicated plate-observation reject action | Done (1.3) -- `recognize plates-reject` |
| Vehicle re-identification by visual description (no plate) | Done (1.5) -- `enrichment/vehicle_appearance.py`, classical hue/saturation histogram + aspect-ratio fingerprint, `recognize vehicles-*`; explicitly documented as a meaningfully weaker fingerprint than face/plate, see `docs/limitations.md` |
| Vehicle telemetry detector (GPS-derived hard braking / speed spike / sudden heading change) | Done (1.5) -- `detection/telemetry_analysis.py` + `detection/telemetry.py`; `ingest/service.py` copies a colocated `.gpx` file into the workspace as a `gps_track` sidecar, matched by presence in the same camera directory, not filename correlation -- see `docs/local-ai.md`'s telemetry section for the "one GPS track per session" scope limit |
| Cryptographic signing of the revision hash chain | Done (1.5) -- `core/signing.py` (Ed25519 via the optional `signing` extra), `workspace signing-init`/`signing-status`, `EventRecord.revision_signature`, public key inlined into exported bundles and verified by `scripts/verify_export_bundle.py` when `cryptography` is present. Off by default (`signing.enabled: false`); proves who signed a revision, not that the key belongs to who you think -- see `docs/local-ai.md` and `docs/threat-model.md` |
| SQLite index schema-upgrade safety | Done (1.6), superseded (1.7) -- real upgrades now happen automatically via Alembic (see below); `workspace reindex [--rebuild]` and `check_schema_drift()` remain as an independent fallback sanity check/manual escape hatch, not the routine path anymore |
| Rapid-approach ("looming") detection via optical flow | Done (1.6) -- `detection/optical_flow_analysis.py` (dense Farneback flow, flow-field divergence, comparative not absolute threshold to reject ego-motion) + `detection/optical_flow.py` (`OpticalFlowDetector`, sidecar-fixture override same as motion/audio/telemetry); a distinct signal from frame-differencing motion detection -- captures "something is closing in on the camera," which frame differencing structurally cannot. Default threshold empirically measured, not guessed -- see `docs/local-ai.md` and the module docstring |
| Project renamed dashcam-sentinel → Gaggle | Done (1.7) -- mechanical rename across every tracked/untracked file, `gaggle` CLI command, README/AGENTS.md repositioned for the broadened camera-agnostic scope; "dashcam" as a word describing dashcam footage/detectors is untouched throughout |
| Alembic SQL migrations (schema upgrades without data loss) | Done (1.7) -- `storage/migrate.py` + `storage/migrations/`; three-way branch on workspace state (fresh/legacy/current), fast-path skip when already at head, every migration hand-verified after `alembic revision --autogenerate`, never trusted blindly. `0001_baseline` captures the pre-Alembic 14-table shape exactly |
| Camera entity + site-scoped cross-camera sync | Done (1.7) -- `schemas/camera.py`, `core/cameras.py::CameraRepository`, `camera list/register/update`; ingest auto-registers a minimal camera record with a deterministic `default_site_id` per ingest run, so an existing dashcam workflow keeps cross-syncing with zero config while a security camera from a separate run stays isolated -- see `normalize/sync.py`'s site partitioning |
| Indoor/outdoor security-camera config profiles | Done (1.7) -- `examples/config/security-outdoor.yaml`/`security-indoor.yaml`, reusing the pre-existing profile system, no new config mechanism; reasoned starting points, not empirically validated against real security footage -- see `docs/limitations.md` |
| Encounter model (cross-modality co-occurrence records) | Done (1.7) -- `schemas/encounter.py`, derived automatically in `enrichment/service.py::EnrichmentService._derive_encounters`, gated by `enrichment.encounters.enabled` (on by default); `recognize encounters --face/--plate/--vehicle`. Explicitly makes no spatial-correspondence claim -- see the module docstring and `docs/limitations.md` |
| Recurring face+vehicle co-occurrence pattern | Done (1.7) -- `patterns/service.py::PatternAnalysisService._recurring_face_vehicle_cooccurrence`, fed by a caller-side (CLI) observation-id -> cluster-id resolution pass so the service itself stays free of any storage dependency; same non-accusatory `hypothesis_only` framing as every other pattern |
| Event-duration cap + scaled ffmpeg audio-extraction timeout | Done (1.8) -- real Vantrue dashcam footage (near-continuous motion throughout a ~5min recording) collapsed into one giant event with a 60s-timeout-tripping derived clip; `core/pipeline.py::_cluster_overlapping_windows` now forces a split past `pipeline.max_event_duration_seconds` (default 120s, `None` disables), and `detection/audio_analysis.py`'s ffmpeg timeout is now a configurable 300s default instead of a hardcoded 60s. Verified against the user's real 3-camera footage: one 299.5s/4,081-signal event became three ~120s events |
| Recognition review + storage-reclamation workflow (face/plate/vehicle) | Done (1.8) -- `RecognitionService.confirm_identity`/`reject_cluster`/`reject_observation`/`purge_reviewed_crops` (`core/recognition.py`), new `review_status`/`crop_purged_at`/`representative_observation_ids` fields (migration `0004_recognition_review`), append-only `recognition_review_log.jsonl`/`recognition_crop_purge_log.jsonl` (invariant 22), CLI (`faces-confirm`/`faces-reject-cluster`/`faces-reject-observation`/`faces-purge-reviewed` + `vehicles-*` mirrors + `plates-purge-reviewed`, `plates-confirm`/`plates-reject` now actor-attributed and logged), and a new review_ui cross-event cluster-browser page (`/recognition/{entity_type}`) surfacing existing merge suggestions inline. Two-step by default (review, then a separate purge sweep), `--purge`/`purge: true` for one-step. Verified end-to-end via real CLI smoke test and a real browser session against live data |
| Deep-learning recognition upgrade: YuNet, AuraFace, fast-alpr | Done (1.9) -- `core/models.py::ModelRegistry` (new, on-demand per-machine model cache under `platformdirs.user_cache_dir`, `device`->precision mapping cpu=int8/cuda=fp16, prefers a prebuilt upstream variant over a local derive, `gaggle models list/download/remove`); `enrichment/face_yunet.py` (YuNet detector, Apache-2.0, new `enrichment.face.detector: yunet` default, `haar` kept as fallback); `enrichment/face_auraface.py` (AuraFace-v1 embedding, Apache-2.0/commercial-safe unlike InsightFace's own models, new `IncrementalFaceEmbeddingClusterer` mirroring the voice/vehicle-appearance centroid-clusterer pattern, `embedding_model: auraface`, `lbph` kept as fallback); `enrichment/plate_fast_alpr.py` (fast-alpr detector+OCR, MIT, international plate formats not Russian-locked, `detector: fast_alpr`, `cascade` kept as fallback; deliberately does NOT route through `ModelRegistry` -- fast-alpr's own preset-based model hub doesn't accept arbitrary paths, so it manages its own one-time download exactly like Whisper already does). `RecognitionService.suggest_face_merges` dispatches on `embedding_model` (LBPH crop-distance vs. AuraFace centroid cosine-distance). Every new model is opt-in, nothing classical removed. Verified end-to-end for real: real YuNet detection (0.94 confidence) and real AuraFace embedding (same-crop distance 0.0, real-face-vs-noise distance ~0.94) on the real face fixture photo, real fast-alpr detection+OCR+region-guess on a real photo (with real first-use model downloads for all three, not mocked), a real CLI `enrich` smoke test for each detector option, and a real-world onnxruntime gap caught and fixed along the way -- see `docs/limitations.md`'s `ConvInteger` note |
| Review-UI/CLI ergonomics pass (shutdown hang, form position, default actor, plate review parity, cluster detach/move, event split, manual sync offset) | Done (1.10) -- real user punch-list, not speculative: bounded `timeout_graceful_shutdown` fixes a review-ui Ctrl+C hang; the review-action form moved to the top of the event-detail page; `core/cli_config.py` + `gaggle config set-actor`/`show` let every `--actor`-taking command (23 of them) resolve a per-machine default instead of retyping a name every time; plate observations gained "not a plate"/text-correction review-ui controls (the service layer already supported this, only surfacing was missing); `RecognitionService.detach_observation`/`move_observation` (new, cluster-count-recomputing, logged) fix the "false positive stuck in the wrong cluster" gap; `core/events.py::EventSplitService.split_event` + `gaggle events split` let a human correct a wrongly-merged multi-camera event (root-caused to `normalize/sync.py`'s pure time-overlap heuristic having no camera/duration-similarity check) without editing the original event, which is preserved and marked `superseded_by_event_ids`; `sync.manual_offset_overrides` lets a per-camera timing correction apply to future `analyze` runs. Verified via real CLI smoke tests and a real browser session against the user's own (since-reset) production workspace. |
| Pedestrian/full-body appearance re-identification | Done (1.10) -- `enrichment/person_appearance.py`, a structural near-copy of `vehicle_appearance.py` (same classical hue/saturation-histogram + aspect-ratio fingerprint technique), surfacing YOLO's already-present COCO "person" class detections as a first-class, re-identifiable signal rather than only feeding vehicle-appearance fingerprinting. Structured attributes only (`dominant_hue_bin`, `height_to_frame_ratio` in `reasoning_metadata`), explicitly never a learned face embedding or an AI-generated description -- a considered, user-confirmed scope decision. YOLO-only detection, no classical fallback (unlike vehicle-appearance) and off by default for that reason. Full CLI (`recognize persons-*`)/review-UI/cluster-management (detach/move/merge-suggestions) parity with vehicle-appearance. Migration `0005_person_appearance` verified against the real workspace's SQLite index (zero-rebuild upgrade). |
| Gunshot/gunfire detection | Done (1.10) -- researched two real approaches (a classical rise-time/crest-factor impulse heuristic vs. a pretrained ONNX audio classifier) and reported a concrete recommendation before writing any code, per explicit user instruction; the user chose the classifier path. `detection/gunshot_analysis.py` (model prep + windowed classification) + `detection/gunshot.py` (`GunshotDetector`, `analyze`-time, unlike enrichment recognition which never affects scoring) via the new optional `sherpa-onnx` dependency (Apache-2.0, k2-fsa's zipformer-small AudioSet tagger, license/hash verified against the real downloaded archive, not assumed) -- off by default, degrades gracefully if the extra isn't installed. New `inference/service.py` rules (`isolated_gunshot_retention` capped at 0.60, `gunshot_plus_motion` for cross-modality corroboration) so a lone classifier opinion can never alone reach medium/high severity (invariant 7). Verified for real: downloaded and hash-checked the real model, inspected its real ONNX I/O contract, ran real inference against the model's own bundled real-world test clips (zero false "gunshot-like" matches on cat/dog/siren/baby-cry/etc.) -- but no real or synthetic gunshot audio existed in this environment to validate a true positive, honestly documented as such. The automated test suite mocks the network/model entirely (this sandbox has no network access), exercising the real windowing/threshold/hash-verification logic against a fake tagger and a fake archive instead. |

### Known gaps for a future pass

See `docs/limitations.md` for the full, categorized list. Highlights:

- Cross-camera sync is a start-alignment heuristic, not measured via
  audio/video cross-correlation.
- Per-clip drift correction within a multi-clip session is a flat
  session-level offset, not linearly interpolated across the session.
- As of 1.9, `yunet`/`auraface`/`fast_alpr` are the config defaults for
  face and plate recognition; the original zero-dependency classical
  detectors (Haar cascade + LBPH clustering, Russian-format cascades +
  generic heuristic) remain as the automatic fallback when the relevant
  extra isn't installed or a model fails to load -- not removed, just no
  longer what ships by default. See `docs/local-ai.md` and
  `docs/limitations.md`.
- Enrichment signals don't trigger re-scoring (by design, see invariant 12
  above) -- severity alone won't reflect "an unusual face/plate was seen."
- Triage classification doesn't account for post-hoc review decisions: a
  clip backing a *rejected* event is still "reviewable" forever (never
  becomes a deletion candidate through `triage`), since classification is
  based purely on signal count, not review outcome. See
  `docs/limitations.md`.
- No "unmerge" command; a mistaken merge can be corrected by re-merging
  differently but not fully undone (the merge log entry is permanent, by
  design). See `docs/limitations.md`.
- Telemetry ingestion assumes one GPS track (`.gpx` file) per camera
  directory per ingest session, matched by presence, not multiple
  overlapping tracks or filename correlation -- see `docs/limitations.md`.
- `purge_event_video`'s cross-event cascade-safety check re-scans every
  event's signals on every call rather than using an incrementally
  maintained index -- fine at single-vehicle-archive scale, not something
  that's been tested at, say, fleet-wide scale. See `docs/limitations.md`.
- `Encounter` records claim co-occurrence, never spatial correspondence --
  none of the four observation schemas store a bounding box, only a crop
  path, so there's no data yet to disambiguate multiple simultaneous
  same-modality entities within one frame. See `docs/limitations.md`.
- Security-camera config profiles (`examples/config/security-*.yaml`) are
  reasoned starting points, not empirically validated against real
  footage -- same honesty standard as the vehicle-appearance/voice/optical-
  flow caveats. See `docs/limitations.md`.
- Live/streaming camera ingestion (RTSP, a directly-attached USB webcam) is
  out of scope -- every source is file-based (`gaggle ingest <directory>`).
  A future `SourceAdapterPlugin`-style plugin type was scoped during the
  1.7 pass's research but deliberately not built; this pass shipped
  file-based security-camera support only, per explicit decision. See
  `docs/limitations.md`.
- Gunshot detection's classifier was never validated against real or
  synthetic gunshot audio in this environment -- only against the
  model's own bundled non-gunshot real-world test clips (a legitimate
  negative control, not a substitute for a true-positive check). See
  `docs/local-ai.md`'s "Gunshot detection" section.

## Conventions for future changes

- **New pipeline stage output = new manifest file + structured log line.**
  Follow the existing pattern (`ingest/`, `normalize/`, `windowing/` all
  write a run-scoped JSON file under their workspace subdirectory and log a
  `structlog` event on completion).
- **New schema field = default value if at all possible**, so old
  `event.json` revisions on disk still validate. If a field must be
  required, that's a schema-version bump (`EVENT_SCHEMA_VERSION` /
  `SIGNAL_SCHEMA_VERSION` / `REVIEW_ACTION_SCHEMA_VERSION`) and needs a
  migration note in `docs/schema.md`.
- **New detector = subclass `detection.base.Detector`**, prefer a sidecar
  fixture path for testability (see `motion.py`/`audio.py` for the
  pattern), and keep the real-analysis algorithm itself in a separate,
  pydantic-free module if it's nontrivial (mirrors `video_analysis.py`/
  `audio_analysis.py`) so it stays independently testable.
- **New inference rule (built-in) = add to
  `InferenceService._apply_builtin_rules`**, with a `confidence_math`
  string that a human can read and verify by hand. If it's meant to be
  optional/third-party, it belongs as an `InferenceRulePlugin` instead.
- **New mutation to an existing event = via `Repository.save_event_revision`
  only**, with a clear `reason` string. Never add a second way to change an
  event's fields.
- **Don't reintroduce a second copy of the source tree.** If you're
  tempted to scaffold a `packages/<name>` directory because the original
  spec mentioned a monorepo layout, don't -- see "Why not a literal
  multi-package monorepo?" in `docs/architecture.md` for why that was
  deliberately consolidated into `src/`.
- **New optional/heavy capability (extra ML backend, another cloud
  service) = follow the `enrichment/` pattern.** A `*_available()` check
  function, a loader that raises a dedicated `*Unavailable` exception (not
  a bare `ImportError`/`RuntimeError`), a best-effort
  `load_*_if_available()` that catches that exception and returns `None`,
  and default-off config gated behind its own `enabled` flag if it touches
  the network or needs a model download. Never let a missing optional
  dependency crash the core pipeline. **And check availability exactly
  once per `EnrichmentService` instance, cached on the instance** (see
  `_vehicle_load_attempted`/`_transcriber_load_attempted`/
  `_tesseract_checked`) **-- never inside a per-frame or per-region loop.**
  A missing dependency checked inside a loop means one warning (fine)
  becomes hundreds of identical warnings plus hundreds of wasted doomed
  subprocess/import attempts (not fine, and a real bug found via user
  testing in 1.1.2 -- see "Repo history context" above).
- **New capability touching faces, plates, or any biometric-adjacent data
  = re-read `docs/forensic-considerations.md`'s "Recognition data: scope
  and intent" first.** No name fields, no external lookups, no networking
  with other cameras/users -- see invariant 11.
- **Line length 100, ruff-formatted, strict mypy on `src/`.** Run
  `ruff check . && ruff format . && mypy src` before considering a change
  done.

## Where to look for more detail

- `docs/getting-started.md` -- tiered setup guide (minimal/recommended/full)
- `docs/pipeline-walkthrough.md` -- the complete narrated workflow, SD
  card to storage-optimized final state; the best single answer to "how
  do I actually use this"
- `docs/architecture.md` -- module map, storage model, revisioning, sync
  algorithm, event clustering, storage lifecycle (ingest modes, triage,
  event-video purge), enrichment, plugins, and the packaging-decision
  rationale
- `docs/schema.md` -- every Pydantic model and how they relate
- `docs/threat-model.md` -- what's actually guaranteed vs. not, attack
  surface table
- `docs/chain-of-custody.md` -- how provenance is recorded and where
- `docs/forensic-considerations.md` -- false-positive philosophy, what
  detectors can/can't tell you, human authority, recognition-data scope
- `docs/limitations.md` -- categorized, honest gap list
- `docs/local-ai.md` -- face/plate/vehicle/transcription/LLM enrichment,
  identity linking/search across sightings, false-positive cleanup
  automation, and the storage-lifecycle triage/purge/deletion workflow,
  full config reference
- `docs/plugin-authoring.md` -- how to write and register a plugin
- `docs/developer-setup.md`, `docs/cli-examples.md` -- practical usage
- `docs/diagrams/pipeline-sequence.md` -- mermaid sequence diagrams for
  ingest->analyze, preserve, and export
