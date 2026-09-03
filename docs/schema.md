# Schema reference

All schemas are Pydantic v2 models under `src/gaggle/schemas/`, all
using `StrictModel` (`extra="forbid"`) so a typo'd field or an unexpected key
from a stale caller fails loudly instead of being silently dropped. All
timestamps use `UTCDateTime` (`datetime`, required to be timezone-aware,
normalized to UTC by an `AfterValidator` -- see `schemas/common.py`).

Full field-by-field source of truth is the code itself (each model is small
and documented inline); this page is a map of how the models relate and why
they're shaped the way they are.

## `signal.py` -- `Signal`

The universal primitive that flows from detection through inference,
scoring, and pattern analysis. **Signals are evidence, never conclusions.**
`schema_version` is independent of `EventRecord.schema_version` so signal
shape can evolve without forcing an event schema bump.

Key fields: `signal_type` (`motion | audio_spike | object_hint | telemetry |
coverage` from the core pipeline; `face_detection | license_plate |
voice_detection | vehicle_detection | transcript_keyword` from enrichment
-- see `docs/local-ai.md`), `confidence` (0-1), `camera_id`, `window_id`
(which `EventWindow` this signal was matched into -- `None` if unmatched,
though no built-in detector currently emits an unmatched signal),
`evidence_references` (list of `ArtifactReference` pointing at the source
media -- empty for `voice_detection` signals specifically, since a voice
observation's identifying artifact is a stored feature vector, not a crop
file; see `recognition.py` below), `reasoning_metadata` (detector version,
evidence source: `sidecar_fixture` vs `computed`).

## `media.py` -- ingest and normalization

* `MediaClip` -- one ingested file. Immutable once written; `observed_start`/
  `observed_end` are the *uncorrected* per-camera timestamps,
  `original_timestamp_source` records how they were derived
  (`filename | mtime | sidecar | manual`) with a `timestamp_confidence`.
  `ingest_mode` (`copy | move | reference`) records how `stored_path` came
  to be -- see `core/config.py::StorageConfig.ingest_mode` and
  `docs/local-ai.md`'s "Choosing an ingest storage mode." This matters for
  deletion safety: a `reference`-mode clip's `stored_path` points outside
  the workspace, so deleting it deletes something the workspace never
  copied.
* `IngestManifest` -- one per `ingest` run; `copied_files: list[MediaClip]`.
* `CameraSync` -- one per recording session (see `docs/architecture.md`'s
  time-sync section). Carries both `original_start/end` and
  `corrected_start/end` side by side, plus `offset_seconds`,
  `drift_seconds_per_hour`, `confidence`, `is_reference`, and a
  plain-language `rationale`. Never silently assumes cameras agree.
* `NormalizedClip` -- wraps a `MediaClip` (`.clip`) with its session's
  correction applied (`corrected_start/end`, `sync_confidence`,
  `sync_rationale`). The wrapped `MediaClip` is never mutated. Convenience
  properties (`camera_id`, `clip_id`, `stored_path`, `sha256`) proxy through
  to `.clip` so callers rarely need `.clip.` boilerplate.
* `NormalizationManifest` -- one per `normalize` run; `clips: list[NormalizedClip]`,
  `camera_sync: list[CameraSync]`.
* `EventWindow` / `WindowManifest` -- deterministic sliding windows over
  corrected clip intervals.

## `event.py` -- the primary forensic artifact

* `Hypothesis` -- an inference rule's explainable output: `rule_name`,
  `label`, `confidence`, `contributing_signal_ids`, `escalation_reasons`,
  and `confidence_math` (a literal string describing the arithmetic, e.g.
  `"mean(motion,audio) + 0.10 corroboration bonus"` -- confidence is never
  an opaque number).
* `SeverityAssessment` -- `severity: low | medium | high`, `confidence`,
  `reasons`, `version`.
* `PreservationStatus` -- `state: pending | preserved | exported`,
  `immutable`, `bundle_path`, `bundle_hash`. `"exported"` is reserved for a
  possible future formal-handoff terminal state; routine exports (via
  `export event`) do not currently set it -- they're recorded as a
  `chain_of_custody` entry instead so preservation and export tracking
  don't get conflated in one field.
* `ReviewSummary` -- `latest_decision: pending | accepted | rejected`,
  `action_count`, `last_reviewed_at`, `last_action_id`. This is a *summary*;
  the full history is the append-only `review/<id>.jsonl` log plus the
  `ReviewAction` records within it, not this struct.
* `EventRecord` -- everything above, plus `chain_of_custody:
  list[ChainOfCustodyEntry]`, `hashes`, `derived_artifacts`,
  `evidence_summary`, and the revisioning fields:
  `revision`, `revision_reason`, `revised_at`, `previous_revision_hash`.
  See `docs/architecture.md` for how revisioning works and
  `docs/chain-of-custody.md` for what belongs in `chain_of_custody` vs. a
  plain revision.

`EVENT_SCHEMA_VERSION` (currently `"1.0.0"`) is separate from the package's
release version and from `AnalysisPipeline.version` (currently `"1.1.0"`,
stored as `EventRecord.pipeline_version`) -- three independently-versioned
things on purpose: schema shape, pipeline algorithm behavior, and package
release each change on their own schedule.

`EventRecord.video_purged_at` (optional, `None` until set) records when
`core/triage.py::TriageService.purge_event_video` removed this event's
video evidence -- the derived clips and, where safe, contributing
originals -- while leaving everything else about the event untouched. See
`schemas/lifecycle.py::EventVideoPurgeRecord`.

## `review.py` -- `ReviewAction`

One immutable record per human action, `action: accept | reject | annotate |
retag | preserve | export`. Written once to the append-only JSONL log and
never edited. `preserve` and `export` actions are recorded here *and* the
CLI/review UI also actually triggers the corresponding side effect in the
same call -- see `docs/chain-of-custody.md`.

## `recognition.py` -- local face/plate/voice re-identification

Deliberately has no identity/name fields anywhere -- see
`docs/forensic-considerations.md`'s "Recognition data: scope and intent."

* `FaceObservation` -- one detected face in one frame: crop path/hash,
  detector confidence, which `FaceCluster` it matched (if any), and the
  LBPH distance to that cluster. `duplicate_of_observation_id` (optional)
  is set only by automated cleanup (`recognize faces-cleanup`), never a
  human action.
* `FaceCluster` -- a locally-generated group of similar-looking
  observations. `label` is free-text, user-set, never auto-populated.
  `merged_into` (optional) points at another cluster's id when a human has
  declared them the same person via `recognize faces-merge` -- the cluster
  itself is never edited or deleted, just aliased. See
  `core/recognition.py` for how this is resolved and aggregated.
* `PlateObservation` -- one detected + OCR'd plate: crop, raw and
  normalized OCR text, OCR confidence, and `review_status` (`auto_accepted
  | needs_review | user_confirmed | user_rejected | duplicate_suppressed`)
  gating whether a human needs to look at it. `duplicate_suppressed` is
  set only by automated cleanup (`recognize plates-cleanup`, see
  `docs/local-ai.md`'s automation section), never by a human action;
  `duplicate_of_observation_id` records which other observation was kept
  instead, so the decision stays inspectable.
* `PlateRecord` -- aggregated view of every observation that normalized to
  the same plate text, with a handful of example crops. Also has an
  optional `merged_into`, same semantics as `FaceCluster`'s.
* `VoiceObservation` -- one detected voice segment: no crop file (there's
  no image to keep), instead `voiceprint` -- the actual MFCC-based feature
  vector -- is stored directly, so voice re-identification survives
  deletion of the source clip the same way LBPH's trained model lets face
  re-identification survive deletion of crop images. See
  `enrichment/voice.py` for what the voiceprint actually is, and its
  module docstring for the honesty caveats around this capability's
  real-world accuracy.
* `VoiceCluster` -- mirrors `FaceCluster` in every respect that matters
  (anonymous by default, user-set `label`, `merged_into` for human-declared
  aliasing).
* `IdentityMergeRecord` -- the forensic-grade one in this module: a
  permanent, append-only record (`workspace/identity_merge_log.jsonl`)
  of a human declaring two clusters/plate/voice records the same identity
  -- entity type, source/target id, actor, timestamp, notes. Never edited
  after being written, mirroring `ReviewAction` and `DeletionRecord`. This
  is written both by a direct `faces-merge`/`plates-merge`/`voices-merge`
  call and by confirming a `MergeSuggestion` (below) -- confirming doesn't
  bypass this log, it's the same code path.
* `MergeSuggestion` -- an automated "these might be the same identity"
  flag (`pending | confirmed | rejected`), generated by
  `RecognitionService.suggest_face_merges`/`suggest_plate_merges`/
  `suggest_voice_merges` and never merged automatically -- confirming one
  performs the real merge (writing its own `IdentityMergeRecord`) and
  marks the suggestion resolved; rejecting performs no merge. `basis` is a
  short, human-readable explanation of why the suggestion was made (never
  just an opaque score), and `similarity_score` is normalized to 0-1 so it
  reads consistently even though the underlying distance metric differs
  per entity type (LBPH distance for faces, text-edit-distance ratio for
  plates, voiceprint cosine distance for voices).

Stored authoritatively in SQLite + small crop files, not filesystem
JSON-with-revisions like events -- see `docs/architecture.md`'s
storage-lifecycle section for why this tier is treated differently.

## `enrichment.py` -- transcription and optional LLM analysis

* `TranscriptSegment` / `AudioTranscript` -- Whisper output, one file per
  clip at `workspace/transcripts/<clip_id>.json`.
* `LLMEnrichment` -- the *only* schema whose data can originate from an
  external network service (see `enrichment/llm_analysis.py`). Always
  off by default; always labeled with exactly which `provider`/`model`/
  `endpoint` produced it so it's never ambiguous which findings are local
  and deterministic versus remote and probabilistic. Never overwrites
  `signals`/`hypotheses`/`scoring`.

## `lifecycle.py` -- storage lifecycle

* `TriageRecord` -- current classification (`reviewable` |
  `benign_pending_deletion` | `deleted`) for one clip. Re-derivable from
  events, stored in SQLite like the timeline index.
* `DeletionRecord` -- forensic-grade: a permanent, append-only record
  (`workspace/deletion_log.jsonl`) written *before* an original's bytes
  are removed, so there's always a durable trace that the file existed,
  its hash, and who confirmed deleting it.
* `EventVideoPurgeRecord` -- also forensic-grade, also append-only
  (`workspace/event_video_purge_log.jsonl`), but event-scoped rather than
  clip-scoped: the summary of one `purge_event_video` call, including
  which derived clips were deleted, which original clips were cascaded
  (each of those also gets its own `DeletionRecord` in the usual deletion
  log) versus retained (still needed by another unpurged event), and
  whether the event had been preserved at the time. See
  `core/triage.py::TriageService.purge_event_video`.

## `common.py`

* `HashDigest` -- `{algorithm, value}`, defaults to `sha256`.
* `ArtifactReference` -- `{path, artifact_type, created_at, sha256, metadata}`.
  Used for source media references on signals, derived clips, sidecar
  fixtures, and normalization manifests alike.
* `ChainOfCustodyEntry` -- `{entry_id, action, actor, timestamp, details,
  input_hashes, output_hashes}`. Append-only within an `EventRecord`'s
  `chain_of_custody` list (a new revision appends to the list; it never
  removes or edits an existing entry).

## Validating a stored `event.json` by hand

```python
import json
from gaggle.schemas.event import EventRecord

payload = json.loads(open("workspace/events/<id>/event.json").read())
event = EventRecord.model_validate(payload)  # raises pydantic.ValidationError on drift
```

Any `event.json` on disk, at any revision, should validate against the
current `EventRecord` model as long as `schema_version` matches. A future
schema bump should add a migration path rather than breaking old revisions
silently -- see the "Future-ready design" note in `docs/limitations.md`.
