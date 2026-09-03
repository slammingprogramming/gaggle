# Chain of custody

Every stage that touches evidence records what it did, on what input, with
what result. This document explains where that record lives and how to read
it.

## Where provenance is recorded

| Question | Where the answer lives |
|---|---|
| Where did this media file come from? | `MediaClip.source_path` (original path at ingest time) and `MediaClip.metadata["source_relpath"]` |
| When was it ingested, and by what? | `IngestManifest.created_at`, `IngestManifest.run_id` |
| What's its hash? | `MediaClip.sha256`, also duplicated into `EventRecord.hashes` for any event built from signals referencing it |
| Was its timestamp corrected, and why? | `CameraSync.rationale` / `NormalizedClip.sync_rationale` (see `docs/architecture.md`'s time-sync section) |
| What generated this event? | `EventRecord.chain_of_custody[0]` (`action="event_generated"`), with `details.ingest_run_id` and `input_hashes` pointing back to every ingested clip |
| Was it preserved? By what, when, into what bundle? | `EventRecord.chain_of_custody` entry with `action="event_preserved"`, `details.bundle_path`/`bundle_hash` |
| Was it exported? To where? | `EventRecord.chain_of_custody` entry with `action="event_exported"`, `details.export_path`/`manifest_hash` |
| What did a human decide, and when? | `review/<event_id>.jsonl` (the full `ReviewAction` log) plus `EventRecord.review_summary` (the latest-state summary) |
| Did review ever overwrite an automated finding? | Never -- see below |

## `chain_of_custody` entries vs. revisions vs. the review log

These three mechanisms look similar and are easy to conflate; they exist for
different reasons:

* **`EventRecord.chain_of_custody`** (a list *inside* the event) records
  discrete provenance-relevant *actions performed on the event as a whole*:
  generation, preservation, export. It grows by appending a new entry as
  part of a new revision -- it is never rewritten in place.
* **Revisions** (`events/<id>/revisions/000N_*.json`) are full point-in-time
  snapshots of the entire `EventRecord`, taken every time *anything* about
  the event changes -- including but not limited to a new chain-of-custody
  entry. A review action that doesn't touch chain-of-custody at all (e.g.
  `annotate`) still produces a new revision because `review_summary`
  changed.
* **The review log** (`review/<id>.jsonl`) is the append-only record of
  individual human actions -- one line per `ReviewAction`, independent of
  the event's revision history, and it is what `review_summary` is derived
  from. It is never truncated or rewritten; new actions are always
  appended.

In short: the review log is "what did a human do," chain-of-custody is
"what did the system do to this evidence," and revisions are "what did the
event look like at each point in time" (a superset that includes the
effects of both of the above).

## Human review never overwrites automated output

This is enforced structurally, not just by convention: `ReviewAction` and
`EventRecord.signals`/`hypotheses`/`scoring` are separate fields, and
`Repository.save_event_revision()` -- the only code path that can change an
existing event -- is called by review actions with an `update` dict that
only ever touches `review_summary` and (for preserve/export) `preservation_status`
/`chain_of_custody`. There is no code path that lets a review action modify
`signals`, `hypotheses`, or `scoring`. If a human disagrees with an
automated finding, that disagreement is recorded as a review action
alongside the original finding -- the original finding is never edited or
removed.

## Reconstructing full custody for an event, end to end

```bash
gaggle review revisions <event-id> --workspace ./workspace   # every revision, in order
gaggle review history <event-id> --workspace ./workspace    # every human action
gaggle export event <event-id> --workspace ./workspace      # bundles all of the above
```

The exported bundle (`export/service.py`) is deliberately self-contained:
full revision history, the review log, and (if preserved) the frozen
preservation bundle, all under one `export_manifest.json` with a hash for
every included file plus a top-level `manifest_hash`. A recipient can
verify that manifest without installing `gaggle` at all:

```bash
python3 scripts/verify_export_bundle.py path/to/event_<id>_<ts>.zip
```

## Known gap: no external timestamping or signing

See `docs/threat-model.md`'s "Chain-of-custody / hash-chain limitations"
section -- the hash chain is internally verifiable but not externally
anchored (no trusted timestamping authority, no cryptographic signature).
This is an intentional, documented v1.0 limitation, not an oversight; the
schema (`previous_revision_hash` on every revision) was designed so signing
can be added later without a breaking schema change.
