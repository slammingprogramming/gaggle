# CLI examples

Every command takes `--workspace <path>`; most read config via
`--config <path.yaml>` (see `examples/config.yaml`) and fall back to
built-in defaults if omitted.

## Setup and ingest

```bash
# Create the workspace layout (idempotent -- safe to rerun)
gaggle workspace init --workspace ./workspace

# Copy media in immutably, hash everything, extract real duration/fps/codec
gaggle ingest /path/to/sd-card --workspace ./workspace --config examples/config.yaml

# Choose how the source is handled instead of the "copy" default -- see
# docs/getting-started.md and docs/pipeline-walkthrough.md's Step 0
gaggle ingest /path/to/sd-card --workspace ./workspace --mode move
gaggle ingest /path/to/sd-card --workspace ./workspace --mode reference
```

## Analysis

```bash
# Runs normalize -> window -> detect -> infer -> score -> build events,
# against the most recent ingest manifest in the workspace. Also runs
# triage automatically unless lifecycle.auto_triage_after_analyze is false.
gaggle analyze --workspace ./workspace --config examples/config.yaml
```

Prints the generated `EventRecord`s as JSON. Each is also written to
`workspace/events/<id>/event.json` (see `docs/architecture.md`).

## Enrichment (local face/plate/vehicle detection, transcription)

```bash
# Face + plate recognition run by default (fully offline, zero setup);
# vehicle detection / transcription / cloud LLM analysis stay off until
# configured -- see docs/local-ai.md
gaggle enrich --workspace ./workspace --config examples/config.yaml

# Re-run enrichment for just one event
gaggle enrich --workspace ./workspace --event-id <event-id>
```

## Storage lifecycle (triage, event-video purge, and deletion)

```bash
gaggle triage run --workspace ./workspace
gaggle triage list --state reviewable --workspace ./workspace
gaggle triage list --state benign_pending_deletion --workspace ./workspace
gaggle triage confirm-deletion <clip-id> --actor "jane" --workspace ./workspace
gaggle triage confirm-deletion --all --actor "jane" --notes "reviewed, all benign" --workspace ./workspace
# add --acknowledge-external if any of those clips were ingested with --mode reference

# Purge a reviewed event's video (derived clips + cascaded originals) while
# keeping event.json/signals/hypotheses/scoring/history forever. Refuses
# unless the event is already preserved, unless --force is passed.
gaggle triage purge-event-video <event-id> --actor "jane" --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision accepted --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision rejected --workspace ./workspace
```

See `docs/pipeline-walkthrough.md` for the guided version of this whole
workflow, including when to use each command.

## Recognition (local face/plate re-identification)

```bash
gaggle recognize faces --workspace ./workspace
gaggle recognize faces --include-merged --workspace ./workspace
gaggle recognize faces-label <cluster-id> "neighbor" --workspace ./workspace
gaggle recognize faces-search <partial-uuid-or-label> --workspace ./workspace
gaggle recognize faces-sightings <cluster-id> --workspace ./workspace
gaggle recognize faces-sightings <cluster-id> --exact --workspace ./workspace

# Declare two clusters/plates are the same person/vehicle (never edits or
# deletes either one -- permanently logged to identity_merge_log.jsonl)
gaggle recognize faces-merge <source-cluster-id> <target-cluster-id> --actor "jane" --workspace ./workspace
gaggle recognize faces-identity <any-member-cluster-id> --workspace ./workspace

gaggle recognize plates --workspace ./workspace
gaggle recognize plates-cleanup --workspace ./workspace
gaggle recognize plates-review --workspace ./workspace
gaggle recognize plates-confirm <observation-id> "ABC1234" --workspace ./workspace
gaggle recognize plates-search ABC1Z34 --workspace ./workspace
gaggle recognize plates-sightings ABC1234 --workspace ./workspace

gaggle recognize plates-merge <source-plate-id> <target-plate-id> --actor "jane" --notes "OCR misread" --workspace ./workspace
gaggle recognize plates-identity ABC1234 --workspace ./workspace
```

See `docs/local-ai.md`'s "Linking sightings to the same person or vehicle"
for the merge/search/identity workflow, and the rest of that document for
what each capability does, its config, and its offline/optional-online
boundaries.

## Review

```bash
# Full review queue, optionally filtered
gaggle review queue --workspace ./workspace
gaggle review queue --workspace ./workspace --severity high
gaggle review queue --workspace ./workspace --camera front --status pending

# Record a decision (append-only; never overwrites signals/hypotheses/scoring)
gaggle review action <event-id> accept --actor "jane" --workspace ./workspace
gaggle review action <event-id> reject --actor "jane" --notes "false positive: tree branch" --workspace ./workspace
gaggle review action <event-id> annotate --actor "jane" --notes "possible near-miss, flag for insurance" --workspace ./workspace

# "preserve" and "export" actions both log the action AND actually trigger
# the corresponding effect -- the audit trail never lies about whether it happened
gaggle review action <event-id> preserve --actor "jane" --workspace ./workspace
gaggle review action <event-id> export --actor "jane" --workspace ./workspace

# Inspect history
gaggle review history <event-id> --workspace ./workspace      # every ReviewAction
gaggle review revisions <event-id> --workspace ./workspace    # every event.json revision

# Interactively walk through everything pending review, one at a time,
# with a prompt for accept/reject/annotate/retag/preserve/export/skip/quit
gaggle review start --actor "jane" --workspace ./workspace
gaggle review start --actor "jane" --severity high --workspace ./workspace
```

## Preservation

```bash
# Equivalent to `review action <id> preserve` but without an accompanying
# review action entry -- useful for scripted/bulk preservation
gaggle preserve <event-id> --workspace ./workspace
```

## Timeline querying

```bash
gaggle timeline query --workspace ./workspace
gaggle timeline query --workspace ./workspace --severity medium --camera rear
gaggle timeline query --workspace ./workspace \
  --start-after 2026-05-01T00:00:00Z --start-before 2026-06-01T00:00:00Z \
  --preservation-state preserved --limit 20
```

## Pattern analysis

```bash
# Metadata-only pattern hypotheses over every generated event
# (repeated camera activity, repeated object labels, temporal clustering)
gaggle patterns analyze --workspace ./workspace
gaggle patterns analyze --workspace ./workspace --cluster-window-seconds 1800 --min-repeat-count 3
```

Results are also written to `workspace/patterns/<timestamp>.json`.

## Export

```bash
# Self-contained, hash-manifested zip: full revision history, review log,
# and the frozen preservation bundle if the event was preserved
gaggle export event <event-id> --workspace ./workspace

# Flat timeline export for spreadsheets/reporting (metadata only, no media)
gaggle export timeline --workspace ./workspace --format csv
gaggle export timeline --workspace ./workspace --format json --severity high
```

## Plugins

```bash
gaggle plugins list
```

## Review UI

```bash
gaggle review-ui --workspace ./workspace --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000` for the review queue, or
`http://127.0.0.1:8000/events/<event-id>` for an individual event's
synchronized multi-camera playback page. The same functionality is
available as JSON under `/api/events`, `/api/events/<id>`,
`/api/events/<id>/review-actions`, `/api/events/<id>/revisions`, and
`/api/timeline` -- see `review_ui/app.py` for the full route list.

## A complete offline workflow, start to finish

```bash
gaggle workspace init --workspace ./workspace
gaggle ingest /media/sd-card --workspace ./workspace
gaggle analyze --workspace ./workspace          # also triages automatically
gaggle enrich --workspace ./workspace            # face/plate recognition, offline by default
gaggle review start --actor "jane" --workspace ./workspace
gaggle triage confirm-deletion --all --actor "jane" --workspace ./workspace
```

No step above requires internet access. (`enrich`'s optional
`vision`/`transcription`/`cloud` capabilities do -- once, to download a
model, or per-call if you explicitly enable cloud LLM analysis -- see
`docs/local-ai.md`.)
