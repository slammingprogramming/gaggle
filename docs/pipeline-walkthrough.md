# Pipeline walkthrough: from SD card to a storage-optimized workspace

This is the guided, narrative version of `docs/cli-examples.md` -- what to
actually run, in order, and why, from inserting an SD card to ending up
with a workspace that keeps everything meaningful and none of the video
you don't need anymore. See `docs/getting-started.md` first if you haven't
decided which capabilities to install yet.

## The shape of the whole workflow

```
ingest -> analyze -> enrich -> review -> cleanup -> preserve -> purge -> export
```

Every stage writes its own artifacts and can be re-run independently (see
`docs/architecture.md`). You don't have to do this all in one sitting --
it's completely normal to `ingest` and `analyze` right after a drive, then
come back days later to `enrich` and `review` a full card's worth of
footage at once.

## Step 0: decide how you want to handle the source footage

Before your first `ingest`, decide which storage mode fits how you use
your SD card (see `core/config.py::StorageConfig.ingest_mode`):

```bash
gaggle ingest /media/sd-card --workspace ./workspace --mode copy       # default
gaggle ingest /media/sd-card --workspace ./workspace --mode move
gaggle ingest /media/sd-card --workspace ./workspace --mode reference
```

* **`copy` (default)** -- duplicates every file into the workspace,
  leaves the SD card untouched. Safest, but needs roughly 2x the footage's
  size in free space during ingest (source + workspace copy). Good default
  if you're not in a hurry to reuse the card.
* **`move`** -- relocates each file into the workspace instead of copying.
  Frees the SD card the moment ingest finishes, one copy total. Good if
  you want to reuse the card right away and don't need the original
  recording location preserved.
* **`reference`** -- doesn't touch the SD card at all; the workspace just
  remembers where each file lives and reads it from there. Zero extra
  disk use, but the workspace now depends on the card staying connected
  and unmodified until you've preserved anything you care about (see Step
  5 below) -- **this is the one mode where "I'm about to reformat the
  card" and "I haven't preserved my flagged events yet" is a genuinely
  dangerous combination.** If you use `reference` mode, preserve promptly.

You can mix modes across different ingest runs into the same workspace --
this isn't a one-time global choice, it's a per-`ingest`-call flag (or set
`storage.ingest_mode` in config as your default and override with `--mode`
per run).

## Step 1: ingest

```bash
gaggle workspace init --workspace ./workspace   # first time only
gaggle ingest /media/sd-card --workspace ./workspace --mode copy
```

Real `ffprobe` metadata extraction happens here (duration, fps, codec) --
if you see `"probe_status": "failed"` in the output, ffmpeg/ffprobe isn't
on `PATH`; see `docs/developer-setup.md`.

## Step 2: analyze

```bash
gaggle analyze --workspace ./workspace --config examples/config.yaml
```

Runs the full deterministic pipeline (normalize -> window -> detect ->
infer -> score -> build events) and, by default, automatically classifies
every clip as `reviewable` or `benign_pending_deletion`
(`lifecycle.auto_triage_after_analyze: true` in config -- set it to
`false` if you'd rather run `triage run` manually). At this point:

```bash
gaggle triage list --state reviewable --workspace ./workspace
gaggle triage list --state benign_pending_deletion --workspace ./workspace
```

tells you the shape of what you're dealing with: how much of the card was
actually eventful versus just driving.

## Step 3: enrich (optional, but this is where face/plate/vehicle/transcript happen)

```bash
gaggle enrich --workspace ./workspace --config examples/config.yaml
```

Only processes events that already exist (i.e. clips that already
produced at least one signal in Step 2) -- benign footage never gets
face/plate/vehicle/transcription processing, keeping this proportional to
what you'll actually review. See `docs/local-ai.md` for what each
capability does and its config.

### Cutting down plate false positives before you review them

Real footage tends to produce a burst of near-identical plate detections
for the same physical plate across the many frames sampled within one
event -- run the cleanup pass before you look at anything by hand:

```bash
gaggle recognize plates-cleanup --workspace ./workspace
```

This collapses those bursts down to one representative observation per
distinct sighting (see `docs/local-ai.md`'s automation section for exactly
how); nothing is deleted, and anything you've already reviewed by hand is
never touched. Then:

```bash
gaggle recognize plates-review --workspace ./workspace
```

shows what's actually left to look at -- much shorter than before cleanup.
For each entry, look at `crop_path` (the actual cropped image tesseract
tried to read) and either confirm or correct it:

```bash
gaggle recognize plates-confirm <observation-id> "ABC1234" --workspace ./workspace
```

If a plate detection is simply wrong (not a plate at all, or garbage OCR
that wasn't caught automatically), there's currently no dedicated
"reject" command for an individual plate observation -- the pragmatic
options are: leave it alone (it stays `needs_review` forever, harmless,
just noise in the queue) or correct it to an empty/placeholder string if
you want it out of your working set. A first-class reject action is a
reasonable thing to add later; see `docs/limitations.md`.

### Reviewing face detections

There's no automated cleanup for faces yet (only plates -- see
`docs/limitations.md`), but the same principle applies manually:

```bash
gaggle recognize faces --workspace ./workspace
```

lists every face *identity* (already deduplicated by cluster, unlike raw
plate observations). If the same real person shows up as two different
clusters (common -- the built-in detector is a classical Haar cascade +
LBPH clusterer, not a deep embedding; see `docs/local-ai.md`), link them:

```bash
gaggle recognize faces-merge <source-cluster-id> <target-cluster-id> --actor "you" --workspace ./workspace
gaggle recognize faces-sightings <target-cluster-id> --workspace ./workspace   # now shows both
```

## Step 4: review events

```bash
gaggle review start --actor "you" --workspace ./workspace
```

Walks every pending-review event one at a time with a summary and prompts
for accept/reject/annotate/retag/preserve/export/skip/quit. See
`docs/cli-examples.md` for the non-interactive equivalent
(`review queue`/`review action`) if you'd rather script this.

## Step 5: preserve what you want to keep

```bash
gaggle review action <event-id> preserve --actor "you" --workspace ./workspace
```

(or `preserve <event-id>` directly, or pick `preserve` during
`review start`). This copies the event's full history -- including its
derived clips -- into a frozen, immutable bundle under `preserved/`. Do
this for anything you want to survive both raw-footage deletion **and**
the video-purge step below; it's the durable copy everything else assumes
exists.

**If you ingested in `reference` mode**, this is the point where it
becomes safe to disconnect/reformat the SD card for events you've
preserved -- the derived clips (the trimmed, relevant footage) now have a
permanent home in the workspace independent of the card. Events you
haven't preserved are still only as safe as the card itself.

## Step 6: the storage-optimization pass -- purge what you don't need anymore

This is the step for "I've reviewed everything I care about, now I want
the disk space back without losing the metadata." Two separate mechanisms,
because they apply to two different kinds of footage:

### Benign footage (never became an event)

```bash
gaggle triage confirm-deletion --all --actor "you" --workspace ./workspace
```

Deletes every original clip that never contributed to any event, after a
hash check and an append-only log entry recorded first. (Add
`--acknowledge-external` if any of those clips were ingested in
`reference` mode -- see Step 0.)

### Reviewed events (the ones with actual video evidence)

This is the bigger space saver, and the one that needed a dedicated
command: an event's own derived clips, plus the original clip(s) that
contributed to it, are usually the single largest thing about it. Purge
the video while keeping `event.json` -- signals, hypotheses, scoring,
chain of custody, every review decision, forever:

```bash
# One event at a time
gaggle triage purge-event-video <event-id> --actor "you" --workspace ./workspace

# Or in bulk, once you've reviewed a batch -- a natural pattern:
gaggle triage purge-reviewed --actor "you" --review-decision accepted --workspace ./workspace
gaggle triage purge-reviewed --actor "you" --review-decision rejected --workspace ./workspace
```

This refuses to run on an event that hasn't been preserved yet (Step 5) --
without a preserved copy, purging would be the only copy of that video
ever destroyed. Pass `--force` to purge anyway if you're sure you don't
need the video at all, not even a frozen copy.

The cascade logic is automatic and safe across events that share footage
(e.g. two flagged incidents in the same 5-minute clip): purging one
event's video only deletes the shared original once *every* event
referencing it has also been purged. You'll never lose evidence still
needed by an unpurged event.

**What's left after this step:** `event.json` (with full signals,
hypotheses, scoring, chain of custody, review history), face/plate
observations and their small crop images, transcripts, and (for anything
you preserved) a frozen bundle under `preserved/`. Gone: the original
full-length clips and the event's own derived clips in the live workspace
tree. For a typical drive that's mostly uneventful, this is a large
reduction from the original SD card's footprint.

## Step 7: export (optional)

```bash
gaggle export event <event-id> --workspace ./workspace     # self-contained, hash-verified bundle
gaggle export timeline --workspace ./workspace --format csv  # metadata-only spreadsheet export
```

## A complete example, start to finish

```bash
gaggle workspace init --workspace ./workspace
gaggle ingest /media/sd-card --workspace ./workspace --mode move
gaggle analyze --workspace ./workspace
gaggle enrich --workspace ./workspace
gaggle recognize plates-cleanup --workspace ./workspace
gaggle review start --actor "you" --workspace ./workspace
gaggle triage confirm-deletion --all --actor "you" --workspace ./workspace
gaggle triage purge-reviewed --actor "you" --review-decision accepted --workspace ./workspace
gaggle triage purge-reviewed --actor "you" --review-decision rejected --workspace ./workspace
```

No step above requires internet access (aside from the one-time model
downloads if you've enabled `vision`/`transcription`, and the optional
cloud LLM step if you've turned that on -- see `docs/getting-started.md`).
