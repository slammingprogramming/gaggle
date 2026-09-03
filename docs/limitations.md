# Limitations

Honest accounting of what v1.0 does not do, organized by whether it's a
deliberate design choice or a known gap worth closing later.

## Deliberate design choices (not planned to change without a reason)

* **No cloud dependency, ever.** By design. Not a limitation to fix.
* **No object/face/plate classification in the built-in pipeline.** The
  built-in object-hint detector reports bounding boxes for large moving
  regions, not identities or classes. A real classifier is meant to arrive
  as a `DetectorPlugin`, not as a built-in default (see
  `docs/plugin-authoring.md` and the ML-avoidance directive in
  `docs/architecture.md`).
* **Time synchronization is a start-alignment + proportional-drift
  heuristic**, not measured clock offset. There is no audio/video
  cross-correlation between cameras. See `normalize/sync.py`'s docstring
  and `docs/architecture.md`.
* **Derived clips are cut with `ffmpeg -c copy`** (stream copy, no
  re-encode) for speed and byte-fidelity, which means cuts snap to the
  nearest keyframe and can run a little longer than the requested window.
  This is treated as a feature (favors over-inclusion) per the
  false-positive philosophy, not a bug to fix.

## Known gaps worth closing in a future release

* ~~No cryptographic signing of the revision hash chain.~~ -- closed: see
  `core/signing.py`, `workspace signing-init`, and the `signing.enabled`
  config flag. Off by default; requires the `signing` extra
  (`pip install gaggle[signing]`). See `docs/local-ai.md`'s
  "Cryptographic signing of the revision hash chain" section for the full
  design, including what's NOT covered: signing proves a revision was
  written by whoever holds the private key, not that the key belongs to
  who you think it does (key custody/distribution is a
  process/organizational control, not something this project can
  enforce), and a revision written before signing was turned on has no
  signature (`revision_signature: null` is expected there, not a bug).
* **No standalone bundle verifier for recipients without the package
  installed.** ~~`export event` produces a hash-manifested zip, but
  verifying it today means writing a short script~~ -- closed: see
  `scripts/verify_export_bundle.py`, a dependency-free (standard-library
  only) script a recipient can run with just `python3`, no
  `gaggle` install required:
  `python3 scripts/verify_export_bundle.py path/to/bundle.zip`.
* **Per-clip drift correction within a multi-clip session is applied as a
  flat, session-level offset**, not interpolated linearly across the
  session's duration. `CameraSync.drift_seconds_per_hour` is computed and
  recorded accurately at the session level, but an individual clip late in
  a long session doesn't currently get a proportionally larger correction
  than one at the start of the same session. For typical dashcam file
  rollover intervals (a few minutes) this is a small effect; it would
  matter more for very long unbroken recording sessions.
* **No audio/video cross-correlation for sync verification.** If two
  cameras both captured the same loud sound, that could in principle be
  used to measure true offset far more precisely than start-alignment.
  This is a natural fit for a future built-in enhancement or a plugin.
* **Ingest still has no protection against maliciously oversized source
  directories** (symlink-loop protection was closed -- see below). Treat
  ingest sources as at least as trusted as any other local file operation
  (see `docs/threat-model.md`).
* **Pattern analysis is a manual/periodic step** (`patterns analyze`), not
  triggered automatically after every `analyze` run. It operates only on
  already-generated events already indexed in the workspace.

## Local AI / recognition capabilities (added post-v1.0)

* **The classical detectors (Haar cascade face detection, LBPH
  re-identification, Russian-format-calibrated plate cascades) are no
  longer what ships by default as of 1.9** -- `yunet`/`auraface`/
  `fast_alpr` are now the config defaults (see below and
  `docs/local-ai.md`). They remain fully supported as the automatic
  fallback when the relevant extra isn't installed or a model fails to
  load, so the caveats below still apply whenever a workspace is running
  in that fallback mode:
  * **Face detection accuracy is bounded by a 2001-era algorithm.** Haar
    cascades are far less accurate than a modern deep-learning face
    detector, especially off-angle, in low light, or at a distance --
    expect missed detections on real dashcam footage, not just false
    ones. This is a deliberate zero-dependency tradeoff (see
    `docs/local-ai.md`), not an oversight; a more accurate detector is a
    natural `DetectorPlugin` extension.
  * **Face re-identification is LBPH texture matching, not a deep
    embedding.** It answers "does this look similar" within one
    workspace's own model, not "is this the same person" with anything
    like the confidence a modern face-embedding network would give.
    Treat cluster assignments as a starting point for human review, not
    a determination.
  * **License plate cascades are Russian-format-calibrated.** The
    bundled OpenCV cascades were trained on Russian plate proportions;
    the heuristic contour-based fallback is format-agnostic but cruder.
    Real-world accuracy on non-Russian plates depends heavily on camera
    angle, resolution, and lighting -- validate against your own footage
    before relying on it, and use `recognize plates-review` for anything
    OCR wasn't confident about.
* **YOLO vehicle detection and Whisper transcription require one-time,
  user-initiated downloads** (a model file / CTranslate2 weights
  respectively) that this project does not ship. Until you provide one,
  those capabilities silently produce nothing (logged once), which is
  correct behavior, not a bug -- see `docs/local-ai.md`.
* **Enrichment signals don't trigger re-scoring.** A face/plate/vehicle
  detection found during `enrich` is added to the event for a human to see,
  but does not change `scoring.severity`/`scoring.confidence`, which stay
  exactly as they were when `analyze` first computed them. This is
  intentional (see `docs/local-ai.md`) but means severity alone doesn't
  reflect "an unusual face/plate was seen" -- check `signals` too.
* **The storage-lifecycle triage classification is binary and per-clip.**
  A clip with even one weak, low-confidence signal is "reviewable" and
  never becomes a deletion candidate through `triage`, even if a human
  later rejects the event entirely. There is currently no "reviewed,
  rejected, now eligible for deletion too" follow-on workflow -- rejected-event
  originals stay in `originals/` indefinitely unless manually removed
  outside this tool. This is a reasonable v1 scope boundary (see
  `docs/local-ai.md`'s storage-lifecycle section), not a decision that
  rejected footage is being deliberately retained forever.
* **No "unmerge" command.** A merge can be corrected by re-merging in a
  different direction, but there's currently no way to fully undo one and
  return two identities to being fully independent again -- the
  `identity_merge_log.jsonl` entry itself is never deleted either (it's
  append-only, like every other audit log in this project), so a mistaken
  merge is visible and traceable, just not automatically reversible.
* **Fuzzy plate search uses a fixed similarity cutoff (0.7), not a
  learned or configurable one.** `recognize plates-search` will surface
  "ABC1234" as a suggestion for a query like "ABC1Z34" but won't catch
  every plausible OCR misread, and could occasionally suggest an unrelated
  plate that happens to share several characters.
* **Ingest storage mode conversion only works one direction.**
  `triage convert-mode <clip_id> --to copy|move` upgrades a `reference`-mode
  clip into a durable, workspace-owned copy after the fact. The reverse
  (`copy`/`move` -> `reference`) is refused outright, deliberately, with no
  override -- it would mean deleting the workspace's only owned copy of a
  file that might already be the sole surviving copy, with no way to
  verify a new external dependency actually has matching bytes. Converting
  also doesn't retroactively fix any already-existing event's
  `evidence_references`, which keep pointing at the old external location
  (append-only provenance is never rewritten) -- it only benefits future
  reads. See `docs/local-ai.md`'s "Choosing an ingest storage mode"
  section.
* **`triage purge-event-video`'s cascade check is a full re-scan of every
  event's signals on every call**, not an incrementally-maintained index.
  For a workspace with a very large number of events this could become
  noticeably slow; it hasn't been a practical problem at the scale this
  project is designed for (a single vehicle's dashcam archive), but it's
  worth knowing this doesn't scale to, say, a fleet-wide shared workspace
  with hundreds of thousands of events.

* **`reference`-mode ingest creates an external dependency the workspace
  can't enforce.** Once ingested, nothing prevents the referenced source
  location (e.g. an SD card) from being modified, disconnected, or wiped
  before you've preserved anything you care about. The system will
  degrade safely if that happens (missing files are logged and skipped,
  never silently treated as present), but "safely" here means "you lose
  access to that evidence," not "nothing bad happens." This is a real
  tradeoff of the mode, not a bug -- see `docs/local-ai.md`'s "Choosing an
  ingest storage mode" for the recommended pairing with prompt
  preservation.
* **Vehicle re-identification by visual description (color, body shape)
  without a legible plate is a coarse, classical fingerprint --
  meaningfully weaker than face or plate matching.**
  `enrichment/vehicle_appearance.py` (`recognize vehicles-*`) matches on a
  dominant hue/saturation histogram plus aspect ratio -- deliberately
  classical, not a learned embedding, consistent with how face/plate/voice
  re-identification are built. It **cannot distinguish two different
  vehicles of the same color and body shape** (e.g. two white sedans of
  the same generation), and is meaningfully more sensitive to
  lighting/angle/dirt/wear than a plate reading. Validated only against
  synthetic vehicle-colored test scenes during development (five hues,
  eight noise seeds each -- see `enrichment/vehicle_appearance.py`'s
  module docstring for the measured same/different-vehicle distance
  distribution), not real footage -- expect to retune
  `enrichment.vehicle_appearance.cluster_distance_threshold` against your
  own footage before relying on it, exactly like voice's caveat below.
  The zero-setup default detection path (a classical "vehicle-shaped
  moving blob" heuristic) is also noisier than face/plate detection's
  cascades -- expect more false-positive region detections; the optional
  YOLO detector (`enrichment.vision`), when enabled, is used opportunistically
  for more precise vehicle boxes.
* **Person (pedestrian) re-identification by visual appearance is a coarse,
  classical fingerprint, weaker still than vehicle-appearance re-ID, and
  requires the optional YOLO detector -- there is no zero-setup fallback.**
  `enrichment/person_appearance.py` (`recognize persons-*`) is the same
  dominant hue/saturation histogram + aspect-ratio technique as
  vehicle-appearance, applied to detected person regions. It **cannot
  distinguish two different people wearing similarly-colored clothing**,
  and is meaningfully more sensitive to lighting/angle/clothing changes
  between sightings than vehicle-appearance matching is to a vehicle's
  paint and shape. Unlike vehicle-appearance, there is no classical
  "person-shaped blob" fallback detector -- this capability only produces
  anything at all when `enrichment.vision.enabled: true` and a YOLO model
  file are both configured, and defaults to `enabled: false` for exactly
  that reason. Validated only against synthetic test scenes during
  development, not real footage -- same caveat class as
  voice/vehicle-appearance re-identification below.
* **The vehicle telemetry detector only recognizes GPX as a real (non-fixture)
  GPS track format, and assumes one track per camera directory per ingest
  session.** `detection/telemetry_analysis.py` parses `.gpx` files (an
  open, stdlib-parseable XML standard -- no new dependency, consistent
  with this project's ffmpeg/tesseract-style "external standard tool over
  a heavy pip dependency" pattern) and hand-computes speed/heading from
  consecutive GPS points via the haversine distance and initial-bearing
  formulas. At ingest, `ingest/service.py` looks for a `.gpx` file
  colocated with each source clip's own directory and copies at most one
  into the workspace as a `gps_track` sidecar, **matched by presence, not
  filename or explicit clip correlation** -- if a camera directory
  contains more than one `.gpx` file, only the first in sorted order is
  used (logged as a warning), and any other GPS-track file formats
  (NMEA, proprietary dashcam telemetry containers) aren't recognized at
  all. This is a deliberate first-pass scope boundary, not a design
  claiming to support arbitrary/multi-session telemetry ingestion -- see
  `docs/local-ai.md`'s telemetry section.
* **The optical-flow "rapid approach" detector is a corroborating
  heuristic, not a reliable collision-warning system.**
  `detection/optical_flow_analysis.py` uses dense Farneback flow and a
  comparative (not absolute) divergence threshold specifically to reject
  ordinary ego-motion, but this remains a coarse classical technique:
  low-texture or night-driving frames starve Farneback of gradient
  information (the same limitation frame differencing already has);
  very fast real-world closures between the default 2Hz samples can
  produce displacement large enough to break Farneback's local-window
  tracking assumption, showing up as noisy rather than cleanly-signed
  divergence; and a genuinely slow, gradual approach can fall below what
  this technique resolves at all (see the module docstring's measured
  true-positive/true-negative distributions -- one of five measured
  growth rates was too subtle to produce any signal). Never sufficient
  alone for high severity (invariant 7), and not validated against real
  dashcam footage in this environment, only synthetic test scenes --
  same caveat class as voice/vehicle-appearance re-identification below.
* **Gunshot detection cannot distinguish a real gunshot from a car door
  slam, an engine backfire, or a firework/firecracker with full
  confidence, and its accuracy against real gunshot audio is
  unvalidated in this environment.** `detection/gunshot_analysis.py`
  (`detection.gunshot.enabled`, off by default) uses a pretrained ONNX
  audio classifier (k2-fsa's zipformer-small AudioSet tagger, via the
  optional `sherpa-onnx` dependency) rather than a classical
  impulse-detection heuristic -- a deliberate choice after researching
  both, since a classical heuristic built on rise-time/crest-factor
  alone cannot reliably separate a gunshot from those other sharp
  acoustic transients either. The classifier was verified for real
  during development (downloaded, license/hash-checked, its real
  ONNX input/output contract inspected, and run against the model's own
  bundled real-world test clips -- zero false "gunshot-like" matches on
  cat/dog/siren/baby-cry/smoke-alarm/etc.), but **no real or synthetic
  gunshot audio was available to test a true positive against** -- unlike
  the classical color/shape fingerprints vehicle-appearance/
  person-appearance use, there is no honest way to synthesize a
  realistic gunshot waveform for a fixture. `Fireworks`/`Firecracker`
  are deliberately excluded from the "gunshot-like" class set (conflating
  them with actual gunfire would be actively misleading), but the
  classifier's own confusion between acoustically-similar classes is
  real and unresolved. A lone gunshot detection is capped at 0.60
  confidence and, per invariant 7, can never alone reach medium/high
  severity -- treat it as a corroborating prompt for review, never a
  confirmed identification. See `docs/local-ai.md`'s "Gunshot detection"
  section.
* **Voiceprinting's real-world accuracy is unvalidated.** Unlike face and
  plate detection (validated against real and realistic synthetic
  imagery), the voice capability was validated only against synthetic
  multi-tone test signals during development, not real recorded human
  speech -- see `docs/local-ai.md`'s "Voice detection and local
  voiceprinting" section and `enrichment/voice.py`'s module docstring for
  the specifics of what was and wasn't tested. Treat every voice match as
  a considerably weaker signal than a face or plate match, and expect to
  retune `enrichment.voice.cluster_distance_threshold` against your own
  footage rather than trusting the shipped default blindly.
* **Merge suggestions are generated on demand, not automatically.**
  `recognize suggest-merges` has to be run explicitly (e.g. after
  `enrich`); nothing watches for newly-fragmented identities and flags
  them on its own. This is a deliberate choice (predictable, inspectable
  runs rather than background magic) but means a stale suggestion queue
  is a "did you forget to run this" problem, not a system malfunction.
* **`Encounter` records claim co-occurrence, never spatial correspondence.**
  `schemas/encounter.py` groups face/plate/voice/vehicle-appearance
  observations that happened close together in time within the same clip
  -- it does *not* mean a co-listed face was near, belongs to, or was
  driving a co-listed vehicle. None of the four observation schemas
  currently store a bounding box on the record itself (only `crop_path`),
  so there's no data yet to disambiguate multiple simultaneous entities of
  the same type spatially within a frame; when more than one observation
  of a modality falls in the same time window, they're paired with the
  other modalities' observations index-wise (1st with 1st, 2nd with 2nd),
  a bookkeeping convenience, not a claim about which entity was actually
  near which. `patterns/service.py`'s recurring face+vehicle
  co-occurrence pattern inherits this same scope limit.
* **Security-camera config profiles are reasoned starting points, not
  empirically validated ones.** `examples/config/security-outdoor.yaml`
  and `security-indoor.yaml` (see `docs/local-ai.md`'s "Security camera
  support" section) adjust `detection.motion_threshold`,
  `detection.optical_flow.roi_divergence_delta_threshold`, and
  `sync.session_gap_seconds` based on general reasoning about
  indoor/outdoor scene noise, the same way `examples/config.yaml`'s
  `quiet_environment` profile already was -- neither has been measured
  against real security-camera footage. Expect to retune both against
  your own footage.
* **Live/streaming camera ingestion (RTSP IP cameras, a directly-attached
  USB webcam) is out of scope.** Every source -- dashcam or security
  camera -- is ingested from already-written files (`gaggle ingest
  <directory>`); there is no continuous-capture or network-camera-polling
  mode. A camera that only offers a live stream needs an external
  recorder/NVR to produce files first. This was an explicit scope decision
  for this pass, not an oversight -- see `docs/architecture.md`'s module
  map and AGENTS.md's status tracker for what a future `SourceAdapterPlugin`
  covering this would look like.
* **`pipeline.max_event_duration_seconds`'s 120s default is a reasoned
  starting point, not validated against every real driving scenario.**
  `core/pipeline.py::_cluster_overlapping_windows` forces a split once a
  merged event's span would exceed this cap, even though the next window
  still temporally overlaps -- necessary because near-continuous real
  motion throughout a long recording otherwise merges into one
  arbitrarily long event (observed directly: a real ~5-minute dashcam
  clip became a single 4,081-signal event before this cap existed). The
  tradeoff: a forced split at the cap boundary can in principle separate
  two halves of one real continuous incident into two events. Set it to
  `null` to restore the old unbounded-merge behavior, or raise/lower it
  per how long a real incident tends to run in your own footage.
* **Recognition crop purging is a disk-space optimization, not evidence
  preservation.** `RecognitionService.purge_reviewed_crops` deletes the
  small JPEG crop image for an already-reviewed observation -- the
  structured observation data (timestamp, camera, confidence, cluster/
  encounter linkage, `crop_sha256` proof of what the image was) is never
  deleted, but the actual pixels are gone. Neither `preservation/` nor
  `export/` currently copy recognition crops into a preserved/exported
  bundle at all (only derived clip video is preserved/exported today) --
  so purging a crop doesn't affect an already-preserved event either way,
  but it also means a preserved/exported bundle never included the crop
  images to begin with. If you need to keep the actual crop pixels for a
  specific reviewed identity, do that before purging (e.g. copy the file
  yourself) -- there is currently no built-in "export recognition crops"
  path.
* **The deep-learning face/plate options (YuNet, AuraFace, fast-alpr) are
  real, working models exercised end to end during development against
  real photos and a real synthetic pipeline run -- but none of the three
  has been independently validated against this project's own real
  American dashcam footage at scale, and their accuracy/threshold
  defaults are reasoned starting points, same honesty standard as every
  other threshold in this project:**
  * `enrichment.face.embedding_cluster_distance_threshold` (default
    0.35, cosine distance over normalized AuraFace embeddings) comes from
    published ArcFace-family benchmark ranges, not this project's own
    footage. It was spot-checked against one real photo (a same-crop
    comparison landed at distance 0.0; a real-face-vs-random-noise
    comparison landed at distance ~0.94, a wide separation) -- reassuring,
    but one data point, not a validation study. Retune against your own
    footage before trusting cluster assignments at face value.
  * fast-alpr's OCR model is trained on international plate formats
    (deliberately not Russian-locked, unlike the classical cascade), but
    has not been benchmarked against real American plates specifically by
    this project -- only confirmed to run correctly end to end (real
    model download, real inference, real region-guess output) on a
    non-plate test photo. If accuracy on your plates proves insufficient,
    fast-plate-ocr's own documentation describes fine-tuning on a custom
    dataset as a next step, out of scope for this project.
  * **A real, verified (not hypothetical) local-inference gap**: deriving
    an int8 model locally from AuraFace's fp32 recognition weights
    produces a file that loads successfully but then fails at
    `onnxruntime` inference-session creation with an unimplemented
    `ConvInteger` kernel, on this project's `onnxruntime` CPU build.
    `core/models.py::ModelRegistry.ensure_model` detects this (by
    actually loading what it just derived, not just trusting the
    conversion call succeeded) and falls back to the fp32 model instead
    of failing outright -- correct, safe behavior, but it means
    `device: cpu` for AuraFace specifically may run at fp32 rather than
    the faster int8 depending on your `onnxruntime` build, not a
    regression, just worth knowing if performance looks different from
    what "int8 on CPU" would suggest.
  * Landmark-based face-crop alignment (YuNet already produces 5-point
    landmarks; AuraFace's embedding accuracy would likely improve from
    using them) is not implemented -- both faces are currently embedded/
    matched from an axis-aligned bounding-box crop only. A documented
    follow-up, not a hidden gap.
* **`suggest_face_merges`'s `embedding_model` argument must match
  whichever backend actually produced the clusters being compared** --
  the CLI (`recognize suggest-merges --entity-type face`) and `enrich`'s
  own dispatch always pass the currently-configured
  `enrichment.face.embedding_model` automatically, so this only becomes a
  footgun if `RecognitionService.suggest_face_merges` is called directly
  (e.g. from a script) with a mismatched value -- LBPH distances and
  AuraFace cosine distances are on completely incompatible numeric scales
  (roughly 0-100+ vs. 0-2), so passing the wrong one wouldn't error, it
  would just produce meaningless similarity scores.
* **Installing `face_recognition`/`plate_recognition` can silently break
  the classical face/plate path too -- a real incident, not a
  hypothetical.** `insightface` and `fast-alpr`'s own dependency chains
  (`albumentations`, `open-image-models`, `fast-plate-ocr`) require
  `opencv-python`/`opencv-python-headless`, unpinned. Those install into
  the exact same `cv2` folder in site-packages as this project's actual
  dependency, `opencv-contrib-python-headless` -- pip has no concept of
  these being mutually exclusive (they're different package names on
  PyPI), so whichever installs last wins and can leave `cv2.face`/
  `cv2.data`/other contrib symbols missing entirely, breaking the
  classical LBPH/Haar-cascade path (`AttributeError: module 'cv2.face'
  has no attribute 'LBPHFaceRecognizer_create'`) even though nothing
  about that path was touched. Documented with the exact fix-up command
  (`pip install --force-reinstall --no-deps
  "opencv-contrib-python-headless>=4.10.0,<5.0.0"`) in
  `docs/getting-started.md`'s deep-learning install section -- run it
  after installing either extra, every time, not just once.

## Testing/verification limitations specific to this project

Everything in this codebase was written and reviewed carefully, including
running the standalone (non-pydantic) computational modules --
`ingest/probe.py`, `detection/video_analysis.py`, `detection/audio_analysis.py`,
`normalize/sync.py`, `core/derived_clips.py`, and several `enrichment/`
modules -- against real generated media during development. The full
pydantic/FastAPI/SQLAlchemy-dependent test suite (`pytest`) could not be
executed in the sandbox environment most of this project was authored in,
due to no network access to install those dependencies. Before relying on
this as stable, run the full suite yourself:

```bash
pip install -e .[dev,vision,cloud]
pytest
ruff check .
mypy src
```

and treat any failures as bugs in this pass, not as expected behavior.

**This limitation is not hypothetical -- it has already caused three real
issues to ship**, all caught by an actual user running the CLI against
real footage rather than by any review or testing done here:

1. A `DetachedInstanceError` crash in `TriageService.classify_all()`,
   caused by SQLAlchemy expiring loaded row attributes on session commit
   while every `TimelineDatabase` `list_*`/`get_*` method's whole design
   assumes callers read those attributes *after* the short-lived query
   session has already closed. Fixed by setting `expire_on_commit=False`
   on the session factory (see
   `storage/database.py::TimelineDatabase.session`).
2. A `ValidationError: timestamp must be timezone-aware` crash in `enrich`,
   caused by a separate, also well-known SQLAlchemy+SQLite limitation:
   SQLite has no native timestamp-with-timezone type, and SQLAlchemy's
   SQLite dialect silently returns *naive* datetimes on read regardless of
   what was written with `DateTime(timezone=True)`. Every timestamp field
   in this project's pydantic schemas requires a timezone-aware value, so
   any code reading a previously-stored timestamp back out of the index
   and feeding it into a new model instance (exactly what
   `EnrichmentService` does to preserve prior stats across observations)
   would eventually hit this. Fixed with a custom `UTCDateTimeColumn` type
   (see `storage/database.py`) applied to every datetime column in the
   schema, which re-attaches UTC on every read and normalizes to UTC on
   every write -- a systemic fix rather than patching each call site.
3. Not a crash but a real usability/efficiency bug: with plate recognition
   enabled (the default) and no `tesseract` binary installed, OCR was
   attempted separately for every detected plate-shaped region in every
   sampled frame -- potentially hundreds of doomed subprocess spawns per
   `enrich` run, each logging an identical warning. This one wasn't a
   library-behavior surprise like the two above; it was simply missing the
   "check an optional dependency's availability once, cache it" pattern
   already used correctly for the vehicle-detection and transcription
   capabilities. Fixed by adding that same caching to plate OCR.

All three were fixed with regression tests added, but the first two in
particular are concrete examples of the kind of bug that static review and
hand-tracing cannot reliably catch -- they depend on runtime library
behavior (SQLAlchemy session lifecycle, SQLite's datetime storage model),
not code structure that's visible from reading the source. Treat this
project's correctness claims accordingly: high confidence in what's been
executed against real data (see the module lists throughout this
document), meaningfully lower confidence in anything that has only been
reviewed -- and if you hit something odd while actually using this tool,
please report it; that's how all three of these were found.
