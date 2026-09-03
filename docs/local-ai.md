# Local AI capabilities

This document covers everything added under `src/gaggle/enrichment/`:
local face/plate re-identification, optional local vehicle detection,
optional local transcription, and optional cloud LLM transcript analysis.
Read `docs/forensic-considerations.md`'s "Recognition data: scope and
intent" section first if you haven't -- it explains what this is and,
importantly, what it deliberately is not.

## Where enrichment fits in the pipeline

```
ingest -> analyze (normalize, window, detect, infer, score, build events)
       -> triage (classify benign vs. reviewable, move/symlink accordingly)
       -> enrich (face, plate, voice, vehicle, transcription, optional LLM)
```

`enrich` is a separate command, not part of `analyze`. Two reasons: it's
comparatively heavy (frame-by-frame CV processing, possibly a Whisper
model), and every capability is independently toggleable, so a low-power
machine can run the core deterministic pipeline at full speed and skip
enrichment, or enable only what it can afford. `enrich` only ever processes
an event's *derived clips* (the short, already-cut segments referenced by
`event.derived_artifacts`) -- never full original files, and never benign
footage that produced zero signals during `analyze`.

New signals discovered by enrichment (`face_detection`, `license_plate`,
`voice_detection`, `vehicle_detection`) are appended to the event as a new
revision (see `docs/architecture.md`'s revisioning section). They do
**not** trigger re-scoring: the original severity stays exactly as
reproducible as it was, and enrichment findings show up as additional,
clearly-labeled (`reasoning_metadata.enrichment_stage: true`) context for
a human reviewer, never a silent retroactive change to *why* an event was
flagged.

## Model management (`gaggle models`)

Three of this project's recognition capabilities have a real deep-learning
option (YuNet face detection, AuraFace face embeddings, fast-alpr plate
detection+OCR -- covered in their own sections below), alongside every
capability's original classical/zero-dependency default, which is never
removed. **No model weights are ever bundled in this package or committed
to the repo.** Each one is fetched on first use -- or explicitly ahead of
time -- into a per-machine cache (`platformdirs.user_cache_dir("gaggle") /
"models"`, not per-workspace, since a model is shared across every
workspace on the machine):

```bash
gaggle models list                          # every known model, cached or not, with sizes
gaggle models download yunet-detector       # explicit pre-fetch, e.g. before going offline
gaggle models download yunet-detector --device cuda   # fetch the fp16 (CUDA) variant instead
gaggle models remove yunet-detector         # free the cache
```

You don't normally need to run `models download` yourself: turning on
`detector: yunet` / `embedding_model: auraface` in config and running
`enrich` triggers the same download automatically, once, the first time
that capability actually runs. A network failure at that moment degrades
gracefully -- one clear warning logged, that capability falls back to its
classical default for the run -- never a hard crash.

**Precision**: `device: cpu` uses int8 (fastest on CPU); `device: cuda`
uses fp16 (CUDA benefits from the higher-throughput format `onnxruntime`'s
CUDAExecutionProvider gives it). Whichever precision has a real pre-built
upstream export is downloaded directly (it has actual accuracy validation
behind it); otherwise the fp32 source is downloaded once and the missing
precision is derived locally (`onnxruntime.quantization.quantize_dynamic`
for int8, `onnxconverter-common` for fp16), cached after that so the
conversion happens at most once per machine. **A real, verified
limitation, not hypothetical**: local int8 derivation can fail for some
model architectures -- confirmed against AuraFace's real recognition
model, where the derived int8 graph loaded as a file but then failed at
inference-session creation with a `ConvInteger` kernel this project's
`onnxruntime` CPU build doesn't implement. When that happens, `ensure_model`
detects it (by actually loading what it just produced, not just trusting
the conversion step returned successfully) and falls back to serving the
fp32 model instead -- slower, but working, rather than a hard failure.

`device: cuda` for any of these three requires additionally installing
`onnxruntime-gpu` in place of the CPU `onnxruntime` package, per each
upstream project's own convention -- gaggle does not manage CUDA/driver
setup itself (same as `vision`'s existing YOLO detector).

## Face detection and local re-identification

```yaml
enrichment:
  face:
    enabled: true
    detector: yunet             # "yunet" (default) or "haar"
    embedding_model: auraface   # "auraface" (default) or "lbph"
    device: cuda                # "cpu" or "cuda" (default) -- only consulted by auraface
    cluster_distance_threshold: 70.0            # LBPH scale; ignored when embedding_model: auraface
    embedding_cluster_distance_threshold: 0.35  # cosine-distance scale; ignored when embedding_model: lbph
    min_detection_confidence: 0.15
```

**Detection** -- `detector: yunet` (the default):

* [YuNet](https://github.com/opencv/opencv_zoo) (`opencv/opencv_zoo`,
  Apache-2.0) -- a real, small (a few hundred KB) deep-learning face
  detector, loaded via `cv2.FaceDetectorYN_create`. No new pip dependency
  at all: it's part of `opencv-contrib-python-headless`, already a core
  dependency for the classical path below. Its model weights are fetched
  on demand (see "Model management" above), not bundled.
* `detector: haar` falls back to OpenCV's bundled Haar cascade
  (`haarcascade_frontalface_default.xml`) -- real, pretrained, zero setup,
  zero network, but less accurate than YuNet, especially off-angle or in
  low light. This is what a zero-extra-dependency install still gets.
* YuNet also produces 5-point facial landmarks per detection; currently
  discarded (not used for cropping or clustering) -- landmark-aligned
  crops would likely improve re-identification accuracy further, a
  documented follow-up, not implemented yet.

**Why YuNet can't use `device: cuda` on a standard install, and what a
CUDA-enabled OpenCV build would actually require**: `cv2.cuda.getCudaEnabledDeviceCount()`
returns `0` on the standard pip `opencv-contrib-python-headless` wheel
regardless of a real GPU being present -- that wheel is never compiled with
`-DWITH_CUDA=ON`. `YuNetDetector` checks this for real before attempting a
CUDA backend and falls back to CPU with a logged warning
(`yunet_cuda_requested_but_opencv_has_no_cuda_support`) rather than silently
misreporting. The only way around this is compiling OpenCV (+ opencv_contrib,
so `cv2.face`/the LBPH fallback keeps working) from source with CUDA enabled
-- checked for real against this project's own dev machine, not assumed:

* **Missing**: MSVC (`cl.exe`) -- required as the host compiler for `nvcc`
  on Windows; MinGW/Strawberry Perl's gcc doesn't work for this. Also
  missing: the NVIDIA CUDA Toolkit itself (`nvcc`, `CUDA_PATH` unset) --
  the pip `nvidia-*` runtime packages `onnxruntime-gpu` uses are
  redistributable DLLs only, not the compiler/dev-headers toolkit a source
  build needs.
* **Present**: CMake (via Strawberry Perl), and a real CUDA-capable GPU
  (RTX 3060 Ti, 8GB VRAM, driver supporting CUDA 13.3 per `nvidia-smi`).
* **Real cost if attempted**: installing VS Build Tools (C++ workload,
  several GB) + the CUDA Toolkit (~3-5GB) + cuDNN dev headers, then a
  from-source OpenCV+opencv_contrib build -- commonly 1-3+ hours of compile
  time on top of the installs, with real risk of build failures (exactly
  why third-party step-by-step guides exist for this). The result is a
  custom wheel that has to be manually rebuilt on every future OpenCV
  version bump, replacing the pip-maintained one this project otherwise
  relies on.
* **Narrow payoff**: this would only speed up YuNet *detection*. AuraFace
  embedding, fast-alpr plate detection, and vehicle-YOLO detection already
  get real CUDA acceleration via `onnxruntime-gpu` independently of
  OpenCV's own build -- see the GPU setup section below. Measured on a real
  event post the N+1-query/WAL-mode enrichment performance fixes: YuNet-on-CPU
  costs roughly 87 seconds per ~2-minute event, the smallest of the
  per-capability costs measured, not a hard blocker.

Given the above, this isn't recommended as a priority -- it's a multi-hour,
multi-GB, fragile undertaking for a comparatively narrow win. Worth
revisiting with a dedicated build session if face-detection speed
specifically becomes a hard blocker on its own; the missing-tooling list
above is the real starting checklist for that, not a guess.

**Re-identification** -- `embedding_model: auraface` (the default):

* A real deep-learning embedding:
  [AuraFace-v1](https://huggingface.co/fal/AuraFace-v1) (`fal/AuraFace-v1`,
  **Apache-2.0, explicitly commercial-use-safe**). Needs the
  `face_recognition` extra; falls back to `embedding_model: lbph`
  automatically (logged once, never silently drops a detection) if the
  extra isn't installed or the model fails to load. Only the
  recognition/embedding model (`glintr100.onnx`) is fetched -- detection
  is already handled above, so the rest of that model pack (InsightFace's
  own detector/landmark/gender-age models, same file layout as
  `buffalo_l`) is never downloaded. Produces a 512-dimensional embedding
  per face, clustered with a running-centroid, cosine-distance clusterer
  (the same design already proven for voice and vehicle-appearance
  re-identification), persisted at
  `workspace/recognition/faces/embedding_model.json`.
* **Why AuraFace and not InsightFace's own `buffalo_l`/ArcFace models**:
  those pretrained recognition weights require a separate commercial
  license from InsightFace despite the loader code (`insightface`, MIT)
  being freely usable -- a real problem to bundle-recommend by default in
  an AGPL tool other people deploy. AuraFace is the same architecture,
  republished by fal under Apache-2.0, no strings attached.
* `embedding_model: lbph` uses an LBPH (Local Binary Pattern Histogram)
  recognizer instead, trained incrementally per-workspace directly from
  detected crops -- not a deep embedding, no weights to download, zero
  extra dependency. "Have I seen a similar-looking face before" for
  personal review, not a strong identity claim. Model state lives at
  `workspace/recognition/faces/model.yml`.
* Because clustering (either kind) is incremental/online, a face's cluster
  assignment can depend on processing order. This is a known, accepted
  characteristic of online clustering, not a bug.

```bash
gaggle recognize faces --workspace ./workspace
gaggle recognize faces-label <cluster-id> "neighbor" --workspace ./workspace
gaggle recognize faces-sightings <cluster-id> --workspace ./workspace
```

Labels are free-text, private, local-only, and never auto-populated --
there is no name lookup anywhere in this system.

## License plate detection and OCR

```yaml
enrichment:
  plate:
    enabled: true
    detector: fast_alpr     # "fast_alpr" (default) or "cascade"
    device: cuda             # "cpu" or "cuda" (default) -- only consulted by fast_alpr
    auto_accept_ocr_confidence: 0.75   # >= this: auto_accepted
    min_ocr_confidence_to_keep: 0.20   # below this: discarded entirely
```

**`detector: fast_alpr`** -- a real deep-learning option via
[fast-alpr](https://github.com/ankandrew/fast-alpr) (`ankandrew/fast-alpr`
+ `ankandrew/fast-plate-ocr`, **MIT**), which performs detection and OCR
together per plate as one library call. Its default OCR model
(`cct-xs-v2-global-model`) is trained on **international plate formats**,
not one region -- the right shape for "focus on American plates but also
work for other regions," unlike the classical cascade path below, which is
calibrated specifically for Russian plate proportions. It also optionally
reports a region/country guess per plate, surfaced as additive
`reasoning_metadata` (`region_guess`/`region_guess_confidence`) alongside
the normal fields -- never replacing them. **Not yet independently
validated against real American dashcam footage** -- see
`docs/limitations.md`. Requires the `plate_recognition` extra (`pip
install gaggle[plate_recognition]`).

**Model acquisition works differently here than YuNet/AuraFace above, on
purpose**: fast-alpr's own model-hub resolution only accepts a closed set
of named presets, which it downloads and caches itself (via
`open_image_models`/`fast_plate_ocr`, both Hugging Face-backed) the first
time `detector: fast_alpr` actually runs -- the same one-time,
library-managed download this project already uses for Whisper
transcription below, not routed through `gaggle models`.

**`detector: cascade`** -- the classical fallback, used automatically if
the `plate_recognition` extra isn't installed or fast-alpr fails to load
(logged once, never silently drops a detection). Combines three
approaches, merged with IoU-based deduplication so hits from different
sources on the same region don't double-count:
  1. Two OpenCV-bundled Haar cascades calibrated for Russian-format
     plates -- real, pretrained, zero-download, but with meaningfully
     lower accuracy on other countries' plate proportions.
  2. **MSER** (Maximally Stable Extremal Regions) -- a classic,
     deterministic blob detector well-suited to text-dense regions. This
     is the primary detector, added specifically after real-world
     feedback that the contour-only heuristic below was missing plates
     entirely in cluttered scenes. Verified during development: a
     synthetic scene with several plate-aspect-ratio decoy rectangles
     (simulating grille slats/trim lines) produced a **0.00 IoU complete
     miss** from the contour heuristic alone, versus a **0.97 IoU**
     correct match from MSER on the same scene -- this is a concrete,
     reproduced fix, not a theoretical improvement (see
     `tests/unit/test_plate_recognition.py::test_mser_finds_the_plate_in_a_junk_heavy_scene_the_old_detector_missed`).
  3. A format-agnostic contour heuristic (Canny edges + morphological
     closing + aspect-ratio filtering). As of this pass, the aspect-ratio
     check runs against each contour's *rotated* minimum-area rectangle
     (`cv2.minAreaRect`), not just its axis-aligned bounding box -- so an
     angled plate (a second specific accuracy complaint) isn't penalized
     just for not being perfectly horizontal. Verified against a
     synthetic plate rotated 20 degrees.

  More candidates reaching OCR than a maximally-precise detector would
  produce is an intentional tradeoff: OCR's own confidence and the
  text-length pre-filter below are what actually reject junk, not the
  detector trying to be perfect on its own.

* **Checking whether detection is actually working on your footage** --
  the concrete answer to "I can't tell what it's finding":

  ```bash
  gaggle recognize plates-debug <event-id> --workspace ./workspace
  ```

  Re-runs detection on that event's real derived clips and saves one
  annotated image per sampled frame that had at least one candidate,
  every region drawn and labeled by source (cascade/MSER/heuristic) and
  confidence. Look at these images directly -- they show you exactly what
  the detector saw, not just what made it into the review queue after OCR
  and confidence filtering. **Always exercises the classical
  cascade/MSER/heuristic path regardless of `detector` config** -- there
  is no fast-alpr equivalent of this debug renderer yet.

* **OCR**: shells out to the `tesseract` binary (matching this project's
  existing ffmpeg/ffprobe pattern rather than adding a heavier pip
  dependency), restricted to an alphanumeric whitelist. Confidence comes
  directly from tesseract's own per-word score. **Requires the `tesseract`
  binary to be installed and on `PATH`** -- this is a real, separate
  install step, not a pip package:
  * **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`
    (already included in the project's Dockerfile).
  * **macOS**: `brew install tesseract`.
  * **Windows**: install from the
    [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
    (the standard Windows distribution), then either check "Add to PATH"
    during setup or add the install directory (typically
    `C:\Program Files\Tesseract-OCR`) to your `PATH` environment variable
    manually and restart your terminal.

  If `tesseract` isn't found, plate detection still runs (it needs no
  external dependency) but OCR is skipped cleanly for the whole run, with
  one clear warning logged (`tesseract_not_found`) -- not a crash, and not
  a warning repeated once per detected region. Face recognition and every
  other enabled capability continue normally regardless.
* **Review workflow**: every OCR reading below `auto_accept_ocr_confidence`
  is marked `needs_review` and kept alongside its crop image for a human to
  confirm, correct, or reject:

```bash
gaggle recognize plates --workspace ./workspace
gaggle recognize plates-review --workspace ./workspace
gaggle recognize plates-confirm <observation-id> "ABC1234" --workspace ./workspace
gaggle recognize plates-reject <observation-id> --workspace ./workspace
gaggle recognize plates-sightings ABC1234 --workspace ./workspace
```

## False-positive cleanup automation

Two layers of automated cleanup, both heuristic and deterministic, neither
using any additional ML model -- consistent with the project's
explainability-over-opacity philosophy:

1. **A pre-filter at detection time.** OCR results outside
   `enrichment.plate.min_plate_text_length`/`max_plate_text_length`
   (default 4-9 characters) are discarded before a `PlateObservation` is
   ever created -- not hidden from review, never stored at all. Most
   single-character or absurdly-long OCR misreads are eliminated this way
   with zero human involvement.
2. **A duplicate-collapsing cleanup pass**, run on demand:

   ```bash
   gaggle recognize plates-cleanup --workspace ./workspace
   ```

   The same physical plate sampled across many frames within one event
   produces a burst of near-identical `PlateObservation`s. This groups
   observations by (event, plate text), clusters each group by how close
   together in time they were seen
   (`enrichment.plate.duplicate_observation_window_seconds`, default 5s),
   and keeps only the highest-OCR-confidence observation per cluster as
   the one needing your attention -- the rest are marked
   `duplicate_suppressed`. **Nothing is deleted**: every observation and
   its crop image stay in the database and on disk, only `review_status`
   changes, and `duplicate_of_observation_id` records which observation
   was kept, so the decision is fully inspectable. A `user_confirmed` or
   `user_rejected` observation (an actual human decision) is never
   revisited by this pass, no matter how it's run.

Run `plates-cleanup` before `plates-review` -- it typically cuts the
review queue down substantially for real footage, since most of what
tesseract successfully reads is the same plate seen repeatedly within one
event, not distinct sightings.

The same cleanup pass exists for faces too:

```bash
gaggle recognize faces-cleanup --workspace ./workspace
```

Groups face observations by (event, cluster) instead of (event, plate
text) -- same time-proximity clustering, same "keep the highest-confidence
one" logic, same guarantee that nothing is deleted (`faces-sightings`
hides suppressed duplicates by default; pass `--include-duplicates` to see
everything).

## Automated merge suggestions -- "these might be the same identity"

The clustering algorithms behind face/plate/voice re-identification are
all classical and imperfect: the same real face can spawn two different
`FaceCluster`s from a lighting change, the same real plate can spawn two
different `PlateRecord`s from an OCR misread, and the same real voice can
spawn two different `VoiceCluster`s from background noise. Rather than
require you to notice these fragmented identities yourself before you can
`faces-merge`/`plates-merge`/`voices-merge` them, a scan can flag likely
candidates for you to confirm or reject:

```bash
gaggle recognize suggest-merges --entity-type face --workspace ./workspace
gaggle recognize suggest-merges --entity-type plate --workspace ./workspace
gaggle recognize suggest-merges --entity-type voice --workspace ./workspace

gaggle recognize merge-suggestions --workspace ./workspace   # list pending
gaggle recognize merge-suggestions-confirm <suggestion-id> --actor "jane" --workspace ./workspace
gaggle recognize merge-suggestions-reject <suggestion-id> --actor "jane" --workspace ./workspace
```

**This never merges anything automatically -- every suggestion sits in a
`pending` queue until a human confirms or rejects it.** Confirming performs
the real merge (the same `merge_faces`/`merge_plates`/`merge_voices` call
you'd make by hand, logged exactly the same way); rejecting performs no
merge and just records that a human looked at it and said no. Neither
action deletes the suggestion itself -- it's kept, marked resolved, as a
permanent record of what was proposed.

**How a suggestion is generated, per entity type:**

* **Faces**: dispatches on `enrichment.face.embedding_model`, matching
  whichever recognition backend is actually configured (they use
  incompatible distance scales -- see the "Face detection and local
  re-identification" section above). With `embedding_model: auraface`
  (the default), stored embedding centroids are compared directly (no
  crop image needed at all, mirroring how plate/voice suggestions already
  work). With `embedding_model: lbph` (the fallback), for each un-merged
  cluster with at least one retained representative crop, that crop is
  compared against the trained LBPH model (read-only, never mutating it)
  to find the nearest *other* cluster; if a cluster's crops have since
  been deleted, it can't generate new suggestions -- its existing
  identity is unaffected, but there's nothing left to visually compare.
  In both cases, a distance just above the auto-merge threshold (which would
  already have combined them at detection time) but within a configurable
  multiplier of it gets suggested.
* **Plates**: pure text similarity (the same `difflib` ratio metric
  `plates-search`'s fuzzy fallback uses) between every pair of un-merged
  plate records. Unlike faces, this needs no image, so it works even for
  plates whose crops have been deleted.
* **Voices**: cluster centroids (the running-mean voiceprint vectors) are
  compared directly, the same distance-band logic as faces. See the voice
  section below for why voice matches deserve more skepticism than face
  or plate ones.

```yaml
enrichment:
  face:
    merge_suggestion_multiplier: 1.6   # x cluster_distance_threshold
  plate:
    merge_suggestion_similarity_threshold: 0.75
  voice:
    merge_suggestion_multiplier: 1.6
```

## Reviewing and reclaiming recognition storage

A real ingest run can produce hundreds of face/plate/vehicle-appearance/
person-appearance detections, many of them false positives (Haar cascades
and the classical plate/vehicle-appearance/person-appearance heuristics are
documented above as meaningfully less accurate than deep-learning
equivalents). Confirming which clusters
are real, picking a representative crop, and discarding the rest is how
you turn that into something actually reviewable -- and, since a crop
image is disposable once you've looked at it and made a decision, how you
reclaim the disk space hundreds of small JPEGs add up to.

**Two-step by design**: reviewing (confirming or rejecting) only changes
an observation's `review_status` -- no file is touched. A separate purge
sweep actually deletes the now-eligible crop images. This mirrors the
existing `triage` benign-clip workflow (classify, then a separate
human-confirmed deletion step) rather than deleting anything the moment
you make a decision. Every confirm/reject/purge action requires an
`--actor` and is permanently logged (`workspace/recognition_review_log.jsonl`,
`workspace/recognition_crop_purge_log.jsonl`) before anything changes --
see AGENTS.md's invariant 22. Pass `--purge` (CLI) or `purge: true` (API)
on a confirm/reject action to do both steps at once when you're sure.

**Confirming a cluster** picks one or more representative observations
(the crop(s) worth keeping) and, optionally, a private label -- every
other observation in the cluster is marked confirmed too, and becomes
purge-eligible:

```bash
gaggle recognize faces-confirm <cluster-id> \
  --representative <observation-id> --actor "jane" --label "mail carrier"
gaggle recognize vehicles-confirm <cluster-id> \
  --representative <observation-id> --actor "jane" --label "neighbor's van"
```

**Rejecting** marks a false positive -- either an entire cluster that was
never a real face/vehicle, or a single observation in an otherwise-good
cluster:

```bash
gaggle recognize faces-reject-cluster <cluster-id> --actor "jane"
gaggle recognize faces-reject-observation <observation-id> --actor "jane"
```

**Plates** have no cluster/representative-crop concept -- each sighting
is its own record, already reviewable one at a time via the existing
`plates-confirm`/`plates-reject` (now actor-attributed and logged the
same way):

```bash
gaggle recognize plates-confirm <observation-id> ABC1234 --actor "jane"
gaggle recognize plates-reject <observation-id> --actor "jane"
```

**The purge sweep** deletes crop images for everything already reviewed
and not yet purged, for one entity type at a time. `--dry-run` shows
exactly what would be deleted (observation ids, paths, hashes) without
touching anything:

```bash
gaggle recognize faces-purge-reviewed --actor "jane" --dry-run
gaggle recognize faces-purge-reviewed --actor "jane"
gaggle recognize vehicles-purge-reviewed --actor "jane"
gaggle recognize plates-purge-reviewed --actor "jane"
```

Every crop is hash-verified against `crop_sha256` before deletion (the
same defensive check `triage confirm-deletion` already does for original
clips) -- a modified or already-missing file is skipped, not force-deleted.
The observation row itself is never touched beyond `crop_purged_at`;
`crop_path` stays as a historical pointer forever.

**Reviewing by identity, across every event** -- `gaggle review-ui`'s
existing per-event enrichment panel only ever shows one event's crops at
a time, which doesn't scale to hundreds of detections spread across many
events. `/recognition/face`, `/recognition/vehicle_appearance`, and
`/recognition/person_appearance` are a dedicated cross-event cluster
browser: every cluster's crops in one
place, checkboxes to pick representative(s), a label field, and
confirm/reject buttons, plus a purge-sweep button with a dry-run preview.
It also surfaces any pending merge suggestion for a cluster inline ("gaggle
thinks this might be the same as cluster X") with one-click confirm/reject,
right next to the manual search-and-merge workflow from the previous
section -- so a suggestion doesn't require a separate CLI step to act on.
The per-event enrichment panel also gained a lightweight "not a
face"/"not a vehicle" button per row, for when you're already looking at
one specific event and spot an obvious false positive.

## Voice detection and local voiceprinting

```yaml
enrichment:
  voice:
    enabled: true   # on by default -- no extra download, just numpy/scipy
```

**Read this before treating any voice match as more than a loose prompt
for human review.** Unlike face and plate recognition, which use real
pretrained detectors (OpenCV's bundled cascades) or a deterministic OCR
engine, there is no pretrained speaker-embedding model reachable without
network access, so this capability is built from scratch using classical
signal processing -- the same pre-deep-learning approach used for speaker
verification before embedding models existed:

1. **Voice activity detection (VAD)**: frames the audio, and flags a frame
   as speech-like if its energy is above an adaptive threshold *and* its
   spectral flatness suggests a broadband, formant-rich signal rather than
   a narrowband tone (a horn, a steady engine hum). Verified against a
   pure 440Hz tone (standing in for exactly that kind of mechanical noise)
   correctly producing zero detected segments.
2. **Voiceprinting**: Mel-Frequency Cepstral Coefficients (MFCCs),
   implemented directly with numpy/scipy (mel filterbank, DCT, the whole
   pipeline), aggregated (mean + std across all frames in a segment) into
   a fixed-length vector.
3. **Clustering**: a simple incremental centroid tracker -- distance from
   a new voiceprint to each existing cluster's running-mean centroid,
   matching if within threshold, a new cluster otherwise. Simpler than the
   LBPH classifier used for faces since voiceprints are already
   fixed-length real vectors.

**This is a meaningfully weaker fingerprint than face or plate
recognition.** It was validated during development against synthetic
multi-tone signals standing in for different "voices" (distinct
fundamental + harmonic structure), repeated across several noise seeds:
same-voice comparisons consistently scored ~0.0002-0.0003 cosine distance,
different-voice comparisons ~0.14 -- a wide, reliable separation that the
default `cluster_distance_threshold: 0.05` sits comfortably inside. An
initial, less careful default of 0.15 was caught during this same
validation pass, close enough to the observed different-voice distance
that it produced a real false merge in testing. None of this has been
validated against real recorded human speech in this environment, though
-- expect to retune the threshold against your own footage, and treat
every voice match as a heuristic prompt for review, never as
identification. See `enrichment/voice.py`'s module docstring for the full
detail.

```bash
gaggle recognize voices --workspace ./workspace
gaggle recognize voices-label <cluster-id> "neighbor's dog barking" --workspace ./workspace
gaggle recognize voices-merge <source-cluster-id> <target-cluster-id> --actor "jane" --workspace ./workspace
gaggle recognize voices-cleanup --workspace ./workspace
gaggle recognize voices-identity <cluster-id> --workspace ./workspace
gaggle recognize voices-search <partial-uuid-or-label> --workspace ./workspace
gaggle recognize voices-sightings <cluster-id> --workspace ./workspace
```

Like faces, voiceprints are stored as actual numeric vectors directly in
the database (`VoiceObservation.voiceprint`), not derived from an audio
file that could later be deleted -- so voice re-identification, like face
re-identification, keeps working even after the source clip is purged or
deleted.

## Vehicle appearance re-identification

```yaml
enrichment:
  vehicle_appearance:
    enabled: true   # on by default -- no extra download, just OpenCV/numpy
```

Plate recognition (above) is this project's primary way to re-identify a
vehicle, but it needs a legible plate. This capability covers the gap: a
coarse, classical appearance fingerprint -- dominant hue/saturation
histogram (computed in HSV, since hue is far more lighting-invariant than
raw BGR for "is this the same color car" comparisons) plus a normalized
aspect ratio -- for recognizing a vehicle seen *without* a readable plate.

**Detection**: two layered sources, the same "best available, never
all-or-nothing" pattern plate detection uses:

1. If the optional YOLO vehicle detector (`vision` extra + a model file)
   is enabled and loaded, its boxes for actual vehicle body classes
   (car/motorcycle/bus/truck) are used -- categorically more precise.
2. Otherwise (the zero-setup default every user gets), a classical
   "vehicle-shaped moving blob" heuristic (Canny edges -> morphological
   closing -> contour extraction, tuned for vehicle-scale regions) finds
   candidate regions. Not real vehicle classification -- labeled as such.

**Read this before treating any vehicle-appearance match as more than a
loose prompt for human review.** This is a meaningfully weaker
fingerprint than face or plate recognition: it cannot distinguish two
different vehicles of the same color and body shape, and is more
sensitive to lighting/angle/dirt than a plate reading. It was validated
during development against synthetic vehicle-colored test scenes across
five distinct hues and eight noise seeds each: same-vehicle comparisons
measured ~0.00000-0.00009 cosine distance, different-vehicle comparisons
~0.353-0.690 -- a wide separation the default
`cluster_distance_threshold: 0.10` sits comfortably inside, biased toward
the same-vehicle end (conservative -- prefer a missed merge over a false
one). None of this has been validated against real footage in this
environment -- expect to retune the threshold against your own footage,
and never treat a match as anything beyond a heuristic prompt for review.
See `enrichment/vehicle_appearance.py`'s module docstring for the full
detail, and `docs/limitations.md` for the honest caveats.

```bash
gaggle recognize vehicles --workspace ./workspace
gaggle recognize vehicles-label <cluster-id> "neighbor's truck" --workspace ./workspace
gaggle recognize vehicles-merge <source-cluster-id> <target-cluster-id> --actor "jane" --workspace ./workspace
gaggle recognize vehicles-cleanup --workspace ./workspace
gaggle recognize vehicles-identity <cluster-id> --workspace ./workspace
gaggle recognize vehicles-search <partial-uuid-or-label> --workspace ./workspace
gaggle recognize vehicles-sightings <cluster-id> --workspace ./workspace
gaggle recognize suggest-merges --entity-type vehicle_appearance --workspace ./workspace
```

Like faces, vehicle-appearance observations keep a crop image
(`VehicleAppearanceObservation.crop_path`) *and* the fingerprint vector
itself stored directly in the database, so re-identification keeps
working even after the source clip is purged or deleted.

## Person appearance re-identification

```yaml
enrichment:
  person_appearance:
    enabled: false   # off by default -- see below for why
```

The pedestrian/full-body counterpart to vehicle-appearance re-identification
above: a coarse, classical appearance fingerprint -- dominant clothing-color
hue/saturation histogram plus a normalized aspect ratio -- for recognizing a
person seen again, including facing away from the camera (YOLO's
bounding-box detection doesn't depend on seeing a face).

**Structured attributes, never a description.** Two classical, computed
attributes are surfaced in the underlying signal's `reasoning_metadata`:
`dominant_hue_bin` (which clothing-color histogram bin dominates) and
`height_to_frame_ratio` (a rough build/distance proxy from the bounding
box). Neither is a free-text or AI-generated description -- see
`enrichment/person_appearance.py`'s module docstring for why that was
deliberately kept out of scope.

**Detection: YOLO only, no classical fallback -- off by default because of
it.** Unlike vehicle-appearance, there is no zero-setup classical
"person-shaped blob" heuristic here (a reliable one is a meaningfully
harder, more false-positive-prone problem than a vehicle- or plate-shaped
blob). This requires `enrichment.vision.enabled: true` plus a real YOLO
model file -- turn both on together to use it. COCO class 0 ("person") was
already detected by the vehicle YOLO detector before this module existed;
this is what surfaces those boxes as their own re-identifiable signal.

**Read this before treating any person-appearance match as more than a
loose prompt for human review** -- the same caveat class as
vehicle-appearance above, but weaker still: it cannot distinguish two
different people wearing similarly-colored clothing, and is more sensitive
to lighting/angle/clothing changes between sightings than face or
vehicle-appearance re-identification. Validated only against synthetic test
data during development, not real footage.

```bash
gaggle recognize persons --workspace ./workspace
gaggle recognize persons-label <cluster-id> "mail carrier" --workspace ./workspace
gaggle recognize persons-merge <source-cluster-id> <target-cluster-id> --actor "jane" --workspace ./workspace
gaggle recognize persons-cleanup --workspace ./workspace
gaggle recognize persons-identity <cluster-id> --workspace ./workspace
gaggle recognize persons-search <partial-uuid-or-label> --workspace ./workspace
gaggle recognize persons-sightings <cluster-id> --workspace ./workspace
gaggle recognize suggest-merges --entity-type person_appearance --workspace ./workspace
```

## Vehicle telemetry detection

```yaml
detection:
  telemetry:
    hard_braking_threshold_mps2: 4.0       # ~0.4g, a commonly-cited threshold
    speed_spike_threshold_mps: 20.0
    heading_change_threshold_deg_per_sec: 45.0
```

If a GPS track is available alongside your dashcam footage, `analyze`
runs a built-in `TelemetryDetector` that flags three kinds of
"interesting" driving moments as weak, corroborating `Signal`s (never, by
themselves, enough to reach high severity -- see invariant 7 in
`AGENTS.md`):

* **Hard braking** -- deceleration between two consecutive GPS fixes at or
  above `hard_braking_threshold_mps2`.
* **Speed spike** -- absolute speed at or above `speed_spike_threshold_mps`
  (a fixed threshold, not a rate-of-change check -- kept simple
  deliberately for this first pass).
* **Sudden heading change** -- the shortest angular distance between two
  consecutive headings, divided by the time between them, at or above
  `heading_change_threshold_deg_per_sec` (a hard swerve or turn).

**Speed and heading are computed directly from consecutive
`(latitude, longitude, time)` points**, not read from a `<speed>`
extension a GPX file may or may not include: speed via the haversine
great-circle distance formula, heading via the initial-bearing formula --
both standard, closed-form, deterministic calculations, hand-implemented
in `detection/telemetry_analysis.py` the same way `enrichment/voice.py`'s
MFCC pipeline is (no learned model, no randomness).

**GPX is the only real (non-fixture) track format recognized.** No
universal dashcam telemetry format exists to target, so this project
targets GPX -- an open, stdlib-parseable (`xml.etree.ElementTree`) XML
standard exported by dedicated GPS loggers, many phone GPS-logging apps,
and some dashcam apps -- rather than inventing a proprietary format or
requiring a new dependency. At ingest time, `IngestService` looks for a
`.gpx` file in the same directory as each source clip and copies at most
one into the workspace as a `gps_track`-type sidecar artifact, matched by
**presence** (one GPS track assumed per camera directory per ingest
session), not by filename correlation to a specific clip -- see
`docs/limitations.md` for the exact scope boundary (what happens with
zero, one, or more than one `.gpx` file present).

Exactly like `detection/motion.py`/`detection/audio.py`, a precomputed
`telemetry_events` array in a `*.samples.json` sidecar fixture (used by
this project's own test suite) takes priority over real GPX parsing when
`detection.use_fixture_signals_when_available` is true (the default) --
set it to `false` to force real GPX analysis even when a fixture sidecar
is present. A clip with no associated GPS track simply produces no
telemetry signals; missing telemetry is a normal outcome, not a detection
failure, exactly like a clip with no audio stream.

There is no dedicated CLI command for this detector -- it runs
automatically as part of `analyze`, alongside motion/audio/object
detection, and its signals show up in `EventRecord.signals` with
`signal_type: "telemetry"` and a `reasoning_metadata` block carrying the
raw measured value (m/s², m/s, or degrees/second) for auditability.

## Rapid-approach ("looming") detection via optical flow

```yaml
detection:
  optical_flow:
    sample_rate_hz: 2.0
    roi_divergence_delta_threshold: 0.015
```

Frame differencing (motion detection, above) answers "did something
change here" -- it cannot distinguish ordinary lateral motion from
something *approaching the camera*, a structurally different,
dashcam-relevant cue (tailgating, a near-miss, a vehicle closing in
fast). This detector uses dense optical flow
(`cv2.calcOpticalFlowFarneback`) between sampled frame pairs and
measures the flow field's *divergence* -- positive divergence means the
flow is expanding outward at that point ("looming"), which frame
differencing structurally cannot capture. Classical, deterministic, no
learned model, same spirit as the motion detector and
`telemetry_analysis.py`'s haversine/bearing math.

**Ego-motion rejection is what keeps this a meaningful signal instead of
constant noise.** A forward-driving dashcam's own motion produces strong
divergence essentially everywhere in frame (radial expansion from the
vanishing point) -- an absolute divergence threshold would flag ordinary
driving constantly. Instead, each sample is scored two ways: globally
(dominated by ego-motion) and over a central ~60% region of interest. A
`rapid_approach` signal fires only when the ROI's divergence exceeds a
*rolling median of recent global divergence* by
`roi_divergence_delta_threshold` -- a comparative threshold, not an
absolute one, so uniform ego-motion (both scalars rising together) stays
quiet while a real object closing in on the ROI stands out.

The default threshold was measured, not guessed, against synthetic
true-positive (an approaching object at five different closing rates)
and true-negative (uniform ego-motion-style zoom at five rates, plus a
static scene) test videos -- see
`detection/optical_flow_analysis.py`'s module docstring for the exact
measured distributions. `reasoning_metadata` on every signal carries
`roi_divergence`/`global_divergence`/`baseline_global_divergence` for
audit.

Like the telemetry detector, there is no dedicated CLI command -- it
runs automatically as part of `analyze` and prefers a precomputed
`optical_flow_events` sidecar fixture when
`detection.use_fixture_signals_when_available` is true (the default).
**Known limitations**: low-texture/night-driving frames starve Farneback
of gradient information (the same failure mode frame differencing
already has); very fast real-world closures between samples can produce
displacement large enough to break Farneback's local-window tracking
assumption, showing up as noisy rather than cleanly-signed divergence; a
genuinely slow, gradual approach can fall below what this technique can
resolve at all. This is a corroborating signal only (invariant 7 in
`AGENTS.md`) -- never sufficient alone for high severity, and not a
claim to catch every real approach.

## Gunshot detection

```yaml
detection:
  gunshot:
    enabled: false   # off by default -- see below for why
    confidence_threshold: 0.5
    window_seconds: 2.0
    hop_seconds: 1.0
```

**Why a classifier, not a classical heuristic.** A gunshot's acoustic
signature (fast rise time, high peak/RMS crest factor, broadband energy
burst) is real, but a classical impulse-detection heuristic built on it
cannot reliably tell a gunshot apart from a car door slam, an engine
backfire, a firework, or a construction impact -- all produce a very
similar sharp transient. This was a real, deliberate tradeoff, made
after researching both options -- see `detection/gunshot_analysis.py`'s
module docstring for the full comparison, and this project's history for
the classical-heuristic alternative that was considered and not chosen.

**The model and its provenance, stated plainly.** This uses k2-fsa's
zipformer-small AudioSet audio-tagging model via the optional
`sherpa-onnx` dependency (`pip install gaggle[gunshot]`) -- both the
Python package and the model itself are Apache-2.0 licensed (verified
directly against the real downloaded archive's own `README.md` and the
package's own PyPI metadata, not assumed). The model is downloaded from
a stable, versioned k2-fsa/sherpa-onnx GitHub Releases URL and its
sha256 hash is pinned and verified before extraction -- see
`core/models.py::ensure_cuda_dlls_preloaded`'s docstring for the exact
kind of DLL-conflict history that makes provenance/isolation matter in
this project; `sherpa-onnx` statically bundles its own onnxruntime, so
it does not conflict with the vision/face_recognition extras' separately
installed onnxruntime.

**Only a curated subset of AudioSet's classes count as "gunshot-like"**:
`Gunshot, gunfire`, `Machine gun`, `Artillery fire`, `Cap gun`.
Deliberately excludes the acoustically-adjacent `Fireworks`/
`Firecracker` classes -- conflating those with actual gunfire would be
actively misleading for a safety signal, not just imprecise.

**Read this before enabling.** This was validated end-to-end during
development by downloading the real model, verifying its real license
and hash, inspecting its real ONNX input/output contract, and running
real inference through it against the model's own bundled real-world
test clips (cat meow, dog bark, siren, baby cry, smoke alarm, etc.) --
every one produced the correct label at high confidence, and none
produced a false "gunshot-like" match. What was **not** possible: no
real or synthetic gunshot audio was available in this environment to
validate a true positive directly. Unlike the classical
color/shape fingerprints `enrichment/vehicle_appearance.py`/
`enrichment/person_appearance.py` use, there's no honest way to
synthesize a realistic gunshot waveform for a test fixture. Treat every
detection as an unvalidated-in-this-environment classifier opinion, not
a confirmed identification -- see `docs/limitations.md`.

**Scoring**: gunshot signals participate in severity scoring like
motion/audio-spike/telemetry (unlike face/plate/vehicle recognition,
which never do -- see `enrichment/service.py`'s module docstring). A
lone gunshot detection, however confident the classifier was, is capped
at 0.60 confidence and -- because `ScoringService` requires at least two
distinct corroborating signal *types* for medium/high severity,
regardless of any single signal's own confidence -- can never alone
reach medium or high severity (AGENTS.md invariant 7). A gunshot-like
sound that coincides with visual motion is a separate, explicit rule
(`gunshot_plus_motion`) with a real confidence bump, the same
corroboration-bonus pattern `motion_plus_audio_spike` already uses.

Off by default: unlike every other built-in detector, this requires the
new `sherpa-onnx` dependency and a real model download on first use.
Degrades gracefully (no signals, logged once) if the extra isn't
installed or the model can't be downloaded -- never a hard failure of
`analyze`, the same pattern `enrichment/face_auraface.py`/
`enrichment/transcription.py` use for their own optional dependencies.

## Linking sightings to the same person or vehicle

The Haar-cascade face detector and the LBPH re-identification clusterer
(deliberately simple, zero-download tools -- see above) will sometimes
split the *same* real face into two `FaceCluster`s because a different
angle or lighting condition fell outside the match threshold. Similarly,
OCR can read the *same* real plate slightly differently across sightings
(a "1" misread as "I") and end up with two `PlateRecord`s for one vehicle.
`recognize *-merge` lets you fix that by hand once you notice it, and every
other recognition command follows the link automatically from then on.

```bash
# "I looked at both crops and they're the same person" / "same car"
gaggle recognize faces-merge <source-cluster-id> <target-cluster-id> --actor "jane" --workspace ./workspace
gaggle recognize plates-merge <source-plate-id> <target-plate-id> --actor "jane" --notes "OCR misread the 2 as Z" --workspace ./workspace
```

**What a merge actually does, and doesn't do:**

* Neither cluster/record is edited, deleted, or renamed. `source` gets a
  `merged_into` pointer to `target` -- an alias, not a rewrite. All of
  `source`'s original observations, stats, and history stay exactly as
  they were.
* The merge itself is permanently logged to
  `workspace/identity_merge_log.jsonl` (the same append-only pattern as
  review actions and deletions): who declared it, when, and any notes --
  this is what makes a merge traceable rather than a silent database edit.
* `target` is treated as the canonical identity going forward. If you
  merge `A -> B` and later merge `B -> C`, both `A` and `B` resolve to
  `C`; `recognize faces-merge`/`plates-merge` refuses any merge that would
  create a cycle (e.g. merging `C` back into `A`).
* Merging is a **human judgment call, not automatic**. Nothing in this
  project re-clusters or auto-merges on your behalf -- the incremental
  clusterer's job is to flag "this looks similar," and merging across its
  mistakes is deliberately left to a person looking at the crops.

**Searching and viewing linked identities:**

```bash
# See the combined picture across every cluster/record merged into one identity
gaggle recognize faces-identity <any-member-cluster-id> --workspace ./workspace
gaggle recognize plates-identity <any-member-plate-id-or-text> --workspace ./workspace

# Sightings follow merges by default; add --exact to see just one literal cluster/record
gaggle recognize faces-sightings <cluster-id> --workspace ./workspace
gaggle recognize plates-sightings ABC1234 --workspace ./workspace --exact

# Search by id, text, or label -- falls back to fuzzy (OCR-typo-tolerant)
# suggestions if nothing matches exactly
gaggle recognize faces-search <partial-uuid-or-label> --workspace ./workspace
gaggle recognize plates-search ABC1Z34 --workspace ./workspace

# Hide clusters/records that are themselves merge-aliases from the default listing
gaggle recognize faces --workspace ./workspace
gaggle recognize faces --include-merged --workspace ./workspace
```

**On privacy, again:** merging makes tracking *more* precise (fewer
fragmented, harder-to-follow partial identities), which is exactly why the
scope boundary in `docs/forensic-considerations.md`'s "Recognition data:
scope and intent" matters more here, not less. An `identity_id` is still
just a locally-generated UUID with an optional private label you chose --
never a real name, never looked up anywhere, never shared. Trackability
within your own footage and respecting the fact that most of the people
and vehicles you're detecting are just going about their day are not in
tension as long as this stays local, unlabeled-by-default, and never
extended into networked or public-facing use.

## Optional local vehicle/object detection (YOLO ONNX)

```yaml
enrichment:
  vision:
    enabled: false
    model_path: /path/to/yolov8n.onnx
    device: cuda   # "cuda" (default) or "cpu" -- falls back to CPU gracefully (logged) if CUDA isn't actually available
    confidence_threshold: 0.35
```

Requires the `vision` extra (`pip install gaggle[vision]`, i.e.
`onnxruntime`) **and** a model file -- neither is bundled, since shipping
model weights would both bloat the package and undermine "offline after
installation." The model download is the one-time, user-initiated network
step:

```bash
pip install gaggle[vision]
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
# or download a pre-exported .onnx directly from a source you trust
```

then point `enrichment.vision.model_path` at the resulting file. Detects
the standard COCO classes; only person/bicycle/car/motorcycle/bus/truck are
treated as relevant by default. If the extra isn't installed or the model
file isn't found, this capability silently produces zero detections (logged
once) rather than failing the run.

**Enabling GPU inference (`device: cuda`)**: the `vision` extra installs
plain `onnxruntime`, which is CPU-only -- it does not include CUDA
support. `device: cuda` is fully wired through to
`onnxruntime.InferenceSession`'s `CUDAExecutionProvider`
(`enrichment/vehicle_yolo.py`, and the same pattern in
`enrichment/face_auraface.py`/`enrichment/plate_fast_alpr.py`), but
actually engaging it takes two real steps -- both verified against a real
RTX 3060 Ti, not assumed:

1. **Swap the package.** `onnxruntime` and `onnxruntime-gpu` install into
   the same `onnxruntime/` folder in site-packages and silently clobber
   each other if both end up installed (a real incident: installing the
   `face_recognition`/`plate_recognition` extras pulled in plain
   `onnxruntime` as a transitive dependency alongside an
   already-installed `onnxruntime-gpu`, and whichever installed last won
   -- pip does not prevent or even warn about this). Always finish with
   only one installed:

   ```bash
   pip uninstall onnxruntime onnxruntime-gpu
   pip install "onnxruntime-gpu[cuda,cudnn]==1.26.0"
   ```

   The `[cuda,cudnn]` extra pulls the real CUDA/cuDNN redistributable DLLs
   as ordinary pip packages (`nvidia-cuda-runtime-cu12`,
   `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc.) -- no separate
   system-wide CUDA Toolkit install needed. **The `==1.26.0` pin is
   deliberate, not a stale leftover -- update it carefully, not
   automatically.** `onnxruntime-gpu` 1.27.0+ switched its `[cuda,cudnn]`
   extra to CUDA 13 packages (`nvidia-cudnn-cu13` etc.) by default; if a
   `torch`-dependent package is also installed (see the cuDNN
   version-conflict note just below for why that matters even though this
   project doesn't use torch itself), a CUDA-13 onnxruntime alongside a
   CUDA-12.x torch build is a real, reproduced source of an `OSError:
   [WinError 127]` crash elsewhere in the same process -- confirmed on a
   real machine, not hypothetical. 1.22.0 through 1.26.0 all resolve to
   CUDA 12.x packages (verified against each version's real PyPI
   metadata); pick whichever is current within that range, or check a new
   version's actual resolved dependencies (`pip install
   "onnxruntime-gpu[cuda,cudnn]==X.Y.Z" --dry-run`) before moving the pin
   past 1.26.0. Package names change between CUDA-major generations, so
   don't hand-guess `nvidia-*` package names yourself (some
   very-similarly-named ones on PyPI, e.g. `nvidia-cublas-cu13`, are
   unrelated placeholder packages, not the real redistributables).

2. **Nothing else to do in code** -- `core/models.py::ensure_cuda_dlls_preloaded()`
   calls `onnxruntime.preload_dlls()` automatically before any
   CUDA-requesting session is created. This matters even with the right
   packages installed: onnxruntime does not search pip-installed nvidia
   site-packages by default, only its own default DLL search paths, so
   `CUDAExecutionProvider` silently falls back to CPU without a
   `preload_dlls()` call happening first, in-process -- confirmed by
   reproducing the exact failure and fix side by side, not assumed from
   the docs. If GPU still isn't engaging after step 1, that's the next
   thing to check.

`device: cuda` is the default in config now, but it degrades gracefully
to CPU (logged) if the package swap in step 1 hasn't been done yet, so
nothing breaks out of the box -- do the swap above once you have a
CUDA-capable GPU and want faster inference.

**A real, reproduced `torch` interaction worth knowing about, if you also
have `torch` installed for something unrelated** (this project never
imports it itself): `preload_dlls()` adds the pip-installed `nvidia-*`
package directories to the process's Windows DLL search path
(`os.add_dll_directory`), process-global and permanent for the rest of
the run. Both `insightface` (an optional `face_recognition`-extra
dependency, via its own optional `albumentations` -> `torch` import) and
`ctranslate2` (the `transcription` extra's actual backend, which
unconditionally imports `torch` in its model-spec module) load `torch`
internally. If `torch`'s own bundled cuDNN and the `onnxruntime-gpu`
cuDNN pulled onto the DLL search path are different CUDA-major
generations, one of `torch`'s DLL dependencies resolves against the
wrong one and raises `OSError: [WinError 127] The specified procedure
could not be found` the next time anything imports `torch` in that
process -- confirmed for real on a real machine, and confirmed to affect
both `insightface` and `ctranslate2` imports, not just one. **The real
fix is the version pin above** (keep `onnxruntime-gpu` on a CUDA-12.x
release to match a `torch+cu12x` build, or match whichever CUDA major
generation your installed `torch` build actually uses); as defense in
depth on top of that, `insightface`/`faster-whisper` availability checks
in this codebase also catch any import failure (not just "not
installed") and fall back gracefully (`embedding_model: lbph` /
transcription skipped for the run, both logged) rather than crashing --
but that fallback exists to survive an environment that's still
misconfigured, not as a substitute for fixing the actual version
mismatch.

Every detector logs
which execution provider actually ended up active
(`vehicle_detector_provider_active`/`auraface_provider_active`, or a
`..._cuda_requested_but_not_active` warning naming the real reason) --
check for that log line rather than assuming `device: cuda` in config
was actually honored.

## Optional local audio transcription (Whisper)

```yaml
enrichment:
  transcription:
    enabled: false
    model_name: base    # tiny | base | small | medium | large-v3
    device: auto        # ctranslate2's own auto-detection (default) -- real GPU if usable, CPU otherwise; a hard "cuda" would raise instead of falling back
    compute_type: int8  # faster on CPU; use "float16" on a supported GPU
```

Requires the `transcription` extra (`pip install
gaggle[transcription]`, i.e. `faster-whisper`). The first use of a
given `model_name` downloads CTranslate2-converted Whisper weights (from
Hugging Face) -- again, a one-time network step, not a runtime dependency.
Transcripts are written to `workspace/transcripts/<event-id>.json` and
never leave the machine unless the separate `enrichment.cloud` feature
below is *also* explicitly enabled.

## Optional cloud LLM transcript analysis

```yaml
enrichment:
  cloud:
    enabled: false        # disabled by default, everywhere
    endpoint: https://openrouter.ai/api/v1/chat/completions
    model: openai/gpt-4o-mini
    api_key_env_var: DASHCAM_SENTINEL_LLM_API_KEY   # never put the key in the YAML file
```

```bash
export DASHCAM_SENTINEL_LLM_API_KEY=sk-...
```

This is the **only** feature in the entire project that sends data over
the network by default when turned on, and it is off by default everywhere.
Requires the `cloud` extra (`pip install gaggle[cloud]`, i.e.
`requests`). When enabled, it sends the transcript **text only** -- never
video, audio, images, or file paths -- to the configured endpoint (any
OpenAI-chat-compatible API: OpenRouter, a self-hosted vLLM/Ollama server,
etc.), asking for a structured summary/extracted-events/extracted-entities/
importance-score JSON response. The result is stored as an `LLMEnrichment`
record at `workspace/transcripts/<event-id>.llm.json` -- a labeled,
versioned, non-authoritative annotation, exactly like any other detector's
output. It never modifies `signals`, `hypotheses`, or `scoring`.

See `docs/threat-model.md` for how this changes the system's trust
boundary when enabled.

## Choosing an ingest storage mode

```bash
gaggle ingest /media/sd-card --workspace ./workspace --mode copy       # default
gaggle ingest /media/sd-card --workspace ./workspace --mode move
gaggle ingest /media/sd-card --workspace ./workspace --mode reference
```

(`core/config.py::StorageConfig.ingest_mode`, overridable per-run with
`--mode`, or set as your default in config.)

* **copy** (default) -- duplicates every file into `workspace/originals/`,
  source untouched. Safest; needs ~2x disk space during ingest.
* **move** -- relocates each file into the workspace instead of copying.
  Frees the source (e.g. an SD card) immediately; one copy total, but it's
  a one-way operation.
* **reference** -- doesn't touch the source at all; the workspace indexes
  the file at its existing location and every stage reads it from there.
  Zero extra disk use, but the workspace now depends on that location
  staying available and unmodified. Sidecar fixture files (if any) still
  land in the workspace, never written back onto the source location.

Deletion safety differs by mode: a `copy`/`move`-mode clip's original
lives inside the workspace, so `triage confirm-deletion` removes a
workspace-owned copy. A `reference`-mode clip's "original" is still at its
external location, so deleting it deletes the user's actual source file --
`confirm_deletion` requires `--acknowledge-external` for exactly this
reason. See `docs/pipeline-walkthrough.md`'s Step 0 for the fuller
tradeoff discussion, including why reference mode pairs naturally with
preserving anything you care about promptly, before disconnecting/
reformatting the source.

### Converting a `reference`-mode clip to a durable copy later

```bash
gaggle triage convert-mode <clip-id> --to copy --actor "you"
gaggle triage convert-mode <clip-id> --to move --actor "you"
```

If you ingested with `--mode reference` and later want a durable
workspace-owned copy for a specific clip (e.g. before disconnecting the
SD card it points at), this converts it in place -- re-hashing the file
at its current external location first and refusing if it's missing or
has changed, exactly like `confirm_deletion`'s hash check before deleting.

**Only `reference -> copy`/`reference -> move` is supported.** Converting
`copy`/`move -> reference` is refused outright, with no override flag --
it would mean deleting the workspace's one and only owned copy of a file
that might already be the sole surviving copy, with no way to verify a
*new* external dependency actually has matching bytes. If you want that
direction, treat it as ingesting a new `reference`-mode copy and deleting
the old one yourself -- a materially different, riskier operation this
command deliberately doesn't automate.

**Caveat worth knowing:** converting doesn't retroactively fix any
already-existing event's `Signal.evidence_references` -- those keep
pointing at the old external path (append-only provenance is never
rewritten after the fact). Conversion only benefits *future* reads
(further `enrich` runs, `preserve`), not historical evidence pointers.

## Storage lifecycle: triage, event-video purge, and human-confirmed deletion

The other half of "maximize usefulness without keeping terabytes around."
See `docs/architecture.md`'s storage-lifecycle section for the full design
and `docs/pipeline-walkthrough.md` for the guided version; in short, two
separate mechanisms for two different kinds of footage:

### Benign footage that never became an event

```bash
gaggle triage run --workspace ./workspace       # classify everything
gaggle triage list --state reviewable --workspace ./workspace
gaggle triage list --state benign_pending_deletion --workspace ./workspace
gaggle triage confirm-deletion --all --actor "jane" --workspace ./workspace
```

`triage run` happens automatically after `analyze` unless you set
`lifecycle.auto_triage_after_analyze: false`. A clip that contributed to
zero signals is physically moved to `workspace/pending_deletion/` (safe --
nothing in any event references it), *unless* it was ingested in
`reference` mode, in which case it's left at its external location and
classified in place. A clip that contributed to at least one signal is
never moved (moving it would break the evidence references already
written into that event's revision history); instead a convenience
symlink appears under `workspace/for_review/` and
`triage list --state reviewable` gives the authoritative listing.

Nothing is ever deleted without an explicit, actor-attributed
`triage confirm-deletion` call, which writes a permanent
`DeletionRecord` to the append-only `workspace/deletion_log.jsonl`
*before* removing the bytes -- see `docs/chain-of-custody.md`.

### Reviewed events -- purging video while keeping metadata forever

Once you've reviewed an event, its video (the event's own derived clips,
plus the original clip(s) that contributed to it) is usually the single
biggest thing about it taking up space:

```bash
gaggle triage purge-event-video <event-id> --actor "jane" --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision accepted --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision rejected --workspace ./workspace
```

Refuses to run unless the event has already been preserved (`preserve
<event-id>` -- a frozen copy of its derived clips already exists under
`preserved/<id>/`), unless you pass `--force` and explicitly accept
losing that video for good. Cascades to the contributing original clip(s)
only when no *other*, still-unpurged event needs them -- purging one
event's video never destroys evidence another unpurged event still
references, even when they share footage. `event.json` -- signals,
hypotheses, scoring, chain of custody, every review decision, full
revision history -- is never touched except to record `video_purged_at`.
Logged permanently to `workspace/event_video_purge_log.jsonl` (mirroring
`deletion_log.jsonl`'s append-only pattern), separately from the
per-clip `DeletionRecord`s any cascaded original deletions still get.

**This is the full answer to "how do I not keep hundreds of gigabytes of
footage around":** `ingest` -> `analyze` -> `enrich` -> review what's
flagged -> `preserve` anything worth keeping -> `triage confirm-deletion
--all` for the never-flagged remainder -> `triage purge-reviewed` for
everything you've finished reviewing. Face crops, plate observations,
transcripts, and full event metadata for everything -- including video
you've since purged or deleted -- stay in the workspace indefinitely, at
a tiny fraction of the original footage's size.

## Cryptographic signing of the revision hash chain

```yaml
signing:
  enabled: false   # off by default
```

Every `EventRecord` revision already links to the canonical JSON hash of
the revision before it (`previous_revision_hash`), so the full history is
internally verifiable -- but internal consistency alone can't prove a
revision wasn't produced (or altered) by someone able to also recompute
consistent hashes. Signing closes that gap: each revision's canonical
payload is signed with an Ed25519 private key held only by this
workspace, so a third party holding just the public key can verify a
revision was written by whoever holds the private key, without trusting
the exporting process itself.

Requires the `signing` extra (`pip install gaggle[signing]`).
Fully offline -- key generation, signing, and verification never touch
the network.

**Setup is explicit, always in this order:**

```bash
gaggle workspace signing-init --workspace ./workspace
# {"private_key_path": "...", "public_key_hex": "..."}
```

Turning on `signing.enabled` *before* running `signing-init` is safe for
read commands, but the next attempt to write a new event revision (an
`analyze` run, a review action, a triage classification, anything that
calls `Repository.save_event`/`save_event_revision`) raises a clear error
telling you to run `signing-init` first -- key generation is never an
implicit side effect of a write. `signing-init` refuses to overwrite an
existing key, since doing so would invalidate verification of every
revision already signed with the old one.

The key lives at `workspace/signing/private_key.pem`, made read-only the
same way every other frozen file in this project is
(`utils/filesystem.py::set_read_only`). It is deliberately outside
`events/`, so `export event`'s file walk (which only touches
`events/<id>/`) can never accidentally bundle it into an exported
archive. Check whether a workspace has a key, and see its public half
(safe to share -- verification only), with:

```bash
gaggle workspace signing-status --workspace ./workspace
```

**What gets signed, and when:** the *entire* canonical revision payload
(the same bytes `write_event_revision` writes to disk), with the
`revision_signature` field itself excluded from what's signed -- the same
self-referential-hash pattern `export/service.py`'s `manifest_hash`
already uses. Every `save_event`/`save_event_revision` call signs fresh
when `signing.enabled` is true; a revision written before signing was
turned on simply has `revision_signature: null` (expected, not an error).

**Verifying an exported bundle:** `export event` inlines the workspace's
public key into `export_manifest.json` as `signing_public_key_hex`
whenever a signing key exists (independent of whether the exporting CLI
invocation itself had `signing.enabled` set) -- a recipient never needs a
second file. `scripts/verify_export_bundle.py` (the same dependency-free,
standard-library-only script that already verifies hash consistency)
additionally verifies every included revision's `revision_signature`
against that public key, *if* the verifying environment also has
`cryptography` installed -- otherwise it prints a clear note that
signature verification was skipped, without affecting the hash checks:

```bash
python3 scripts/verify_export_bundle.py path/to/event_<id>_<ts>.zip
# note: 3 revision(s) had a valid Ed25519 signature
# OK: ... is internally consistent (all hashes verified)
```

**What signing does NOT prove** (see `docs/threat-model.md`'s full
discussion): that the private key belongs to who you think it does --
key custody and distribution is a process/organizational control outside
this project's scope, the same limitation any public-key scheme has. It
also does not protect against an attacker who has both filesystem access
to the workspace *and* the private key file -- that attacker could edit
and re-sign a fabricated history. Signing raises the bar specifically
against tampering with an already-exported bundle, or against someone who
can edit files in a workspace but doesn't have the signing key.

## Security camera support

Gaggle isn't limited to a vehicle's own dashcams. `camera_id` has always
been a free-form string everywhere in the schema, and ingest is purely
filesystem/directory-tree based, so pointing `gaggle ingest` at a directory
of security-camera clips (motion-triggered exports, an NVR dump, a
doorbell's downloaded footage) works exactly the same way it does for an SD
card's `front`/`rear`/`interior` folders -- every detector, recognition
model, review workflow, and export path documented above applies equally,
indoors or outdoors. Live/streaming ingestion (RTSP, a directly-attached
USB webcam) is not supported yet -- see `docs/limitations.md`.

### The camera registry

`gaggle camera register/update/list` attaches optional metadata to a
`camera_id`: a human label, a `source_type` (`dashcam` | `security_ip` |
`security_usb` | `nvr_export` | `doorbell` | `other`), whether it's
`indoor`/`outdoor`, and a `site_id`. Registration is never required --
`ingest` auto-registers a minimal record (`source_type="other"`) the first
time it sees a new `camera_id`, and every existing capability works with
zero registration exactly as before this entity existed. Registering is
purely for your own bookkeeping and for `site_id`, which does have a real
effect (see below).

```bash
gaggle camera list --workspace ./workspace
gaggle camera register porch --workspace ./workspace \
  --source-type security_ip --label "Front porch" --outdoor --site-id home
gaggle camera update porch --workspace ./workspace --notes "installed 2026-06-01"
```

### Site-scoped time synchronization

`docs/architecture.md`'s cross-camera time-sync algorithm was built for one
vehicle's simultaneous dashcams: two cameras' overlapping recording times
are treated as evidence they were powered on together and get aligned to
each other. That assumption doesn't hold for independent cameras with
unrelated clocks -- a neighbor's porch camera and your dashcam could
plausibly overlap in time without ever having anything to do with each
other, and aligning them would be actively wrong.

Every camera discovered within one `gaggle ingest` run is assigned the same
auto-derived `site_id` (a hash of the source directory), so cameras from
the same source (an SD card's `front`/`rear`/`interior` folders, or one
NVR's export directory) keep cross-syncing exactly as before, with zero
configuration. A camera ingested in a *separate* run gets a different
`site_id` and is never cross-synced against cameras from the other run.
Only cameras sharing a `site_id` are ever considered candidates for
alignment; `camera update --site-id` lets you override this by hand (e.g.
to group two cameras from separate ingest runs that really are at the same
physical location).

### Indoor/outdoor example profiles

Two starting-point config profiles ship under `examples/config/`, on top of
the existing profile system documented in `examples/config.yaml`:

- **`security-outdoor.yaml`**: raises `detection.motion_threshold` (0.20 ->
  0.35) and `detection.optical_flow.roi_divergence_delta_threshold` (0.015
  -> 0.025) to tolerate the constant low-level motion noise outdoor scenes
  have that indoor ones don't -- wind-blown foliage, shifting shadows,
  passing headlights -- and extends `sync.session_gap_seconds` (120s ->
  600s) since motion-triggered clips from one camera can be sparse, many
  minutes apart, unlike a dashcam's continuous recording.
- **`security-indoor.yaml`**: lowers `detection.motion_threshold` (0.20 ->
  0.12), since indoor lighting is controlled and mostly static, so a
  person entering a room at a normal pace needs a lower bar to register as
  motion than an outdoor scene's noise floor requires.

Both are reasoned starting points, not values empirically validated against
real security footage -- see `docs/limitations.md`. Vehicle telemetry and
optical-flow rapid-approach detection need no indoor-specific override:
a static indoor camera simply has no GPS track and little of the ego-motion
those detectors are tuned for, so they produce no signals rather than false
ones, the same graceful degradation as a dashcam clip with no GPX sidecar.

`examples/security_camera_sample/` is a small synthetic example (mirroring
how `examples/sample_media/` demonstrates the dashcam path) -- one
motion-triggered "porch" clip with a `*.samples.json` fixture standing in
for real motion/audio/object-hint detection:

```bash
gaggle ingest examples/security_camera_sample --workspace ./workspace \
  --config examples/config/security-outdoor.yaml
```

## A complete end-to-end workflow

```bash
gaggle workspace init --workspace ./workspace
gaggle workspace signing-init --workspace ./workspace  # optional, if signing.enabled
gaggle ingest /media/sd-card --workspace ./workspace --mode move
gaggle analyze --workspace ./workspace        # also runs triage
gaggle enrich --workspace ./workspace          # face/plate/voice/(vehicle)/(transcript)
gaggle recognize plates-cleanup --workspace ./workspace
gaggle recognize faces-cleanup --workspace ./workspace
gaggle recognize suggest-merges --entity-type face --workspace ./workspace
gaggle recognize suggest-merges --entity-type plate --workspace ./workspace
gaggle recognize merge-suggestions --workspace ./workspace
gaggle review start --actor "jane" --workspace ./workspace
gaggle triage confirm-deletion --all --actor "jane" --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision accepted --workspace ./workspace
gaggle triage purge-reviewed --actor "jane" --review-decision rejected --workspace ./workspace
```

See `docs/pipeline-walkthrough.md` for the fully narrated version of this
same sequence. Every step above works fully offline except the one-time
model downloads for `vision`/`transcription` (if you enable them) and the
optional `cloud` LLM step (if you enable and configure it). Nothing here
requires internet access to function day-to-day.
