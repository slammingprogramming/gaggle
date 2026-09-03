# Getting started: choosing your setup

`gaggle` is built so that almost everything is optional beyond
the deterministic core pipeline. This page is a decision guide: what each
tier of setup actually buys you, what it costs (disk, compute, one-time
downloads), and how to pick a starting point without having to read the
whole codebase first. See `docs/pipeline-walkthrough.md` for what to
actually *do* once you've decided; this page is about *what to install*.

## The three tiers at a glance

| Tier | What you get | What it costs | Install |
|---|---|---|---|
| **Minimal** | Motion/audio detection, rule-based inference, severity scoring, review, preservation, export, storage lifecycle (triage/purge/deletion) | Nothing beyond ffmpeg | `pip install -e .` + ffmpeg on `PATH` |
| **Recommended** | Everything in Minimal, plus local face/plate detection and re-identification, with a confidence-gated review queue and duplicate-cleanup automation | tesseract on `PATH` (a real install step, see below); modest extra CPU time per `enrich` run | same as Minimal + tesseract |
| **Full** | Everything in Recommended, plus local vehicle/object detection (YOLO), local audio transcription (Whisper), and optional cloud LLM transcript analysis | A YOLO model file and/or a Whisper model download (one-time); real CPU/GPU time for transcription; the cloud LLM piece sends transcript text to a remote endpoint if you turn it on | `pip install -e .[vision,transcription,cloud]` + model setup |

Every tier is fully offline after its one-time setup step, *except* the
optional cloud LLM piece in Full, which is off by default even if you
install the extra. Nothing here requires choosing one tier and sticking
with it forever -- every capability has its own `enabled` flag in config,
so you can start Minimal and turn things on individually whenever you
want, without reinstalling or re-ingesting anything.

## Minimal: just the deterministic pipeline

```bash
pip install -e .
```

Requires `ffmpeg`/`ffprobe` on `PATH` (see `docs/developer-setup.md` for
platform-specific install instructions). That's it. You get:

- Immutable ingest with real metadata extraction
- Cross-camera time synchronization
- Real motion detection (OpenCV) and audio-spike detection (scipy)
- Rule-based, explainable severity scoring
- Full review workflow, preservation, export
- The storage-lifecycle triage/purge/deletion system

This is the right starting point if you want to see the core pipeline
work end to end before deciding whether the recognition features are
worth the extra setup, or if you specifically don't want face/plate
recognition running on your footage at all (see
`docs/forensic-considerations.md`'s "Recognition data: scope and intent"
if that's a deliberate choice for you, not just a default).

To disable face/plate recognition explicitly (they're on by default once
installed, since they need no extra setup) rather than just not installing
anything extra:

```yaml
# config.yaml
profiles:
  default:
    enrichment:
      face:
        enabled: false
      plate:
        enabled: false
```

## Recommended: add local face/plate recognition

```bash
pip install -e .
```

Face and plate recognition (`enrichment.face`/`enrichment.plate`) need no
extra Python package -- they're built entirely on OpenCV (already a core
dependency) and, for plate OCR specifically, the external `tesseract`
binary. Install that separately:

* **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`
* **macOS**: `brew install tesseract`
* **Windows**: install from the
  [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki),
  then add it to `PATH` (see `docs/developer-setup.md` for the exact steps)

Without tesseract, plate *detection* still runs (finds plate-shaped
regions) but OCR is skipped cleanly with one warning -- face recognition
is entirely unaffected either way.

This tier is worth it if you want to answer "have I seen this face or
plate before" questions about your own footage. It adds:

- Local face detection + on-device re-identification, zero cloud, zero
  extra pip dependency (the default detector, YuNet, downloads a small
  model file from GitHub on first use -- see `docs/local-ai.md`'s "Model
  management" section; `detector: haar` avoids even that if you want
  truly zero network ever)
- Local license plate detection + OCR, with a confidence-gated review
  queue for uncertain readings
- Identity linking (`recognize faces-merge`/`plates-merge`) so fragmented
  detections of the same real person/vehicle can be tied together under
  one searchable identity
- Duplicate-cleanup automation (`recognize plates-cleanup`) to cut down
  how much you have to review by hand

Read `docs/forensic-considerations.md`'s "Recognition data: scope and
intent" before relying on this for anything beyond casual personal
review -- it explains what this is (local pattern re-identification) and
what it deliberately isn't (identity lookup, networked surveillance), plus
jurisdiction-specific legal considerations around face recognition
specifically.

## Full: add vehicle detection, transcription, and/or cloud analysis

Each of these three is independent -- install and enable whichever you
actually want, not all-or-nothing.

### Local vehicle/object detection (YOLO)

```bash
pip install -e .[vision]
```

Needs a model file too (not bundled, to keep the package small and the
core install offline-only):

```bash
pip install ultralytics   # one-time, only needed to export a model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
```

Then point config at it:

```yaml
enrichment:
  vision:
    enabled: true
    model_path: /path/to/yolov8n.onnx
    device: cuda   # default; falls back to CPU gracefully (logged) without onnxruntime-gpu + a CUDA device
```

Worth it if you want structured vehicle/pedestrian classification beyond
the built-in heuristic motion-region detector. See
`docs/local-ai.md`'s vehicle detection section for supported export
formats and class list.

### Local audio transcription (Whisper)

```bash
pip install -e .[transcription]
```

The first use of a given model size downloads CTranslate2-converted
Whisper weights from Hugging Face (one-time, needs network access that
one time):

```yaml
enrichment:
  transcription:
    enabled: true
    model_name: base   # tiny is fastest/least accurate; large-v3 is the opposite
    device: auto   # default; real GPU if usable, CPU otherwise
```

Worth it if audio content (conversations, arguments, horn honks, spoken
context) matters for your review, not just what's visible on camera.

### Deep-learning face/plate recognition (YuNet, AuraFace, fast-alpr)

```bash
pip install -e .[face_recognition]     # YuNet detection + AuraFace embeddings
pip install -e .[plate_recognition]    # fast-alpr detection + OCR
pip install --force-reinstall --no-deps "opencv-contrib-python-headless>=4.10.0,<5.0.0"
```

**That third command is not optional -- run it every time, right after
installing either extra above.** `insightface` and `fast-alpr`'s own
dependency chains (`albumentations`, `open-image-models`,
`fast-plate-ocr`) pull in `opencv-python`/`opencv-python-headless`
unpinned, which install into the *same* `cv2` folder this project's real
dependency, `opencv-contrib-python-headless`, uses -- pip has no way to
detect this as a conflict since they're different package names, so
whichever one installs last silently wins and can leave you with a `cv2`
build that's missing `cv2.face`/`cv2.data`/other contrib symbols the
classical face/plate path needs (`AttributeError: module 'cv2.face' has
no attribute 'LBPHFaceRecognizer_create'` is the exact symptom -- a real
failure hit and fixed during this project's own development, not a
hypothetical). Re-running the pinned reinstall above after any
`pip install`/`pip install -U` that touches these extras restores the
correct build; it's a no-op if nothing actually got clobbered.

The classical, zero-dependency face/plate recognition described above
keeps working exactly as before if you skip this. These extras add a real
deep-learning option for each, opt-in via config:

```yaml
enrichment:
  face:
    detector: yunet          # already the default -- no new dependency at all
    embedding_model: auraface   # already the default -- needs the face_recognition extra above, falls back to lbph without it
  plate:
    detector: fast_alpr         # already the default -- needs the plate_recognition extra above, falls back to the classical cascade without it
```

Model weights are never bundled -- they're fetched automatically (or via
`gaggle models download`, see `docs/local-ai.md`'s "Model management"
section) the first time each capability actually runs, into a
per-machine cache, not per-workspace. `device: cuda` for any of these
additionally requires `onnxruntime-gpu` in place of the CPU package.
Worth it if you want meaningfully better detection accuracy and/or
semi-automated identity linking than the classical path offers; not worth
it if you're happy with the zero-setup default and don't want the extra
~1-2GB of ML dependencies.

### Optional cloud LLM transcript analysis

```bash
pip install -e .[cloud]
```

```yaml
enrichment:
  cloud:
    enabled: true
    endpoint: https://openrouter.ai/api/v1/chat/completions
    model: openai/gpt-4o-mini
```

```bash
export DASHCAM_SENTINEL_LLM_API_KEY=sk-...
```

This is the **only** feature in the entire project that sends data over
the network by default once turned on -- and it's off by default even
after installing the extra, requiring both `enabled: true` *and* a
configured endpoint *and* an API key. It only ever sends transcript
*text* (requires transcription to be enabled and producing something to
send), never video, audio, or file paths. Worth it if you want an
automated first-pass summary of what was said during flagged events; not
worth it (skip this tier) if you want to keep this tool's data flow
entirely local, which is a completely reasonable choice the other two
"Full" capabilities don't require you to give up.

## Installing everything at once

```bash
pip install -e .[all]
pip install --force-reinstall --no-deps "opencv-contrib-python-headless>=4.10.0,<5.0.0"
```

(Equivalent to `dev,vision,transcription,cloud,signing,face_recognition,plate_recognition`
-- `dev` adds test/lint tooling, not a runtime capability, see
`docs/developer-setup.md`.) The second command is the same
`opencv-contrib-python-headless` fix-up described just above -- `all`
includes `face_recognition`/`plate_recognition`, so it needs it too.
Classical face/plate recognition need no extra flag, just tesseract on
`PATH` as above; the deep-learning options above need
`detector`/`embedding_model` set in config to actually turn on. Still
requires the YOLO model file separately (Whisper, YuNet, AuraFace, and
fast-alpr all download their own weights automatically on first use), and
the cloud LLM piece stays off until you explicitly configure it --
installing the extras never silently enables anything.

## A quick reference: what happens if a capability isn't set up?

Every optional capability degrades cleanly rather than crashing the
pipeline, checked once per run and logged once, not retried per frame:

| Capability | If not installed/configured | What still works |
|---|---|---|
| Face/plate recognition (classical) | (Nothing to install -- these use only OpenCV) | Disable via config if you don't want them |
| YuNet model download (`detector: yunet`, the default) | No network on first use: falls back to `haar` for the run, one warning logged | Everything else |
| `face_recognition` extra / `embedding_model: auraface` | Falls back to LBPH for the run, one warning logged | Face detection, everything else |
| `plate_recognition` extra / `detector: fast_alpr` | Falls back to the classical cascade for the run, one warning logged | Everything else |
| tesseract (classical plate OCR specifically) | Plate detection runs, OCR skipped, one warning logged | Face recognition, everything else |
| `vision` extra / YOLO model | Vehicle detection produces zero signals, one warning logged | Everything else, including the built-in motion-region detector |
| `transcription` extra / Whisper model | Transcription skipped, one warning logged | Everything else |
| `cloud` extra / endpoint / API key | LLM analysis skipped, one warning logged | Everything else, including the transcript itself (still saved locally) |
| `ffmpeg`/`ffprobe` | Ingest falls back to a conservative duration estimate; detection skips real analysis | The pipeline still runs end to end, just without real media analysis -- **not recommended**, this is the one dependency that isn't really optional in practice |

See `docs/pipeline-walkthrough.md` for what to actually run once you've
picked a tier.
