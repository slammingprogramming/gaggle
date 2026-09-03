# Changelog

All notable changes to Gaggle are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
doesn't yet follow strict Semantic Versioning guarantees pre-1.0 tag on
GitHub, but version numbers only ever increase and are never reused.

For deep implementation rationale behind any entry below, see
`AGENTS.md`'s "Repo history context" section, which keeps a much more
detailed internal narrative per pass.

## [1.10.0]

### Added

- Pedestrian/full-body appearance re-identification (`enrichment/person_appearance.py`,
  `recognize persons-*`) -- structured attributes only (dominant clothing
  color, build), never a learned face embedding or an AI-generated
  description. Off by default (requires the `vision` extra + a YOLO model).
- Gunshot/gunfire detection (`detection/gunshot_analysis.py`, the new
  `gunshot` extra) via a local ONNX audio classifier. Off by default;
  contributes to severity scoring but can never alone reach medium/high
  severity.
- `gaggle config set-actor`/`show` -- a per-machine default `--actor` so
  it doesn't need retyping on every review/recognition command.
- `gaggle events split` -- corrects an event that wrongly bundled clips
  from separate recording sessions, without editing the original event.
- `sync.manual_offset_overrides` config for correcting one camera's
  timing sync going forward.
- Cluster `detach`/`move` operations for face/vehicle/person-appearance
  recognition review, plus plate "not a plate"/text-correction controls
  in the review UI.

### Fixed

- review-ui no longer hangs on Ctrl+C shutdown.
- Deleting a read-only file (a real Windows failure mode) during
  triage/purge no longer crashes.

## [1.9.0]

### Added

- Real deep-learning recognition upgrades, all optional and falling back
  automatically to the existing classical detector if not installed:
  YuNet face detection, AuraFace face-embedding re-identification
  (Apache-2.0/commercial-safe), fast-alpr license-plate detection+OCR
  (international plate formats).
- `core/models.py::ModelRegistry` -- on-demand per-machine model cache,
  `gaggle models list/download/remove`.

## [1.8.0]

### Added

- Recognition review + storage-reclamation workflow: confirm/reject
  clusters and observations, purge reviewed crops to reclaim disk space,
  with a full append-only audit trail.
- Event-duration cap so near-continuous motion doesn't collapse an
  entire long recording into one giant event.

## [1.7.0]

### Changed

- Project renamed dashcam-sentinel -> Gaggle, broadening scope from
  dashcam-only to camera-agnostic (dashcam, indoor/outdoor security
  camera).
- SQLite schema upgrades now go through real Alembic migrations instead
  of a rebuild-if-needed fallback.

### Added

- Camera entities and site-scoped cross-camera sync, so an indoor camera
  from a separate installation doesn't get incorrectly time-synced
  against an unrelated dashcam session.
- Cross-modality `Encounter` records (co-occurrence, never spatial
  correspondence) and a recurring face+vehicle co-occurrence pattern.

## [1.6.0]

### Added

- Rapid-approach ("looming") detection via dense optical flow --
  captures an object closing in on the camera, which frame-differencing
  motion detection structurally cannot.
- Plugin architecture wired into the pipeline: detectors, inference
  rules, exporters, and review extensions, discovered via standard
  Python entry points with failure isolation.

## [1.5.0]

### Added

- Vehicle re-identification by visual appearance (color/body shape) when
  no legible plate is available.
- GPS-derived vehicle telemetry detection (hard braking, speed spikes,
  sudden heading changes) from a colocated GPX track.
- Optional Ed25519 cryptographic signing of the event revision hash
  chain.

## [1.4.0] and earlier

- Initial pipeline: immutable ingest, cross-camera time sync, motion/
  audio-spike/object-hint detection, rule-based inference and scoring,
  preservation, review UI, timeline querying, pattern detection, export.
- Local face/plate recognition (classical, zero-setup by default) and
  local voice activity detection + voiceprinting.
- Storage lifecycle: triage (benign vs. reviewable), human-confirmed
  deletion, event-video purge with safe cross-event cascade.
- Identity linking, merge suggestions, and search across recognized
  faces/plates/voices/vehicles.

[1.10.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.10.0
[1.9.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.9.0
[1.8.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.8.0
[1.7.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.7.0
[1.6.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.6.0
[1.5.0]: https://github.com/slammingprogramming/gaggle/releases/tag/v1.5.0
