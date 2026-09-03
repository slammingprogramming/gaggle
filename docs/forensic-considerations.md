# Forensic considerations

## False-positive philosophy

The system is deliberately biased toward **retaining suspicious or
uncertain evidence over discarding it**. Missing evidence is expensive and
irreversible; preserving footage that turns out not to matter costs disk
space. Concretely, in `scoring/service.py`:

* **Low severity** is allowed to be noisy. A single, isolated, weakly
  corroborated motion signal is retained (`inference/service.py`'s
  `isolated_motion_retention` rule, capped at 0.60 confidence) with an
  explicit escalation reason: *"false positives tolerated at low severity
  to avoid missed evidence."*
* **Medium severity** requires corroboration -- at least two distinct
  signal types (e.g. motion + audio, or motion + a moving-region hint).
* **High severity** requires strong corroboration across multiple signal
  types, explicitly reasoned in `SeverityAssessment.reasons`.

No single weak signal can reach high severity on its own; every escalation
path in `inference/service.py` requires either multiple signal types or
multiple cameras agreeing.

## Human judgment is authoritative

Every automated output in this system is a **hypothesis**, never a
conclusion:

* Detectors produce `Signal`s (evidence).
* The inference engine produces `Hypothesis`es (explainable interpretations
  of signals, with a `confidence_math` string spelling out the arithmetic).
* Scoring produces a `SeverityAssessment` (a triage aid, not a verdict).
* Pattern analysis produces `Pattern`s, always marked `hypothesis_only: true`.

None of these determine guilt, fault, or any legal conclusion. A human
reviewer's `ReviewAction` is the only thing that represents an actual
decision about an event, and even that decision doesn't overwrite the
automated findings it's responding to (see `docs/chain-of-custody.md`).

## What the built-in detectors can and cannot tell you

* **Motion** (`detection/motion.py`) reports *that* the frame changed
  meaningfully between samples, via grayscale frame differencing. It does
  not know what moved.
* **Audio spikes** (`detection/audio.py`) report *that* the audio energy
  crossed a threshold in a time window. It does not know what made the
  sound.
* **Object hints** (`detection/object_detection.py`), absent a sidecar or a
  plugin, report *that* a large contiguous region changed, with a bounding
  box -- labeled `unclassified_moving_region`, deliberately not "car,"
  "person," or any named class. There is no object classifier in the
  built-in pipeline (see `docs/architecture.md`'s note on avoiding
  ML-first design). A real classifier is a legitimate `DetectorPlugin`
  extension, but its output would still be a `Signal` with a confidence
  score, subject to the same corroboration rules as everything else.

None of the built-in detectors perform identification (no face recognition,
no license plate recognition, no biometric matching of any kind).

## Multi-camera correlation

`inference/service.py`'s `cross_camera_motion_correlation` rule is the
primary mechanism: motion observed on two or more distinct cameras within
the same (merged) time window is treated as stronger evidence than motion
on one camera alone, with an explicit `+0.20` corroboration bonus recorded
in `confidence_math`. This depends on `normalize/sync.py` having produced a
reasonable time alignment across cameras first -- see
`docs/architecture.md`'s time-synchronization section for what that
alignment can and cannot guarantee.

## Reviewing evidence efficiently

The review UI (`review_ui/app.py`) exists specifically so a reviewer doesn't
have to read raw JSON to make a decision: it renders synchronized
multi-camera playback (all derived clips for an event, play/pause/seek
locked together via `<video data-sync>` elements), the contributing
signals and hypotheses in a table, and a one-click review-action form.
Metadata that a human can't efficiently inspect isn't useful metadata --
this was an explicit design priority, not an afterthought (see the
project's original design directives).

## Recognition data: scope and intent

Face and license-plate re-identification (`src/gaggle/enrichment/`)
sit closer to a genuine ethical line than anything else in this project, so
the boundary is stated explicitly rather than left implicit:

**What this is:** local pattern re-identification within footage *you*
captured, entirely on your own machine, for your own review. "Have I seen
this face or plate before, and when" -- a recurring-signal aid, directly
analogous to features already shipped in mainstream consumer products (e.g.
a home security camera's "familiar faces," or a dashcam app's plate-alert
feature).

**What this explicitly is not, and never will become without a deliberate,
separately-justified design change:**

* **Not identification.** There is no name resolution, no linking a face or
  plate to a real-world identity, no reverse lookup against any external
  database or service, and no field anywhere in the schema for one. A face
  becomes an anonymous `FaceCluster` the user may privately label with
  whatever nickname they choose (see `schemas/recognition.py`) -- the
  system itself never claims to know who anyone is.
* **Not networked.** Nothing here shares observations with other cameras,
  other users, or any third party. The recognition database is per-
  workspace, single-user, local-only. Building any form of shared/networked
  ALPR or facial-recognition capability -- the thing that makes
  commercial mass-surveillance ALPR networks what they are -- is out of
  scope for this project and would require a fundamentally different (and
  separately scrutinized) design, not an incremental extension of what's
  here.
* **Not a surveillance deployment tool.** This is built for footage from a
  vehicle *you* drive, reviewed by *you*. Pointing a camera running this
  software at a public space specifically to build a standing log of
  everyone who passes is a materially different use case with materially
  different legal and ethical obligations than reviewing your own dashcam
  footage after a near-miss.

**Legal considerations you should review before enabling face
recognition specifically** (license plates are generally less regulated,
but check your jurisdiction): several jurisdictions treat facial
biometric data as a specially protected category with obligations that can
apply even to a private individual's personal recording -- for example,
Illinois' Biometric Information Privacy Act (BIPA) and the EU/UK GDPR's
treatment of biometric data as a "special category." Whether and how these
apply to a personal dashcam depends on your jurisdiction, whether you
record other identifiable people, and how you use or share the resulting
data. This is not legal advice; if face recognition matters for your use
case, understand your local law before enabling it. `enrichment.face.enabled`
defaults to `true` for convenience, but can be turned off entirely in
config with no loss of any other capability.

## Limitations that affect forensic weight

See `docs/limitations.md` and `docs/threat-model.md` for the full list.
The most consequential ones for anyone relying on this system's output in a
dispute:

* Time synchronization across cameras is a heuristic (session-start
  alignment), not measured clock offset -- there's no audio/video
  cross-correlation to verify it.
* The hash chain on event revisions is internally consistent but not
  cryptographically signed or externally anchored.
* Derived clips are cut on keyframe boundaries (stream copy, no re-encode),
  so they may run slightly longer than the exact detected window --
  intentionally, to avoid cutting off evidence, but worth knowing when
  citing an exact timestamp.
