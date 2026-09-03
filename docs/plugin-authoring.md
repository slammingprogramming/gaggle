# Plugin authoring

Four extension points, each a `typing.Protocol` in
`src/gaggle/plugins/base.py`, discovered via
[`importlib.metadata.entry_points()`](https://docs.python.org/3/library/importlib.metadata.html#entry-points)
at pipeline construction time. A plugin is an ordinary, independently
installable Python package -- it does not need to depend on
`gaggle` internals beyond the schemas it needs to produce.

## The four protocols

```python
class DetectorPlugin(Protocol):
    name: str
    version: str
    def detect(self, windows: Iterable[EventWindow], clips: Iterable[NormalizedClip]) -> list[Signal]: ...

class InferenceRulePlugin(Protocol):
    name: str
    version: str
    def apply(self, signals: Iterable[Signal]) -> list[Hypothesis]: ...

class ExporterPlugin(Protocol):
    name: str
    version: str
    format_id: str
    def export(self, event_path: str, destination: str) -> str: ...

class ReviewExtensionPlugin(Protocol):
    name: str
    version: str
    def on_review_action(self, action: ReviewAction, event: EventRecord) -> None: ...
```

A detector plugin must produce real `Signal` instances -- evidence, never a
conclusion -- and should be a pure function of its inputs so results stay
reproducible. This is exactly where a real ML object/vehicle classifier
belongs: it consumes the same `NormalizedClip`/`EventWindow` inputs the
built-in detectors get, and emits `Signal(signal_type="object_hint", ...)`
with a real confidence and `reasoning_metadata` explaining the model and
version used.

An inference rule plugin runs *after* the built-in rules
(`inference/service.py::InferenceService.infer`) and never sees or modifies
another rule's output -- each rule's hypotheses are independently auditable.

## Registering a plugin

In your plugin package's `pyproject.toml`:

```toml
[project.entry-points."gaggle.plugins.detectors"]
my_object_detector = "my_package.detector:MyObjectDetector"

[project.entry-points."gaggle.plugins.inference_rules"]
my_rule = "my_package.rules:MyRule"

[project.entry-points."gaggle.plugins.exporters"]
my_exporter = "my_package.export:MyExporter"

[project.entry-points."gaggle.plugins.review_extensions"]
my_review_hook = "my_package.review:MyReviewExtension"
```

The entry point group names are also available as constants in
`gaggle.plugins.registry` (`DETECTOR_PLUGIN_GROUP`,
`INFERENCE_RULE_PLUGIN_GROUP`, `EXPORTER_PLUGIN_GROUP`,
`REVIEW_EXTENSION_PLUGIN_GROUP`) so you don't have to retype the strings.

The target can be a class (it will be instantiated with no arguments) or an
already-constructed instance/factory function. Once your plugin package is
`pip install`ed into the same environment as `gaggle`, it is
picked up automatically -- no configuration file changes needed.

## Verifying registration

```bash
gaggle plugins list
```

prints every currently-loadable detector, inference rule, exporter, and
review-extension plugin.

## Failure isolation

`plugins/registry.py::load_plugins()` wraps every plugin's `.load()` call
and instantiation in a `try/except`; a plugin that raises is logged
(`plugin_load_failed`, with the plugin's entry-point name and the error) and
skipped, never allowed to crash the built-in pipeline. Write your plugin to
raise clear exceptions rather than fail silently -- the isolation happens at
the call site, not inside your code, so your own errors should still be
loud and specific.

At runtime, `AnalysisPipeline.analyze()` and `InferenceService.infer()` also
isolate exceptions raised while *running* an already-loaded detector/rule
plugin (not just while loading it), for the same reason.

## A minimal example detector plugin

```python
# my_package/detector.py
from collections.abc import Iterable
from gaggle.schemas.media import EventWindow, NormalizedClip
from gaggle.schemas.signal import Signal

class AlwaysLowConfidenceDetector:
    name = "my_package.always_low"
    version = "1.0.0"

    def detect(
        self, windows: Iterable[EventWindow], clips: Iterable[NormalizedClip]
    ) -> list[Signal]:
        # A real plugin would analyze `clips` here. This one is illustrative.
        return []
```

Note the real `DetectorPlugin.detect` signature takes `windows` and `clips`
directly (unlike the built-in `Detector` base class in `detection/base.py`,
which takes a single `DetectionInputs` bundle including `config`) -- plugins
are expected to bring their own configuration rather than reaching into
`gaggle`'s `RuntimeConfig`, keeping them decoupled from the host
project's config schema.

## Exporter plugins

`ExporterPlugin.export(event_path, destination) -> str` should read the
event directory at `event_path` (the same layout documented in
`docs/architecture.md`) and write to `destination`, returning the final
output path (which doesn't have to exactly match `destination` -- e.g. a
plugin can pick its own extension -- but must exist, or the dispatch
raises `ExportError`).

`export event --format <plugin format_id>` dispatches to a matching
loaded plugin instead of the built-in zip bundle; omit `--format` (or pass
none) for the built-in format. A plugin that raises, or that reports an
output path that doesn't actually exist, is isolated -- the CLI surfaces
a clear `ExportError`, never a raw traceback from third-party code (see
invariant 8's plugin-isolation carve-out). Every plugin export still
appends a `chain_of_custody` entry to the event, exactly like the
built-in bundle format.

`export timeline`'s `--format` stays fixed to `csv`/`json` -- the
`ExporterPlugin` Protocol is scoped to one `event_path` at a time (see
above), which doesn't naturally extend to a multi-event timeline export,
so timeline export isn't wired to plugin dispatch.

`tests/unit/test_export.py`'s `_MarkerExporter` is a minimal, real,
working example of the plugin contract (reads `event_path`, writes a
trivial marker file to `destination`, returns it) if you want something
concrete to start from beyond this file's skeleton.

## Review-extension plugins

`ReviewExtensionPlugin.on_review_action(action, event)` is called by
`core/review.py::ReviewService.append_action()` strictly *after* the
review action has already been durably appended to the review log and
folded into a new event revision. This ordering is deliberate: a review
extension can observe and react to a human decision (send a
notification, mirror it into an external ticketing system, write a
site-specific audit copy) but can never block, delay, or alter the
decision itself -- by the time your plugin runs, the append-only review
log and the new `EventRecord` revision are already committed.

Because of that, a review extension that raises is isolated the same
way a broken detector/rule/exporter plugin is (`review_extension_plugin_
failed` is logged with the plugin name and the review action id), but
the *review action itself* still succeeds and is returned to the caller
normally -- unlike an exporter plugin failure, which does surface as a
CLI-visible `ExportError` because exporting is the action being
requested. Here, persisting the review decision is the action being
requested; your extension is a secondary observer of it.

```python
# my_package/review.py
from gaggle.schemas.event import EventRecord
from gaggle.schemas.review import ReviewAction

class LogToStderrExtension:
    name = "my_package.log_to_stderr"
    version = "1.0.0"

    def on_review_action(self, action: ReviewAction, event: EventRecord) -> None:
        print(f"review action {action.action} on event {event.event_id}")
```

`tests/unit/test_review_extensions.py` has a working, tested example
(`RecordingExtension`) plus the failure-isolation test if you want a
concrete reference beyond the skeleton above.
