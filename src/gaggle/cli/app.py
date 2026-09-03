from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast
from uuid import UUID

import typer
import uvicorn

from gaggle.core import cli_config
from gaggle.core.cameras import CameraRepository
from gaggle.core.config import RuntimeConfig, load_config
from gaggle.core.events import EventSplitError, EventSplitService
from gaggle.core.models import Device, ModelRegistry, ModelUnavailableError
from gaggle.core.pipeline import AnalysisPipeline
from gaggle.core.recognition import MergeError, RecognitionService, ReviewError
from gaggle.core.review import ReviewService
from gaggle.core.signing import (
    EventSigner,
    SigningUnavailableError,
    generate_signing_key,
    save_private_key_pem,
)
from gaggle.core.signing import public_key_hex as _signing_public_key_hex
from gaggle.core.triage import TriageService
from gaggle.enrichment.service import EnrichmentService
from gaggle.export.service import ExportError, ExportService
from gaggle.ingest.service import IngestService
from gaggle.patterns.service import EncounterIdentityPair, PatternAnalysisService
from gaggle.plugins.registry import (
    DETECTOR_PLUGIN_GROUP,
    EXPORTER_PLUGIN_GROUP,
    INFERENCE_RULE_PLUGIN_GROUP,
    REVIEW_EXTENSION_PLUGIN_GROUP,
    load_plugins,
)
from gaggle.review_ui.app import create_app
from gaggle.schemas.camera import Camera, CameraSourceType
from gaggle.schemas.event import EventRecord
from gaggle.schemas.media import IngestManifest
from gaggle.storage.database import TimelineQuery
from gaggle.storage.filesystem import WorkspacePaths
from gaggle.storage.repository import Repository
from gaggle.timeline.service import TimelineService
from gaggle.utils.filesystem import set_read_only
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import configure_logging
from gaggle.utils.time import utc_now

app = typer.Typer(help="Offline forensic dashcam analysis system")
review_app = typer.Typer(help="Append-only review commands")
workspace_app = typer.Typer(help="Workspace setup commands")
timeline_app = typer.Typer(help="Timeline querying")
patterns_app = typer.Typer(help="Metadata-only pattern analysis")
export_app = typer.Typer(help="Structured forensic metadata / evidence bundle export")
plugins_app = typer.Typer(help="Plugin discovery")
triage_app = typer.Typer(
    help="Storage lifecycle: benign vs. reviewable triage, human-confirmed deletion"
)
recognize_app = typer.Typer(help="Local face/plate re-identification queries")
camera_app = typer.Typer(help="Camera registry: optional metadata about camera_id sources")
models_app = typer.Typer(help="Deep-learning recognition model download/cache management")
config_app = typer.Typer(help="Per-machine CLI convenience configuration (not workspace state)")
events_app = typer.Typer(help="Human-review corrections to already-generated events")
app.add_typer(review_app, name="review")
app.add_typer(workspace_app, name="workspace")
app.add_typer(timeline_app, name="timeline")
app.add_typer(patterns_app, name="patterns")
app.add_typer(export_app, name="export")
app.add_typer(plugins_app, name="plugins")
app.add_typer(triage_app, name="triage")
app.add_typer(recognize_app, name="recognize")
app.add_typer(camera_app, name="camera")
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(events_app, name="events")

WorkspaceOption = Annotated[Path, typer.Option("--workspace")]
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", exists=True, file_okay=True, dir_okay=False, readable=True),
]
ReviewActionLiteral = Literal["accept", "reject", "annotate", "retag", "preserve", "export"]
ActorOption = Annotated[str | None, typer.Option("--actor")]


def _resolve_actor(actor: str | None) -> str:
    """Every command with an `ActorOption` calls this to fall back to the
    configured default actor (`gaggle config set-actor`) when `--actor`
    isn't passed -- see `core/cli_config.py::resolve_actor`."""

    try:
        return cli_config.resolve_actor(actor)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error


def _load_runtime(config_path: Path | None) -> RuntimeConfig:
    configure_logging()
    return load_config(config_path)


def _build_signer(workspace: Path, runtime: RuntimeConfig) -> EventSigner | None:
    """Best-effort: returns None whenever signing is off, or the key hasn't
    been generated yet. A missing key when signing is actually enabled is
    deliberately NOT an error here -- read-only commands (and even
    `signing-init` itself) must keep working before the key exists. It's
    ``Repository._maybe_sign`` that raises a clear error, and only at the
    moment an event write is actually attempted."""

    if not runtime.signing.enabled:
        return None
    key_path = WorkspacePaths(workspace).signing_private_key_path
    if not key_path.exists():
        return None
    return EventSigner.load(key_path)


def _repository(workspace: Path, runtime: RuntimeConfig | None = None) -> Repository:
    if runtime is None:
        runtime = _load_runtime(None)
    signer = _build_signer(workspace, runtime)
    repository = Repository(workspace, signer=signer, signing_enabled=runtime.signing.enabled)
    repository.initialize()
    return repository


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


@workspace_app.command("init")
def workspace_init(workspace: WorkspaceOption) -> None:
    _repository(workspace)
    typer.echo(f"initialized workspace at {workspace}")


@workspace_app.command("signing-init")
def workspace_signing_init(workspace: WorkspaceOption) -> None:
    """Generate a new Ed25519 signing keypair for this workspace's event
    revision chain. Must be run once, explicitly, before setting
    `signing.enabled: true` and expecting new revisions to actually get
    signed -- key generation is never an implicit side effect of a write.
    Refuses to overwrite an existing key (that would invalidate
    verification of every revision already signed with the old one)."""

    repository = _repository(workspace)
    key_path = repository.workspace.signing_private_key_path
    if key_path.exists():
        raise typer.BadParameter(
            f"a signing key already exists at {key_path}; refusing to overwrite it "
            "(this would invalidate verification of every revision signed with the "
            "old key)"
        )
    try:
        private_key = generate_signing_key()
    except SigningUnavailableError as error:
        raise typer.BadParameter(str(error)) from error
    save_private_key_pem(key_path, private_key)
    set_read_only(key_path)
    typer.echo(
        json.dumps(
            {
                "private_key_path": str(key_path),
                "public_key_hex": _signing_public_key_hex(private_key),
            },
            indent=2,
        )
    )


@workspace_app.command("signing-status")
def workspace_signing_status(workspace: WorkspaceOption) -> None:
    """Show whether this workspace has a signing key, and its public key
    (safe to share -- verification-only) if so."""

    repository = _repository(workspace)
    key_path = repository.workspace.signing_private_key_path
    if not key_path.exists():
        typer.echo(json.dumps({"key_exists": False}, indent=2))
        return
    try:
        signer = EventSigner.load(key_path)
    except SigningUnavailableError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "key_exists": True,
                "private_key_path": str(key_path),
                "public_key_hex": signer.public_key_hex,
            },
            indent=2,
        )
    )


@workspace_app.command("reindex")
def workspace_reindex(
    workspace: WorkspaceOption,
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild",
            help=(
                "delete and recreate timeline/index.sqlite3 from scratch before "
                "re-syncing -- a manual escape hatch for schema drift somehow still "
                "detected, or index corruption unrelated to schema, not the routine "
                "upgrade path. Never touches events/, originals/, or anything "
                "else outside the SQLite index file; the filesystem stays the "
                "source of truth throughout."
            ),
        ),
    ] = False,
) -> None:
    """Rebuild the SQLite index from the filesystem (the source of truth).

    Schema upgrades now happen automatically, via Alembic, on every
    command that opens a workspace (see `storage/migrate.py`) -- you do
    not need to run this after upgrading gaggle. This command is for two
    other things: (1) a normal re-sync of already-existing index tables
    from event.json files on disk, always safe, never touches evidence;
    (2) `check_schema_drift()` (an independent sanity check, separate
    from Alembic's own version tracking) reports anything still
    mismatched -- which should never happen in normal operation -- and
    `--rebuild` is the fix for that case.
    """

    repository = _repository(workspace)
    drift = repository.database.check_schema_drift()
    if drift:
        typer.echo(
            "warning: schema drift detected in timeline/index.sqlite3 "
            f"({len(drift)} table(s)); pass --rebuild to fix",
            err=True,
        )

    if rebuild:
        database_path = repository.workspace.database
        repository.close()
        database_path.unlink(missing_ok=True)
        repository = _repository(workspace)

    count = repository.reindex()
    typer.echo(
        json.dumps(
            {
                "reindexed_event_count": count,
                "rebuilt": rebuild,
                "schema_drift_detected": [
                    {"table": table_name, "missing_columns": missing_columns}
                    for table_name, missing_columns in drift
                ],
            },
            indent=2,
        )
    )


@camera_app.command("list")
def camera_list(workspace: WorkspaceOption) -> None:
    """List every registered camera. Registration is always optional --
    ingest auto-registers a minimal record on first-seen camera_id, and
    this only shows what's known so far; an un-registered camera_id still
    works everywhere else in the pipeline, it just won't appear here."""

    repository = _repository(workspace)
    cameras = CameraRepository(repository).list()
    typer.echo(
        json.dumps(
            [
                {
                    "camera_id": c.camera_id,
                    "label": c.label,
                    "source_type": c.source_type,
                    "indoor": c.indoor,
                    "site_id": c.site_id,
                    "created_at": c.created_at.isoformat(),
                    "notes": c.notes,
                }
                for c in cameras
            ],
            indent=2,
        )
    )


@camera_app.command("register")
def camera_register(
    camera_id: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
    source_type: Annotated[
        CameraSourceType,
        typer.Option(
            "--source-type",
            help="dashcam | security_ip | security_usb | nvr_export | doorbell | other",
        ),
    ] = "other",
    label: Annotated[str | None, typer.Option("--label")] = None,
    indoor: Annotated[
        bool | None,
        typer.Option("--indoor/--outdoor", help="omit to leave indoor/outdoor unknown"),
    ] = None,
    site_id: Annotated[
        str | None,
        typer.Option(
            "--site-id",
            help="cameras sharing a site_id are candidates for cross-camera time sync",
        ),
    ] = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Register (or fully overwrite) a camera's metadata. Unlike ingest's
    auto-registration (which never overwrites), this is an explicit user
    action, so it always writes exactly what's passed."""

    repository = _repository(workspace)
    camera = Camera(
        camera_id=camera_id,
        label=label,
        source_type=source_type,
        indoor=indoor,
        site_id=site_id,
        created_at=utc_now(),
        notes=notes,
    )
    CameraRepository(repository).register(camera)
    typer.echo(json.dumps(camera.model_dump(mode="json"), indent=2))


@camera_app.command("update")
def camera_update(
    camera_id: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
    source_type: Annotated[CameraSourceType | None, typer.Option("--source-type")] = None,
    label: Annotated[str | None, typer.Option("--label")] = None,
    indoor: Annotated[bool | None, typer.Option("--indoor/--outdoor")] = None,
    site_id: Annotated[str | None, typer.Option("--site-id")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
) -> None:
    """Edit an already-registered camera's metadata -- only the fields
    actually passed are changed. Fails with a clear error if the camera
    was never registered; use `camera register` for a first write."""

    repository = _repository(workspace)
    updated = CameraRepository(repository).update(
        camera_id,
        label=label,
        source_type=source_type,
        indoor=indoor,
        site_id=site_id,
        notes=notes,
    )
    if updated is None:
        raise typer.BadParameter(
            f"no camera registered under id {camera_id!r}; use 'camera register' first"
        )
    typer.echo(
        json.dumps(
            {
                "camera_id": updated.camera_id,
                "label": updated.label,
                "source_type": updated.source_type,
                "indoor": updated.indoor,
                "site_id": updated.site_id,
                "created_at": updated.created_at.isoformat(),
                "notes": updated.notes,
            },
            indent=2,
        )
    )


@models_app.command("list")
def models_list() -> None:
    """Show every known deep-learning recognition model, per precision:
    cached locally or not, disk size, license. Independent of any
    workspace -- models are cached once per machine, not per workspace."""

    typer.echo(json.dumps(ModelRegistry().status(), indent=2))


@models_app.command("download")
def models_download(
    name: Annotated[
        str | None,
        typer.Argument(help="model name (see 'models list'); omit to fetch everything known"),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="cpu (int8) or cuda (fp16)")] = "cpu",
) -> None:
    """Explicitly pre-fetch one or every known model -- e.g. before
    running fully offline. Detectors already do this automatically on
    first use when their model is missing; this is for pre-warming the
    cache deliberately."""

    if device not in ("cpu", "cuda"):
        raise typer.BadParameter("--device must be 'cpu' or 'cuda'")
    registry = ModelRegistry()
    names = [name] if name else registry.known_models()
    results: dict[str, str] = {}
    for model_name in names:
        try:
            path = registry.ensure_model(model_name, device=cast(Device, device))
            results[model_name] = str(path)
        except ModelUnavailableError as error:
            raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(results, indent=2))


@models_app.command("remove")
def models_remove(name: Annotated[str, typer.Argument()]) -> None:
    """Delete a model's cached files (every precision variant) to free
    disk space. It will be re-downloaded automatically next time it's
    needed."""

    try:
        ModelRegistry().remove_model(name)
    except ModelUnavailableError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"removed {name}")


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument()],
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            help=(
                "copy (default, safest, needs 2x disk space) | move (relocate into the "
                "workspace, frees the source immediately) | reference (index files in place, "
                "no copy at all -- see docs/local-ai.md before using this for anything you "
                "can't re-ingest later). Overrides storage.ingest_mode from config."
            ),
        ),
    ] = None,
) -> None:
    runtime = _load_runtime(config)
    if mode is not None:
        if mode not in ("copy", "move", "reference"):
            raise typer.BadParameter("--mode must be 'copy', 'move', or 'reference'")
        runtime.storage.ingest_mode = mode  # type: ignore[assignment]
    repository = _repository(workspace, runtime)
    manifest = IngestService(
        repository.workspace, runtime, database=repository.database
    ).ingest_directory(source)
    repository.index_ingest_manifest(manifest)
    typer.echo(json.dumps(manifest.model_dump(mode="json"), indent=2))


def _covered_clip_ids(repository: Repository) -> set[UUID]:
    """Every clip_id already referenced by an existing `EventRecord`'s
    derived_artifacts -- i.e. clips that have already gone through
    `analyze` and contributed to a real event (mirrors how
    `core/pipeline.py::_build_events` records `source_clip_id` in the
    first place)."""

    covered: set[UUID] = set()
    for event in repository.list_events():
        for artifact in event.derived_artifacts:
            source_clip_id = artifact.metadata.get("source_clip_id")
            if source_clip_id:
                covered.add(UUID(str(source_clip_id)))
    return covered


def _partition_pending_manifests(
    manifests: list[IngestManifest], covered_clip_ids: set[UUID]
) -> tuple[list[IngestManifest], list[IngestManifest]]:
    """Splits manifests into (pending, already_analyzed).

    A manifest counts as already analyzed if *any* of its clips already
    contributed to an existing event -- true by construction, since
    `AnalysisPipeline.analyze()` always processes a manifest's clips as
    one all-or-nothing batch, never partially (both before and after this
    function existed). A manifest with zero covered clips but at least
    one clip is either genuinely new, or was previously analyzed and
    found entirely benign (a benign clip never creates an event) --
    either way it's safe to (re)run: a benign clip just reconfirms zero
    signals, and there's no event to duplicate. A manifest with *no*
    clips at all (e.g. an ingest that deduplicated away to nothing) has
    nothing to analyze, ever, and is treated as already-analyzed too --
    otherwise it would be reprocessed by every future `analyze` call
    forever for no reason.

    This replaces an earlier, real bug: manifest filenames are
    `{run_id}.json`, a random UUID (see `ingest/service.py`), not a
    timestamp, so picking "the most recent manifest" by sorting filenames
    (or even by `created_at`) only ever looks at one manifest per
    `analyze` call -- any other still-pending manifest is silently never
    analyzed. Processing every manifest with zero covered clips, not just
    the newest one, is what actually fixes that class of bug rather than
    changing which single manifest gets picked.
    """

    pending: list[IngestManifest] = []
    already_analyzed: list[IngestManifest] = []
    for manifest in manifests:
        clip_ids = {clip.clip_id for clip in manifest.copied_files}
        if not clip_ids or clip_ids & covered_clip_ids:
            already_analyzed.append(manifest)
        else:
            pending.append(manifest)
    return pending, already_analyzed


def _merge_ingest_manifests(manifests: list[IngestManifest]) -> IngestManifest:
    """Combines multiple pending ingest manifests into one, so
    `AnalysisPipeline.analyze()` sees every pending clip in a single
    pass. This matters for correctness, not just convenience:
    `normalize/service.py` only ever looks at the manifest it's handed,
    so merging first is what lets cross-camera/session time-correlation
    (`normalize/sync.py`) work correctly across clips that came from
    different `ingest` calls, not just within one.
    """

    copied_files = [clip for manifest in manifests for clip in manifest.copied_files]
    hashes = [digest for manifest in manifests for digest in manifest.hashes]
    source_roots = sorted({manifest.source_root for manifest in manifests})
    return IngestManifest(
        run_id=new_uuid(),
        created_at=utc_now(),
        source_root="; ".join(source_roots),
        copied_files=copied_files,
        hashes=hashes,
    )


@app.command()
def analyze(workspace: WorkspaceOption, config: ConfigOption = None) -> None:
    """Analyze every clip that hasn't been analyzed yet, across every
    ingest run so far -- safe to call any number of times, in any order
    relative to `ingest`/`enrich`. A clip that already contributed to an
    event is never reprocessed; a manifest with nothing new is silently
    skipped. If nothing is pending, this is a clean no-op, not an error.
    """

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    manifest_paths = list(repository.workspace.ingest.glob("*.json"))
    if not manifest_paths:
        raise typer.BadParameter("no ingest manifests found; run 'ingest' first")
    manifests = [
        IngestManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in manifest_paths
    ]
    pending, _already_analyzed = _partition_pending_manifests(
        manifests, _covered_clip_ids(repository)
    )
    if not pending:
        typer.echo(json.dumps({"message": "nothing new to analyze", "events": []}, indent=2))
        return
    merged_manifest = _merge_ingest_manifests(pending)
    events = AnalysisPipeline(repository, runtime).analyze(merged_manifest)
    if runtime.lifecycle.auto_triage_after_analyze:
        TriageService(repository).classify_all()
    typer.echo(json.dumps([event.model_dump(mode="json") for event in events], indent=2))


@app.command()
def enrich(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    event_id: Annotated[UUID | None, typer.Option("--event-id")] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Rerun every enabled capability even if already marked complete on an event "
            "(e.g. after switching a detector/model). Default: skip what's already done.",
        ),
    ] = False,
) -> None:
    """Run local face/plate/vehicle detection, transcription, and (if configured)
    LLM transcript analysis over already-analyzed events.

    Only ever touches events that already exist (i.e. clips that already
    produced at least one signal) -- benign footage never gets enrichment
    processing. Each capability is independently toggled in config
    (`enrichment.face/plate/vision/transcription/cloud.enabled`); disabled
    ones are skipped with no error. Heavier than `analyze`, so it's a
    separate command rather than automatic -- run it whenever convenient,
    including well after `analyze` and `triage`.

    Safe to call any number of times, in any order relative to
    `ingest`/`analyze`: a capability that already completed on an event is
    skipped (not rerun, not duplicated) unless `--force` is passed; a
    capability that was never attempted -- including one just newly
    enabled in config -- always runs the next time this is called.
    """

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    service = EnrichmentService(repository, runtime)
    event_ids = [event_id] if event_id else [e.event_id for e in repository.list_events()]
    results = [service.enrich_event(eid, force=force).model_dump(mode="json") for eid in event_ids]
    typer.echo(json.dumps(results, indent=2))


@app.command("preserve")
def preserve_event(
    event_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    config: ConfigOption = None,
) -> None:
    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    pipeline = AnalysisPipeline(repository, runtime)
    updated_event = pipeline.preserve_event(event_id)
    typer.echo(json.dumps(updated_event.model_dump(mode="json"), indent=2))


@triage_app.command("run")
def triage_run(workspace: WorkspaceOption) -> None:
    """Classify every ingested clip as reviewable or benign-pending-deletion.

    Runs automatically after `analyze` unless
    `lifecycle.auto_triage_after_analyze` is set to false; this command
    exists to rerun it explicitly (e.g. after `enrich` adds new signals to
    events, or just to refresh listings).
    """

    repository = _repository(workspace)
    records = TriageService(repository).classify_all()
    typer.echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))


@triage_app.command("list")
def triage_list(
    workspace: WorkspaceOption,
    state: Annotated[
        str | None,
        typer.Option("--state", help="reviewable | benign_pending_deletion | deleted"),
    ] = None,
) -> None:
    repository = _repository(workspace)
    records = TriageService(repository).list_state(state)  # type: ignore[arg-type]
    typer.echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))


@triage_app.command("confirm-deletion")
def triage_confirm_deletion(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    clip_id: Annotated[UUID | None, typer.Argument()] = None,
    all_pending: Annotated[bool, typer.Option("--all")] = False,
    notes: Annotated[str, typer.Option("--notes")] = "",
    acknowledge_external: Annotated[
        bool,
        typer.Option(
            "--acknowledge-external",
            help=(
                "required to delete clips ingested with --mode reference, since that "
                "deletes a file outside the workspace (the original source location), "
                "not a workspace-owned copy"
            ),
        ),
    ] = False,
) -> None:
    """Permanently delete the original bytes of one (or, with --all, every)
    clip classified benign-pending-deletion. Always writes a DeletionRecord
    to the append-only deletion log first -- see docs/architecture.md."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    service = TriageService(repository)
    if all_pending:
        records = service.confirm_deletion_all(actor, notes, acknowledge_external)
    elif clip_id is not None:
        records = [service.confirm_deletion(clip_id, actor, notes, acknowledge_external)]
    else:
        raise typer.BadParameter("provide a clip_id or pass --all")
    typer.echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))


@triage_app.command("convert-mode")
def triage_convert_ingest_mode(
    clip_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    to_mode: Annotated[
        str,
        typer.Option(
            "--to",
            help=(
                "'copy' or 'move' -- only converting from 'reference' to a workspace-owned "
                "mode is supported. Converting 'copy'/'move' to 'reference' is refused "
                "outright (would delete the workspace's only owned copy)."
            ),
        ),
    ],
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Convert a `reference`-mode clip into a durable, workspace-owned copy.

    Existing events' evidence references keep pointing at the old external
    location (append-only provenance is never rewritten) -- this only
    benefits future reads. See docs/local-ai.md's "Choosing an ingest
    storage mode" section."""

    actor = _resolve_actor(actor)
    if to_mode not in ("copy", "move"):
        raise typer.BadParameter("--to must be 'copy' or 'move'")
    mode_literal: Literal["copy", "move"] = "move" if to_mode == "move" else "copy"
    repository = _repository(workspace)
    try:
        row = TriageService(repository).convert_ingest_mode(clip_id, mode_literal, actor, notes)
    except (ValueError, RuntimeError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "clip_id": row.clip_id,
                "ingest_mode": row.ingest_mode,
                "stored_path": row.stored_path,
            },
            indent=2,
        )
    )


@triage_app.command("purge-event-video")
def triage_purge_event_video(
    event_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="purge even if the event has not been preserved yet (loses that video for good)",
        ),
    ] = False,
) -> None:
    """Delete an event's video (its derived clips, and any contributing
    original clips no other event still needs) while keeping event.json,
    signals, hypotheses, scoring, and full history forever. Refuses unless
    the event has been preserved already, unless --force is passed. See
    docs/local-ai.md's storage-optimization workflow."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = TriageService(repository).purge_event_video(event_id, actor, notes, force)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@triage_app.command("purge-reviewed")
def triage_purge_reviewed(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    review_decision: Annotated[str | None, typer.Option("--review-decision")] = None,
    preservation_state: Annotated[str | None, typer.Option("--preservation-state")] = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Bulk-purge video for every event matching the given filters that
    hasn't been purged yet. A common pattern: preserve and accept/reject
    everything you care about first, then run

        triage purge-reviewed --actor you --review-decision accepted --workspace .
        triage purge-reviewed --actor you --review-decision rejected --workspace .

    to reclaim space from events you've already finished with. Events not
    yet preserved are skipped (not force-purged) unless --force is passed."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    query = TimelineQuery(
        severity=severity,
        review_decision=review_decision,
        preservation_state=preservation_state,
    )
    records = TriageService(repository).purge_event_video_bulk(query, actor, notes, force)
    typer.echo(json.dumps([r.model_dump(mode="json") for r in records], indent=2))


@review_app.command("queue")
def review_queue(
    workspace: WorkspaceOption,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    camera: Annotated[str | None, typer.Option("--camera")] = None,
    status: Annotated[str | None, typer.Option("--status", help="review decision filter")] = None,
) -> None:
    repository = _repository(workspace)
    query = TimelineQuery(severity=severity, camera_id=camera, review_decision=status)
    events = TimelineService(repository.database).list_events(query)
    typer.echo(json.dumps(events, indent=2, default=str))


@review_app.command("history")
def review_history(event_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption) -> None:
    repository = _repository(workspace)
    actions = repository.list_review_actions(event_id)
    typer.echo(json.dumps([a.model_dump(mode="json") for a in actions], indent=2))


@review_app.command("revisions")
def review_revisions(
    event_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption
) -> None:
    repository = _repository(workspace)
    revisions = repository.list_event_revisions(event_id)
    summary = [
        {
            "revision": r.revision,
            "revision_reason": r.revision_reason,
            "revised_at": r.revised_at.isoformat() if r.revised_at else None,
            "previous_revision_hash": r.previous_revision_hash,
        }
        for r in revisions
    ]
    typer.echo(json.dumps(summary, indent=2))


@review_app.command("action")
def review_action(
    event_id: Annotated[UUID, typer.Argument()],
    action: Annotated[ReviewActionLiteral, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    config: ConfigOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Record a review action.

    ``preserve`` and ``export`` are recorded as review actions AND actually
    trigger the corresponding pipeline effect (creating an immutable
    preservation bundle / an export archive) -- the two are always kept in
    sync so a "preserve" action in the audit trail never lies about whether
    preservation actually happened.
    """

    actor = _resolve_actor(actor)
    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    review_service = ReviewService(repository)
    record, updated_event = review_service.append_action(event_id, action, actor, notes=notes)
    result: dict[str, object] = {"review_action": record.model_dump(mode="json")}

    if action == "preserve":
        pipeline = AnalysisPipeline(repository, runtime)
        preserved = pipeline.preserve_event(event_id)
        result["preservation_status"] = preserved.preservation_status.model_dump(mode="json")
    elif action == "export":
        export_result = ExportService(repository).export_event_bundle(event_id)
        result["export_path"] = str(export_result.path)
        result["export_manifest_hash"] = export_result.manifest_hash
    else:
        result["review_summary"] = updated_event.review_summary.model_dump(mode="json")

    typer.echo(json.dumps(result, indent=2, default=str))


@review_app.command("start")
def review_start(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    config: ConfigOption = None,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
) -> None:
    """Interactively walk through every pending-review event, one at a time.

    Prints a summary of each event and prompts for an action. This is the
    "go through everything waiting for me" entry point; `review
    queue`/`review action` remain available for scripted/non-interactive
    use.
    """

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    query = TimelineQuery(severity=severity, review_decision="pending")
    pending = TimelineService(repository.database).list_events(query)
    if not pending:
        typer.echo("Nothing pending review.")
        return

    review_service = ReviewService(repository)
    valid_actions = {"accept", "reject", "annotate", "retag", "preserve", "export"}
    typer.echo(f"{len(pending)} event(s) pending review.\n")

    for index, row in enumerate(pending, start=1):
        event_id = UUID(str(row["event_id"]))
        event = repository.load_event(event_id)
        typer.echo(f"--- [{index}/{len(pending)}] event {event_id} ---")
        typer.echo(
            f"severity: {event.scoring.severity} (confidence {event.scoring.confidence:.2f})"
        )
        typer.echo(f"cameras: {', '.join(event.involved_cameras)}")
        typer.echo(f"when: {event.event_start.isoformat()} -> {event.event_end.isoformat()}")
        typer.echo(f"summary: {event.evidence_summary}")
        typer.echo(f"signals: {len(event.signals)}, hypotheses: {len(event.hypotheses)}")

        choice = (
            typer.prompt(
                "action [accept/reject/annotate/retag/preserve/export/skip/quit]", default="skip"
            )
            .strip()
            .lower()
        )
        if choice == "quit":
            typer.echo("Stopping review session.")
            break
        if choice in ("skip", ""):
            continue
        if choice not in valid_actions:
            typer.echo(f"unrecognized action '{choice}', skipping")
            continue
        action: ReviewActionLiteral = choice

        notes = typer.prompt("notes (optional)", default="", show_default=False)
        review_service.append_action(event_id, action, actor, notes=notes)
        if action == "preserve":
            AnalysisPipeline(repository, _load_runtime(config)).preserve_event(event_id)
        elif action == "export":
            ExportService(repository).export_event_bundle(event_id)
        typer.echo(f"recorded: {action}\n")

    typer.echo("Review session complete.")


@timeline_app.command("query")
def timeline_query(
    workspace: WorkspaceOption,
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    camera: Annotated[str | None, typer.Option("--camera")] = None,
    review_decision: Annotated[str | None, typer.Option("--review-decision")] = None,
    preservation_state: Annotated[str | None, typer.Option("--preservation-state")] = None,
    start_after: Annotated[str | None, typer.Option("--start-after")] = None,
    start_before: Annotated[str | None, typer.Option("--start-before")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    repository = _repository(workspace)
    query = TimelineQuery(
        severity=severity,
        camera_id=camera,
        review_decision=review_decision,
        preservation_state=preservation_state,
        start_after=_parse_iso_datetime(start_after),
        start_before=_parse_iso_datetime(start_before),
        limit=limit,
    )
    events = TimelineService(repository.database).list_events(query)
    typer.echo(json.dumps(events, indent=2, default=str))


@patterns_app.command("analyze")
def patterns_analyze(
    workspace: WorkspaceOption,
    cluster_window_seconds: Annotated[float, typer.Option("--cluster-window-seconds")] = 3600.0,
    min_repeat_count: Annotated[int, typer.Option("--min-repeat-count")] = 2,
) -> None:
    repository = _repository(workspace)
    events = repository.list_events()
    encounter_identities = _resolve_encounter_identities(repository, events)
    patterns = PatternAnalysisService().analyze(
        events,
        cluster_window_seconds=cluster_window_seconds,
        min_repeat_count=min_repeat_count,
        encounter_identities=encounter_identities,
    )
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = repository.workspace.patterns / f"{timestamp}.json"
    repository.workspace.write_json(output_path, {"patterns": patterns})
    typer.echo(json.dumps(patterns, indent=2))


def _resolve_encounter_identities(
    repository: Repository, events: list[EventRecord]
) -> list[EncounterIdentityPair]:
    """Resolve every event's Encounters' raw face/vehicle-appearance
    observation ids to their canonical (merge-resolved) cluster ids -- the
    caller-side lookup pass `PatternAnalysisService` deliberately doesn't
    do itself, so that module stays free of any storage dependency. See
    `patterns/service.py::EncounterIdentityPair`."""

    recognition = RecognitionService(repository)
    pairs: list[EncounterIdentityPair] = []
    for event in events:
        for encounter in repository.database.list_encounters_for_event(event.event_id):
            face_cluster_id: str | None = None
            if encounter.face_observation_id is not None:
                face_observation = repository.database.get_face_observation(
                    UUID(encounter.face_observation_id)
                )
                if face_observation is not None and face_observation.cluster_id is not None:
                    face_cluster_id = str(
                        recognition.resolve_face_identity(UUID(face_observation.cluster_id))
                    )
            vehicle_cluster_id: str | None = None
            if encounter.vehicle_appearance_observation_id is not None:
                vehicle_observation = repository.database.get_vehicle_appearance_observation(
                    UUID(encounter.vehicle_appearance_observation_id)
                )
                if vehicle_observation is not None and vehicle_observation.cluster_id is not None:
                    vehicle_cluster_id = str(
                        recognition.resolve_vehicle_appearance_identity(
                            UUID(vehicle_observation.cluster_id)
                        )
                    )
            pairs.append(EncounterIdentityPair(face_cluster_id, vehicle_cluster_id))
    return pairs


@export_app.command("event")
def export_event(
    event_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    export_format: Annotated[
        str | None,
        typer.Option(
            "--format",
            help=(
                "Unset (default): the built-in hash-verified zip bundle. Any other value is "
                "looked up against loaded exporter plugins by format_id (see "
                "docs/plugin-authoring.md) -- 'plugins list' shows what's loaded."
            ),
        ),
    ] = None,
) -> None:
    repository = _repository(workspace)
    try:
        result = ExportService(repository).export_event_bundle(event_id, export_format)
    except ExportError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "path": str(result.path),
                "manifest_hash": result.manifest_hash,
                "file_count": result.file_count,
            },
            indent=2,
        )
    )


@export_app.command("timeline")
def export_timeline(
    workspace: WorkspaceOption,
    export_format: Annotated[str, typer.Option("--format")] = "csv",
    severity: Annotated[str | None, typer.Option("--severity")] = None,
    camera: Annotated[str | None, typer.Option("--camera")] = None,
) -> None:
    if export_format not in ("csv", "json"):
        raise typer.BadParameter("--format must be 'csv' or 'json'")
    repository = _repository(workspace)
    query = TimelineQuery(severity=severity, camera_id=camera)
    format_literal: Literal["csv", "json"] = "csv" if export_format == "csv" else "json"
    path = ExportService(repository).export_timeline(query, format_literal)
    typer.echo(str(path))


@plugins_app.command("list")
def plugins_list() -> None:
    typer.echo(
        json.dumps(
            {
                "detectors": [
                    getattr(p, "name", repr(p)) for p in load_plugins(DETECTOR_PLUGIN_GROUP)
                ],
                "inference_rules": [
                    getattr(p, "name", repr(p)) for p in load_plugins(INFERENCE_RULE_PLUGIN_GROUP)
                ],
                "exporters": [
                    getattr(p, "name", repr(p)) for p in load_plugins(EXPORTER_PLUGIN_GROUP)
                ],
                "review_extensions": [
                    getattr(p, "name", repr(p)) for p in load_plugins(REVIEW_EXTENSION_PLUGIN_GROUP)
                ],
            },
            indent=2,
        )
    )


@app.command("review-ui")
def review_ui(
    workspace: WorkspaceOption,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    # The `uvicorn.run(...)` shortcut leaves `timeout_graceful_shutdown` at
    # its default (wait indefinitely / require a second Ctrl+C) -- a real,
    # observed "hangs on 'shutting down'" symptom on Windows whenever a
    # connection (e.g. a browser tab left open on a video) is still alive
    # at the moment of the first interrupt. Building the Config/Server
    # explicitly bounds that wait so one Ctrl+C always exits within 5s.
    config = uvicorn.Config(
        create_app(workspace), host=host, port=port, timeout_graceful_shutdown=5
    )
    uvicorn.Server(config).run()


@recognize_app.command("faces")
def recognize_faces(
    workspace: WorkspaceOption,
    include_merged: Annotated[bool, typer.Option("--include-merged")] = False,
) -> None:
    """List every local face identity: id, observation count, first/last seen,
    and (if set) the private label the user gave it. No identity resolution
    is ever performed -- see docs/local-ai.md. Clusters merged into another
    identity via `faces-merge` are hidden by default; pass --include-merged
    to see every raw cluster including aliases."""

    repository = _repository(workspace)
    clusters = repository.database.list_face_clusters()
    if not include_merged:
        clusters = [c for c in clusters if not c.merged_into]
    typer.echo(
        json.dumps(
            [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "observation_count": c.observation_count,
                    "first_seen_at": c.first_seen_at.isoformat() if c.first_seen_at else None,
                    "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
                    "merged_into": c.merged_into,
                    "representative_crops": c.representative_crops_csv.split(",")
                    if c.representative_crops_csv
                    else [],
                }
                for c in clusters
            ],
            indent=2,
        )
    )


@recognize_app.command("faces-label")
def recognize_faces_label(
    cluster_id: Annotated[UUID, typer.Argument()],
    label: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
) -> None:
    """Attach a private, local-only nickname to a face cluster (e.g. 'neighbor').
    Never resolves or infers a real identity -- this is purely for the
    user's own future reference."""

    repository = _repository(workspace)
    repository.database.set_face_cluster_label(cluster_id, label)
    typer.echo(f"labeled {cluster_id} as '{label}'")


@recognize_app.command("faces-merge")
def recognize_faces_merge(
    source_cluster_id: Annotated[UUID, typer.Argument()],
    target_cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Declare that two face clusters are the same person.

    Neither cluster is edited or deleted -- `source` is marked as an alias
    of `target`, and the merge itself is permanently logged
    (workspace/identity_merge_log.jsonl) with who made the call and when.
    `faces-sightings`/`faces-identity` follow this link automatically.
    """

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).merge_faces(
            source_cluster_id, target_cluster_id, actor, notes
        )
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"merged {source_cluster_id} -> {target_cluster_id}")


@recognize_app.command("faces-confirm")
def recognize_faces_confirm(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    representative: Annotated[
        list[UUID],
        typer.Option("--representative", help="repeatable; at least one required"),
    ],
    actor: ActorOption = None,
    label: Annotated[str | None, typer.Option("--label")] = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="also delete every non-representative crop in this cluster immediately",
        ),
    ] = False,
) -> None:
    """Confirm a face cluster is a real, consistent identity: pick its
    representative crop(s) and (optionally) label it. Every observation
    in the cluster is marked confirmed; non-representative crops become
    eligible for `faces-purge-reviewed` (or pass --purge to delete them
    right away)."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).confirm_identity(
            cluster_id, representative, actor, "face", label=label, notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-reject-cluster")
def recognize_faces_reject_cluster(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete every crop in this cluster immediately")
    ] = False,
) -> None:
    """Mark an entire face cluster a false positive (e.g. a cluster of
    misdetections that was never a face at all)."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_cluster(
            cluster_id, actor, "face", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-reject-observation")
def recognize_faces_reject_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete this crop immediately")
    ] = False,
) -> None:
    """Mark a single face observation a false positive -- for a mixed
    cluster where only some crops are wrong, without touching the rest of
    the cluster's review state."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_observation(
            observation_id, actor, "face", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-detach-observation")
def recognize_faces_detach_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Remove a single observation from its cluster entirely -- unlike
    `faces-reject-observation` (which flags it but still counts it toward
    the cluster), this clears its cluster assignment and recomputes the
    former cluster's observation_count/representative crops. The
    observation row itself is never deleted."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).detach_observation(
            observation_id, actor, "face", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-move-observation")
def recognize_faces_move_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    to: Annotated[UUID, typer.Option("--to", help="the target cluster id to move it into")],
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Reassign a single observation from its current cluster to another
    -- for a crop auto-clustered under the wrong identity that a human can
    see actually belongs elsewhere. Recomputes observation_count on both
    the source and target clusters."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).move_observation(
            observation_id, to, actor, "face", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-purge-reviewed")
def recognize_faces_purge_reviewed(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="show what would be purged without deleting")
    ] = False,
) -> None:
    """Delete crop images for every already-reviewed (confirmed
    non-representative, or rejected) face observation not yet purged.
    The observation rows themselves are never deleted."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    record = RecognitionService(repository).purge_reviewed_crops(
        "face", actor, notes=notes, dry_run=dry_run
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("faces-cleanup")
def recognize_faces_cleanup(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    window_seconds: Annotated[
        float | None,
        typer.Option(
            "--window-seconds",
            help="override enrichment.face.duplicate_observation_window_seconds",
        ),
    ] = None,
) -> None:
    """Automatically collapse near-duplicate face observations within the
    same event and cluster, mirroring `plates-cleanup`.

    Nothing is deleted; `faces-sightings` hides suppressed duplicates by
    default afterward (pass --include-duplicates to see everything)."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    effective_window = (
        window_seconds
        if window_seconds is not None
        else runtime.enrichment.face.duplicate_observation_window_seconds
    )
    result = RecognitionService(repository).cleanup_duplicate_face_observations(effective_window)
    typer.echo(
        json.dumps(
            {
                "suppressed_count": len(result.suppressed_observation_ids),
                "kept_count": len(result.kept_observation_ids),
                "clusters_with_duplicates": result.clusters_with_duplicates,
            },
            indent=2,
        )
    )


@recognize_app.command("faces-identity")
def recognize_faces_identity(
    cluster_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Show the aggregated identity for a face cluster: every member cluster
    id merged into it, combined observation count, and first/last seen
    across the whole group."""

    repository = _repository(workspace)
    try:
        identity = RecognitionService(repository).get_face_identity(cluster_id)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "identity_id": str(identity.identity_id),
                "member_cluster_ids": [str(c) for c in identity.member_cluster_ids],
                "label": identity.label,
                "observation_count": identity.observation_count,
                "first_seen_at": identity.first_seen_at.isoformat()
                if identity.first_seen_at
                else None,
                "last_seen_at": identity.last_seen_at.isoformat()
                if identity.last_seen_at
                else None,
                "representative_crop_paths": identity.representative_crop_paths,
            },
            indent=2,
        )
    )


@recognize_app.command("faces-search")
def recognize_faces_search(
    query: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Search face clusters by id (full or partial) or label. Falls back to
    fuzzy suggestions if nothing matches exactly."""

    repository = _repository(workspace)
    result = RecognitionService(repository).search_faces(query)
    typer.echo(
        json.dumps(
            {
                "exact_matches": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.exact_matches
                ],
                "fuzzy_suggestions": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.fuzzy_suggestions
                ],
            },
            indent=2,
        )
    )


@recognize_app.command("faces-sightings")
def recognize_faces_sightings(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    exact: Annotated[
        bool, typer.Option("--exact", help="only this literal cluster, ignore merges")
    ] = False,
    include_duplicates: Annotated[
        bool,
        typer.Option(
            "--include-duplicates",
            help="also show observations suppressed by `faces-cleanup` as redundant repeats",
        ),
    ] = False,
) -> None:
    """List every observation (time, camera, crop) matched to a face identity.

    By default this follows any `faces-merge` links, so it shows sightings
    across every cluster merged into the same identity, and hides
    observations `faces-cleanup` marked as redundant repeats within a
    burst. Pass --exact to see only this literal cluster's own
    observations, or --include-duplicates to see everything."""

    repository = _repository(workspace)
    observations = RecognitionService(repository).list_face_sightings(
        cluster_id, follow_merges=not exact, include_duplicates=include_duplicates
    )
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "event_id": o.event_id,
                    "cluster_id": o.cluster_id,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                    "crop_path": o.crop_path,
                    "detector_confidence": o.detector_confidence,
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("plates")
def recognize_plates(
    workspace: WorkspaceOption,
    include_merged: Annotated[bool, typer.Option("--include-merged")] = False,
) -> None:
    """List every recognized license plate identity and how many times it's
    been seen. Records merged into another identity via `plates-merge` are
    hidden by default; pass --include-merged to see every raw record."""

    repository = _repository(workspace)
    records = repository.database.list_plate_records()
    if not include_merged:
        records = [r for r in records if not r.merged_into]
    typer.echo(
        json.dumps(
            [
                {
                    "plate_id": r.plate_id,
                    "normalized_text": r.normalized_text,
                    "label": r.label,
                    "observation_count": r.observation_count,
                    "first_seen_at": r.first_seen_at.isoformat() if r.first_seen_at else None,
                    "last_seen_at": r.last_seen_at.isoformat() if r.last_seen_at else None,
                    "merged_into": r.merged_into,
                }
                for r in records
            ],
            indent=2,
        )
    )


@recognize_app.command("plates-review")
def recognize_plates_review(workspace: WorkspaceOption) -> None:
    """List low-confidence plate OCR observations awaiting human confirmation."""

    repository = _repository(workspace)
    observations = repository.database.list_plate_observations(review_status="needs_review")
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "raw_ocr_text": o.raw_ocr_text,
                    "normalized_text": o.normalized_text,
                    "ocr_confidence": o.ocr_confidence,
                    "crop_path": o.crop_path,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("plates-cleanup")
def recognize_plates_cleanup(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    window_seconds: Annotated[
        float | None,
        typer.Option(
            "--window-seconds",
            help="override enrichment.plate.duplicate_observation_window_seconds",
        ),
    ] = None,
) -> None:
    """Automatically collapse near-duplicate plate observations before you
    review them by hand.

    The same physical plate sighted across many sampled frames within one
    event produces one observation per frame -- this groups those by
    (event, plate text), clusters them by how close together in time they
    were seen, and marks all but the highest-confidence observation per
    cluster as `duplicate_suppressed`. Nothing is deleted, and any
    observation you've already confirmed or rejected by hand is never
    touched. Run this before `plates-review` to cut down how much you have
    to look at."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    effective_window = (
        window_seconds
        if window_seconds is not None
        else runtime.enrichment.plate.duplicate_observation_window_seconds
    )
    result = RecognitionService(repository).cleanup_duplicate_plate_observations(effective_window)
    typer.echo(
        json.dumps(
            {
                "suppressed_count": len(result.suppressed_observation_ids),
                "kept_count": len(result.kept_observation_ids),
                "clusters_with_duplicates": result.clusters_with_duplicates,
                "suppressed_observation_ids": [str(i) for i in result.suppressed_observation_ids],
            },
            indent=2,
        )
    )


@recognize_app.command("plates-confirm")
def recognize_plates_confirm(
    observation_id: Annotated[UUID, typer.Argument()],
    corrected_text: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="also delete this observation's crop image immediately, in one step",
        ),
    ] = False,
) -> None:
    """Confirm or correct a low-confidence plate OCR reading after viewing
    its crop. Logged to `recognition_review_log.jsonl`. The crop image
    itself is kept until a `plates-purge-reviewed` sweep (or pass
    --purge to delete it right away)."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).confirm_plate_observation(
            observation_id, corrected_text, actor, notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"confirmed {observation_id} as '{corrected_text.upper()}'")


@recognize_app.command("plates-reject")
def recognize_plates_reject(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="also delete this observation's crop image immediately, in one step",
        ),
    ] = False,
) -> None:
    """Mark a plate observation as not a real plate (or an unreadable one you
    don't want cluttering the review queue). Logged to
    `recognition_review_log.jsonl`. The crop image itself is kept until a
    `plates-purge-reviewed` sweep (or pass --purge to delete it right away)."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).reject_plate_observation(
            observation_id, actor, notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"rejected {observation_id}")


@recognize_app.command("plates-purge-reviewed")
def recognize_plates_purge_reviewed(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="show what would be purged without deleting")
    ] = False,
) -> None:
    """Delete crop images for every already-reviewed (confirmed or
    rejected) plate observation not yet purged. Plates have no
    cluster/representative-crop concept -- every reviewed observation is
    individually purge-eligible. The observation rows themselves are
    never deleted."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    record = RecognitionService(repository).purge_reviewed_crops(
        "plate", actor, notes=notes, dry_run=dry_run
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("plates-merge")
def recognize_plates_merge(
    source_plate_id: Annotated[UUID, typer.Argument()],
    target_plate_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Declare that two plate records are the same vehicle (e.g. an OCR
    misread split one real plate into two records). Neither record is
    edited or deleted; the merge is permanently logged."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).merge_plates(source_plate_id, target_plate_id, actor, notes)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"merged {source_plate_id} -> {target_plate_id}")


@recognize_app.command("plates-identity")
def recognize_plates_identity(
    plate_id_or_text: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Show the aggregated identity for a plate: every plate text/id merged
    into it, combined observation count, and first/last seen across the
    whole group. Accepts either a plate id (UUID) or plate text."""

    repository = _repository(workspace)
    service = RecognitionService(repository)
    record = service.resolve_plate_input(plate_id_or_text)
    if record is None:
        raise typer.BadParameter(f"no such plate: {plate_id_or_text}")
    try:
        identity = service.get_plate_identity(UUID(record.plate_id))
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "identity_id": str(identity.identity_id),
                "normalized_texts": identity.normalized_texts,
                "member_plate_ids": [str(p) for p in identity.member_plate_ids],
                "label": identity.label,
                "observation_count": identity.observation_count,
                "first_seen_at": identity.first_seen_at.isoformat()
                if identity.first_seen_at
                else None,
                "last_seen_at": identity.last_seen_at.isoformat()
                if identity.last_seen_at
                else None,
            },
            indent=2,
        )
    )


@recognize_app.command("plates-search")
def recognize_plates_search(
    query: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Search plate records by text, id, or label. Falls back to fuzzy
    suggestions (tolerating likely OCR misreads) if nothing matches exactly."""

    repository = _repository(workspace)
    result = RecognitionService(repository).search_plates(query)
    exact_matches = [
        {"plate_id": r.plate_id, "normalized_text": r.normalized_text}  # type: ignore[attr-defined]
        for r in result.exact_matches
    ]
    fuzzy_suggestions = [
        {"plate_id": r.plate_id, "normalized_text": r.normalized_text}  # type: ignore[attr-defined]
        for r in result.fuzzy_suggestions
    ]
    typer.echo(
        json.dumps(
            {
                "exact_matches": exact_matches,
                "fuzzy_suggestions": fuzzy_suggestions,
            },
            indent=2,
        )
    )


@recognize_app.command("plates-debug")
def recognize_plates_debug(
    event_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="defaults to workspace/patterns/plate_debug/<event-id>"),
    ] = None,
) -> None:
    """Re-run plate detection on an event's derived clips and save annotated
    frames so you can actually see what the detector found (and didn't) --
    every candidate region drawn and labeled by source and confidence. Use
    this to check whether detection is working on your real footage rather
    than guessing from the review queue alone. See docs/local-ai.md."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    target_dir = output_dir or (repository.workspace.patterns / "plate_debug" / str(event_id))
    written = EnrichmentService(repository, runtime).debug_render_plate_detections(
        event_id, target_dir
    )
    typer.echo(
        json.dumps(
            {"frame_count": len(written), "output_dir": str(target_dir)},
            indent=2,
        )
    )


@recognize_app.command("plates-sightings")
def recognize_plates_sightings(
    plate_id_or_text: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
    exact: Annotated[
        bool, typer.Option("--exact", help="only this literal plate record, ignore merges")
    ] = False,
) -> None:
    """List every observation of a specific plate identity (accepts a plate id
    or plate text). By default follows any `plates-merge` links; pass
    --exact to see only this literal record's own observations."""

    repository = _repository(workspace)
    observations = RecognitionService(repository).list_plate_sightings(
        plate_id_or_text, follow_merges=not exact
    )
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "event_id": o.event_id,
                    "normalized_text": o.normalized_text,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                    "crop_path": o.crop_path,
                    "review_status": o.review_status,
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("voices")
def recognize_voices(
    workspace: WorkspaceOption,
    include_merged: Annotated[bool, typer.Option("--include-merged")] = False,
) -> None:
    """List every local voice identity: id, observation count, first/last
    seen, and (if set) the private label. See docs/local-ai.md's voice
    section before treating any match as more than a heuristic prompt for
    review -- this is a meaningfully weaker fingerprint than face/plate
    recognition. Clusters merged into another identity via `voices-merge`
    are hidden by default; pass --include-merged to see every raw
    cluster."""

    repository = _repository(workspace)
    clusters = repository.database.list_voice_clusters()
    if not include_merged:
        clusters = [c for c in clusters if not c.merged_into]
    typer.echo(
        json.dumps(
            [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "observation_count": c.observation_count,
                    "first_seen_at": c.first_seen_at.isoformat() if c.first_seen_at else None,
                    "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
                    "merged_into": c.merged_into,
                }
                for c in clusters
            ],
            indent=2,
        )
    )


@recognize_app.command("voices-label")
def recognize_voices_label(
    cluster_id: Annotated[UUID, typer.Argument()],
    label: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
) -> None:
    """Attach a private, local-only nickname to a voice cluster."""

    repository = _repository(workspace)
    repository.database.set_voice_cluster_label(cluster_id, label)
    typer.echo(f"labeled {cluster_id} as '{label}'")


@recognize_app.command("voices-merge")
def recognize_voices_merge(
    source_cluster_id: Annotated[UUID, typer.Argument()],
    target_cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Declare that two voice clusters are the same speaker. Neither cluster
    is edited or deleted; the merge is permanently logged."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).merge_voices(
            source_cluster_id, target_cluster_id, actor, notes
        )
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"merged {source_cluster_id} -> {target_cluster_id}")


@recognize_app.command("voices-cleanup")
def recognize_voices_cleanup(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    window_seconds: Annotated[
        float | None,
        typer.Option(
            "--window-seconds",
            help="override enrichment.voice.duplicate_observation_window_seconds",
        ),
    ] = None,
) -> None:
    """Automatically collapse near-duplicate voice observations within the
    same event and cluster (e.g. one continuous utterance split into
    several adjacent VAD segments), mirroring `faces-cleanup`."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    effective_window = (
        window_seconds
        if window_seconds is not None
        else runtime.enrichment.voice.duplicate_observation_window_seconds
    )
    result = RecognitionService(repository).cleanup_duplicate_voice_observations(effective_window)
    typer.echo(
        json.dumps(
            {
                "suppressed_count": len(result.suppressed_observation_ids),
                "kept_count": len(result.kept_observation_ids),
                "clusters_with_duplicates": result.clusters_with_duplicates,
            },
            indent=2,
        )
    )


@recognize_app.command("voices-identity")
def recognize_voices_identity(
    cluster_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Show the aggregated identity for a voice cluster: every member
    cluster id merged into it, combined observation count, first/last seen."""

    repository = _repository(workspace)
    try:
        identity = RecognitionService(repository).get_voice_identity(cluster_id)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "identity_id": str(identity.identity_id),
                "member_cluster_ids": [str(c) for c in identity.member_cluster_ids],
                "label": identity.label,
                "observation_count": identity.observation_count,
                "first_seen_at": identity.first_seen_at.isoformat()
                if identity.first_seen_at
                else None,
                "last_seen_at": identity.last_seen_at.isoformat()
                if identity.last_seen_at
                else None,
            },
            indent=2,
        )
    )


@recognize_app.command("voices-search")
def recognize_voices_search(
    query: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Search voice clusters by id (full or partial) or label."""

    repository = _repository(workspace)
    result = RecognitionService(repository).search_voices(query)
    typer.echo(
        json.dumps(
            {
                "exact_matches": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.exact_matches
                ],
                "fuzzy_suggestions": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.fuzzy_suggestions
                ],
            },
            indent=2,
        )
    )


@recognize_app.command("voices-sightings")
def recognize_voices_sightings(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    exact: Annotated[
        bool, typer.Option("--exact", help="only this literal cluster, ignore merges")
    ] = False,
    include_duplicates: Annotated[bool, typer.Option("--include-duplicates")] = False,
) -> None:
    """List every observation (time, camera, segment) matched to a voice
    identity. By default follows `voices-merge` links and hides
    `voices-cleanup`-suppressed duplicates."""

    repository = _repository(workspace)
    observations = RecognitionService(repository).list_voice_sightings(
        cluster_id, follow_merges=not exact, include_duplicates=include_duplicates
    )
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "event_id": o.event_id,
                    "cluster_id": o.cluster_id,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                    "segment_start_seconds": o.segment_start_seconds,
                    "segment_end_seconds": o.segment_end_seconds,
                    "energy_confidence": o.energy_confidence,
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("vehicles")
def recognize_vehicles(
    workspace: WorkspaceOption,
    include_merged: Annotated[bool, typer.Option("--include-merged")] = False,
) -> None:
    """List every local vehicle-appearance identity: id, observation count,
    first/last seen, and (if set) the private label. See docs/local-ai.md's
    vehicle-appearance section before treating any match as more than a
    heuristic prompt for review -- it cannot distinguish two vehicles of
    the same color and body shape. Clusters merged into another identity
    via `vehicles-merge` are hidden by default; pass --include-merged to
    see every raw cluster."""

    repository = _repository(workspace)
    clusters = repository.database.list_vehicle_appearance_clusters()
    if not include_merged:
        clusters = [c for c in clusters if not c.merged_into]
    typer.echo(
        json.dumps(
            [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "observation_count": c.observation_count,
                    "first_seen_at": c.first_seen_at.isoformat() if c.first_seen_at else None,
                    "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
                    "merged_into": c.merged_into,
                }
                for c in clusters
            ],
            indent=2,
        )
    )


@recognize_app.command("vehicles-label")
def recognize_vehicles_label(
    cluster_id: Annotated[UUID, typer.Argument()],
    label: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
) -> None:
    """Attach a private, local-only nickname to a vehicle-appearance cluster."""

    repository = _repository(workspace)
    repository.database.set_vehicle_appearance_cluster_label(cluster_id, label)
    typer.echo(f"labeled {cluster_id} as '{label}'")


@recognize_app.command("vehicles-merge")
def recognize_vehicles_merge(
    source_cluster_id: Annotated[UUID, typer.Argument()],
    target_cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Declare that two vehicle-appearance clusters are the same vehicle.
    Neither cluster is edited or deleted; the merge is permanently logged."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).merge_vehicle_appearances(
            source_cluster_id, target_cluster_id, actor, notes
        )
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"merged {source_cluster_id} -> {target_cluster_id}")


@recognize_app.command("vehicles-confirm")
def recognize_vehicles_confirm(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    representative: Annotated[
        list[UUID],
        typer.Option("--representative", help="repeatable; at least one required"),
    ],
    actor: ActorOption = None,
    label: Annotated[str | None, typer.Option("--label")] = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="also delete every non-representative crop in this cluster immediately",
        ),
    ] = False,
) -> None:
    """Confirm a vehicle-appearance cluster is a real, consistent vehicle:
    pick its representative crop(s) and (optionally) label it. Mirrors
    `faces-confirm` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).confirm_identity(
            cluster_id,
            representative,
            actor,
            "vehicle_appearance",
            label=label,
            notes=notes,
            purge=purge,
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-reject-cluster")
def recognize_vehicles_reject_cluster(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete every crop in this cluster immediately")
    ] = False,
) -> None:
    """Mark an entire vehicle-appearance cluster a false positive. Mirrors
    `faces-reject-cluster` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_cluster(
            cluster_id, actor, "vehicle_appearance", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-reject-observation")
def recognize_vehicles_reject_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete this crop immediately")
    ] = False,
) -> None:
    """Mark a single vehicle-appearance observation a false positive.
    Mirrors `faces-reject-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_observation(
            observation_id, actor, "vehicle_appearance", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-detach-observation")
def recognize_vehicles_detach_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Remove a single vehicle-appearance observation from its cluster
    entirely. Mirrors `faces-detach-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).detach_observation(
            observation_id, actor, "vehicle_appearance", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-move-observation")
def recognize_vehicles_move_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    to: Annotated[UUID, typer.Option("--to", help="the target cluster id to move it into")],
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Reassign a single vehicle-appearance observation to another
    cluster. Mirrors `faces-move-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).move_observation(
            observation_id, to, actor, "vehicle_appearance", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-purge-reviewed")
def recognize_vehicles_purge_reviewed(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="show what would be purged without deleting")
    ] = False,
) -> None:
    """Delete crop images for every already-reviewed vehicle-appearance
    observation not yet purged. Mirrors `faces-purge-reviewed` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    record = RecognitionService(repository).purge_reviewed_crops(
        "vehicle_appearance", actor, notes=notes, dry_run=dry_run
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("vehicles-cleanup")
def recognize_vehicles_cleanup(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    window_seconds: Annotated[
        float | None,
        typer.Option(
            "--window-seconds",
            help="override enrichment.vehicle_appearance.duplicate_observation_window_seconds",
        ),
    ] = None,
) -> None:
    """Automatically collapse near-duplicate vehicle-appearance observations
    within the same event and cluster, mirroring `faces-cleanup`/
    `voices-cleanup`."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    effective_window = (
        window_seconds
        if window_seconds is not None
        else runtime.enrichment.vehicle_appearance.duplicate_observation_window_seconds
    )
    result = RecognitionService(repository).cleanup_duplicate_vehicle_appearance_observations(
        effective_window
    )
    typer.echo(
        json.dumps(
            {
                "suppressed_count": len(result.suppressed_observation_ids),
                "kept_count": len(result.kept_observation_ids),
                "clusters_with_duplicates": result.clusters_with_duplicates,
            },
            indent=2,
        )
    )


@recognize_app.command("vehicles-identity")
def recognize_vehicles_identity(
    cluster_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Show the aggregated identity for a vehicle-appearance cluster: every
    member cluster id merged into it, combined observation count,
    first/last seen, representative crops."""

    repository = _repository(workspace)
    try:
        identity = RecognitionService(repository).get_vehicle_appearance_identity(cluster_id)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "identity_id": str(identity.identity_id),
                "member_cluster_ids": [str(c) for c in identity.member_cluster_ids],
                "label": identity.label,
                "observation_count": identity.observation_count,
                "first_seen_at": identity.first_seen_at.isoformat()
                if identity.first_seen_at
                else None,
                "last_seen_at": identity.last_seen_at.isoformat()
                if identity.last_seen_at
                else None,
                "representative_crop_paths": identity.representative_crop_paths,
            },
            indent=2,
        )
    )


@recognize_app.command("vehicles-search")
def recognize_vehicles_search(
    query: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Search vehicle-appearance clusters by id (full or partial) or label."""

    repository = _repository(workspace)
    result = RecognitionService(repository).search_vehicle_appearances(query)
    typer.echo(
        json.dumps(
            {
                "exact_matches": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.exact_matches
                ],
                "fuzzy_suggestions": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.fuzzy_suggestions
                ],
            },
            indent=2,
        )
    )


@recognize_app.command("vehicles-sightings")
def recognize_vehicles_sightings(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    exact: Annotated[
        bool, typer.Option("--exact", help="only this literal cluster, ignore merges")
    ] = False,
    include_duplicates: Annotated[bool, typer.Option("--include-duplicates")] = False,
) -> None:
    """List every observation (time, camera, crop, confidence) matched to a
    vehicle-appearance identity. By default follows `vehicles-merge` links
    and hides `vehicles-cleanup`-suppressed duplicates."""

    repository = _repository(workspace)
    observations = RecognitionService(repository).list_vehicle_appearance_sightings(
        cluster_id, follow_merges=not exact, include_duplicates=include_duplicates
    )
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "event_id": o.event_id,
                    "cluster_id": o.cluster_id,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                    "crop_path": o.crop_path,
                    "detector_confidence": o.detector_confidence,
                    "embedding_distance_to_cluster": o.embedding_distance_to_cluster,
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("persons")
def recognize_persons(
    workspace: WorkspaceOption,
    include_merged: Annotated[bool, typer.Option("--include-merged")] = False,
) -> None:
    """List every local person-appearance identity: id, observation count,
    first/last seen, and (if set) the private label. Structured attributes
    only (dominant clothing color, height-in-frame ratio) -- never a
    learned face embedding or an AI-generated description; see
    `enrichment/person_appearance.py`'s module docstring. Clusters merged
    into another identity via `persons-merge` are hidden by default; pass
    --include-merged to see every raw cluster."""

    repository = _repository(workspace)
    clusters = repository.database.list_person_appearance_clusters()
    if not include_merged:
        clusters = [c for c in clusters if not c.merged_into]
    typer.echo(
        json.dumps(
            [
                {
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "observation_count": c.observation_count,
                    "first_seen_at": c.first_seen_at.isoformat() if c.first_seen_at else None,
                    "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
                    "merged_into": c.merged_into,
                }
                for c in clusters
            ],
            indent=2,
        )
    )


@recognize_app.command("persons-label")
def recognize_persons_label(
    cluster_id: Annotated[UUID, typer.Argument()],
    label: Annotated[str, typer.Argument()],
    workspace: WorkspaceOption,
) -> None:
    """Attach a private, local-only nickname to a person-appearance cluster."""

    repository = _repository(workspace)
    repository.database.set_person_appearance_cluster_label(cluster_id, label)
    typer.echo(f"labeled {cluster_id} as '{label}'")


@recognize_app.command("persons-merge")
def recognize_persons_merge(
    source_cluster_id: Annotated[UUID, typer.Argument()],
    target_cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Declare that two person-appearance clusters are the same person.
    Neither cluster is edited or deleted; the merge is permanently logged."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).merge_person_appearances(
            source_cluster_id, target_cluster_id, actor, notes
        )
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"merged {source_cluster_id} -> {target_cluster_id}")


@recognize_app.command("persons-confirm")
def recognize_persons_confirm(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    representative: Annotated[
        list[UUID],
        typer.Option("--representative", help="repeatable; at least one required"),
    ],
    actor: ActorOption = None,
    label: Annotated[str | None, typer.Option("--label")] = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool,
        typer.Option(
            "--purge",
            help="also delete every non-representative crop in this cluster immediately",
        ),
    ] = False,
) -> None:
    """Confirm a person-appearance cluster is a real, consistent person:
    pick its representative crop(s) and (optionally) label it. Mirrors
    `vehicles-confirm` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).confirm_identity(
            cluster_id,
            representative,
            actor,
            "person_appearance",
            label=label,
            notes=notes,
            purge=purge,
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-reject-cluster")
def recognize_persons_reject_cluster(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete every crop in this cluster immediately")
    ] = False,
) -> None:
    """Mark an entire person-appearance cluster a false positive. Mirrors
    `vehicles-reject-cluster` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_cluster(
            cluster_id, actor, "person_appearance", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-reject-observation")
def recognize_persons_reject_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    purge: Annotated[
        bool, typer.Option("--purge", help="also delete this crop immediately")
    ] = False,
) -> None:
    """Mark a single person-appearance observation a false positive.
    Mirrors `vehicles-reject-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).reject_observation(
            observation_id, actor, "person_appearance", notes=notes, purge=purge
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-detach-observation")
def recognize_persons_detach_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Remove a single person-appearance observation from its cluster
    entirely. Mirrors `vehicles-detach-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).detach_observation(
            observation_id, actor, "person_appearance", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-move-observation")
def recognize_persons_move_observation(
    observation_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    to: Annotated[UUID, typer.Option("--to", help="the target cluster id to move it into")],
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Reassign a single person-appearance observation to another cluster.
    Mirrors `vehicles-move-observation` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        record = RecognitionService(repository).move_observation(
            observation_id, to, actor, "person_appearance", notes=notes
        )
    except ReviewError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-purge-reviewed")
def recognize_persons_purge_reviewed(
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="show what would be purged without deleting")
    ] = False,
) -> None:
    """Delete crop images for every already-reviewed person-appearance
    observation not yet purged. Mirrors `vehicles-purge-reviewed` exactly."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    record = RecognitionService(repository).purge_reviewed_crops(
        "person_appearance", actor, notes=notes, dry_run=dry_run
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))


@recognize_app.command("persons-cleanup")
def recognize_persons_cleanup(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    window_seconds: Annotated[
        float | None,
        typer.Option(
            "--window-seconds",
            help="override enrichment.person_appearance.duplicate_observation_window_seconds",
        ),
    ] = None,
) -> None:
    """Automatically collapse near-duplicate person-appearance observations
    within the same event and cluster, mirroring `vehicles-cleanup`."""

    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    effective_window = (
        window_seconds
        if window_seconds is not None
        else runtime.enrichment.person_appearance.duplicate_observation_window_seconds
    )
    result = RecognitionService(repository).cleanup_duplicate_person_appearance_observations(
        effective_window
    )
    typer.echo(
        json.dumps(
            {
                "suppressed_count": len(result.suppressed_observation_ids),
                "kept_count": len(result.kept_observation_ids),
                "clusters_with_duplicates": result.clusters_with_duplicates,
            },
            indent=2,
        )
    )


@recognize_app.command("persons-identity")
def recognize_persons_identity(
    cluster_id: Annotated[UUID, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Show the aggregated identity for a person-appearance cluster: every
    member cluster id merged into it, combined observation count,
    first/last seen, representative crops."""

    repository = _repository(workspace)
    try:
        identity = RecognitionService(repository).get_person_appearance_identity(cluster_id)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        json.dumps(
            {
                "identity_id": str(identity.identity_id),
                "member_cluster_ids": [str(c) for c in identity.member_cluster_ids],
                "label": identity.label,
                "observation_count": identity.observation_count,
                "first_seen_at": identity.first_seen_at.isoformat()
                if identity.first_seen_at
                else None,
                "last_seen_at": identity.last_seen_at.isoformat()
                if identity.last_seen_at
                else None,
                "representative_crop_paths": identity.representative_crop_paths,
            },
            indent=2,
        )
    )


@recognize_app.command("persons-search")
def recognize_persons_search(
    query: Annotated[str, typer.Argument()], workspace: WorkspaceOption
) -> None:
    """Search person-appearance clusters by id (full or partial) or label."""

    repository = _repository(workspace)
    result = RecognitionService(repository).search_person_appearances(query)
    typer.echo(
        json.dumps(
            {
                "exact_matches": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.exact_matches
                ],
                "fuzzy_suggestions": [
                    {"cluster_id": c.cluster_id, "label": c.label}  # type: ignore[attr-defined]
                    for c in result.fuzzy_suggestions
                ],
            },
            indent=2,
        )
    )


@recognize_app.command("persons-sightings")
def recognize_persons_sightings(
    cluster_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    exact: Annotated[
        bool, typer.Option("--exact", help="only this literal cluster, ignore merges")
    ] = False,
    include_duplicates: Annotated[bool, typer.Option("--include-duplicates")] = False,
) -> None:
    """List every observation (time, camera, crop, confidence) matched to a
    person-appearance identity. By default follows `persons-merge` links
    and hides `persons-cleanup`-suppressed duplicates."""

    repository = _repository(workspace)
    observations = RecognitionService(repository).list_person_appearance_sightings(
        cluster_id, follow_merges=not exact, include_duplicates=include_duplicates
    )
    typer.echo(
        json.dumps(
            [
                {
                    "observation_id": o.observation_id,
                    "event_id": o.event_id,
                    "cluster_id": o.cluster_id,
                    "camera_id": o.camera_id,
                    "observed_at": o.observed_at.isoformat(),
                    "crop_path": o.crop_path,
                    "detector_confidence": o.detector_confidence,
                    "embedding_distance_to_cluster": o.embedding_distance_to_cluster,
                }
                for o in observations
            ],
            indent=2,
        )
    )


@recognize_app.command("suggest-merges")
def recognize_suggest_merges(
    workspace: WorkspaceOption,
    config: ConfigOption = None,
    entity_type: Annotated[
        str,
        typer.Option(
            "--entity-type",
            help="'face', 'plate', 'voice', 'vehicle_appearance', or 'person_appearance'",
        ),
    ] = "face",
    face_distance_threshold: Annotated[
        float | None, typer.Option("--face-distance-threshold")
    ] = None,
    face_suggestion_multiplier: Annotated[
        float | None, typer.Option("--face-suggestion-multiplier")
    ] = None,
    plate_similarity_threshold: Annotated[
        float | None, typer.Option("--plate-similarity-threshold")
    ] = None,
    voice_distance_threshold: Annotated[
        float | None, typer.Option("--voice-distance-threshold")
    ] = None,
    voice_suggestion_multiplier: Annotated[
        float | None, typer.Option("--voice-suggestion-multiplier")
    ] = None,
    vehicle_distance_threshold: Annotated[
        float | None, typer.Option("--vehicle-distance-threshold")
    ] = None,
    vehicle_suggestion_multiplier: Annotated[
        float | None, typer.Option("--vehicle-suggestion-multiplier")
    ] = None,
    person_distance_threshold: Annotated[
        float | None, typer.Option("--person-distance-threshold")
    ] = None,
    person_suggestion_multiplier: Annotated[
        float | None, typer.Option("--person-suggestion-multiplier")
    ] = None,
) -> None:
    """Scan for pairs of clusters/records that look like they might be the
    same identity and add them to the review queue (`merge-suggestions`) --
    never merges anything automatically. Run this periodically (e.g. after
    `enrich`) rather than expecting it to happen on its own."""

    if entity_type not in ("face", "plate", "voice", "vehicle_appearance", "person_appearance"):
        raise typer.BadParameter(
            "--entity-type must be 'face', 'plate', 'voice', 'vehicle_appearance', "
            "or 'person_appearance'"
        )
    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    service = RecognitionService(repository)
    if entity_type == "face":
        face_config = runtime.enrichment.face
        default_face_threshold = (
            face_config.embedding_cluster_distance_threshold
            if face_config.embedding_model == "auraface"
            else face_config.cluster_distance_threshold
        )
        suggestions = service.suggest_face_merges(
            face_distance_threshold
            if face_distance_threshold is not None
            else default_face_threshold,
            face_suggestion_multiplier
            if face_suggestion_multiplier is not None
            else face_config.merge_suggestion_multiplier,
            embedding_model=face_config.embedding_model,
        )
    elif entity_type == "voice":
        voice_config = runtime.enrichment.voice
        suggestions = service.suggest_voice_merges(
            voice_distance_threshold
            if voice_distance_threshold is not None
            else voice_config.cluster_distance_threshold,
            voice_suggestion_multiplier
            if voice_suggestion_multiplier is not None
            else voice_config.merge_suggestion_multiplier,
        )
    elif entity_type == "vehicle_appearance":
        vehicle_config = runtime.enrichment.vehicle_appearance
        suggestions = service.suggest_vehicle_appearance_merges(
            vehicle_distance_threshold
            if vehicle_distance_threshold is not None
            else vehicle_config.cluster_distance_threshold,
            vehicle_suggestion_multiplier
            if vehicle_suggestion_multiplier is not None
            else vehicle_config.merge_suggestion_multiplier,
        )
    elif entity_type == "person_appearance":
        person_config = runtime.enrichment.person_appearance
        suggestions = service.suggest_person_appearance_merges(
            person_distance_threshold
            if person_distance_threshold is not None
            else person_config.cluster_distance_threshold,
            person_suggestion_multiplier
            if person_suggestion_multiplier is not None
            else person_config.merge_suggestion_multiplier,
        )
    else:
        plate_config = runtime.enrichment.plate
        suggestions = service.suggest_plate_merges(
            plate_similarity_threshold
            if plate_similarity_threshold is not None
            else plate_config.merge_suggestion_similarity_threshold
        )
    typer.echo(json.dumps([s.model_dump(mode="json") for s in suggestions], indent=2))


@recognize_app.command("merge-suggestions")
def recognize_merge_suggestions(
    workspace: WorkspaceOption,
    entity_type: Annotated[str | None, typer.Option("--entity-type")] = None,
    status: Annotated[str | None, typer.Option("--status")] = "pending",
) -> None:
    """List merge suggestions awaiting review (or, with --status, any status)."""

    repository = _repository(workspace)
    rows = RecognitionService(repository).list_merge_suggestions(entity_type, status)
    typer.echo(
        json.dumps(
            [
                {
                    "suggestion_id": r.suggestion_id,
                    "entity_type": r.entity_type,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "similarity_score": r.similarity_score,
                    "basis": r.basis,
                    "status": r.status,
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ],
            indent=2,
        )
    )


@recognize_app.command("merge-suggestions-confirm")
def recognize_merge_suggestions_confirm(
    suggestion_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Accept a merge suggestion: performs the actual merge (logged exactly
    like a manual `faces-merge`/`plates-merge` would be) and marks the
    suggestion resolved."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).confirm_merge_suggestion(suggestion_id, actor, notes)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"confirmed suggestion {suggestion_id}")


@recognize_app.command("merge-suggestions-reject")
def recognize_merge_suggestions_reject(
    suggestion_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    actor: ActorOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Decline a merge suggestion: no merge happens; the suggestion is kept,
    marked rejected, as a record that a human looked at it and said no."""

    actor = _resolve_actor(actor)
    repository = _repository(workspace)
    try:
        RecognitionService(repository).reject_merge_suggestion(suggestion_id, actor, notes)
    except MergeError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"rejected suggestion {suggestion_id}")


@recognize_app.command("encounters")
def recognize_encounters(
    workspace: WorkspaceOption,
    face: Annotated[
        UUID | None, typer.Option("--face", help="a face cluster id (or merge target)")
    ] = None,
    plate: Annotated[str | None, typer.Option("--plate", help="a plate id or plate text")] = None,
    vehicle: Annotated[
        UUID | None,
        typer.Option("--vehicle", help="a vehicle-appearance cluster id (or merge target)"),
    ] = None,
) -> None:
    """Show every Encounter (see `docs/local-ai.md`'s "Security camera
    support" section) involving a given face/plate/vehicle identity --
    exactly one of --face/--plate/--vehicle must be passed. An Encounter
    only means "these were observed in the same clip within a few seconds
    of each other" -- it never claims any spatial correspondence between
    them (e.g. that a co-listed face belongs to a co-listed vehicle)."""

    options = (("--face", face), ("--plate", plate), ("--vehicle", vehicle))
    selected = [name for name, value in options if value is not None]
    if len(selected) != 1:
        raise typer.BadParameter("pass exactly one of --face, --plate, --vehicle")

    repository = _repository(workspace)
    recognition = RecognitionService(repository)
    if face is not None:
        observation_ids = [UUID(o.observation_id) for o in recognition.list_face_sightings(face)]
        encounters = repository.database.list_encounters_by_face_observation_ids(observation_ids)
    elif plate is not None:
        observation_ids = [UUID(o.observation_id) for o in recognition.list_plate_sightings(plate)]
        encounters = repository.database.list_encounters_by_plate_observation_ids(observation_ids)
    else:
        assert vehicle is not None
        observation_ids = [
            UUID(o.observation_id) for o in recognition.list_vehicle_appearance_sightings(vehicle)
        ]
        encounters = repository.database.list_encounters_by_vehicle_appearance_observation_ids(
            observation_ids
        )

    typer.echo(
        json.dumps(
            [
                {
                    "encounter_id": e.encounter_id,
                    "event_id": e.event_id,
                    "clip_id": e.clip_id,
                    "camera_id": e.camera_id,
                    "observed_at": e.observed_at.isoformat(),
                    "face_observation_id": e.face_observation_id,
                    "plate_observation_id": e.plate_observation_id,
                    "voice_observation_id": e.voice_observation_id,
                    "vehicle_appearance_observation_id": e.vehicle_appearance_observation_id,
                }
                for e in encounters
            ],
            indent=2,
        )
    )


@config_app.command("set-actor")
def config_set_actor(name: Annotated[str, typer.Argument()]) -> None:
    """Set the default `--actor` used by every command that omits it.

    Stored per-machine (`gaggle.core.cli_config.config_path()`), not in any
    workspace -- run this once and every `--actor`-taking command falls
    back to it, while `--actor <other name>` still overrides per call."""

    cli_config.set_default_actor(name)
    typer.echo(f"default actor set to '{name}'")


@config_app.command("show")
def config_show() -> None:
    typer.echo(json.dumps({"default_actor": cli_config.get_default_actor()}, indent=2))


@config_app.command("set-sync-offset")
def config_set_sync_offset(
    camera_id: Annotated[str, typer.Argument()],
    seconds: Annotated[float, typer.Argument()],
) -> None:
    """Print the config.yaml snippet for a manual per-camera timing-sync
    correction (see `SyncConfig.manual_offset_overrides`'s docstring in
    `core/config.py` for when this is the right tool). Prints rather than
    writes the file directly -- this project has no comment-preserving
    YAML writer, and a blind rewrite would strip every comment from a
    real config.yaml; pasting four lines in by hand is safer than that."""

    typer.echo(
        "Add (or extend) this under your active profile in config.yaml:\n\n"
        "  sync:\n"
        "    manual_offset_overrides:\n"
        f"      {camera_id}: {seconds}\n\n"
        "Only affects clips normalized by a future 'gaggle analyze' run -- an "
        "event already built with the wrong sync needs 'gaggle events split' "
        "instead, since 'analyze' is idempotent and won't reprocess "
        "already-covered clips just because this config changed."
    )


@events_app.command("split")
def events_split(
    event_id: Annotated[UUID, typer.Argument()],
    workspace: WorkspaceOption,
    group: Annotated[
        list[str],
        typer.Option(
            "--group",
            help=(
                "comma-separated clip_ids for one split-off event; pass --group at least "
                "twice, and every clip_id this event's derived_artifacts reference must "
                "appear in exactly one group"
            ),
        ),
    ],
    actor: ActorOption = None,
    config: ConfigOption = None,
    notes: Annotated[str, typer.Option("--notes")] = "",
) -> None:
    """Split an event that incorrectly bundled clips from separate
    recording sessions (see `core/events.py`'s module docstring for why
    this happens) into independent replacement events -- one per --group.
    The original event is never deleted; it gets a final revision
    recording which events it was split into."""

    actor = _resolve_actor(actor)
    runtime = _load_runtime(config)
    repository = _repository(workspace, runtime)
    clip_id_groups = [
        [UUID(clip_id.strip()) for clip_id in group_str.split(",") if clip_id.strip()]
        for group_str in group
    ]
    try:
        new_events = EventSplitService(repository, runtime).split_event(
            event_id, clip_id_groups, actor, notes=notes
        )
    except EventSplitError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps([e.model_dump(mode="json") for e in new_events], indent=2))


def main() -> None:
    app()
