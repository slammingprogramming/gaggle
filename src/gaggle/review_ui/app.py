from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast
from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from gaggle.core import cli_config
from gaggle.core.config import load_config
from gaggle.core.events import EventSplitError, EventSplitService
from gaggle.core.pipeline import AnalysisPipeline
from gaggle.core.recognition import ClusterEntityType, MergeError, RecognitionService, ReviewError
from gaggle.core.review import ReviewService
from gaggle.export.service import ExportService
from gaggle.schemas.common import ArtifactReference
from gaggle.schemas.event import EventRecord
from gaggle.schemas.review import ReviewAction
from gaggle.storage.database import (
    FaceClusterRow,
    FaceObservationRow,
    PersonAppearanceClusterRow,
    PersonAppearanceObservationRow,
    PlateObservationRow,
    TimelineQuery,
    VehicleAppearanceClusterRow,
    VehicleAppearanceObservationRow,
    VoiceObservationRow,
)
from gaggle.storage.repository import Repository
from gaggle.timeline.service import TimelineService


class ReviewActionRequest(BaseModel):
    action: Literal["accept", "reject", "annotate", "retag", "preserve", "export"]
    actor: str
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class ConfirmClusterRequest(BaseModel):
    representative_observation_ids: list[UUID]
    actor: str
    label: str | None = None
    notes: str = ""
    purge: bool = False


class RejectRequest(BaseModel):
    actor: str
    notes: str = ""
    purge: bool = False


class PurgeReviewedRequest(BaseModel):
    actor: str
    notes: str = ""
    dry_run: bool = False


class ConfirmPlateObservationRequest(BaseModel):
    corrected_text: str
    actor: str
    notes: str = ""
    purge: bool = False


class SplitEventRequest(BaseModel):
    clip_id_groups: list[list[UUID]]
    actor: str
    notes: str = ""


class DetachObservationRequest(BaseModel):
    actor: str
    notes: str = ""


class MoveObservationRequest(BaseModel):
    target_cluster_id: UUID
    actor: str
    notes: str = ""


class MergeSuggestionActionRequest(BaseModel):
    actor: str
    notes: str = ""


_CLUSTER_ENTITY_TYPES = ("face", "vehicle_appearance", "person_appearance")
_CROP_ENTITY_TYPES = ("face", "plate", "vehicle_appearance", "person_appearance")


def _require_cluster_entity_type(entity_type: str) -> ClusterEntityType:
    if entity_type not in _CLUSTER_ENTITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"entity_type must be one of {_CLUSTER_ENTITY_TYPES} (plates have no clusters)",
        )
    return cast(ClusterEntityType, entity_type)


def create_app(workspace_root: Path) -> FastAPI:
    repository = Repository(workspace_root)
    repository.initialize()
    review_service = ReviewService(repository)
    recognition_service = RecognitionService(repository)
    app = FastAPI(title="gaggle review")

    def _resolve_workspace_path(path_str: str) -> Path:
        resolved = Path(path_str).resolve()
        root = repository.workspace.root.resolve()
        if not resolved.is_relative_to(root):
            raise HTTPException(status_code=403, detail="path is outside the workspace")
        if not resolved.exists():
            raise HTTPException(status_code=404, detail="media file not found")
        return resolved

    @app.get("/api/config/default-actor")
    def get_default_actor() -> dict[str, str | None]:
        """Read-only: the per-machine default actor set via `gaggle config
        set-actor`, if any -- the review form and inline-reject prompt both
        pre-fill from this so the user isn't retyping their name on every
        action (still overridable per action)."""

        return {"default_actor": cli_config.get_default_actor()}

    @app.get("/", response_class=HTMLResponse)
    def index(
        severity: str | None = None, camera: str | None = None, status: str | None = None
    ) -> str:
        query = TimelineQuery(severity=severity, camera_id=camera, review_decision=status)
        rows = TimelineService(repository.database).list_events(query)
        return _render_index(rows)

    @app.get("/events/{event_id}", response_class=HTMLResponse)
    def event_detail(event_id: UUID) -> str:
        try:
            event = repository.load_event(event_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="event not found") from error
        review_history = repository.list_review_actions(event_id)
        return _render_event_detail(
            event,
            review_history,
            face_observations=repository.database.list_face_observations_for_event(event_id),
            plate_observations=repository.database.list_plate_observations_for_event(event_id),
            voice_observations=repository.database.list_voice_observations_for_event(event_id),
            vehicle_observations=(
                repository.database.list_vehicle_appearance_observations_for_event(event_id)
            ),
            person_observations=(
                repository.database.list_person_appearance_observations_for_event(event_id)
            ),
            transcript=_read_json_if_exists(repository.workspace.transcripts / f"{event_id}.json"),
            llm_enrichment=_read_json_if_exists(
                repository.workspace.transcripts / f"{event_id}.llm.json"
            ),
        )

    @app.get("/api/events")
    def list_events(
        severity: str | None = None, camera: str | None = None, status: str | None = None
    ) -> list[dict[str, object]]:
        query = TimelineQuery(severity=severity, camera_id=camera, review_decision=status)
        return TimelineService(repository.database).list_events(query)

    @app.get("/api/events/{event_id}")
    def get_event(event_id: UUID) -> dict[str, object]:
        try:
            return repository.load_event(event_id).model_dump(mode="json")
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="event not found") from error

    @app.get("/api/events/{event_id}/revisions")
    def get_event_revisions(event_id: UUID) -> list[dict[str, object]]:
        return [
            {
                "revision": rev.revision,
                "revision_reason": rev.revision_reason,
                "revised_at": rev.revised_at.isoformat() if rev.revised_at else None,
                "previous_revision_hash": rev.previous_revision_hash,
            }
            for rev in repository.list_event_revisions(event_id)
        ]

    @app.get("/api/events/{event_id}/review-actions")
    def list_review_actions(event_id: UUID) -> list[dict[str, object]]:
        return [
            action.model_dump(mode="json") for action in repository.list_review_actions(event_id)
        ]

    @app.post("/api/events/{event_id}/review-actions")
    def add_review_action(event_id: UUID, request: ReviewActionRequest) -> dict[str, object]:
        record, updated_event = review_service.append_action(
            event_id=event_id,
            action=request.action,
            actor=request.actor,
            notes=request.notes,
            tags=request.tags,
        )
        result: dict[str, object] = {"review_action": record.model_dump(mode="json")}
        if request.action == "preserve":
            runtime = load_config(None)
            pipeline = AnalysisPipeline(repository, runtime)
            preserved = pipeline.preserve_event(event_id)
            result["preservation_status"] = preserved.preservation_status.model_dump(mode="json")
        elif request.action == "export":
            export_result = ExportService(repository).export_event_bundle(event_id)
            result["export_path"] = str(export_result.path)
        else:
            result["review_summary"] = updated_event.review_summary.model_dump(mode="json")
        return result

    @app.post("/api/events/{event_id}/split")
    def split_event(event_id: UUID, request: SplitEventRequest) -> dict[str, object]:
        """See `core/events.py`'s module docstring for why an event
        sometimes needs this: normalize/sync.py's pure time-overlap
        heuristic can bundle clips from separate recording sessions into
        one event, and this is the human-review correction for it."""

        runtime = load_config(None)
        try:
            new_events = EventSplitService(repository, runtime).split_event(
                event_id, request.clip_id_groups, request.actor, notes=request.notes
            )
        except EventSplitError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"new_event_ids": [str(e.event_id) for e in new_events]}

    @app.get("/api/events/{event_id}/media/{artifact_index}")
    def get_event_media(event_id: UUID, artifact_index: int) -> FileResponse:
        event = repository.load_event(event_id)
        if artifact_index < 0 or artifact_index >= len(event.derived_artifacts):
            raise HTTPException(status_code=404, detail="no such derived artifact")
        artifact = event.derived_artifacts[artifact_index]
        resolved = _resolve_workspace_path(artifact.path)
        return FileResponse(resolved, media_type="video/mp4")

    @app.get("/api/events/{event_id}/crop/{entity_type}/{observation_id}")
    def get_observation_crop(
        event_id: UUID, entity_type: str, observation_id: UUID
    ) -> FileResponse:
        """Serve a face/plate/vehicle-appearance observation's crop image.

        Voice observations have no crop (see `enrichment/voice.py`'s
        docstring on why). Resolves through `_resolve_workspace_path`
        exactly like `get_event_media` -- never serve a stored path
        directly without that containment check.
        """

        getters = {
            "face": repository.database.get_face_observation,
            "plate": repository.database.get_plate_observation,
            "vehicle_appearance": repository.database.get_vehicle_appearance_observation,
            "person_appearance": repository.database.get_person_appearance_observation,
        }
        getter = getters.get(entity_type)
        if getter is None:
            raise HTTPException(status_code=404, detail=f"unknown entity_type '{entity_type}'")
        row = getter(observation_id)
        if row is None or row.event_id != str(event_id):
            raise HTTPException(status_code=404, detail="no such observation for this event")
        resolved = _resolve_workspace_path(row.crop_path)
        return FileResponse(resolved, media_type="image/jpeg")

    @app.get("/recognition/{entity_type}", response_class=HTMLResponse)
    def recognition_clusters_page(entity_type: str) -> str:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        clusters = _list_clusters_for_review(repository, cluster_entity_type)
        return _render_recognition_clusters_page(cluster_entity_type, clusters)

    @app.get("/api/recognition/{entity_type}/clusters")
    def api_recognition_clusters(entity_type: str) -> list[dict[str, object]]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        return _list_clusters_for_review(repository, cluster_entity_type)

    @app.get("/api/recognition/{entity_type}/observations/{observation_id}/crop")
    def get_recognition_observation_crop(entity_type: str, observation_id: UUID) -> FileResponse:
        """Cluster-scoped crop serving, unlike `get_observation_crop`
        (event-scoped) -- a cluster spans many events, so this doesn't
        require knowing which event an observation came from."""

        getters = {
            "face": repository.database.get_face_observation,
            "plate": repository.database.get_plate_observation,
            "vehicle_appearance": repository.database.get_vehicle_appearance_observation,
            "person_appearance": repository.database.get_person_appearance_observation,
        }
        getter = getters.get(entity_type)
        if getter is None:
            raise HTTPException(status_code=404, detail=f"unknown entity_type '{entity_type}'")
        row = getter(observation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="no such observation")
        resolved = _resolve_workspace_path(row.crop_path)
        return FileResponse(resolved, media_type="image/jpeg")

    @app.post("/api/recognition/{entity_type}/clusters/{cluster_id}/confirm")
    def confirm_recognition_cluster(
        entity_type: str, cluster_id: UUID, request: ConfirmClusterRequest
    ) -> dict[str, object]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        try:
            record = recognition_service.confirm_identity(
                cluster_id,
                request.representative_observation_ids,
                request.actor,
                cluster_entity_type,
                label=request.label,
                notes=request.notes,
                purge=request.purge,
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/{entity_type}/clusters/{cluster_id}/reject")
    def reject_recognition_cluster(
        entity_type: str, cluster_id: UUID, request: RejectRequest
    ) -> dict[str, object]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        try:
            record = recognition_service.reject_cluster(
                cluster_id,
                request.actor,
                cluster_entity_type,
                notes=request.notes,
                purge=request.purge,
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    # NOTE: these two plate routes must be registered *before* the generic
    # `/api/recognition/{entity_type}/observations/{observation_id}/reject`
    # route below -- FastAPI/Starlette matches routes in registration
    # order, not by literal-vs-parameter specificity, so a plate request
    # would otherwise be swallowed by the generic route and rejected by
    # `_require_cluster_entity_type` (plate has no cluster concept, so
    # it's deliberately excluded from `ClusterEntityType`).
    @app.post("/api/recognition/plate/observations/{observation_id}/reject")
    def reject_plate_recognition_observation(
        observation_id: UUID, request: RejectRequest
    ) -> dict[str, object]:
        try:
            record = recognition_service.reject_plate_observation(
                observation_id, request.actor, notes=request.notes, purge=request.purge
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/plate/observations/{observation_id}/confirm")
    def confirm_plate_recognition_observation(
        observation_id: UUID, request: ConfirmPlateObservationRequest
    ) -> dict[str, object]:
        """Confirm a plate OCR reading as-is, or correct it -- pass the
        corrected text either way (the same text if just confirming)."""

        try:
            record = recognition_service.confirm_plate_observation(
                observation_id,
                request.corrected_text,
                request.actor,
                notes=request.notes,
                purge=request.purge,
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/{entity_type}/observations/{observation_id}/reject")
    def reject_recognition_observation(
        entity_type: str, observation_id: UUID, request: RejectRequest
    ) -> dict[str, object]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        try:
            record = recognition_service.reject_observation(
                observation_id,
                request.actor,
                cluster_entity_type,
                notes=request.notes,
                purge=request.purge,
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/{entity_type}/observations/{observation_id}/detach")
    def detach_recognition_observation(
        entity_type: str, observation_id: UUID, request: DetachObservationRequest
    ) -> dict[str, object]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        try:
            record = recognition_service.detach_observation(
                observation_id, request.actor, cluster_entity_type, notes=request.notes
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/{entity_type}/observations/{observation_id}/move")
    def move_recognition_observation(
        entity_type: str, observation_id: UUID, request: MoveObservationRequest
    ) -> dict[str, object]:
        cluster_entity_type = _require_cluster_entity_type(entity_type)
        try:
            record = recognition_service.move_observation(
                observation_id,
                request.target_cluster_id,
                request.actor,
                cluster_entity_type,
                notes=request.notes,
            )
        except ReviewError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return record.model_dump(mode="json")

    @app.post("/api/recognition/{entity_type}/purge-reviewed")
    def purge_recognition_reviewed(
        entity_type: str, request: PurgeReviewedRequest
    ) -> dict[str, object]:
        if entity_type not in _CROP_ENTITY_TYPES:
            raise HTTPException(
                status_code=400, detail=f"entity_type must be one of {_CROP_ENTITY_TYPES}"
            )
        record = recognition_service.purge_reviewed_crops(
            cast(Literal["face", "plate", "vehicle_appearance"], entity_type),
            request.actor,
            notes=request.notes,
            dry_run=request.dry_run,
        )
        return record.model_dump(mode="json")

    @app.post("/api/recognition/merge-suggestions/{suggestion_id}/confirm")
    def confirm_recognition_merge_suggestion(
        suggestion_id: UUID, request: MergeSuggestionActionRequest
    ) -> dict[str, object]:
        try:
            recognition_service.confirm_merge_suggestion(
                suggestion_id, request.actor, request.notes
            )
        except MergeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"suggestion_id": str(suggestion_id), "status": "confirmed"}

    @app.post("/api/recognition/merge-suggestions/{suggestion_id}/reject")
    def reject_recognition_merge_suggestion(
        suggestion_id: UUID, request: MergeSuggestionActionRequest
    ) -> dict[str, object]:
        try:
            recognition_service.reject_merge_suggestion(suggestion_id, request.actor, request.notes)
        except MergeError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"suggestion_id": str(suggestion_id), "status": "rejected"}

    @app.get("/api/events/{event_id}/transcript")
    def get_event_transcript(event_id: UUID) -> dict[str, object]:
        transcript_path = repository.workspace.transcripts / f"{event_id}.json"
        transcript = _read_json_if_exists(transcript_path)
        if transcript is None:
            raise HTTPException(status_code=404, detail="no transcript for this event")
        return transcript

    @app.get("/api/timeline")
    def api_timeline(
        severity: str | None = None,
        camera: str | None = None,
        review_decision: str | None = None,
        preservation_state: str | None = None,
    ) -> list[dict[str, object]]:
        query = TimelineQuery(
            severity=severity,
            camera_id=camera,
            review_decision=review_decision,
            preservation_state=preservation_state,
        )
        return TimelineService(repository.database).list_events(query)

    return app


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _list_clusters_for_review(
    repository: Repository, entity_type: ClusterEntityType
) -> list[dict[str, object]]:
    """Cross-event cluster browsing data for `/recognition/{entity_type}`
    -- the actual answer to "review these hundreds of detections grouped
    by identity," since the per-event enrichment panel only ever shows
    one event's crops at a time. Unreviewed clusters (no representative
    chosen yet, and not every observation rejected) sort first, largest
    first within each group -- the ones most worth a reviewer's time.

    Also surfaces any pending merge suggestion touching each cluster
    (`RecognitionService.suggest_face_merges`/
    `suggest_vehicle_appearance_merges` -- this mechanism predates the
    cluster browser, it just was never surfaced in a browsing UI before)
    so a reviewer sees "gaggle thinks this might be cluster X" inline,
    next to the manual merge control.
    """

    db = repository.database
    clusters: (
        list[FaceClusterRow] | list[VehicleAppearanceClusterRow] | list[PersonAppearanceClusterRow]
    )
    list_observations: Callable[
        [list[UUID]],
        list[FaceObservationRow]
        | list[VehicleAppearanceObservationRow]
        | list[PersonAppearanceObservationRow],
    ]
    if entity_type == "face":
        clusters = db.list_face_clusters()
        list_observations = db.list_face_observations_by_cluster_ids
    elif entity_type == "vehicle_appearance":
        clusters = db.list_vehicle_appearance_clusters()
        list_observations = db.list_vehicle_appearance_observations_by_cluster_ids
    else:
        clusters = db.list_person_appearance_clusters()
        list_observations = db.list_person_appearance_observations_by_cluster_ids

    suggestions_by_cluster: dict[str, list[dict[str, object]]] = {}
    for suggestion in db.list_merge_suggestions(entity_type=entity_type, status="pending"):
        entry = {
            "suggestion_id": suggestion.suggestion_id,
            "similarity_score": suggestion.similarity_score,
            "basis": suggestion.basis,
        }
        suggestions_by_cluster.setdefault(suggestion.source_id, []).append(
            {**entry, "other_cluster_id": suggestion.target_id}
        )
        suggestions_by_cluster.setdefault(suggestion.target_id, []).append(
            {**entry, "other_cluster_id": suggestion.source_id}
        )

    items: list[dict[str, object]] = []
    for cluster in clusters:
        if cluster.merged_into:
            continue  # an alias of another cluster -- review the canonical one instead
        observations: list[
            FaceObservationRow | VehicleAppearanceObservationRow | PersonAppearanceObservationRow
        ] = list(list_observations([UUID(cluster.cluster_id)]))
        reviewed = bool(cluster.representative_observation_ids_csv) or (
            bool(observations) and all(o.review_status == "user_rejected" for o in observations)
        )
        items.append(
            {
                "cluster_id": cluster.cluster_id,
                "label": cluster.label,
                "observation_count": len(observations),
                "reviewed": reviewed,
                "representative_observation_ids": (
                    cluster.representative_observation_ids_csv.split(",")
                    if cluster.representative_observation_ids_csv
                    else []
                ),
                "observations": [
                    {
                        "observation_id": o.observation_id,
                        "review_status": o.review_status,
                        "camera_id": o.camera_id,
                        "observed_at": o.observed_at.isoformat(),
                    }
                    for o in sorted(observations, key=lambda o: o.observed_at)
                ],
                "suggestions": suggestions_by_cluster.get(cluster.cluster_id, []),
            }
        )
    items.sort(key=lambda item: (item["reviewed"], -cast(int, item["observation_count"])))
    return items


def _render_index(rows: list[dict[str, object]]) -> str:
    items = "".join(
        f"<tr>"
        f"<td><a href='/events/{html.escape(str(row['event_id']))}'>"
        f"{html.escape(str(row['event_id']))[:8]}</a></td>"
        f"<td class='sev sev-{html.escape(str(row['severity']))}'>"
        f"{html.escape(str(row['severity']))}</td>"
        f"<td>{html.escape(str(row['start_time']))}</td>"
        f"<td>{html.escape(','.join(row['cameras']))}</td>"  # type: ignore[arg-type]
        f"<td>{html.escape(str(row['preservation_state']))}</td>"
        f"<td>{html.escape(str(row['review_decision']))}</td>"
        f"</tr>"
        for row in rows
    )
    return f"""<html><head><title>gaggle review queue</title>{_STYLE}</head>
<body>
<h1>gaggle review queue</h1>
<p>{len(rows)} event(s). Filter with <code>?severity=high</code>,
<code>?camera=front</code>, <code>?status=pending</code>.</p>
<p>Review recognition detections by identity, across every event:
<a href="/recognition/face">faces</a> &middot;
<a href="/recognition/vehicle_appearance">vehicle appearance</a> &middot;
<a href="/recognition/person_appearance">person appearance</a></p>
<table>
<thead><tr><th>Event</th><th>Severity</th><th>Start</th><th>Cameras</th>
<th>Preservation</th><th>Review</th></tr></thead>
<tbody>{items}</tbody>
</table>
</body></html>"""


def _render_recognition_clusters_page(
    entity_type: ClusterEntityType, clusters: list[dict[str, object]]
) -> str:
    label_noun = {"face": "face", "vehicle_appearance": "vehicle", "person_appearance": "person"}[
        entity_type
    ]
    unreviewed_count = sum(1 for c in clusters if not c["reviewed"])
    cards = "".join(_render_cluster_card(entity_type, cluster) for cluster in clusters)
    return f"""<html><head><title>gaggle recognition review -- {label_noun}s</title>
{_STYLE}{_RECOGNITION_STYLE}{_RECOGNITION_SCRIPT}</head>
<body>
<p><a href="/">&larr; back to event queue</a></p>
<h1>Review {label_noun} clusters</h1>
<p>{len(clusters)} cluster(s), {unreviewed_count} not yet reviewed. Pick the crop(s) that best
represent each real {label_noun} and confirm, or reject a cluster/observation that was never a
real {label_noun} at all -- see docs/local-ai.md's "Reviewing and reclaiming recognition
storage" section. Reviewing never deletes anything by itself; a purge sweep does that
afterward.</p>
<p>
<button data-purge-sweep="{entity_type}" data-dry-run="true">Preview purge (dry run)</button>
<button data-purge-sweep="{entity_type}" data-dry-run="false">Run purge sweep</button>
<span id="purge-result"></span>
</p>
<div class="clusters">{cards}</div>
</body></html>"""


def _render_suggestion_item(s: dict[str, object]) -> str:
    suggestion_id = html.escape(cast(str, s["suggestion_id"]))
    other_cluster_id = html.escape(cast(str, s["other_cluster_id"]))[:8]
    similarity = cast(float, s["similarity_score"])
    basis = html.escape(cast(str, s["basis"]))
    return (
        f'<div class="suggestion">might be the same as cluster {other_cluster_id} '
        f"(similarity {similarity:.2f}): {basis}\n"
        f'<button data-confirm-suggestion="{suggestion_id}">confirm merge</button>\n'
        f'<button data-reject-suggestion="{suggestion_id}">not the same</button>\n'
        f"</div>"
    )


def _render_cluster_card(entity_type: ClusterEntityType, cluster: dict[str, object]) -> str:
    cluster_id = cast(str, cluster["cluster_id"])
    label = cast(str | None, cluster["label"])
    reviewed = cast(bool, cluster["reviewed"])
    representative_ids = set(cast(list[str], cluster["representative_observation_ids"]))
    observations = cast(list[dict[str, object]], cluster["observations"])
    suggestions = cast(list[dict[str, object]], cluster["suggestions"])

    crop_items = "".join(
        f"""<div class="crop-item">
<label><input type="checkbox" data-representative-checkbox
value="{html.escape(cast(str, o["observation_id"]))}"
{"checked" if o["observation_id"] in representative_ids else ""}>
<img class="crop" loading="lazy"
src="/api/recognition/{entity_type}/observations/{html.escape(cast(str, o["observation_id"]))}/crop"
alt="{entity_type} crop"></label>
<div class="crop-meta">{html.escape(cast(str, o["camera_id"]))}
&middot; {html.escape(cast(str, o["review_status"]))}</div>
<button data-reject-observation="{html.escape(cast(str, o["observation_id"]))}"
data-entity-type="{entity_type}">not a {("face" if entity_type == "face" else "vehicle")}</button>
<button data-detach-observation="{html.escape(cast(str, o["observation_id"]))}"
data-entity-type="{entity_type}"
title="remove from this cluster entirely, without still counting it toward it">detach</button>
<input type="text" data-move-target-input
placeholder="target cluster id" size="10">
<button data-move-observation="{html.escape(cast(str, o["observation_id"]))}"
data-entity-type="{entity_type}">move</button>
</div>"""
        for o in observations
    )

    suggestion_items = "".join(_render_suggestion_item(s) for s in suggestions)
    label_value = html.escape(label or "")

    return f"""<div class="cluster-card {"reviewed" if reviewed else ""}"
data-cluster-id="{html.escape(cluster_id)}" data-entity-type="{entity_type}">
<h3>{html.escape(cluster_id)[:8]} {f"&mdash; {html.escape(label)}" if label else ""}
<span class="badge">{"reviewed" if reviewed else "needs review"}</span></h3>
<div class="crops">{crop_items}</div>
{suggestion_items}
<div class="cluster-actions">
<input type="text" data-label-input placeholder="label (optional)" value="{label_value}">
<input type="text" data-actor-input placeholder="your name" required>
<button data-confirm-cluster>confirm selected as representative</button>
<button data-reject-cluster>reject entire cluster</button>
</div>
<div class="cluster-result" data-cluster-result></div>
</div>"""


def _render_camera_video(event_id: UUID, artifact_index: int, artifact: ArtifactReference) -> str:
    camera_id = str(artifact.metadata.get("camera_id", "unknown"))
    source_clip_id = html.escape(str(artifact.metadata.get("source_clip_id", "")))
    group_picker = (
        f"<label>split group: <select data-split-group-input data-source-clip-id="
        f"'{source_clip_id}'><option value=''>(not this one)</option>"
        f"<option value='1'>1</option><option value='2'>2</option>"
        f"<option value='3'>3</option><option value='4'>4</option></select></label>"
        if source_clip_id
        else ""
    )
    return (
        f"<div class='camera'><h3>{html.escape(camera_id)}</h3>"
        f"<video data-sync controls muted preload='metadata' "
        f"src='/api/events/{event_id}/media/{artifact_index}'></video>{group_picker}</div>"
    )


def _crop_img_tag(event_id: UUID, entity_type: str, observation_id: str) -> str:
    src = f"/api/events/{event_id}/crop/{entity_type}/{observation_id}"
    return f"<img class='crop' src='{src}' loading='lazy' alt='{html.escape(entity_type)} crop'>"


def _render_reject_observation_button(entity_type: str, observation_id: str, noun: str) -> str:
    return (
        f'<button data-reject-observation-inline="{html.escape(observation_id)}" '
        f'data-entity-type="{entity_type}">not a {noun}</button>'
    )


def _render_face_rows(event_id: UUID, observations: list[FaceObservationRow]) -> str:
    return "".join(
        f"<tr><td>{_crop_img_tag(event_id, 'face', o.observation_id)}</td>"
        f"<td>{html.escape(o.cluster_id or '')}</td>"
        f"<td>{html.escape(o.camera_id)}</td>"
        f"<td>{o.detector_confidence:.2f}</td>"
        f"<td>{html.escape(o.observed_at.isoformat())}</td>"
        f"<td>{html.escape(o.review_status)}</td>"
        f"<td>{_render_reject_observation_button('face', o.observation_id, 'face')}</td></tr>"
        for o in observations
    )


def _render_plate_rows(event_id: UUID, observations: list[PlateObservationRow]) -> str:
    return "".join(
        f"<tr><td>{_crop_img_tag(event_id, 'plate', o.observation_id)}</td>"
        f"<td>{html.escape(o.normalized_text)}</td>"
        f"<td>{html.escape(o.review_status)}</td>"
        f"<td>{o.ocr_confidence:.2f}</td>"
        f"<td>{html.escape(o.camera_id)}</td>"
        f"<td>{html.escape(o.observed_at.isoformat())}</td>"
        f"<td>"
        f'<input type="text" data-plate-text-input="{html.escape(o.observation_id)}" '
        f'value="{html.escape(o.normalized_text)}" size="10">'
        f'<button data-confirm-plate-text-inline="{html.escape(o.observation_id)}">'
        f"confirm/correct</button> "
        f"{_render_reject_observation_button('plate', o.observation_id, 'plate')}"
        f"</td></tr>"
        for o in observations
    )


def _render_voice_rows(observations: list[VoiceObservationRow]) -> str:
    # No crop column -- voice observations have no image (see
    # enrichment/voice.py's module docstring for why).
    return "".join(
        f"<tr><td>{html.escape(o.cluster_id or '')}</td>"
        f"<td>{html.escape(o.camera_id)}</td>"
        f"<td>{o.energy_confidence:.2f}</td>"
        f"<td>{o.segment_start_seconds:.2f}</td><td>{o.segment_end_seconds:.2f}</td>"
        f"<td>{html.escape(o.observed_at.isoformat())}</td></tr>"
        for o in observations
    )


def _render_vehicle_rows(
    event_id: UUID, observations: list[VehicleAppearanceObservationRow]
) -> str:
    return "".join(
        f"<tr><td>{_crop_img_tag(event_id, 'vehicle_appearance', o.observation_id)}</td>"
        f"<td>{html.escape(o.cluster_id or '')}</td>"
        f"<td>{html.escape(o.camera_id)}</td>"
        f"<td>{o.detector_confidence:.2f}</td>"
        f"<td>{html.escape(o.observed_at.isoformat())}</td>"
        f"<td>{html.escape(o.review_status)}</td>"
        f"<td>{
            _render_reject_observation_button('vehicle_appearance', o.observation_id, 'vehicle')
        }</td></tr>"
        for o in observations
    )


def _render_person_rows(event_id: UUID, observations: list[PersonAppearanceObservationRow]) -> str:
    return "".join(
        f"<tr><td>{_crop_img_tag(event_id, 'person_appearance', o.observation_id)}</td>"
        f"<td>{html.escape(o.cluster_id or '')}</td>"
        f"<td>{html.escape(o.camera_id)}</td>"
        f"<td>{o.detector_confidence:.2f}</td>"
        f"<td>{html.escape(o.observed_at.isoformat())}</td>"
        f"<td>{html.escape(o.review_status)}</td>"
        f"<td>{
            _render_reject_observation_button('person_appearance', o.observation_id, 'person')
        }</td></tr>"
        for o in observations
    )


def _render_transcript_section(transcript: dict[str, object] | None) -> str:
    if transcript is None:
        return ""
    segments = transcript.get("segments", [])
    if not isinstance(segments, list):
        segments = []
    segment_rows = "".join(
        f"<tr><td>{float(seg.get('start_offset_seconds', 0.0)):.2f}</td>"
        f"<td>{float(seg.get('end_offset_seconds', 0.0)):.2f}</td>"
        f"<td>{html.escape(str(seg.get('text', '')))}</td>"
        f"<td>{float(seg.get('confidence', 0.0)):.2f}</td></tr>"
        for seg in segments
        if isinstance(seg, dict)
    )
    return f"""
<h2>Transcript</h2>
<table><thead><tr><th>start</th><th>end</th><th>text</th><th>confidence</th></tr>
</thead><tbody>{segment_rows}</tbody></table>
"""


def _render_llm_section(llm_enrichment: dict[str, object] | None) -> str:
    if llm_enrichment is None:
        return ""
    summary = html.escape(str(llm_enrichment.get("summary", "")))
    importance = llm_enrichment.get("importance_score", "n/a")
    return f"""
<h2>Cloud LLM transcript analysis
<span class='disclaimer'>(non-authoritative -- see docs/local-ai.md)</span></h2>
<p>{summary}</p>
<p>importance score: {html.escape(str(importance))}</p>
"""


def _render_event_detail(
    event: EventRecord,
    review_history: list[ReviewAction],
    face_observations: list[FaceObservationRow],
    plate_observations: list[PlateObservationRow],
    voice_observations: list[VoiceObservationRow],
    vehicle_observations: list[VehicleAppearanceObservationRow],
    person_observations: list[PersonAppearanceObservationRow],
    transcript: dict[str, object] | None,
    llm_enrichment: dict[str, object] | None,
) -> str:
    cameras = event.involved_cameras
    video_tags = "".join(
        _render_camera_video(event.event_id, index, artifact)
        for index, artifact in enumerate(event.derived_artifacts)
        if artifact.artifact_type == "derived_clip"
    )
    if not video_tags:
        video_tags = "<p><em>No derived clips were generated for this event.</em></p>"

    signal_rows = "".join(
        f"<tr><td>{html.escape(s.signal_type)}</td><td>{html.escape(str(s.camera_id))}</td>"
        f"<td>{s.confidence:.2f}</td><td>{html.escape(s.timestamp_start.isoformat())}</td></tr>"
        for s in event.signals
    )
    hypothesis_rows = "".join(
        f"<tr><td>{html.escape(h.rule_name)}</td><td>{html.escape(h.label)}</td>"
        f"<td>{h.confidence:.2f}</td><td>{html.escape(h.confidence_math)}</td></tr>"
        for h in event.hypotheses
    )
    history_rows = "".join(
        f"<tr><td>{html.escape(a.timestamp.isoformat())}</td>"
        f"<td>{html.escape(a.action)}</td>"
        f"<td>{html.escape(a.actor)}</td>"
        f"<td>{html.escape(a.notes)}</td></tr>"
        for a in review_history
    )

    enrichment_section = ""
    if (
        face_observations
        or plate_observations
        or voice_observations
        or vehicle_observations
        or person_observations
    ):
        enrichment_section = f"""
<h2>Recognition (local, never identification -- see docs/forensic-considerations.md)</h2>
{
            f'''<h3>Faces ({len(face_observations)})</h3>
<table><thead><tr><th>crop</th><th>cluster</th><th>camera</th><th>confidence</th><th>observed</th>
<th>status</th><th></th></tr>
</thead><tbody>{_render_face_rows(event.event_id, face_observations)}</tbody></table>'''
            if face_observations
            else ""
        }
{
            f'''<h3>Plates ({len(plate_observations)})</h3>
<table><thead><tr><th>crop</th><th>text</th><th>status</th><th>confidence</th><th>camera</th>
<th>observed</th><th></th></tr></thead><tbody>{
                _render_plate_rows(event.event_id, plate_observations)
            }</tbody></table>'''
            if plate_observations
            else ""
        }
{
            f'''<h3>Voices ({len(voice_observations)})</h3>
<table><thead><tr><th>cluster</th><th>camera</th><th>confidence</th><th>start</th><th>end</th>
<th>observed</th></tr></thead><tbody>{_render_voice_rows(voice_observations)}</tbody></table>'''
            if voice_observations
            else ""
        }
{
            f'''<h3>Vehicle appearance ({len(vehicle_observations)})</h3>
<table><thead><tr><th>crop</th><th>cluster</th><th>camera</th><th>confidence</th><th>observed</th>
<th>status</th><th></th></tr>
</thead><tbody>{_render_vehicle_rows(event.event_id, vehicle_observations)}</tbody></table>'''
            if vehicle_observations
            else ""
        }
{
            f'''<h3>Person appearance ({len(person_observations)})</h3>
<table><thead><tr><th>crop</th><th>cluster</th><th>camera</th><th>confidence</th><th>observed</th>
<th>status</th><th></th></tr>
</thead><tbody>{_render_person_rows(event.event_id, person_observations)}</tbody></table>'''
            if person_observations
            else ""
        }
"""

    split_banner = ""
    if event.superseded_by_event_ids:
        split_links = " ".join(
            f"<a href='/events/{sid}'>{html.escape(str(sid))[:8]}</a>"
            for sid in event.superseded_by_event_ids
        )
        split_banner = (
            "<p class='sev sev-high'>This event was split into: "
            f"{split_links} -- everything below is preserved unchanged for the record, "
            "but is no longer the current view of what happened.</p>"
        )

    split_section = ""
    derived_clip_count = sum(
        1 for a in event.derived_artifacts if a.artifact_type == "derived_clip"
    )
    if not event.superseded_by_event_ids and derived_clip_count >= 2:
        split_section = """
<h3>Split this event</h3>
<p>If this event actually bundled clips from two or more unrelated recording
sessions, assign each camera-video above to a split group (using the "split
group" picker under its player), then submit -- the original event is kept
unchanged, and one new independent event is created per group.</p>
<button data-split-event>Split into groups</button>
<div id='split-result'></div>
"""

    return f"""<html><head><title>Event {event.event_id}</title>{_STYLE}{_SCRIPT}
{_RECOGNITION_INLINE_SCRIPT}</head>
<body>
<p><a href="/">&larr; back to queue</a></p>
<h1>Event {event.event_id}</h1>
<p class='sev sev-{event.scoring.severity}'>severity: {event.scoring.severity}
(confidence {event.scoring.confidence:.2f})</p>
{split_banner}
<p>{html.escape(event.evidence_summary)}</p>
<p>cameras: {", ".join(cameras)} | preservation: {event.preservation_status.state}
| review: {event.review_summary.latest_decision}
({event.review_summary.action_count} action(s)) | revision: {event.revision}</p>

<h2>Review</h2>
<form id='review-form'>
<select name='action'>
<option value='accept'>accept</option>
<option value='reject'>reject</option>
<option value='annotate'>annotate</option>
<option value='retag'>retag</option>
<option value='preserve'>preserve</option>
<option value='export'>export</option>
</select>
<input name='actor' placeholder='your name' required>
<input name='notes' placeholder='notes'>
<button type='submit'>submit</button>
</form>
<div id='review-result'></div>

<h2>Synchronized playback</h2>
<div class='cameras'>{video_tags}</div>
{split_section}

<h2>Signals</h2>
<table><thead><tr><th>type</th><th>camera</th><th>confidence</th><th>start</th></tr>
</thead><tbody>{signal_rows}</tbody></table>

<h2>Hypotheses</h2>
<table><thead><tr><th>rule</th><th>label</th><th>confidence</th><th>math</th></tr>
</thead><tbody>{hypothesis_rows}</tbody></table>
{enrichment_section}{_render_transcript_section(transcript)}{_render_llm_section(llm_enrichment)}
<h3>Review history (append-only)</h3>
<table><thead><tr><th>when</th><th>action</th><th>actor</th><th>notes</th></tr>
</thead><tbody>{history_rows}</tbody></table>

<script>
window.EVENT_ID = "{event.event_id}";
</script>
</body></html>"""


_STYLE = """<style>
body { font-family: -apple-system, sans-serif; margin: 2rem; color: #1a1a1a; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }
th { background: #f4f4f4; }
.sev-high { color: #b00020; font-weight: bold; }
.sev-medium { color: #b06000; font-weight: bold; }
.sev-low { color: #555; }
.cameras { display: flex; flex-wrap: wrap; gap: 1rem; }
.camera video { width: 360px; max-width: 100%; background: #000; }
form input, form select, form button { margin-right: 0.5rem; padding: 0.3rem; }
.crop { max-width: 80px; max-height: 80px; display: block; }
.disclaimer { font-weight: normal; font-size: 0.75rem; color: #888; }
</style>"""

_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/config/default-actor')
    .then((r) => r.json())
    .then((data) => {
      if (!data.default_actor) return;
      const actorInput = document.querySelector('#review-form [name="actor"]');
      if (actorInput && !actorInput.value) actorInput.value = data.default_actor;
    })
    .catch(() => {});

  const videos = Array.from(document.querySelectorAll('video[data-sync]'));
  function syncFrom(source) {
    videos.forEach((v) => {
      if (v !== source && Math.abs(v.currentTime - source.currentTime) > 0.25) {
        v.currentTime = source.currentTime;
      }
    });
  }
  videos.forEach((v) => {
    v.addEventListener('play', () => videos.forEach((o) => { if (o !== v) o.play(); }));
    v.addEventListener('pause', () => videos.forEach((o) => { if (o !== v) o.pause(); }));
    v.addEventListener('seeked', () => syncFrom(v));
    v.addEventListener('timeupdate', () => syncFrom(v));
  });

  const form = document.getElementById('review-form');
  if (form) {
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const data = new FormData(form);
      const payload = {
        action: data.get('action'),
        actor: data.get('actor'),
        notes: data.get('notes') || '',
      };
      const response = await fetch(`/api/events/${window.EVENT_ID}/review-actions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const resultBox = document.getElementById('review-result');
      if (response.ok) {
        resultBox.textContent = 'Recorded. Reload to see updated history.';
      } else {
        resultBox.textContent = `Failed: ${response.status}`;
      }
    });
  }

  document.body.addEventListener('click', async (event) => {
    const splitBtn = event.target.closest('[data-split-event]');
    if (!splitBtn) return;
    const groups = {};
    document.querySelectorAll('[data-split-group-input]').forEach((el) => {
      const groupNumber = el.value;
      const clipId = el.dataset.sourceClipId;
      if (!groupNumber || !clipId) return;
      (groups[groupNumber] = groups[groupNumber] || []).push(clipId);
    });
    const clipIdGroups = Object.values(groups);
    const resultEl = document.getElementById('split-result');
    if (clipIdGroups.length < 2) {
      resultEl.textContent = 'assign at least 2 different split groups first';
      return;
    }
    const actorInput = document.querySelector('#review-form [name="actor"]');
    const actor = actorInput ? actorInput.value.trim() : '';
    if (!actor) {
      resultEl.textContent = 'enter your name in the Review actor field first';
      return;
    }
    const response = await fetch(`/api/events/${window.EVENT_ID}/split`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clip_id_groups: clipIdGroups, actor }),
    });
    if (response.ok) {
      resultEl.textContent = 'split -- reloading...';
      setTimeout(() => window.location.reload(), 600);
    } else {
      const detail = await response.json().catch(() => ({}));
      resultEl.textContent = `failed: ${detail.detail || response.status}`;
    }
  });
});
</script>"""

_RECOGNITION_STYLE = """<style>
.clusters { display: flex; flex-direction: column; gap: 1rem; }
.cluster-card { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; }
.cluster-card.reviewed { opacity: 0.6; }
.cluster-card h3 { margin: 0 0 0.5rem; }
.badge { font-size: 0.7rem; font-weight: normal; color: #fff; background: #888;
  border-radius: 4px; padding: 0.1rem 0.4rem; margin-left: 0.5rem; }
.reviewed .badge { background: #2e7d32; }
.crops { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 0.5rem; }
.crop-item { text-align: center; font-size: 0.75rem; }
.crop-meta { color: #666; }
.suggestion { background: #fff8e1; border: 1px solid #ffe082; border-radius: 4px;
  padding: 0.4rem 0.6rem; margin: 0.3rem 0; font-size: 0.85rem; }
.cluster-actions { display: flex; gap: 0.5rem; align-items: center; margin-top: 0.5rem; }
.cluster-result { font-size: 0.8rem; color: #666; margin-top: 0.3rem; }
</style>"""

_RECOGNITION_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded', () => {
  fetch('/api/config/default-actor')
    .then((r) => r.json())
    .then((data) => {
      if (!data.default_actor) return;
      document.querySelectorAll('[data-actor-input]').forEach((el) => {
        if (!el.value) el.value = data.default_actor;
      });
    })
    .catch(() => {});

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return response;
  }

  function cardFor(el) {
    return el.closest('.cluster-card');
  }

  function actorAndLabel(card) {
    const actor = card.querySelector('[data-actor-input]').value.trim();
    const label = card.querySelector('[data-label-input]').value.trim();
    return { actor, label: label || null };
  }

  document.body.addEventListener('click', async (event) => {
    const confirmBtn = event.target.closest('[data-confirm-cluster]');
    const rejectClusterBtn = event.target.closest('[data-reject-cluster]');
    const rejectObsBtn = event.target.closest('[data-reject-observation]');
    const detachObsBtn = event.target.closest('[data-detach-observation]');
    const moveObsBtn = event.target.closest('[data-move-observation]');
    const confirmSuggBtn = event.target.closest('[data-confirm-suggestion]');
    const rejectSuggBtn = event.target.closest('[data-reject-suggestion]');
    const purgeBtn = event.target.closest('[data-purge-sweep]');

    if (confirmBtn) {
      const card = cardFor(confirmBtn);
      const entityType = card.dataset.entityType;
      const clusterId = card.dataset.clusterId;
      const { actor, label } = actorAndLabel(card);
      const representative = Array.from(
        card.querySelectorAll('[data-representative-checkbox]:checked')
      ).map((el) => el.value);
      if (!actor || representative.length === 0) {
        card.querySelector('[data-cluster-result]').textContent =
          'pick at least one representative crop and enter your name first';
        return;
      }
      const response = await postJson(
        `/api/recognition/${entityType}/clusters/${clusterId}/confirm`,
        { representative_observation_ids: representative, actor, label }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? 'confirmed -- reloading...' : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (rejectClusterBtn) {
      const card = cardFor(rejectClusterBtn);
      const entityType = card.dataset.entityType;
      const clusterId = card.dataset.clusterId;
      const { actor } = actorAndLabel(card);
      if (!actor) {
        card.querySelector('[data-cluster-result]').textContent = 'enter your name first';
        return;
      }
      const response = await postJson(
        `/api/recognition/${entityType}/clusters/${clusterId}/reject`, { actor }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? 'rejected -- reloading...' : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (rejectObsBtn) {
      const card = cardFor(rejectObsBtn);
      const entityType = rejectObsBtn.dataset.entityType;
      const observationId = rejectObsBtn.dataset.rejectObservation;
      const { actor } = actorAndLabel(card);
      if (!actor) {
        card.querySelector('[data-cluster-result]').textContent = 'enter your name first';
        return;
      }
      const response = await postJson(
        `/api/recognition/${entityType}/observations/${observationId}/reject`, { actor }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? 'observation rejected -- reloading...' : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (detachObsBtn) {
      const card = cardFor(detachObsBtn);
      const entityType = detachObsBtn.dataset.entityType;
      const observationId = detachObsBtn.dataset.detachObservation;
      const { actor } = actorAndLabel(card);
      if (!actor) {
        card.querySelector('[data-cluster-result]').textContent = 'enter your name first';
        return;
      }
      const response = await postJson(
        `/api/recognition/${entityType}/observations/${observationId}/detach`, { actor }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? 'detached -- reloading...' : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (moveObsBtn) {
      const card = cardFor(moveObsBtn);
      const entityType = moveObsBtn.dataset.entityType;
      const observationId = moveObsBtn.dataset.moveObservation;
      const targetInput = moveObsBtn.closest('.crop-item').querySelector(
        '[data-move-target-input]'
      );
      const targetClusterId = targetInput ? targetInput.value.trim() : '';
      const { actor } = actorAndLabel(card);
      if (!actor || !targetClusterId) {
        card.querySelector('[data-cluster-result]').textContent =
          'enter your name and a target cluster id first';
        return;
      }
      const response = await postJson(
        `/api/recognition/${entityType}/observations/${observationId}/move`,
        { target_cluster_id: targetClusterId, actor }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? 'moved -- reloading...' : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (confirmSuggBtn || rejectSuggBtn) {
      const card = cardFor(confirmSuggBtn || rejectSuggBtn);
      const { actor } = actorAndLabel(card);
      if (!actor) {
        card.querySelector('[data-cluster-result]').textContent = 'enter your name first';
        return;
      }
      const suggestionId = confirmSuggBtn
        ? confirmSuggBtn.dataset.confirmSuggestion
        : rejectSuggBtn.dataset.rejectSuggestion;
      const action = confirmSuggBtn ? 'confirm' : 'reject';
      const response = await postJson(
        `/api/recognition/merge-suggestions/${suggestionId}/${action}`, { actor }
      );
      card.querySelector('[data-cluster-result]').textContent =
        response.ok ? `suggestion ${action}ed -- reloading...` : `failed: ${response.status}`;
      if (response.ok) setTimeout(() => window.location.reload(), 600);
    } else if (purgeBtn) {
      const entityType = purgeBtn.dataset.purgeSweep;
      const dryRun = purgeBtn.dataset.dryRun === 'true';
      const actor = window.prompt('your name, to log this purge sweep:');
      if (!actor) return;
      const response = await postJson(
        `/api/recognition/${entityType}/purge-reviewed`, { actor, dry_run: dryRun }
      );
      const resultEl = document.getElementById('purge-result');
      if (response.ok) {
        const record = await response.json();
        const count = record.purged_observation_ids.length;
        resultEl.textContent = dryRun
          ? ` would purge ${count} crop(s)`
          : ` purged ${count} crop(s) -- reloading...`;
        if (!dryRun) setTimeout(() => window.location.reload(), 600);
      } else {
        resultEl.textContent = ` failed: ${response.status}`;
      }
    }
  });
});
</script>"""

_RECOGNITION_INLINE_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded', () => {
  let defaultActor = null;
  fetch('/api/config/default-actor')
    .then((r) => r.json())
    .then((data) => { defaultActor = data.default_actor; })
    .catch(() => {});

  document.body.addEventListener('click', async (event) => {
    const rejectBtn = event.target.closest('[data-reject-observation-inline]');
    const confirmPlateBtn = event.target.closest('[data-confirm-plate-text-inline]');
    if (!rejectBtn && !confirmPlateBtn) return;

    if (rejectBtn) {
      const actor = window.prompt('your name, to log this rejection:', defaultActor || '');
      if (!actor) return;
      const entityType = rejectBtn.dataset.entityType;
      const observationId = rejectBtn.dataset.rejectObservationInline;
      const response = await fetch(
        `/api/recognition/${entityType}/observations/${observationId}/reject`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actor }),
        }
      );
      if (response.ok) {
        window.location.reload();
      } else {
        window.alert(`failed: ${response.status}`);
      }
      return;
    }

    const observationId = confirmPlateBtn.dataset.confirmPlateTextInline;
    const textInput = document.querySelector(
      `[data-plate-text-input="${observationId}"]`
    );
    const correctedText = textInput ? textInput.value.trim() : '';
    if (!correctedText) {
      window.alert('enter the plate text first');
      return;
    }
    const actor = window.prompt('your name, to log this confirmation:', defaultActor || '');
    if (!actor) return;
    const response = await fetch(
      `/api/recognition/plate/observations/${observationId}/confirm`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corrected_text: correctedText, actor }),
      }
    );
    if (response.ok) {
      window.location.reload();
    } else {
      window.alert(`failed: ${response.status}`);
    }
  });
});
</script>"""
