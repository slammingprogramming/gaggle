"""Identity linking and search for recognized faces and license plates.

See `docs/local-ai.md`'s "Linking sightings to the same person or vehicle"
section for the full picture; in short:

* A `FaceCluster`/`PlateRecord` is never deleted or rewritten when merged.
  `merged_into` just marks it as an alias of another cluster/record.
  `resolve_face_identity`/`resolve_plate_identity` follow that chain to the
  canonical root, and `get_face_identity`/`get_plate_identity` aggregate
  stats and sightings across every cluster/record that resolves to the
  same root, computed fresh on every read -- nothing about the original
  per-cluster data is ever rewritten or lost, exactly the same
  append-only-in-spirit approach used for event revisions.
* Every merge is a human decision, permanently logged to the append-only
  `identity_merge_log.jsonl`: who declared two things the same, and when.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

import cv2

from gaggle.enrichment.face import IncrementalFaceClusterer
from gaggle.enrichment.face_auraface import IncrementalFaceEmbeddingClusterer
from gaggle.enrichment.person_appearance import IncrementalPersonAppearanceClusterer
from gaggle.enrichment.vehicle_appearance import IncrementalVehicleAppearanceClusterer
from gaggle.enrichment.voice import IncrementalVoiceClusterer
from gaggle.schemas.recognition import IdentityMergeRecord, MergeSuggestion
from gaggle.schemas.recognition_review import RecognitionCropPurgeRecord, RecognitionReviewRecord
from gaggle.storage.database import (
    FaceClusterRow,
    FaceObservationRow,
    MergeSuggestionRow,
    PersonAppearanceClusterRow,
    PersonAppearanceObservationRow,
    PlateObservationRow,
    PlateRecordRow,
    VehicleAppearanceClusterRow,
    VehicleAppearanceObservationRow,
    VoiceClusterRow,
    VoiceObservationRow,
)
from gaggle.storage.repository import Repository
from gaggle.utils.hashing import hash_file
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

ClusterEntityType = Literal["face", "vehicle_appearance", "person_appearance"]
CropEntityType = Literal["face", "plate", "vehicle_appearance", "person_appearance"]

LOGGER = get_logger(__name__)
# Cycle/runaway-chain guard, belt-and-suspenders on top of _merge's own check.
_MAX_MERGE_CHAIN_LENGTH = 64


class MergeError(ValueError):
    """Raised for invalid merge requests (self-merge, unknown id, cycle)."""


class ReviewError(ValueError):
    """Raised for invalid recognition-review requests (unknown cluster/
    observation, a representative id that doesn't belong to the cluster
    being confirmed)."""


@dataclass(frozen=True, slots=True)
class FaceIdentity:
    identity_id: UUID  # the canonical (root) cluster_id
    member_cluster_ids: list[UUID]
    label: str | None
    observation_count: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None
    representative_crop_paths: list[str]


@dataclass(frozen=True, slots=True)
class PlateIdentity:
    identity_id: UUID  # the canonical (root) plate_id
    normalized_texts: list[str]  # every distinct plate text folded into this identity
    member_plate_ids: list[UUID]
    label: str | None
    observation_count: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None


@dataclass(frozen=True, slots=True)
class SearchResult:
    exact_matches: list[object]
    fuzzy_suggestions: list[object]


@dataclass(frozen=True, slots=True)
class CleanupResult:
    suppressed_observation_ids: list[UUID]
    kept_observation_ids: list[UUID]
    clusters_with_duplicates: int


@dataclass(frozen=True, slots=True)
class _ReviewHandlers:
    """Bundles the per-entity-type storage calls `confirm_identity`/
    `reject_cluster`/`reject_observation`/`purge_reviewed_crops` need,
    the same parameterize-with-callables approach `_merge`/`_resolve`
    already use to share one implementation across face/plate/voice/
    vehicle-appearance -- see that section below."""

    get_observation: Callable[
        [UUID],
        FaceObservationRow
        | VehicleAppearanceObservationRow
        | PersonAppearanceObservationRow
        | None,
    ]
    list_observations_by_cluster: Callable[
        [list[UUID]],
        list[FaceObservationRow]
        | list[VehicleAppearanceObservationRow]
        | list[PersonAppearanceObservationRow],
    ]
    set_cluster_representative: Callable[[UUID, list[UUID], list[str]], None]
    set_cluster_label: Callable[[UUID, str], None]
    set_observation_review_status: Callable[[UUID, str], None]
    mark_observation_crop_purged: Callable[[UUID], None]
    list_observations_eligible_for_purge: Callable[
        [],
        list[FaceObservationRow]
        | list[VehicleAppearanceObservationRow]
        | list[PersonAppearanceObservationRow],
    ]
    get_cluster: Callable[
        [UUID], FaceClusterRow | VehicleAppearanceClusterRow | PersonAppearanceClusterRow | None
    ]
    set_observation_cluster: Callable[[UUID, UUID | None], None]
    set_cluster_observation_count: Callable[[UUID, int], None]


class RecognitionService:
    def __init__(self, repository: Repository) -> None:
        self.repository = repository

    # -- faces --------------------------------------------------------------

    def resolve_face_identity(self, cluster_id: UUID) -> UUID:
        return self._resolve(cluster_id, self.repository.database.get_face_cluster)

    def merge_faces(self, source_id: UUID, target_id: UUID, actor: str, notes: str = "") -> None:
        self._merge(
            source_id,
            target_id,
            actor,
            notes,
            entity_type="face",
            get_row=self.repository.database.get_face_cluster,
            set_merge=self.repository.database.set_face_cluster_merge,
            resolve=self.resolve_face_identity,
        )

    def get_face_identity(self, cluster_id: UUID) -> FaceIdentity:
        root = self.resolve_face_identity(cluster_id)
        members = [
            row
            for row in self.repository.database.list_face_clusters()
            if self.resolve_face_identity(UUID(row.cluster_id)) == root
        ]
        if not members:
            raise MergeError(f"no such face cluster: {cluster_id}")
        label = next((m.label for m in members if m.label), None)
        crops: list[str] = []
        for member in members:
            if member.representative_crops_csv:
                crops.extend(member.representative_crops_csv.split(","))
        first_seen = min((m.first_seen_at for m in members if m.first_seen_at), default=None)
        last_seen = max((m.last_seen_at for m in members if m.last_seen_at), default=None)
        return FaceIdentity(
            identity_id=root,
            member_cluster_ids=sorted(UUID(m.cluster_id) for m in members),
            label=label,
            observation_count=sum(m.observation_count for m in members),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            representative_crop_paths=crops[-4:],
        )

    def list_face_sightings(
        self, cluster_id: UUID, follow_merges: bool = True, include_duplicates: bool = True
    ) -> list[FaceObservationRow]:
        if not follow_merges:
            return self.repository.database.list_face_observations(cluster_id, include_duplicates)
        root = self.resolve_face_identity(cluster_id)
        member_ids = [
            UUID(row.cluster_id)
            for row in self.repository.database.list_face_clusters()
            if self.resolve_face_identity(UUID(row.cluster_id)) == root
        ]
        sightings: list[FaceObservationRow] = []
        for member_id in member_ids:
            sightings.extend(
                self.repository.database.list_face_observations(member_id, include_duplicates)
            )
        sightings.sort(key=lambda o: o.observed_at)
        return sightings

    def search_faces(self, query: str, fuzzy_cutoff: float = 0.7) -> SearchResult:
        exact = self.repository.database.search_face_clusters(query)
        if exact:
            return SearchResult(exact_matches=list(exact), fuzzy_suggestions=[])
        all_clusters = self.repository.database.list_face_clusters()
        candidates: dict[str, FaceClusterRow] = {row.cluster_id: row for row in all_clusters}
        for row in all_clusters:
            if row.label:
                candidates[row.label] = row
        close = difflib.get_close_matches(query, list(candidates.keys()), n=5, cutoff=fuzzy_cutoff)
        return SearchResult(exact_matches=[], fuzzy_suggestions=[candidates[c] for c in close])

    # -- plates ---------------------------------------------------------------

    def resolve_plate_identity(self, plate_id: UUID) -> UUID:
        return self._resolve(plate_id, self.repository.database.get_plate_record)

    def merge_plates(self, source_id: UUID, target_id: UUID, actor: str, notes: str = "") -> None:
        self._merge(
            source_id,
            target_id,
            actor,
            notes,
            entity_type="plate",
            get_row=self.repository.database.get_plate_record,
            set_merge=self.repository.database.set_plate_record_merge,
            resolve=self.resolve_plate_identity,
        )

    def confirm_plate_observation(
        self,
        observation_id: UUID,
        corrected_text: str,
        actor: str,
        notes: str = "",
        purge: bool = False,
    ) -> RecognitionReviewRecord:
        """Confirm/correct a plate OCR reading after a human has viewed
        its crop. Unlike face/vehicle-appearance, a plate observation has
        no cluster/representative-crop concept -- each sighting is
        reviewed and its crop becomes purge-eligible individually."""

        if self.repository.database.get_plate_observation(observation_id) is None:
            raise ReviewError(f"no such plate observation: {observation_id}")
        self.repository.database.confirm_plate_observation(observation_id, corrected_text.upper())
        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="confirmed",
            entity_type="plate",
            cluster_id=None,
            observation_ids=[observation_id],
            label=corrected_text.upper(),
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_plate_observation_confirmed",
            observation_id=str(observation_id),
            corrected_text=corrected_text.upper(),
            actor=actor,
        )
        if purge:
            self.purge_reviewed_crops("plate", actor, notes=notes)
        return record

    def reject_plate_observation(
        self, observation_id: UUID, actor: str, notes: str = "", purge: bool = False
    ) -> RecognitionReviewRecord:
        """Mark a plate observation not a real (or not a usable) plate
        reading."""

        if self.repository.database.get_plate_observation(observation_id) is None:
            raise ReviewError(f"no such plate observation: {observation_id}")
        self.repository.database.mark_plate_observation_rejected(observation_id)
        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="rejected",
            entity_type="plate",
            cluster_id=None,
            observation_ids=[observation_id],
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_plate_observation_rejected",
            observation_id=str(observation_id),
            actor=actor,
        )
        if purge:
            self.purge_reviewed_crops("plate", actor, notes=notes)
        return record

    def get_plate_identity(self, plate_id: UUID) -> PlateIdentity:
        root = self.resolve_plate_identity(plate_id)
        members = [
            row
            for row in self.repository.database.list_plate_records()
            if self.resolve_plate_identity(UUID(row.plate_id)) == root
        ]
        if not members:
            raise MergeError(f"no such plate record: {plate_id}")
        label = next((m.label for m in members if m.label), None)
        first_seen = min((m.first_seen_at for m in members if m.first_seen_at), default=None)
        last_seen = max((m.last_seen_at for m in members if m.last_seen_at), default=None)
        return PlateIdentity(
            identity_id=root,
            normalized_texts=sorted({m.normalized_text for m in members}),
            member_plate_ids=sorted(UUID(m.plate_id) for m in members),
            label=label,
            observation_count=sum(m.observation_count for m in members),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
        )

    def list_plate_sightings(
        self, plate_id_or_text: str, follow_merges: bool = True
    ) -> list[PlateObservationRow]:
        record = self.resolve_plate_input(plate_id_or_text)
        if record is None:
            return []
        if not follow_merges:
            return self.repository.database.list_plate_observations(
                normalized_text=record.normalized_text
            )
        root = self.resolve_plate_identity(UUID(record.plate_id))
        members = [
            row
            for row in self.repository.database.list_plate_records()
            if self.resolve_plate_identity(UUID(row.plate_id)) == root
        ]
        sightings: list[PlateObservationRow] = []
        for member in members:
            sightings.extend(
                self.repository.database.list_plate_observations(
                    normalized_text=member.normalized_text
                )
            )
        sightings.sort(key=lambda o: o.observed_at)
        return sightings

    def search_plates(self, query: str, fuzzy_cutoff: float = 0.7) -> SearchResult:
        exact = self.repository.database.search_plate_records(query)
        if exact:
            return SearchResult(exact_matches=list(exact), fuzzy_suggestions=[])
        all_records = self.repository.database.list_plate_records()
        candidates: dict[str, PlateRecordRow] = {row.normalized_text: row for row in all_records}
        close = difflib.get_close_matches(
            query.upper(), list(candidates.keys()), n=5, cutoff=fuzzy_cutoff
        )
        return SearchResult(exact_matches=[], fuzzy_suggestions=[candidates[c] for c in close])

    # -- voices -----------------------------------------------------------

    def resolve_voice_identity(self, cluster_id: UUID) -> UUID:
        return self._resolve(cluster_id, self.repository.database.get_voice_cluster)

    def merge_voices(self, source_id: UUID, target_id: UUID, actor: str, notes: str = "") -> None:
        self._merge(
            source_id,
            target_id,
            actor,
            notes,
            entity_type="voice",
            get_row=self.repository.database.get_voice_cluster,
            set_merge=self.repository.database.set_voice_cluster_merge,
            resolve=self.resolve_voice_identity,
        )

    def get_voice_identity(self, cluster_id: UUID) -> FaceIdentity:
        """Returns a `FaceIdentity` (the shape is identical: identity id,
        member cluster ids, label, counts, first/last seen) -- voices don't
        have representative crop images the way faces do, so
        `representative_crop_paths` is always empty for a voice identity."""

        root = self.resolve_voice_identity(cluster_id)
        members = [
            row
            for row in self.repository.database.list_voice_clusters()
            if self.resolve_voice_identity(UUID(row.cluster_id)) == root
        ]
        if not members:
            raise MergeError(f"no such voice cluster: {cluster_id}")
        label = next((m.label for m in members if m.label), None)
        first_seen = min((m.first_seen_at for m in members if m.first_seen_at), default=None)
        last_seen = max((m.last_seen_at for m in members if m.last_seen_at), default=None)
        return FaceIdentity(
            identity_id=root,
            member_cluster_ids=sorted(UUID(m.cluster_id) for m in members),
            label=label,
            observation_count=sum(m.observation_count for m in members),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            representative_crop_paths=[],
        )

    def list_voice_sightings(
        self, cluster_id: UUID, follow_merges: bool = True, include_duplicates: bool = True
    ) -> list[VoiceObservationRow]:
        if not follow_merges:
            return self.repository.database.list_voice_observations(cluster_id, include_duplicates)
        root = self.resolve_voice_identity(cluster_id)
        member_ids = [
            UUID(row.cluster_id)
            for row in self.repository.database.list_voice_clusters()
            if self.resolve_voice_identity(UUID(row.cluster_id)) == root
        ]
        sightings: list[VoiceObservationRow] = []
        for member_id in member_ids:
            sightings.extend(
                self.repository.database.list_voice_observations(member_id, include_duplicates)
            )
        sightings.sort(key=lambda o: o.observed_at)
        return sightings

    def search_voices(self, query: str, fuzzy_cutoff: float = 0.7) -> SearchResult:
        exact = self.repository.database.search_voice_clusters(query)
        if exact:
            return SearchResult(exact_matches=list(exact), fuzzy_suggestions=[])
        all_clusters = self.repository.database.list_voice_clusters()
        candidates: dict[str, VoiceClusterRow] = {row.cluster_id: row for row in all_clusters}
        for row in all_clusters:
            if row.label:
                candidates[row.label] = row
        close = difflib.get_close_matches(query, list(candidates.keys()), n=5, cutoff=fuzzy_cutoff)
        return SearchResult(exact_matches=[], fuzzy_suggestions=[candidates[c] for c in close])

    def cleanup_duplicate_voice_observations(self, window_seconds: float = 5.0) -> CleanupResult:
        """Collapse near-duplicate voice observations within the same event
        and cluster -- the same speaker talking continuously can produce
        several adjacent VAD segments. Mirrors
        `cleanup_duplicate_face_observations` exactly; see that method's
        docstring."""

        actionable = [
            observation
            for observation in self.repository.database.list_all_voice_observations()
            if observation.duplicate_of_observation_id is None
        ]
        groups: dict[tuple[str | None, str | None], list[VoiceObservationRow]] = {}
        for observation in actionable:
            key = (observation.event_id, observation.cluster_id)
            groups.setdefault(key, []).append(observation)

        suppressed: list[UUID] = []
        kept: list[UUID] = []
        groups_with_duplicates = 0
        for observations in groups.values():
            observations.sort(key=lambda o: o.observed_at)
            for cluster in _cluster_by_time(observations, window_seconds):
                if len(cluster) < 2:
                    kept.append(UUID(cluster[0].observation_id))
                    continue
                groups_with_duplicates += 1
                best = max(cluster, key=lambda o: o.energy_confidence)
                best_id = UUID(best.observation_id)
                kept.append(best_id)
                for observation in cluster:
                    if observation.observation_id == best.observation_id:
                        continue
                    self.repository.database.mark_voice_observation_duplicate(
                        UUID(observation.observation_id), duplicate_of=best_id
                    )
                    suppressed.append(UUID(observation.observation_id))

        LOGGER.info(
            "voice_duplicate_cleanup_completed",
            suppressed_count=len(suppressed),
            kept_count=len(kept),
            clusters_with_duplicates=groups_with_duplicates,
        )
        return CleanupResult(
            suppressed_observation_ids=suppressed,
            kept_observation_ids=kept,
            clusters_with_duplicates=groups_with_duplicates,
        )

    def suggest_voice_merges(
        self,
        cluster_distance_threshold: float = 0.05,
        suggestion_multiplier: float = 1.6,
    ) -> list[MergeSuggestion]:
        """Flag pairs of voice clusters that might be the same speaker, for
        human review -- never merged automatically. Mirrors
        `suggest_face_merges`, but compares cluster centroids directly
        (voiceprints are already fixed-length vectors, no representative
        crop image needed) via
        `IncrementalVoiceClusterer.predict_nearest_cluster`.
        """

        clusters = [
            cluster
            for cluster in self.repository.database.list_voice_clusters()
            if not cluster.merged_into
        ]
        if len(clusters) < 2:
            return []

        clusterer = IncrementalVoiceClusterer(
            self.repository.workspace.voice_model_path,
            distance_threshold=cluster_distance_threshold,
        )
        suggestion_ceiling = cluster_distance_threshold * suggestion_multiplier
        existing_pending = {
            frozenset((row.source_id, row.target_id))
            for row in self.repository.database.list_merge_suggestions(
                entity_type="voice", status="pending"
            )
        }

        created: list[MergeSuggestion] = []
        for cluster in clusters:
            centroid = clusterer.get_cluster_centroid(cluster.cluster_id)
            if centroid is None:
                continue
            matched_cluster_id, distance = clusterer.predict_nearest_cluster(
                centroid, exclude_cluster_id=cluster.cluster_id
            )
            if matched_cluster_id is None:
                continue
            if not (cluster_distance_threshold < distance <= suggestion_ceiling):
                continue
            pair_key = frozenset((cluster.cluster_id, matched_cluster_id))
            if pair_key in existing_pending:
                continue
            span = suggestion_ceiling - cluster_distance_threshold
            similarity = 1.0 - (distance - cluster_distance_threshold) / span if span else 0.0
            suggestion = MergeSuggestion(
                suggestion_id=new_uuid(),
                entity_type="voice",
                source_id=UUID(cluster.cluster_id),
                target_id=UUID(matched_cluster_id),
                similarity_score=round(max(0.0, min(1.0, similarity)), 6),
                basis=(
                    f"voiceprint cosine distance {distance:.4f} to cluster "
                    f"{matched_cluster_id[:8]} (auto-merge threshold "
                    f"{cluster_distance_threshold:.4f})"
                ),
                created_at=utc_now(),
            )
            self.repository.database.insert_merge_suggestion(suggestion)
            created.append(suggestion)
            existing_pending.add(pair_key)

        LOGGER.info("voice_merge_suggestions_generated", count=len(created))
        return created

    # -- vehicle appearance -------------------------------------------------

    def resolve_vehicle_appearance_identity(self, cluster_id: UUID) -> UUID:
        return self._resolve(cluster_id, self.repository.database.get_vehicle_appearance_cluster)

    def merge_vehicle_appearances(
        self, source_id: UUID, target_id: UUID, actor: str, notes: str = ""
    ) -> None:
        self._merge(
            source_id,
            target_id,
            actor,
            notes,
            entity_type="vehicle_appearance",
            get_row=self.repository.database.get_vehicle_appearance_cluster,
            set_merge=self.repository.database.set_vehicle_appearance_cluster_merge,
            resolve=self.resolve_vehicle_appearance_identity,
        )

    def get_vehicle_appearance_identity(self, cluster_id: UUID) -> FaceIdentity:
        """Returns a `FaceIdentity` (the shape is identical: identity id,
        member cluster ids, label, counts, first/last seen, representative
        crops -- vehicle-appearance observations do have crop images, like
        faces, unlike voice)."""

        root = self.resolve_vehicle_appearance_identity(cluster_id)
        members = [
            row
            for row in self.repository.database.list_vehicle_appearance_clusters()
            if self.resolve_vehicle_appearance_identity(UUID(row.cluster_id)) == root
        ]
        if not members:
            raise MergeError(f"no such vehicle appearance cluster: {cluster_id}")
        label = next((m.label for m in members if m.label), None)
        crops: list[str] = []
        for member in members:
            if member.representative_crops_csv:
                crops.extend(member.representative_crops_csv.split(","))
        first_seen = min((m.first_seen_at for m in members if m.first_seen_at), default=None)
        last_seen = max((m.last_seen_at for m in members if m.last_seen_at), default=None)
        return FaceIdentity(
            identity_id=root,
            member_cluster_ids=sorted(UUID(m.cluster_id) for m in members),
            label=label,
            observation_count=sum(m.observation_count for m in members),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            representative_crop_paths=crops[-4:],
        )

    def list_vehicle_appearance_sightings(
        self, cluster_id: UUID, follow_merges: bool = True, include_duplicates: bool = True
    ) -> list[VehicleAppearanceObservationRow]:
        if not follow_merges:
            return self.repository.database.list_vehicle_appearance_observations(
                cluster_id, include_duplicates
            )
        root = self.resolve_vehicle_appearance_identity(cluster_id)
        member_ids = [
            UUID(row.cluster_id)
            for row in self.repository.database.list_vehicle_appearance_clusters()
            if self.resolve_vehicle_appearance_identity(UUID(row.cluster_id)) == root
        ]
        sightings: list[VehicleAppearanceObservationRow] = []
        for member_id in member_ids:
            sightings.extend(
                self.repository.database.list_vehicle_appearance_observations(
                    member_id, include_duplicates
                )
            )
        sightings.sort(key=lambda o: o.observed_at)
        return sightings

    def search_vehicle_appearances(self, query: str, fuzzy_cutoff: float = 0.7) -> SearchResult:
        exact = self.repository.database.search_vehicle_appearance_clusters(query)
        if exact:
            return SearchResult(exact_matches=list(exact), fuzzy_suggestions=[])
        all_clusters = self.repository.database.list_vehicle_appearance_clusters()
        candidates: dict[str, VehicleAppearanceClusterRow] = {
            row.cluster_id: row for row in all_clusters
        }
        for row in all_clusters:
            if row.label:
                candidates[row.label] = row
        close = difflib.get_close_matches(query, list(candidates.keys()), n=5, cutoff=fuzzy_cutoff)
        return SearchResult(exact_matches=[], fuzzy_suggestions=[candidates[c] for c in close])

    def cleanup_duplicate_vehicle_appearance_observations(
        self, window_seconds: float = 5.0
    ) -> CleanupResult:
        """Collapse near-duplicate vehicle-appearance observations within
        the same event and cluster -- the same parked/passing vehicle
        sampled across many frames produces one observation per frame.
        Mirrors `cleanup_duplicate_face_observations`/
        `cleanup_duplicate_voice_observations` exactly; see either's
        docstring."""

        actionable = [
            observation
            for observation in self.repository.database.list_all_vehicle_appearance_observations()
            if observation.duplicate_of_observation_id is None
        ]
        groups: dict[tuple[str | None, str | None], list[VehicleAppearanceObservationRow]] = {}
        for observation in actionable:
            key = (observation.event_id, observation.cluster_id)
            groups.setdefault(key, []).append(observation)

        suppressed: list[UUID] = []
        kept: list[UUID] = []
        groups_with_duplicates = 0
        for observations in groups.values():
            observations.sort(key=lambda o: o.observed_at)
            for cluster in _cluster_by_time(observations, window_seconds):
                if len(cluster) < 2:
                    kept.append(UUID(cluster[0].observation_id))
                    continue
                groups_with_duplicates += 1
                best = max(cluster, key=lambda o: o.detector_confidence)
                best_id = UUID(best.observation_id)
                kept.append(best_id)
                for observation in cluster:
                    if observation.observation_id == best.observation_id:
                        continue
                    self.repository.database.mark_vehicle_appearance_observation_duplicate(
                        UUID(observation.observation_id), duplicate_of=best_id
                    )
                    suppressed.append(UUID(observation.observation_id))

        LOGGER.info(
            "vehicle_appearance_duplicate_cleanup_completed",
            suppressed_count=len(suppressed),
            kept_count=len(kept),
            clusters_with_duplicates=groups_with_duplicates,
        )
        return CleanupResult(
            suppressed_observation_ids=suppressed,
            kept_observation_ids=kept,
            clusters_with_duplicates=groups_with_duplicates,
        )

    def suggest_vehicle_appearance_merges(
        self,
        cluster_distance_threshold: float = 0.10,
        suggestion_multiplier: float = 1.6,
    ) -> list[MergeSuggestion]:
        """Flag pairs of vehicle-appearance clusters that might be the same
        vehicle, for human review -- never merged automatically. Mirrors
        `suggest_voice_merges`: compares cluster centroids directly via
        `IncrementalVehicleAppearanceClusterer.predict_nearest_cluster`.
        """

        clusters = [
            cluster
            for cluster in self.repository.database.list_vehicle_appearance_clusters()
            if not cluster.merged_into
        ]
        if len(clusters) < 2:
            return []

        clusterer = IncrementalVehicleAppearanceClusterer(
            self.repository.workspace.vehicle_appearance_model_path,
            distance_threshold=cluster_distance_threshold,
        )
        suggestion_ceiling = cluster_distance_threshold * suggestion_multiplier
        existing_pending = {
            frozenset((row.source_id, row.target_id))
            for row in self.repository.database.list_merge_suggestions(
                entity_type="vehicle_appearance", status="pending"
            )
        }

        created: list[MergeSuggestion] = []
        for cluster in clusters:
            centroid = clusterer.get_cluster_centroid(cluster.cluster_id)
            if centroid is None:
                continue
            matched_cluster_id, distance = clusterer.predict_nearest_cluster(
                centroid, exclude_cluster_id=cluster.cluster_id
            )
            if matched_cluster_id is None:
                continue
            if not (cluster_distance_threshold < distance <= suggestion_ceiling):
                continue
            pair_key = frozenset((cluster.cluster_id, matched_cluster_id))
            if pair_key in existing_pending:
                continue
            span = suggestion_ceiling - cluster_distance_threshold
            similarity = 1.0 - (distance - cluster_distance_threshold) / span if span else 0.0
            suggestion = MergeSuggestion(
                suggestion_id=new_uuid(),
                entity_type="vehicle_appearance",
                source_id=UUID(cluster.cluster_id),
                target_id=UUID(matched_cluster_id),
                similarity_score=round(max(0.0, min(1.0, similarity)), 6),
                basis=(
                    f"appearance fingerprint cosine distance {distance:.4f} to cluster "
                    f"{matched_cluster_id[:8]} (auto-merge threshold "
                    f"{cluster_distance_threshold:.4f})"
                ),
                created_at=utc_now(),
            )
            self.repository.database.insert_merge_suggestion(suggestion)
            created.append(suggestion)
            existing_pending.add(pair_key)

        LOGGER.info("vehicle_appearance_merge_suggestions_generated", count=len(created))
        return created

    # -- person appearance --------------------------------------------------

    def resolve_person_appearance_identity(self, cluster_id: UUID) -> UUID:
        return self._resolve(cluster_id, self.repository.database.get_person_appearance_cluster)

    def merge_person_appearances(
        self, source_id: UUID, target_id: UUID, actor: str, notes: str = ""
    ) -> None:
        self._merge(
            source_id,
            target_id,
            actor,
            notes,
            entity_type="person_appearance",
            get_row=self.repository.database.get_person_appearance_cluster,
            set_merge=self.repository.database.set_person_appearance_cluster_merge,
            resolve=self.resolve_person_appearance_identity,
        )

    def get_person_appearance_identity(self, cluster_id: UUID) -> FaceIdentity:
        """Returns a `FaceIdentity` (the shape is identical: identity id,
        member cluster ids, label, counts, first/last seen, representative
        crops -- person-appearance observations do have crop images, like
        faces and vehicle-appearance)."""

        root = self.resolve_person_appearance_identity(cluster_id)
        members = [
            row
            for row in self.repository.database.list_person_appearance_clusters()
            if self.resolve_person_appearance_identity(UUID(row.cluster_id)) == root
        ]
        if not members:
            raise MergeError(f"no such person appearance cluster: {cluster_id}")
        label = next((m.label for m in members if m.label), None)
        crops: list[str] = []
        for member in members:
            if member.representative_crops_csv:
                crops.extend(member.representative_crops_csv.split(","))
        first_seen = min((m.first_seen_at for m in members if m.first_seen_at), default=None)
        last_seen = max((m.last_seen_at for m in members if m.last_seen_at), default=None)
        return FaceIdentity(
            identity_id=root,
            member_cluster_ids=sorted(UUID(m.cluster_id) for m in members),
            label=label,
            observation_count=sum(m.observation_count for m in members),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            representative_crop_paths=crops[-4:],
        )

    def list_person_appearance_sightings(
        self, cluster_id: UUID, follow_merges: bool = True, include_duplicates: bool = True
    ) -> list[PersonAppearanceObservationRow]:
        if not follow_merges:
            return self.repository.database.list_person_appearance_observations(
                cluster_id, include_duplicates
            )
        root = self.resolve_person_appearance_identity(cluster_id)
        member_ids = [
            UUID(row.cluster_id)
            for row in self.repository.database.list_person_appearance_clusters()
            if self.resolve_person_appearance_identity(UUID(row.cluster_id)) == root
        ]
        sightings: list[PersonAppearanceObservationRow] = []
        for member_id in member_ids:
            sightings.extend(
                self.repository.database.list_person_appearance_observations(
                    member_id, include_duplicates
                )
            )
        sightings.sort(key=lambda o: o.observed_at)
        return sightings

    def search_person_appearances(self, query: str, fuzzy_cutoff: float = 0.7) -> SearchResult:
        exact = self.repository.database.search_person_appearance_clusters(query)
        if exact:
            return SearchResult(exact_matches=list(exact), fuzzy_suggestions=[])
        all_clusters = self.repository.database.list_person_appearance_clusters()
        candidates: dict[str, PersonAppearanceClusterRow] = {
            row.cluster_id: row for row in all_clusters
        }
        for row in all_clusters:
            if row.label:
                candidates[row.label] = row
        close = difflib.get_close_matches(query, list(candidates.keys()), n=5, cutoff=fuzzy_cutoff)
        return SearchResult(exact_matches=[], fuzzy_suggestions=[candidates[c] for c in close])

    def cleanup_duplicate_person_appearance_observations(
        self, window_seconds: float = 5.0
    ) -> CleanupResult:
        """Collapse near-duplicate person-appearance observations within
        the same event and cluster -- the same pedestrian sampled across
        many frames produces one observation per frame. Mirrors
        `cleanup_duplicate_vehicle_appearance_observations` exactly; see
        that method's docstring."""

        actionable = [
            observation
            for observation in self.repository.database.list_all_person_appearance_observations()
            if observation.duplicate_of_observation_id is None
        ]
        groups: dict[tuple[str | None, str | None], list[PersonAppearanceObservationRow]] = {}
        for observation in actionable:
            key = (observation.event_id, observation.cluster_id)
            groups.setdefault(key, []).append(observation)

        suppressed: list[UUID] = []
        kept: list[UUID] = []
        groups_with_duplicates = 0
        for observations in groups.values():
            observations.sort(key=lambda o: o.observed_at)
            for cluster in _cluster_by_time(observations, window_seconds):
                if len(cluster) < 2:
                    kept.append(UUID(cluster[0].observation_id))
                    continue
                groups_with_duplicates += 1
                best = max(cluster, key=lambda o: o.detector_confidence)
                best_id = UUID(best.observation_id)
                kept.append(best_id)
                for observation in cluster:
                    if observation.observation_id == best.observation_id:
                        continue
                    self.repository.database.mark_person_appearance_observation_duplicate(
                        UUID(observation.observation_id), duplicate_of=best_id
                    )
                    suppressed.append(UUID(observation.observation_id))

        LOGGER.info(
            "person_appearance_duplicate_cleanup_completed",
            suppressed_count=len(suppressed),
            kept_count=len(kept),
            clusters_with_duplicates=groups_with_duplicates,
        )
        return CleanupResult(
            suppressed_observation_ids=suppressed,
            kept_observation_ids=kept,
            clusters_with_duplicates=groups_with_duplicates,
        )

    def suggest_person_appearance_merges(
        self,
        cluster_distance_threshold: float = 0.10,
        suggestion_multiplier: float = 1.6,
    ) -> list[MergeSuggestion]:
        """Flag pairs of person-appearance clusters that might be the same
        person, for human review -- never merged automatically. Mirrors
        `suggest_vehicle_appearance_merges`: compares cluster centroids
        directly via
        `IncrementalPersonAppearanceClusterer.predict_nearest_cluster`.
        """

        clusters = [
            cluster
            for cluster in self.repository.database.list_person_appearance_clusters()
            if not cluster.merged_into
        ]
        if len(clusters) < 2:
            return []

        clusterer = IncrementalPersonAppearanceClusterer(
            self.repository.workspace.person_appearance_model_path,
            distance_threshold=cluster_distance_threshold,
        )
        suggestion_ceiling = cluster_distance_threshold * suggestion_multiplier
        existing_pending = {
            frozenset((row.source_id, row.target_id))
            for row in self.repository.database.list_merge_suggestions(
                entity_type="person_appearance", status="pending"
            )
        }

        created: list[MergeSuggestion] = []
        for cluster in clusters:
            centroid = clusterer.get_cluster_centroid(cluster.cluster_id)
            if centroid is None:
                continue
            matched_cluster_id, distance = clusterer.predict_nearest_cluster(
                centroid, exclude_cluster_id=cluster.cluster_id
            )
            if matched_cluster_id is None:
                continue
            if not (cluster_distance_threshold < distance <= suggestion_ceiling):
                continue
            pair_key = frozenset((cluster.cluster_id, matched_cluster_id))
            if pair_key in existing_pending:
                continue
            span = suggestion_ceiling - cluster_distance_threshold
            similarity = 1.0 - (distance - cluster_distance_threshold) / span if span else 0.0
            suggestion = MergeSuggestion(
                suggestion_id=new_uuid(),
                entity_type="person_appearance",
                source_id=UUID(cluster.cluster_id),
                target_id=UUID(matched_cluster_id),
                similarity_score=round(max(0.0, min(1.0, similarity)), 6),
                basis=(
                    f"appearance fingerprint cosine distance {distance:.4f} to cluster "
                    f"{matched_cluster_id[:8]} (auto-merge threshold "
                    f"{cluster_distance_threshold:.4f})"
                ),
                created_at=utc_now(),
            )
            self.repository.database.insert_merge_suggestion(suggestion)
            created.append(suggestion)
            existing_pending.add(pair_key)

        LOGGER.info("person_appearance_merge_suggestions_generated", count=len(created))
        return created

    # -- shared -----------------------------------------------------------

    def _resolve(
        self,
        entity_id: UUID,
        get_row: Callable[
            [UUID],
            FaceClusterRow
            | PlateRecordRow
            | VoiceClusterRow
            | VehicleAppearanceClusterRow
            | PersonAppearanceClusterRow
            | None,
        ],
    ) -> UUID:
        current = entity_id
        seen = {current}
        for _ in range(_MAX_MERGE_CHAIN_LENGTH):
            row = get_row(current)
            if row is None or not row.merged_into:
                return current
            next_id = UUID(row.merged_into)
            if next_id in seen:
                LOGGER.error("identity_merge_cycle_detected", entity_id=str(entity_id))
                return current  # defensively break the cycle rather than loop forever
            seen.add(next_id)
            current = next_id
        LOGGER.error("identity_merge_chain_too_long", entity_id=str(entity_id))
        return current

    def _merge(
        self,
        source_id: UUID,
        target_id: UUID,
        actor: str,
        notes: str,
        *,
        entity_type: str,
        get_row: Callable[
            [UUID],
            FaceClusterRow
            | PlateRecordRow
            | VoiceClusterRow
            | VehicleAppearanceClusterRow
            | PersonAppearanceClusterRow
            | None,
        ],
        set_merge: Callable[[UUID, UUID], None],
        resolve: Callable[[UUID], UUID],
    ) -> None:
        if source_id == target_id:
            raise MergeError("cannot merge an identity into itself")
        if get_row(source_id) is None:
            raise MergeError(f"no such {entity_type}: {source_id}")
        if get_row(target_id) is None:
            raise MergeError(f"no such {entity_type}: {target_id}")
        if resolve(target_id) == source_id:
            raise MergeError(f"merging {source_id} into {target_id} would create a cycle")
        set_merge(source_id, target_id)
        record = IdentityMergeRecord(
            merge_id=new_uuid(),
            entity_type=entity_type,  # type: ignore[arg-type]
            source_id=source_id,
            target_id=target_id,
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_identity_merge_record(record)
        LOGGER.info(
            "identity_merged",
            entity_type=entity_type,
            source=str(source_id),
            target=str(target_id),
            actor=actor,
        )

    def resolve_plate_input(self, plate_id_or_text: str) -> PlateRecordRow | None:
        """Accept either a plate id (UUID) or plate text and look up the record."""

        try:
            plate_uuid = UUID(plate_id_or_text)
        except ValueError:
            return self.repository.database.get_plate_record_by_text(plate_id_or_text.upper())
        return self.repository.database.get_plate_record(plate_uuid)

    # -- false-positive cleanup automation ------------------------------------

    def cleanup_duplicate_face_observations(self, window_seconds: float = 5.0) -> CleanupResult:
        """Collapse near-duplicate face observations within the same event.

        Mirrors `cleanup_duplicate_plate_observations` (see that method's
        docstring for the full rationale): the same real face sighted
        across many sampled frames within one event produces one
        `FaceObservation` per frame. This groups observations by (event,
        cluster), clusters each group by `window_seconds` time proximity,
        and marks all but the highest-detector-confidence observation per
        cluster with `duplicate_of_observation_id` set -- never deleted,
        just excluded from `faces-sightings` by default
        (`include_duplicates=False`). Unlike plates, face observations have
        no human-decision status to preserve, so there's nothing this pass
        needs to avoid overriding -- it only ever touches observations that
        haven't already been marked a duplicate.
        """

        actionable = [
            observation
            for observation in self.repository.database.list_all_face_observations()
            if observation.duplicate_of_observation_id is None
        ]
        groups: dict[tuple[str | None, str | None], list[FaceObservationRow]] = {}
        for observation in actionable:
            key = (observation.event_id, observation.cluster_id)
            groups.setdefault(key, []).append(observation)

        suppressed: list[UUID] = []
        kept: list[UUID] = []
        groups_with_duplicates = 0
        for observations in groups.values():
            observations.sort(key=lambda o: o.observed_at)
            for cluster in _cluster_by_time(observations, window_seconds):
                if len(cluster) < 2:
                    kept.append(UUID(cluster[0].observation_id))
                    continue
                groups_with_duplicates += 1
                best = max(cluster, key=lambda o: o.detector_confidence)
                best_id = UUID(best.observation_id)
                kept.append(best_id)
                for observation in cluster:
                    if observation.observation_id == best.observation_id:
                        continue
                    self.repository.database.mark_face_observation_duplicate(
                        UUID(observation.observation_id), duplicate_of=best_id
                    )
                    suppressed.append(UUID(observation.observation_id))

        LOGGER.info(
            "face_duplicate_cleanup_completed",
            suppressed_count=len(suppressed),
            kept_count=len(kept),
            clusters_with_duplicates=groups_with_duplicates,
        )
        return CleanupResult(
            suppressed_observation_ids=suppressed,
            kept_observation_ids=kept,
            clusters_with_duplicates=groups_with_duplicates,
        )

    def cleanup_duplicate_plate_observations(self, window_seconds: float = 5.0) -> CleanupResult:
        """Collapse near-duplicate plate observations within the same event.

        The same physical plate sighted across many sampled frames within
        one event produces one `PlateObservation` per frame -- a burst of
        near-identical rows for what a human would consider a single
        sighting. This groups observations by (event, plate text), clusters
        each group by `window_seconds` time proximity, and keeps only the
        highest-OCR-confidence observation per cluster as the one still
        needing a human's attention. The rest are marked
        `duplicate_suppressed`, not deleted -- every observation and its
        crop image stay in the database and on disk; only `review_status`
        changes, and `duplicate_of_observation_id` records which
        observation was kept, so the decision stays fully inspectable and
        in principle reversible.

        Only ever touches observations still in `needs_review` or
        `auto_accepted`; a `user_confirmed` or `user_rejected` observation
        (an actual human decision) is never revisited by this pass.
        """

        actionable = [
            observation
            for observation in self.repository.database.list_plate_observations()
            if observation.review_status in ("needs_review", "auto_accepted")
        ]
        groups: dict[tuple[str | None, str], list[PlateObservationRow]] = {}
        for observation in actionable:
            key = (observation.event_id, observation.normalized_text)
            groups.setdefault(key, []).append(observation)

        suppressed: list[UUID] = []
        kept: list[UUID] = []
        groups_with_duplicates = 0
        for observations in groups.values():
            observations.sort(key=lambda o: o.observed_at)
            for cluster in _cluster_by_time(observations, window_seconds):
                if len(cluster) < 2:
                    kept.append(UUID(cluster[0].observation_id))
                    continue
                groups_with_duplicates += 1
                best = max(cluster, key=lambda o: o.ocr_confidence)
                best_id = UUID(best.observation_id)
                kept.append(best_id)
                for observation in cluster:
                    if observation.observation_id == best.observation_id:
                        continue
                    self.repository.database.mark_plate_observation_duplicate_suppressed(
                        UUID(observation.observation_id), duplicate_of=best_id
                    )
                    suppressed.append(UUID(observation.observation_id))

        LOGGER.info(
            "plate_duplicate_cleanup_completed",
            suppressed_count=len(suppressed),
            kept_count=len(kept),
            clusters_with_duplicates=groups_with_duplicates,
        )
        return CleanupResult(
            suppressed_observation_ids=suppressed,
            kept_observation_ids=kept,
            clusters_with_duplicates=groups_with_duplicates,
        )

    # -- automated merge suggestions ------------------------------------------

    def suggest_face_merges(
        self,
        cluster_distance_threshold: float = 70.0,
        suggestion_multiplier: float = 1.6,
        embedding_model: Literal["lbph", "auraface"] = "lbph",
    ) -> list[MergeSuggestion]:
        """Flag pairs of face clusters that look like they might be the same
        person, for a human to confirm or reject -- never merged automatically.

        Dispatches on `embedding_model` (mirrors
        `enrichment.face.embedding_model`, and must be passed the matching
        one -- `cluster_distance_threshold` is on a completely different
        scale between the two, LBPH's raw distance metric vs. AuraFace's
        cosine distance over normalized embeddings):

        * `lbph` (default): loads each un-merged cluster's most recent
          representative crop (already normalized the same way detection
          normalizes crops, so no reprocessing needed) and checks which
          *other* trained cluster the LBPH model considers it closest to,
          via the read-only `IncrementalFaceClusterer.predict_nearest_cluster`.
          Silently produces no suggestion for a cluster whose representative
          crop has since been deleted (see `docs/limitations.md`) -- its
          existing cluster identity is unaffected, but a fresh suggestion
          involving it can't be generated without at least one retained crop
          to compare.
        * `auraface`: compares stored cluster centroids directly, mirroring
          `suggest_voice_merges`/`suggest_vehicle_appearance_merges` exactly
          (an embedding is already a fixed-length vector -- no crop image or
          re-embedding needed to compare two clusters). Silently produces no
          suggestion for a cluster with no stored centroid (never observed
          under `auraface`, e.g. it predates switching to this
          `embedding_model`).

        Either way: a distance at or below `cluster_distance_threshold`
        would already have been auto-merged into the same cluster at
        detection time, so it can't appear here; anything up to
        `suggestion_multiplier` times that threshold is close enough to be
        worth a human's attention without being confident enough to merge
        automatically. A suggestion already pending for the same pair is
        never duplicated; a previously confirmed or rejected one doesn't
        block a fresh suggestion (the model may have changed since).
        """

        clusters = [
            cluster
            for cluster in self.repository.database.list_face_clusters()
            if not cluster.merged_into
        ]
        if len(clusters) < 2:
            return []

        find_match: Callable[[FaceClusterRow], tuple[str | None, float]]
        basis_label: str
        if embedding_model == "auraface":
            embedding_clusterer = IncrementalFaceEmbeddingClusterer(
                self.repository.workspace.face_embedding_model_path,
                distance_threshold=cluster_distance_threshold,
            )

            def find_match(cluster: FaceClusterRow) -> tuple[str | None, float]:
                centroid = embedding_clusterer.get_cluster_centroid(cluster.cluster_id)
                if centroid is None:
                    return None, 0.0
                return embedding_clusterer.predict_nearest_cluster(
                    centroid, exclude_cluster_id=cluster.cluster_id
                )

            basis_label = "AuraFace cosine distance"
        else:
            lbph_clusterer = IncrementalFaceClusterer(
                self.repository.workspace.face_model_path,
                distance_threshold=cluster_distance_threshold,
            )

            def find_match(cluster: FaceClusterRow) -> tuple[str | None, float]:
                if not cluster.representative_crops_csv:
                    return None, 0.0
                crop_path = Path(cluster.representative_crops_csv.split(",")[0])
                if not crop_path.exists():
                    return None, 0.0
                crop_image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
                if crop_image is None:
                    return None, 0.0
                return lbph_clusterer.predict_nearest_cluster(
                    crop_image, exclude_cluster_id=cluster.cluster_id
                )

            basis_label = "LBPH distance"

        suggestion_ceiling = cluster_distance_threshold * suggestion_multiplier
        existing_pending = {
            frozenset((row.source_id, row.target_id))
            for row in self.repository.database.list_merge_suggestions(
                entity_type="face", status="pending"
            )
        }

        created: list[MergeSuggestion] = []
        for cluster in clusters:
            matched_cluster_id, distance = find_match(cluster)
            if matched_cluster_id is None:
                continue
            if not (cluster_distance_threshold < distance <= suggestion_ceiling):
                continue
            pair_key = frozenset((cluster.cluster_id, matched_cluster_id))
            if pair_key in existing_pending:
                continue
            span = suggestion_ceiling - cluster_distance_threshold
            similarity = 1.0 - (distance - cluster_distance_threshold) / span if span else 0.0
            suggestion = MergeSuggestion(
                suggestion_id=new_uuid(),
                entity_type="face",
                source_id=UUID(cluster.cluster_id),
                target_id=UUID(matched_cluster_id),
                similarity_score=round(max(0.0, min(1.0, similarity)), 6),
                basis=(
                    f"{basis_label} {distance:.4f} to cluster {matched_cluster_id[:8]} "
                    f"(auto-merge threshold {cluster_distance_threshold:.4f})"
                ),
                created_at=utc_now(),
            )
            self.repository.database.insert_merge_suggestion(suggestion)
            created.append(suggestion)
            existing_pending.add(pair_key)

        LOGGER.info("face_merge_suggestions_generated", count=len(created))
        return created

    def suggest_plate_merges(self, similarity_threshold: float = 0.75) -> list[MergeSuggestion]:
        """Flag pairs of plate records whose text is close enough to plausibly
        be the same plate misread differently by OCR -- e.g. a "1" read as
        "I" on one sighting. Never merged automatically.

        Pure text-similarity comparison (`difflib.SequenceMatcher.ratio`,
        the same metric `recognize plates-search`'s fuzzy fallback uses),
        so unlike face suggestions this doesn't depend on any crop image
        still existing -- plate text is durable by nature, no image needed
        to compare it.
        """

        records = [
            record
            for record in self.repository.database.list_plate_records()
            if not record.merged_into
        ]
        if len(records) < 2:
            return []

        existing_pending = {
            frozenset((row.source_id, row.target_id))
            for row in self.repository.database.list_merge_suggestions(
                entity_type="plate", status="pending"
            )
        }

        created: list[MergeSuggestion] = []
        for index, first in enumerate(records):
            for second in records[index + 1 :]:
                ratio = difflib.SequenceMatcher(
                    None, first.normalized_text, second.normalized_text
                ).ratio()
                if ratio < similarity_threshold:
                    continue
                pair_key = frozenset((first.plate_id, second.plate_id))
                if pair_key in existing_pending:
                    continue
                suggestion = MergeSuggestion(
                    suggestion_id=new_uuid(),
                    entity_type="plate",
                    source_id=UUID(first.plate_id),
                    target_id=UUID(second.plate_id),
                    similarity_score=round(ratio, 6),
                    basis=(
                        f"text similarity {ratio:.2f} between "
                        f"'{first.normalized_text}' and '{second.normalized_text}'"
                    ),
                    created_at=utc_now(),
                )
                self.repository.database.insert_merge_suggestion(suggestion)
                created.append(suggestion)
                existing_pending.add(pair_key)

        LOGGER.info("plate_merge_suggestions_generated", count=len(created))
        return created

    def list_merge_suggestions(
        self, entity_type: str | None = None, status: str | None = "pending"
    ) -> list[MergeSuggestionRow]:
        return self.repository.database.list_merge_suggestions(entity_type, status)

    def confirm_merge_suggestion(self, suggestion_id: UUID, actor: str, notes: str = "") -> None:
        """Accept a suggestion: perform the actual merge (which writes its own
        `IdentityMergeRecord`, independent of this suggestion) and mark the
        suggestion resolved."""

        row = self._require_pending_suggestion(suggestion_id)
        source_id, target_id = UUID(row.source_id), UUID(row.target_id)
        if row.entity_type == "face":
            self.merge_faces(source_id, target_id, actor, notes)
        elif row.entity_type == "plate":
            self.merge_plates(source_id, target_id, actor, notes)
        elif row.entity_type == "voice":
            self.merge_voices(source_id, target_id, actor, notes)
        elif row.entity_type == "vehicle_appearance":
            self.merge_vehicle_appearances(source_id, target_id, actor, notes)
        elif row.entity_type == "person_appearance":
            self.merge_person_appearances(source_id, target_id, actor, notes)
        else:
            raise MergeError(f"unsupported entity_type for confirmation: {row.entity_type}")
        self.repository.database.resolve_merge_suggestion(
            suggestion_id, "confirmed", actor, utc_now()
        )

    def reject_merge_suggestion(self, suggestion_id: UUID, actor: str, notes: str = "") -> None:
        """Decline a suggestion: no merge is performed, the suggestion is marked
        `rejected` and retained (not deleted) as a record of what was proposed
        and that a human looked at it and said no."""

        self._require_pending_suggestion(suggestion_id)
        self.repository.database.resolve_merge_suggestion(
            suggestion_id, "rejected", actor, utc_now()
        )

    def _require_pending_suggestion(self, suggestion_id: UUID) -> MergeSuggestionRow:
        row = self.repository.database.get_merge_suggestion(suggestion_id)
        if row is None:
            raise MergeError(f"no such merge suggestion: {suggestion_id}")
        if row.status != "pending":
            raise MergeError(f"merge suggestion {suggestion_id} is already {row.status}")
        return row

    # -- human review & storage reclamation ------------------------------------
    #
    # See AGENTS.md invariant 22: confirming/rejecting always writes a
    # `RecognitionReviewRecord` before changing any row; purging a crop
    # always writes a `RecognitionCropPurgeRecord` before unlinking the
    # file. Neither ever deletes or rewrites an observation row itself --
    # only `review_status`/`crop_purged_at` change, `crop_path` stays as a
    # historical pointer. Two-step by default (review, then a separate
    # purge sweep); `purge=True` does both in one call for when a reviewer
    # is sure.

    def _review_handlers(self, entity_type: ClusterEntityType) -> _ReviewHandlers:
        db = self.repository.database
        if entity_type == "face":
            return _ReviewHandlers(
                get_observation=db.get_face_observation,
                list_observations_by_cluster=db.list_face_observations_by_cluster_ids,
                set_cluster_representative=db.set_face_cluster_representative,
                set_cluster_label=db.set_face_cluster_label,
                set_observation_review_status=db.set_face_observation_review_status,
                mark_observation_crop_purged=db.mark_face_observation_crop_purged,
                list_observations_eligible_for_purge=db.list_face_observations_eligible_for_purge,
                get_cluster=db.get_face_cluster,
                set_observation_cluster=db.set_face_observation_cluster,
                set_cluster_observation_count=db.set_face_cluster_observation_count,
            )
        if entity_type == "vehicle_appearance":
            return _ReviewHandlers(
                get_observation=db.get_vehicle_appearance_observation,
                list_observations_by_cluster=db.list_vehicle_appearance_observations_by_cluster_ids,
                set_cluster_representative=db.set_vehicle_appearance_cluster_representative,
                set_cluster_label=db.set_vehicle_appearance_cluster_label,
                set_observation_review_status=db.set_vehicle_appearance_observation_review_status,
                mark_observation_crop_purged=db.mark_vehicle_appearance_observation_crop_purged,
                list_observations_eligible_for_purge=(
                    db.list_vehicle_appearance_observations_eligible_for_purge
                ),
                get_cluster=db.get_vehicle_appearance_cluster,
                set_observation_cluster=db.set_vehicle_appearance_observation_cluster,
                set_cluster_observation_count=db.set_vehicle_appearance_cluster_observation_count,
            )
        return _ReviewHandlers(
            get_observation=db.get_person_appearance_observation,
            list_observations_by_cluster=db.list_person_appearance_observations_by_cluster_ids,
            set_cluster_representative=db.set_person_appearance_cluster_representative,
            set_cluster_label=db.set_person_appearance_cluster_label,
            set_observation_review_status=db.set_person_appearance_observation_review_status,
            mark_observation_crop_purged=db.mark_person_appearance_observation_crop_purged,
            list_observations_eligible_for_purge=(
                db.list_person_appearance_observations_eligible_for_purge
            ),
            get_cluster=db.get_person_appearance_cluster,
            set_observation_cluster=db.set_person_appearance_observation_cluster,
            set_cluster_observation_count=db.set_person_appearance_cluster_observation_count,
        )

    def confirm_identity(
        self,
        cluster_id: UUID,
        representative_observation_ids: list[UUID],
        actor: str,
        entity_type: ClusterEntityType,
        label: str | None = None,
        notes: str = "",
        purge: bool = False,
    ) -> RecognitionReviewRecord:
        """Confirm `cluster_id` is a real, consistent identity: pick its
        representative crop(s) (must belong to this cluster), optionally
        attach a label, and mark every observation in the cluster
        `user_confirmed`. Non-representative observations become eligible
        for crop purging via `purge_reviewed_crops` (or immediately if
        `purge=True`). Writes one `RecognitionReviewRecord` covering the
        whole cluster's observation ids, not one per observation.
        """

        if not representative_observation_ids:
            raise ReviewError("at least one representative_observation_id is required")
        handlers = self._review_handlers(entity_type)
        observations = handlers.list_observations_by_cluster([cluster_id])
        if not observations:
            raise ReviewError(f"no {entity_type} observations found for cluster {cluster_id}")
        observation_ids = {UUID(o.observation_id) for o in observations}
        missing = set(representative_observation_ids) - observation_ids
        if missing:
            raise ReviewError(
                f"representative observation(s) {sorted(missing)} do not belong to "
                f"cluster {cluster_id}"
            )
        representative_crop_paths = [
            o.crop_path
            for o in observations
            if UUID(o.observation_id) in representative_observation_ids
        ]
        handlers.set_cluster_representative(
            cluster_id, representative_observation_ids, representative_crop_paths
        )
        if label is not None:
            handlers.set_cluster_label(cluster_id, label)
        for observation_id in observation_ids:
            handlers.set_observation_review_status(observation_id, "user_confirmed")

        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="confirmed",
            entity_type=entity_type,
            cluster_id=cluster_id,
            observation_ids=sorted(observation_ids),
            label=label,
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_identity_confirmed",
            entity_type=entity_type,
            cluster_id=str(cluster_id),
            observation_count=len(observation_ids),
            actor=actor,
        )
        if purge:
            self.purge_reviewed_crops(entity_type, actor, notes=notes)
        return record

    def reject_cluster(
        self,
        cluster_id: UUID,
        actor: str,
        entity_type: ClusterEntityType,
        notes: str = "",
        purge: bool = False,
    ) -> RecognitionReviewRecord:
        """Mark every observation in `cluster_id` a false positive (e.g.
        an entire cluster of misdetections that was never actually a
        face/vehicle). Writes one `RecognitionReviewRecord` covering the
        whole cluster."""

        handlers = self._review_handlers(entity_type)
        observations = handlers.list_observations_by_cluster([cluster_id])
        if not observations:
            raise ReviewError(f"no {entity_type} observations found for cluster {cluster_id}")
        observation_ids = [UUID(o.observation_id) for o in observations]
        for observation_id in observation_ids:
            handlers.set_observation_review_status(observation_id, "user_rejected")

        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="rejected",
            entity_type=entity_type,
            cluster_id=cluster_id,
            observation_ids=sorted(observation_ids),
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_cluster_rejected",
            entity_type=entity_type,
            cluster_id=str(cluster_id),
            observation_count=len(observation_ids),
            actor=actor,
        )
        if purge:
            self.purge_reviewed_crops(entity_type, actor, notes=notes)
        return record

    def reject_observation(
        self,
        observation_id: UUID,
        actor: str,
        entity_type: ClusterEntityType,
        notes: str = "",
        purge: bool = False,
    ) -> RecognitionReviewRecord:
        """Mark a single observation a false positive -- for a mixed
        cluster where only some crops are wrong, without touching the
        rest of the cluster's review state."""

        handlers = self._review_handlers(entity_type)
        if handlers.get_observation(observation_id) is None:
            raise ReviewError(f"no such {entity_type} observation: {observation_id}")
        handlers.set_observation_review_status(observation_id, "user_rejected")

        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="rejected",
            entity_type=entity_type,
            cluster_id=None,
            observation_ids=[observation_id],
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_observation_rejected",
            entity_type=entity_type,
            observation_id=str(observation_id),
            actor=actor,
        )
        if purge:
            self.purge_reviewed_crops(entity_type, actor, notes=notes)
        return record

    def detach_observation(
        self,
        observation_id: UUID,
        actor: str,
        entity_type: ClusterEntityType,
        notes: str = "",
    ) -> RecognitionReviewRecord:
        """Remove a single observation from its cluster entirely.

        Unlike `reject_observation` (which only flags `review_status` and
        leaves the observation still counted toward its cluster's
        `observation_count`/representative crops), this clears
        `cluster_id` and recomputes the former cluster's count -- for a
        crop that was auto-clustered as a false positive and shouldn't
        contribute to that identity's stats at all. The observation row
        itself is never deleted -- it becomes clusterless, a permanent
        record that it was reviewed and found not to belong there."""

        handlers = self._review_handlers(entity_type)
        observation = handlers.get_observation(observation_id)
        if observation is None:
            raise ReviewError(f"no such {entity_type} observation: {observation_id}")
        former_cluster_id = UUID(observation.cluster_id) if observation.cluster_id else None

        handlers.set_observation_cluster(observation_id, None)
        handlers.set_observation_review_status(observation_id, "user_rejected")
        if former_cluster_id is not None:
            remaining = handlers.list_observations_by_cluster([former_cluster_id])
            handlers.set_cluster_observation_count(former_cluster_id, len(remaining))

        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="detached",
            entity_type=entity_type,
            cluster_id=former_cluster_id,
            observation_ids=[observation_id],
            actor=actor,
            timestamp=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_observation_detached",
            entity_type=entity_type,
            observation_id=str(observation_id),
            former_cluster_id=str(former_cluster_id) if former_cluster_id else None,
            actor=actor,
        )
        return record

    def move_observation(
        self,
        observation_id: UUID,
        target_cluster_id: UUID,
        actor: str,
        entity_type: ClusterEntityType,
        notes: str = "",
    ) -> RecognitionReviewRecord:
        """Reassign a single observation from its current cluster to
        `target_cluster_id` -- for a crop auto-clustered under the wrong
        identity that a human can see actually belongs elsewhere.
        Recomputes `observation_count` on both the source and target
        clusters; the source cluster id (if any) is recorded in `notes`
        for full traceability alongside the target in `cluster_id`."""

        handlers = self._review_handlers(entity_type)
        observation = handlers.get_observation(observation_id)
        if observation is None:
            raise ReviewError(f"no such {entity_type} observation: {observation_id}")
        if handlers.get_cluster(target_cluster_id) is None:
            raise ReviewError(f"no such {entity_type} cluster: {target_cluster_id}")
        source_cluster_id = UUID(observation.cluster_id) if observation.cluster_id else None
        if source_cluster_id == target_cluster_id:
            raise ReviewError("observation is already in that cluster")

        handlers.set_observation_cluster(observation_id, target_cluster_id)
        if source_cluster_id is not None:
            remaining = handlers.list_observations_by_cluster([source_cluster_id])
            handlers.set_cluster_observation_count(source_cluster_id, len(remaining))
        updated_target = handlers.list_observations_by_cluster([target_cluster_id])
        handlers.set_cluster_observation_count(target_cluster_id, len(updated_target))

        record = RecognitionReviewRecord(
            review_id=new_uuid(),
            action="moved",
            entity_type=entity_type,
            cluster_id=target_cluster_id,
            observation_ids=[observation_id],
            actor=actor,
            timestamp=utc_now(),
            notes=(
                f"moved from cluster {source_cluster_id}. {notes}"
                if source_cluster_id is not None
                else notes
            ),
        )
        self.repository.workspace.append_recognition_review_record(record)
        LOGGER.info(
            "recognition_observation_moved",
            entity_type=entity_type,
            observation_id=str(observation_id),
            source_cluster_id=str(source_cluster_id) if source_cluster_id else None,
            target_cluster_id=str(target_cluster_id),
            actor=actor,
        )
        return record

    def purge_reviewed_crops(
        self,
        entity_type: CropEntityType,
        actor: str,
        notes: str = "",
        dry_run: bool = False,
    ) -> RecognitionCropPurgeRecord:
        """Delete every already-reviewed, purge-eligible crop image from
        disk (see the eligibility queries in `storage/database.py`) --
        hash-verifies each file before deleting (the same defensive check
        `TriageService.confirm_deletion` already does), writes the
        `RecognitionCropPurgeRecord` *before* unlinking. The observation
        row itself, including `crop_path` as a historical pointer, is
        never deleted -- only `crop_purged_at` is set.

        `dry_run=True` returns exactly what *would* be purged without
        touching any file or writing the record -- the CLI/review_ui
        `--dry-run` preview.
        """

        db = self.repository.database
        eligible: list[
            FaceObservationRow
            | PlateObservationRow
            | VehicleAppearanceObservationRow
            | PersonAppearanceObservationRow
        ]
        mark_purged: Callable[[UUID], None]
        if entity_type == "plate":
            eligible = list(db.list_plate_observations_eligible_for_purge())
            mark_purged = db.mark_plate_observation_crop_purged
        else:
            handlers = self._review_handlers(entity_type)
            eligible = list(handlers.list_observations_eligible_for_purge())
            mark_purged = handlers.mark_observation_crop_purged

        purged_ids: list[UUID] = []
        purged_paths: list[str] = []
        purged_hashes: list[str] = []
        for observation in eligible:
            crop_path = Path(observation.crop_path)
            if not crop_path.exists():
                continue  # already gone somehow -- nothing to purge, not an error
            actual_hash = hash_file(crop_path)
            if actual_hash != observation.crop_sha256:
                LOGGER.warning(
                    "recognition_crop_purge_hash_mismatch_skipped",
                    observation_id=observation.observation_id,
                    path=str(crop_path),
                )
                continue
            purged_ids.append(UUID(observation.observation_id))
            purged_paths.append(str(crop_path))
            purged_hashes.append(actual_hash)

        if dry_run:
            return RecognitionCropPurgeRecord(
                purge_id=new_uuid(),
                entity_type=entity_type,
                purged_observation_ids=purged_ids,
                purged_crop_paths=purged_paths,
                purged_crop_hashes=purged_hashes,
                confirmed_by=actor,
                confirmed_at=utc_now(),
                notes=f"[dry run] {notes}" if notes else "[dry run]",
            )

        record = RecognitionCropPurgeRecord(
            purge_id=new_uuid(),
            entity_type=entity_type,
            purged_observation_ids=purged_ids,
            purged_crop_paths=purged_paths,
            purged_crop_hashes=purged_hashes,
            confirmed_by=actor,
            confirmed_at=utc_now(),
            notes=notes,
        )
        self.repository.workspace.append_recognition_crop_purge_record(record)
        for observation_id, path in zip(purged_ids, purged_paths, strict=True):
            Path(path).unlink()
            mark_purged(observation_id)
        LOGGER.info(
            "recognition_crops_purged", entity_type=entity_type, count=len(purged_ids), actor=actor
        )
        return record


def _cluster_by_time[
    T: (
        FaceObservationRow,
        PlateObservationRow,
        VoiceObservationRow,
        VehicleAppearanceObservationRow,
        PersonAppearanceObservationRow,
    )
](observations: list[T], window_seconds: float) -> list[list[T]]:
    """Group time-sorted observations into bursts no more than `window_seconds` apart.

    Shared by `cleanup_duplicate_face_observations`, `cleanup_duplicate_plate_observations`,
    and `cleanup_duplicate_voice_observations` -- all three row types expose
    the same `observed_at` field, which is all this needs. Bound to those
    three row types (rather than a vague `Any`) so a caller passing a list
    of one type gets back a list of lists of that same type.
    """

    clusters: list[list[T]] = []
    current: list[T] = []
    for observation in observations:
        if current:
            gap = (observation.observed_at - current[-1].observed_at).total_seconds()
            if gap > window_seconds:
                clusters.append(current)
                current = []
        current.append(observation)
    if current:
        clusters.append(current)
    return clusters
