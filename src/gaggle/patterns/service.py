from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from gaggle.schemas.event import EventRecord


@dataclass(frozen=True, slots=True)
class EncounterIdentityPair:
    """One Encounter's face/vehicle-appearance identity, already resolved
    to canonical cluster ids by the caller (see `cli/app.py`'s `patterns
    analyze` command, which does the observation-id -> cluster-id lookup
    via `RecognitionService`) -- keeps this module free of any
    `Repository`/`TimelineDatabase` import, exactly like every other
    pattern method here, which operates purely on already-computed
    metadata rather than querying storage directly.
    """

    face_cluster_id: str | None
    vehicle_cluster_id: str | None


@dataclass(frozen=True, slots=True)
class Pattern:
    """A metadata-only pattern hypothesis, never a conclusion.

    Patterns are explicitly hypotheses (``hypothesis_only=True`` is always
    present in ``to_dict()``) -- e.g. "this camera fired repeatedly" is a
    fact worth a human's attention, not proof of anything on its own.
    """

    pattern_type: str
    confidence: float
    version: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "hypothesis_only": True,
            "version": self.version,
            **self.details,
        }


class PatternAnalysisService:
    """Detects metadata-only patterns across previously generated events.

    All three built-in patterns operate purely on already-computed event
    metadata (camera ids, timestamps, object-hint labels) -- no new video
    or audio analysis happens here, consistent with "pattern detection is
    metadata-only" from the project's design directives.
    """

    version = "1.2.0"

    def analyze(
        self,
        events: list[EventRecord],
        cluster_window_seconds: float = 3600.0,
        min_repeat_count: int = 2,
        encounter_identities: list[EncounterIdentityPair] | None = None,
    ) -> list[dict[str, Any]]:
        patterns: list[Pattern] = []
        patterns.extend(self._repeated_camera_activity(events, min_repeat_count))
        patterns.extend(self._repeated_object_labels(events, min_repeat_count))
        patterns.extend(self._repeated_vehicle_appearances(events, min_repeat_count))
        patterns.extend(self._temporal_clusters(events, cluster_window_seconds, min_repeat_count))
        if encounter_identities is not None:
            patterns.extend(
                self._recurring_face_vehicle_cooccurrence(encounter_identities, min_repeat_count)
            )
        return [pattern.to_dict() for pattern in patterns]

    def _repeated_camera_activity(
        self, events: list[EventRecord], min_repeat_count: int
    ) -> list[Pattern]:
        camera_counter = Counter(camera for event in events for camera in event.involved_cameras)
        return [
            Pattern(
                pattern_type="repeated_camera_activity",
                confidence=min(1.0, count / 10.0),
                version=self.version,
                details={"camera_id": camera_id, "count": count},
            )
            for camera_id, count in sorted(camera_counter.items())
            if count >= min_repeat_count
        ]

    def _repeated_object_labels(
        self, events: list[EventRecord], min_repeat_count: int
    ) -> list[Pattern]:
        label_counter: Counter[str] = Counter()
        for event in events:
            for signal in event.signals:
                if signal.signal_type != "object_hint":
                    continue
                label = signal.reasoning_metadata.get("label")
                if isinstance(label, str):
                    label_counter[label] += 1
        return [
            Pattern(
                pattern_type="repeated_object_label",
                confidence=min(1.0, count / 10.0),
                version=self.version,
                details={"label": label, "count": count},
            )
            for label, count in sorted(label_counter.items())
            if count >= min_repeat_count
        ]

    def _repeated_vehicle_appearances(
        self, events: list[EventRecord], min_repeat_count: int
    ) -> list[Pattern]:
        """A vehicle-appearance cluster seen across several events is a
        legitimate hypothesis-only pattern -- "this same vehicle keeps
        showing up" -- mirroring `_repeated_object_labels` but keyed on the
        re-identification cluster id from `enrichment/vehicle_appearance.py`
        rather than a raw label. Like every pattern here, this is metadata-only:
        no new video/image analysis happens, it only counts signals already
        produced by a prior `enrich` run.
        """

        counter: Counter[str] = Counter()
        seen_at: dict[str, list[datetime]] = defaultdict(list)
        for event in events:
            for signal in event.signals:
                if signal.signal_type != "vehicle_appearance":
                    continue
                cluster_id = signal.reasoning_metadata.get("cluster_id")
                if not isinstance(cluster_id, str):
                    continue
                counter[cluster_id] += 1
                seen_at[cluster_id].append(event.event_start)
        return [
            Pattern(
                pattern_type="repeated_vehicle_appearance",
                confidence=min(1.0, count / 10.0),
                version=self.version,
                details={
                    "cluster_id": cluster_id,
                    "count": count,
                    "first_seen_at": min(seen_at[cluster_id]).isoformat(),
                    "last_seen_at": max(seen_at[cluster_id]).isoformat(),
                },
            )
            for cluster_id, count in sorted(counter.items())
            if count >= min_repeat_count
        ]

    def _recurring_face_vehicle_cooccurrence(
        self, encounter_identities: list[EncounterIdentityPair], min_repeat_count: int
    ) -> list[Pattern]:
        """A specific (face, vehicle) identity pair observed together
        across several distinct Encounters -- e.g. the same person
        consistently arriving with the same vehicle. Purely a
        co-occurrence count, exactly as non-accusatory as every other
        pattern here: an Encounter itself makes no spatial-correspondence
        claim (see `schemas/encounter.py`), so this pattern never claims
        the face belongs to or was driving the vehicle -- only that they
        were repeatedly observed together within the same clip.
        """

        counter: Counter[tuple[str, str]] = Counter()
        for pair in encounter_identities:
            if pair.face_cluster_id is None or pair.vehicle_cluster_id is None:
                continue
            counter[(pair.face_cluster_id, pair.vehicle_cluster_id)] += 1
        return [
            Pattern(
                pattern_type="recurring_face_vehicle_cooccurrence",
                confidence=min(1.0, count / 10.0),
                version=self.version,
                details={
                    "face_cluster_id": face_cluster_id,
                    "vehicle_cluster_id": vehicle_cluster_id,
                    "count": count,
                },
            )
            for (face_cluster_id, vehicle_cluster_id), count in sorted(counter.items())
            if count >= min_repeat_count
        ]

    def _temporal_clusters(
        self,
        events: list[EventRecord],
        cluster_window_seconds: float,
        min_repeat_count: int,
    ) -> list[Pattern]:
        ordered = sorted(events, key=lambda event: event.event_start)
        window = timedelta(seconds=cluster_window_seconds)
        clusters: list[list[EventRecord]] = []
        current: list[EventRecord] = []
        for event in ordered:
            if current and event.event_start - current[-1].event_start > window:
                clusters.append(current)
                current = []
            current.append(event)
        if current:
            clusters.append(current)

        return [
            Pattern(
                pattern_type="temporal_clustering",
                confidence=min(1.0, len(cluster) / 5.0),
                version=self.version,
                details={
                    "event_count": len(cluster),
                    "cluster_start": cluster[0].event_start.isoformat(),
                    "cluster_end": cluster[-1].event_end.isoformat(),
                    "event_ids": [str(event.event_id) for event in cluster],
                },
            )
            for cluster in clusters
            if len(cluster) >= min_repeat_count
        ]
