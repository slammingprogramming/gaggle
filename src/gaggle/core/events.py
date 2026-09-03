"""Human-review correction for events that incorrectly bundled clips from
separate recording sessions into one.

`normalize/sync.py::_group_overlapping_sessions` and
`core/pipeline.py::_cluster_overlapping_windows` both merge purely on
**time overlap** -- no camera/session-boundary check, no
duration-similarity check (see `normalize/sync.py`'s module docstring:
"assumption that dashcams sharing a vehicle are usually powered on
together"). That is intentional, documented heuristic design, not a bug
to fix by tightening the algorithm itself -- doing so risks breaking
legitimately-merged multi-camera events. When the heuristic gets it
wrong, the correction is a human decision, not an algorithm change --
the same "flag for human judgment rather than guess" philosophy already
used for merge suggestions and the low-confidence OCR review queue.

`EventSplitService.split_event` is that correction: given an event whose
signals/derived clips actually came from unrelated sessions, a human
partitions its clips into groups and gets back one independent
`EventRecord` per group. The original event is never deleted or edited
beyond one final revision recording `superseded_by_event_ids` -- every
signal, derived clip, and revision it already had stays exactly as it
was, fully readable and auditable.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

from gaggle.core.config import RuntimeConfig
from gaggle.schemas.common import ChainOfCustodyEntry
from gaggle.schemas.event import EventRecord, Hypothesis, PreservationStatus
from gaggle.scoring.service import ScoringService
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid
from gaggle.utils.logging import get_logger
from gaggle.utils.time import utc_now

LOGGER = get_logger(__name__)


class EventSplitError(ValueError):
    """Raised for an invalid split request: unknown event, an
    already-split event, fewer than 2 groups, a clip_id repeated across
    groups, a clip_id that doesn't belong to this event, or a group that
    doesn't cover every one of this event's clips (a partition, not a
    pick-some-and-discard-the-rest)."""


class EventSplitService:
    def __init__(self, repository: Repository, runtime: RuntimeConfig) -> None:
        self.repository = repository
        self.scoring = ScoringService(runtime)

    def split_event(
        self,
        event_id: UUID,
        clip_id_groups: list[list[UUID]],
        actor: str,
        notes: str = "",
    ) -> list[EventRecord]:
        if len(clip_id_groups) < 2:
            raise EventSplitError("split requires at least 2 groups")
        event = self.repository.load_event(event_id)
        if event.superseded_by_event_ids:
            raise EventSplitError(
                f"event {event_id} was already split into {event.superseded_by_event_ids}"
            )

        clip_sha256_by_id: dict[UUID, str] = {}
        for artifact in event.derived_artifacts:
            source_clip_id = artifact.metadata.get("source_clip_id")
            source_sha256 = artifact.metadata.get("source_sha256")
            if source_clip_id and source_sha256:
                clip_sha256_by_id[UUID(str(source_clip_id))] = str(source_sha256)
        if not clip_sha256_by_id:
            raise EventSplitError(f"event {event_id} has no derived clips to split")

        all_grouped_ids = [clip_id for group in clip_id_groups for clip_id in group]
        if len(all_grouped_ids) != len(set(all_grouped_ids)):
            raise EventSplitError("a clip_id appears in more than one group")
        if set(all_grouped_ids) != set(clip_sha256_by_id):
            raise EventSplitError(
                "clip_id_groups must exactly partition this event's own clips -- "
                f"expected {sorted(str(c) for c in clip_sha256_by_id)}, "
                f"got {sorted(str(c) for c in all_grouped_ids)}"
            )

        new_events = [
            self._build_split_event(event, group, clip_sha256_by_id) for group in clip_id_groups
        ]
        for new_event in new_events:
            self.repository.save_event(new_event)

        self.repository.save_event_revision(
            event_id,
            reason="split",
            update={"superseded_by_event_ids": [e.event_id for e in new_events]},
        )
        LOGGER.info(
            "event_split",
            event_id=str(event_id),
            new_event_ids=[str(e.event_id) for e in new_events],
            actor=actor,
            notes=notes,
        )
        return new_events

    def _build_split_event(
        self,
        original: EventRecord,
        clip_ids: list[UUID],
        clip_sha256_by_id: dict[UUID, str],
    ) -> EventRecord:
        group_sha256s = {clip_sha256_by_id[clip_id] for clip_id in clip_ids}
        group_signals = [
            signal
            for signal in original.signals
            if signal.evidence_references and signal.evidence_references[0].sha256 in group_sha256s
        ]
        if not group_signals:
            raise EventSplitError(f"no signals trace to clip group {clip_ids}")
        sorted_signals = sorted(group_signals, key=lambda s: (s.timestamp_start, str(s.id)))
        signal_ids = {signal.id for signal in sorted_signals}

        # A hypothesis correlating signals from *both* sides of the split
        # no longer makes sense post-split -- only carry over ones whose
        # every contributing signal is present in this group.
        group_hypotheses: list[Hypothesis] = [
            hypothesis
            for hypothesis in original.hypotheses
            if hypothesis.contributing_signal_ids
            and set(hypothesis.contributing_signal_ids).issubset(signal_ids)
        ]
        scoring = self.scoring.score(group_hypotheses, sorted_signals)
        involved_cameras = sorted({s.camera_id for s in sorted_signals if s.camera_id is not None})

        new_event_id = new_uuid()
        clips_dir = self.repository.workspace.event_clips_dir(new_event_id)
        clips_dir.mkdir(parents=True, exist_ok=True)
        copied_artifacts = []
        for artifact in original.derived_artifacts:
            source_clip_id = artifact.metadata.get("source_clip_id")
            if not source_clip_id or UUID(str(source_clip_id)) not in clip_ids:
                continue
            source_path = Path(artifact.path)
            destination = clips_dir / source_path.name
            shutil.copy2(source_path, destination)
            copied_artifacts.append(artifact.model_copy(update={"path": str(destination)}))

        signal_types = sorted({s.signal_type for s in sorted_signals})
        labels = [h.label for h in group_hypotheses]
        return EventRecord(
            event_id=new_event_id,
            created_at=utc_now(),
            pipeline_version=original.pipeline_version,
            config_snapshot=original.config_snapshot,
            event_start=min(s.timestamp_start for s in sorted_signals),
            event_end=max(s.timestamp_end for s in sorted_signals),
            involved_cameras=involved_cameras,
            signals=sorted_signals,
            hypotheses=group_hypotheses,
            scoring=scoring,
            preservation_status=PreservationStatus(state="pending", immutable=False),
            chain_of_custody=[
                ChainOfCustodyEntry(
                    entry_id=new_uuid(),
                    action="split_from_event",
                    actor="gaggle",
                    timestamp=utc_now(),
                    details={"source_event_id": str(original.event_id)},
                )
            ],
            hashes=[
                signal.evidence_references[0].sha256
                for signal in sorted_signals
                if signal.evidence_references and signal.evidence_references[0].sha256 is not None
            ],
            derived_artifacts=copied_artifacts,
            evidence_summary=(
                f"Split from event {original.event_id}. "
                f"Signals: {', '.join(signal_types)}. "
                f"Hypotheses: {', '.join(labels) if labels else 'none'}."
            ),
            metadata={"split_from_event_id": str(original.event_id)},
        )
