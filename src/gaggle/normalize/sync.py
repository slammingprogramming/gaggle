"""Deterministic, explainable multi-camera time synchronization.

Dashcam clocks are not trustworthy: cameras start recording at slightly
different moments after ignition, their clocks drift, and some don't even
have a real-time clock at all (falling back to file mtime). This module
never assumes cameras are already synchronized. Instead it:

1. Groups each camera's clips into *sessions* — contiguous recording runs
   with no gap larger than ``session_gap_seconds`` between one clip's
   observed end and the next clip's observed start. A session models one
   continuous power-on cycle for that camera.
2. Groups sessions from different cameras into *sync groups* when their
   time ranges overlap, on the assumption that dashcams sharing a vehicle
   are usually powered on together.
3. Within each sync group, deterministically selects a reference session —
   the one with the highest average per-clip timestamp confidence, ties
   broken alphabetically by camera id — and aligns every other session in
   the group to it.
4. Computes an offset (difference between session start times) and a
   drift-per-hour estimate (from the proportional difference in session
   span, i.e. how much faster/slower the two sessions appear to run
   relative to each other over their shared duration).

This is a heuristic, not a ground-truth clock recovery algorithm — there is
no audio/video cross-correlation here (a legitimate future upgrade path via
a plugin). Every corrected timestamp keeps its original alongside it, and
every correction carries a plain-language rationale and a confidence score
so a human reviewer can see exactly why a timestamp was adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class ClipTimeInfo:
    """The minimal per-clip information the sync algorithm needs."""

    clip_id: str
    camera_id: str
    observed_start: datetime
    observed_end: datetime
    timestamp_confidence: float


@dataclass(frozen=True, slots=True)
class SessionSyncResult:
    camera_id: str
    session_id: str
    clip_ids: list[str]
    original_start: datetime
    original_end: datetime
    corrected_start: datetime
    corrected_end: datetime
    offset_seconds: float
    drift_seconds_per_hour: float
    confidence: float
    is_reference: bool
    reference_camera_id: str | None
    rationale: str


def compute_camera_sync(
    clips: list[ClipTimeInfo],
    session_gap_seconds: float = 120.0,
    site_id_by_camera: dict[str, str] | None = None,
) -> list[SessionSyncResult]:
    """Compute session grouping and cross-camera alignment for ``clips``.

    ``site_id_by_camera`` scopes which cameras are even considered for
    cross-camera alignment: two sessions are only checked for time overlap
    (and therefore only ever grouped together) if their cameras share a
    site. A camera absent from the mapping defaults to its own private site
    (keyed by its own camera_id) -- the safe default for callers that don't
    pass a mapping at all, since two unrelated cameras with coincidentally
    overlapping clocks (e.g. a neighbor's camera and your dashcam) must
    never be spuriously aligned just because their timestamps overlap. See
    ``ingest/service.py``'s ``default_site_id`` derivation for how real
    ingest runs populate this so a single dashcam rig's simultaneous
    cameras keep cross-syncing with zero configuration.
    """

    site_id_by_camera = site_id_by_camera or {}
    sessions = _group_into_sessions(clips, session_gap_seconds, site_id_by_camera)

    results: list[SessionSyncResult] = []
    for site_sessions in _partition_by_site(sessions):
        for group in _group_overlapping_sessions(site_sessions):
            reference = _select_reference(group)
            for session in group:
                results.append(_align_session(session, reference))
    results.sort(key=lambda r: (r.camera_id, r.original_start))
    return results


@dataclass
class _Session:
    camera_id: str
    site_id: str
    session_index: int
    clip_ids: list[str]
    start: datetime
    end: datetime
    avg_confidence: float

    @property
    def session_id(self) -> str:
        return f"{self.camera_id}#{self.session_index:03d}"

    @property
    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def _group_into_sessions(
    clips: list[ClipTimeInfo], session_gap_seconds: float, site_id_by_camera: dict[str, str]
) -> list[_Session]:
    by_camera: dict[str, list[ClipTimeInfo]] = {}
    for clip in clips:
        by_camera.setdefault(clip.camera_id, []).append(clip)

    sessions: list[_Session] = []
    for camera_id in sorted(by_camera):
        site_id = site_id_by_camera.get(camera_id, camera_id)
        camera_clips = sorted(by_camera[camera_id], key=lambda c: c.observed_start)
        current: list[ClipTimeInfo] = []
        session_index = 0
        for clip in camera_clips:
            if (
                current
                and (clip.observed_start - current[-1].observed_end).total_seconds()
                > session_gap_seconds
            ):
                sessions.append(_build_session(camera_id, site_id, session_index, current))
                session_index += 1
                current = []
            current.append(clip)
        if current:
            sessions.append(_build_session(camera_id, site_id, session_index, current))
    return sessions


def _build_session(
    camera_id: str, site_id: str, session_index: int, clips: list[ClipTimeInfo]
) -> _Session:
    return _Session(
        camera_id=camera_id,
        site_id=site_id,
        session_index=session_index,
        clip_ids=[clip.clip_id for clip in clips],
        start=min(clip.observed_start for clip in clips),
        end=max(clip.observed_end for clip in clips),
        avg_confidence=sum(clip.timestamp_confidence for clip in clips) / len(clips),
    )


def _partition_by_site(sessions: list[_Session]) -> list[list[_Session]]:
    by_site: dict[str, list[_Session]] = {}
    for session in sessions:
        by_site.setdefault(session.site_id, []).append(session)
    return [by_site[key] for key in sorted(by_site)]


def _group_overlapping_sessions(sessions: list[_Session]) -> list[list[_Session]]:
    ordered = sorted(sessions, key=lambda s: s.start)
    groups: list[list[_Session]] = []
    for session in ordered:
        placed = False
        for group in groups:
            if any(_overlaps(session, member) for member in group):
                group.append(session)
                placed = True
                break
        if not placed:
            groups.append([session])
    return groups


def _overlaps(a: _Session, b: _Session) -> bool:
    return a.start < b.end and b.start < a.end


def _select_reference(group: list[_Session]) -> _Session:
    return sorted(group, key=lambda s: (-s.avg_confidence, s.camera_id, s.session_index))[0]


def _align_session(session: _Session, reference: _Session) -> SessionSyncResult:
    if session is reference:
        return SessionSyncResult(
            camera_id=session.camera_id,
            session_id=session.session_id,
            clip_ids=list(session.clip_ids),
            original_start=session.start,
            original_end=session.end,
            corrected_start=session.start,
            corrected_end=session.end,
            offset_seconds=0.0,
            drift_seconds_per_hour=0.0,
            confidence=session.avg_confidence,
            is_reference=True,
            reference_camera_id=None,
            rationale=(
                "selected as sync reference for its group "
                f"(highest average timestamp confidence: {session.avg_confidence:.2f})"
            ),
        )

    offset_seconds = (reference.start - session.start).total_seconds()
    corrected_start = session.start + timedelta(seconds=offset_seconds)
    corrected_end = session.end + timedelta(seconds=offset_seconds)

    drift_seconds_per_hour = 0.0
    if session.duration_seconds > 0 and reference.duration_seconds > 0:
        span_delta = reference.duration_seconds - session.duration_seconds
        drift_seconds_per_hour = (span_delta / session.duration_seconds) * 3600.0
        corrected_end = corrected_end + timedelta(seconds=span_delta)

    confidence = round(min(session.avg_confidence, reference.avg_confidence) * 0.9, 6)
    rationale = (
        f"aligned to reference camera '{reference.camera_id}' by matching session start "
        f"(offset {offset_seconds:+.2f}s); drift estimated from proportional session-span "
        f"difference ({drift_seconds_per_hour:+.2f}s/hr). No cross-camera signal correlation "
        "was available, so this is a start-alignment heuristic, not a measured clock offset."
    )
    return SessionSyncResult(
        camera_id=session.camera_id,
        session_id=session.session_id,
        clip_ids=list(session.clip_ids),
        original_start=session.start,
        original_end=session.end,
        corrected_start=corrected_start,
        corrected_end=corrected_end,
        offset_seconds=offset_seconds,
        drift_seconds_per_hour=drift_seconds_per_hour,
        confidence=confidence,
        is_reference=False,
        reference_camera_id=reference.camera_id,
        rationale=rationale,
    )
