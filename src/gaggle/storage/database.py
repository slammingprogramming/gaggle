from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    delete,
    event,
    inspect,
    select,
)
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

from gaggle.schemas.camera import Camera
from gaggle.schemas.encounter import Encounter
from gaggle.schemas.event import EventRecord
from gaggle.schemas.media import MediaClip
from gaggle.schemas.recognition import (
    FaceCluster,
    FaceObservation,
    MergeSuggestion,
    PersonAppearanceCluster,
    PersonAppearanceObservation,
    PlateObservation,
    PlateRecord,
    VehicleAppearanceCluster,
    VehicleAppearanceObservation,
    VoiceCluster,
    VoiceObservation,
)
from gaggle.schemas.review import ReviewAction
from gaggle.storage.migrate import ensure_schema_up_to_date
from gaggle.utils.time import utc_now


class UTCDateTimeColumn(TypeDecorator[datetime]):
    """A ``DateTime`` column that guarantees a timezone-aware (UTC) value
    on both write and read, working around a well-known SQLAlchemy+SQLite
    limitation.

    SQLite has no native timestamp-with-timezone type. SQLAlchemy's
    ``DateTime(timezone=True)`` accepts and stores a timezone-aware value,
    but its SQLite dialect's read-back parser does not reconstruct the UTC
    offset -- every value comes back as a *naive* ``datetime``, regardless
    of what was written. Every timestamp field in this project's pydantic
    schemas (``gaggle.schemas.common.UTCDateTime``) requires a
    timezone-aware value, so a naive datetime read from this index and fed
    into one of those models fails validation. This was a real crash in
    production (``enrich`` reading a previously-stored ``FaceCluster``'s
    ``first_seen_at`` back out of SQLite), not a hypothetical.

    This type re-attaches UTC on every read, and normalizes to UTC on every
    write, so callers never need to think about it. Use this instead of
    ``DateTime(timezone=True)`` for every datetime column in this module.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # Every write site in this codebase passes timezone-aware
            # values already; this is a defensive fallback, not the
            # expected path, so treat naive input as already UTC rather
            # than raising deep inside a write call.
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class EventIndexRow(Base):
    """Index-only projection of an EventRecord for fast querying.

    This table is a query accelerator, never a source of truth: every field
    here is re-derived from the latest ``event.json`` revision on disk on
    every write. If this database were deleted entirely, it could be
    rebuilt in full from the filesystem alone (see
    ``gaggle.storage.repository.Repository.reindex``).
    """

    __tablename__ = "event_index"

    event_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_path: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    end_time: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    preservation_state: Mapped[str] = mapped_column(String, nullable=False)
    review_decision: Mapped[str] = mapped_column(String, nullable=False)
    camera_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cameras_csv: Mapped[str] = mapped_column(String, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class ReviewActionRow(Base):
    __tablename__ = "review_action"

    action_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    notes: Mapped[str] = mapped_column(String, nullable=False)


class MediaIndexRow(Base):
    __tablename__ = "media_index"

    clip_id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    stored_path: Mapped[str] = mapped_column(String, nullable=False)
    sha256: Mapped[str] = mapped_column(String, index=True, nullable=False)
    start_time: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    end_time: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    ingest_mode: Mapped[str] = mapped_column(String, nullable=False, default="copy")


class CameraRow(Base):
    """A registered camera source. See ``gaggle.schemas.camera.Camera``
    for the full field-by-field rationale. Never required to exist for
    the pipeline to work -- ``camera_id`` remains a free-form string
    everywhere else; this table only holds optional, user-editable
    metadata about it."""

    __tablename__ = "camera"

    camera_id: Mapped[str] = mapped_column(String, primary_key=True)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="other")
    indoor: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    site_id: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    notes: Mapped[str] = mapped_column(String, nullable=False, default="")


class TriageRow(Base):
    """Storage-lifecycle triage state for one clip. Re-derivable from events; see
    ``gaggle.schemas.lifecycle.TriageRecord``."""

    __tablename__ = "triage"

    clip_id: Mapped[str] = mapped_column(String, primary_key=True)
    camera_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    state: Mapped[str] = mapped_column(String, index=True, nullable=False)
    signal_count: Mapped[int] = mapped_column(Integer, nullable=False)
    event_ids_csv: Mapped[str] = mapped_column(String, nullable=False, default="")
    classified_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    reason: Mapped[str] = mapped_column(String, nullable=False)


class FaceClusterRow(Base):
    """Recognition data (see ``docs/architecture.md``'s storage-lifecycle
    section for why this tier is stored authoritatively in SQLite + small
    crop files rather than as filesystem JSON-per-record like events)."""

    __tablename__ = "face_cluster"

    cluster_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    representative_crops_csv: Mapped[str] = mapped_column(String, nullable=False, default="")
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    representative_observation_ids_csv: Mapped[str] = mapped_column(
        String, nullable=False, default=""
    )


class FaceObservationRow(Base):
    __tablename__ = "face_observation"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    crop_path: Mapped[str] = mapped_column(String, nullable=False)
    crop_sha256: Mapped[str] = mapped_column(String, nullable=False)
    detector_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_distance_to_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    detector_version: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_of_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    review_status: Mapped[str] = mapped_column(
        String, nullable=False, default="needs_review", index=True
    )
    crop_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)


class PlateRecordRow(Base):
    __tablename__ = "plate_record"

    plate_id: Mapped[str] = mapped_column(String, primary_key=True)
    normalized_text: Mapped[str] = mapped_column(String, nullable=False, index=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    example_crops_csv: Mapped[str] = mapped_column(String, nullable=False, default="")
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True, index=True)


class PlateObservationRow(Base):
    __tablename__ = "plate_observation"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    crop_path: Mapped[str] = mapped_column(String, nullable=False)
    crop_sha256: Mapped[str] = mapped_column(String, nullable=False)
    raw_ocr_text: Mapped[str] = mapped_column(String, nullable=False)
    normalized_text: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ocr_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    detector_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    review_status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_corrected_text: Mapped[str | None] = mapped_column(String, nullable=True)
    detector_version: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_of_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    crop_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)


class VoiceClusterRow(Base):
    __tablename__ = "voice_cluster"

    cluster_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True, index=True)


class VoiceObservationRow(Base):
    __tablename__ = "voice_observation"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    segment_start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    segment_end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    voiceprint_json: Mapped[str] = mapped_column(String, nullable=False)
    energy_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    detector_version: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_of_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)


class VehicleAppearanceClusterRow(Base):
    """Mirrors `FaceClusterRow` (crops kept, unlike voice) -- see
    `enrichment/vehicle_appearance.py`'s module docstring."""

    __tablename__ = "vehicle_appearance_cluster"

    cluster_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    representative_crops_csv: Mapped[str] = mapped_column(String, nullable=False, default="")
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    representative_observation_ids_csv: Mapped[str] = mapped_column(
        String, nullable=False, default=""
    )


class VehicleAppearanceObservationRow(Base):
    __tablename__ = "vehicle_appearance_observation"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    crop_path: Mapped[str] = mapped_column(String, nullable=False)
    crop_sha256: Mapped[str] = mapped_column(String, nullable=False)
    fingerprint_json: Mapped[str] = mapped_column(String, nullable=False)
    detector_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_distance_to_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    detector_version: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_of_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    review_status: Mapped[str] = mapped_column(
        String, nullable=False, default="needs_review", index=True
    )
    crop_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)


class PersonAppearanceClusterRow(Base):
    """Mirrors `VehicleAppearanceClusterRow` exactly -- see
    `enrichment/person_appearance.py`'s module docstring."""

    __tablename__ = "person_appearance_cluster"

    cluster_id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    label: Mapped[str | None] = mapped_column(String, nullable=True)
    representative_crops_csv: Mapped[str] = mapped_column(String, nullable=False, default="")
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    merged_into: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    representative_observation_ids_csv: Mapped[str] = mapped_column(
        String, nullable=False, default=""
    )


class PersonAppearanceObservationRow(Base):
    """Mirrors `VehicleAppearanceObservationRow` exactly."""

    __tablename__ = "person_appearance_observation"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_id: Mapped[str] = mapped_column(String, nullable=False)
    event_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    crop_path: Mapped[str] = mapped_column(String, nullable=False)
    crop_sha256: Mapped[str] = mapped_column(String, nullable=False)
    fingerprint_json: Mapped[str] = mapped_column(String, nullable=False)
    detector_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_distance_to_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    cluster_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    detector_version: Mapped[str] = mapped_column(String, nullable=False)
    duplicate_of_observation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    review_status: Mapped[str] = mapped_column(
        String, nullable=False, default="needs_review", index=True
    )
    crop_purged_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)


class EncounterRow(Base):
    """A derived cross-modality grouping of observations. See
    ``schemas/encounter.py::Encounter`` for the full scope and the
    explicit no-spatial-correspondence caveat -- this table only ever
    stores what ``enrichment/service.py::EnrichmentService._derive_encounters``
    computed; nothing here is ever user-edited."""

    __tablename__ = "encounter"

    encounter_id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    clip_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    face_observation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    plate_observation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    voice_observation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    vehicle_appearance_observation_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    person_appearance_observation_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )


class MergeSuggestionRow(Base):
    """An automated merge suggestion awaiting human confirmation or rejection.
    See ``schemas/recognition.py::MergeSuggestion`` for the full design
    rationale."""

    __tablename__ = "merge_suggestion"

    suggestion_id: Mapped[str] = mapped_column(String, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    target_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    basis: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTimeColumn(), nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTimeColumn(), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String, nullable=True)


@dataclass(frozen=True, slots=True)
class TimelineQuery:
    """Filters for ``TimelineDatabase.query_events``. All filters are ANDed."""

    severity: str | None = None
    camera_id: str | None = None
    review_decision: str | None = None
    preservation_state: str | None = None
    start_after: datetime | None = None
    start_before: datetime | None = None
    limit: int | None = None


def _set_sqlite_pragmas(dbapi_connection: object, _connection_record: object) -> None:
    """Every enrichment write (one cluster upsert + one observation insert
    per detection, one row per encounter) is its own short-lived session
    that commits independently -- by design, see `session()`'s docstring.
    SQLite's *default* journal mode does a full fsync on every one of
    those commits; on a slow/external/network drive that's a real,
    measured bottleneck (a single real event with ~1300 vehicle-appearance
    detections took several minutes of wall-clock time that turned out to
    be almost entirely commit-fsync overhead, not model inference -- see
    `enrichment/service.py`'s per-capability timing). WAL mode moves each
    commit to a sequential log-file append instead of an in-place fsync,
    and `synchronous=NORMAL` (safe and recommended by SQLite's own docs
    when paired with WAL) only risks losing the most recent commits on an
    OS-level crash or power loss, never corruption -- an acceptable
    tradeoff here since this database is a rebuildable index, not primary
    evidence (see `docs/architecture.md` and invariant 5 in AGENTS.md);
    the worst case after such a crash is `enrich` redoing a capability
    that hadn't reached its `enrichment_completed` marker yet, which it
    already does safely by design."""

    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


class TimelineDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.engine = create_engine(f"sqlite:///{path}", future=True)
        event.listens_for(self.engine, "connect")(_set_sqlite_pragmas)
        self._session_factory = sessionmaker(self.engine, future=True, expire_on_commit=False)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ensure_schema_up_to_date(self.engine, self.path)

    def close(self) -> None:
        """Dispose the engine's connection pool.

        Required before deleting the underlying sqlite file out from under
        this instance (see ``workspace reindex --rebuild`` in `cli/app.py`)
        -- on Windows in particular, an open pooled connection will keep
        the file locked and block ``Path.unlink()`` otherwise.
        """

        self.engine.dispose()

    def check_schema_drift(self) -> list[tuple[str, list[str]]]:
        """Compare every ``Row`` model's expected columns against what
        actually exists in this sqlite file.

        ``initialize()`` only ever calls ``create_all()``, which creates a
        missing table but never alters an existing one -- a workspace from
        an older gaggle version that gains a new column on an
        already-existing table would otherwise hit a bare
        ``sqlite3.OperationalError: no such column`` on first query,
        with no forewarning. Returns ``(table_name, missing_column_names)``
        for every table with a mismatch; an empty list means no drift.
        Deliberately not called from ``initialize()`` itself (which runs on
        nearly every CLI invocation) -- this is an explicit, on-demand
        diagnostic, run via ``workspace reindex``.
        """

        inspector = inspect(self.engine)
        drift: list[tuple[str, list[str]]] = []
        for table_name, table in Base.metadata.tables.items():
            if not inspector.has_table(table_name):
                drift.append((table_name, sorted(table.columns.keys())))
                continue
            actual_columns = {column["name"] for column in inspector.get_columns(table_name)}
            expected_columns = set(table.columns.keys())
            missing = sorted(expected_columns - actual_columns)
            if missing:
                drift.append((table_name, missing))
        return drift

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Every ``list_*``/``get_*`` method below returns ORM row objects
        for the caller to read *after* this context manager has already
        closed the session (that's the whole point of the pattern -- one
        short-lived session per call). ``expire_on_commit=False`` on the
        session factory is what makes that safe: without it, SQLAlchemy
        expires every loaded attribute on commit, and reading an expired
        attribute on a detached (session-closed) instance raises
        ``DetachedInstanceError``. Do not remove that setting."""

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def upsert_event(self, event: EventRecord, event_path: Path) -> None:
        row = EventIndexRow(
            event_id=str(event.event_id),
            event_path=str(event_path),
            start_time=event.event_start,
            end_time=event.event_end,
            severity=event.scoring.severity,
            confidence=event.scoring.confidence,
            preservation_state=event.preservation_status.state,
            review_decision=event.review_summary.latest_decision,
            camera_count=len(event.involved_cameras),
            cameras_csv=",".join(event.involved_cameras),
            revision=event.revision,
        )
        with self.session() as session:
            session.merge(row)

    def upsert_media(self, clip: MediaClip) -> None:
        row = MediaIndexRow(
            clip_id=str(clip.clip_id),
            camera_id=clip.camera_id,
            stored_path=clip.stored_path,
            sha256=clip.sha256,
            start_time=clip.observed_start,
            end_time=clip.observed_end,
            ingest_mode=clip.ingest_mode,
        )
        with self.session() as session:
            session.merge(row)

    def register_camera_if_absent(self, camera: Camera) -> None:
        """Insert `camera` only if no row for this camera_id already
        exists -- never overwrites a user's already-registered richer
        metadata (including a manually-edited `site_id`). Used by
        `ingest/service.py`'s auto-registration on first-seen camera_id."""

        with self.session() as session:
            existing = session.get(CameraRow, camera.camera_id)
            if existing is not None:
                return
            session.add(
                CameraRow(
                    camera_id=camera.camera_id,
                    label=camera.label,
                    source_type=camera.source_type,
                    indoor=camera.indoor,
                    site_id=camera.site_id,
                    created_at=camera.created_at,
                    notes=camera.notes,
                )
            )

    def upsert_camera(self, camera: Camera) -> None:
        """Explicit, unconditional write -- used by `camera register`,
        where overwriting is the whole point (unlike auto-registration)."""

        row = CameraRow(
            camera_id=camera.camera_id,
            label=camera.label,
            source_type=camera.source_type,
            indoor=camera.indoor,
            site_id=camera.site_id,
            created_at=camera.created_at,
            notes=camera.notes,
        )
        with self.session() as session:
            session.merge(row)

    def get_camera(self, camera_id: str) -> CameraRow | None:
        with self.session() as session:
            return session.get(CameraRow, camera_id)

    def list_cameras(self) -> list[CameraRow]:
        with self.session() as session:
            rows = session.execute(select(CameraRow).order_by(CameraRow.camera_id))
            return list(rows.scalars())

    def site_id_by_camera(self) -> dict[str, str]:
        """Every registered camera's `site_id`, for `normalize/sync.py`'s
        site-scoped grouping. A camera with no `site_id` set is simply
        absent from this mapping -- callers treat that as "its own site"."""

        with self.session() as session:
            rows = session.execute(
                select(CameraRow.camera_id, CameraRow.site_id).where(CameraRow.site_id.is_not(None))
            )
            return {camera_id: site_id for camera_id, site_id in rows if site_id is not None}

    def append_review_action(self, action: ReviewAction) -> None:
        with self.session() as session:
            session.add(
                ReviewActionRow(
                    action_id=str(action.action_id),
                    event_id=str(action.event_id),
                    action=action.action,
                    actor=action.actor,
                    timestamp=action.timestamp,
                    notes=action.notes,
                )
            )

    def list_events(self) -> list[EventIndexRow]:
        return self.query_events(TimelineQuery())

    def query_events(self, query: TimelineQuery) -> list[EventIndexRow]:
        statement = select(EventIndexRow).order_by(EventIndexRow.start_time)
        if query.severity is not None:
            statement = statement.where(EventIndexRow.severity == query.severity)
        if query.review_decision is not None:
            statement = statement.where(EventIndexRow.review_decision == query.review_decision)
        if query.preservation_state is not None:
            statement = statement.where(
                EventIndexRow.preservation_state == query.preservation_state
            )
        if query.start_after is not None:
            statement = statement.where(EventIndexRow.start_time >= query.start_after)
        if query.start_before is not None:
            statement = statement.where(EventIndexRow.start_time <= query.start_before)
        if query.camera_id is not None:
            # cameras_csv is a small denormalized list; SQLite LIKE keeps this a
            # single indexed-table scan without a join table for the common case
            # of a handful of cameras per vehicle.
            statement = statement.where(EventIndexRow.cameras_csv.contains(query.camera_id))
        if query.limit is not None:
            statement = statement.limit(query.limit)
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_review_actions(self, event_id: UUID) -> list[ReviewActionRow]:
        with self.session() as session:
            rows = session.execute(
                select(ReviewActionRow)
                .where(ReviewActionRow.event_id == str(event_id))
                .order_by(ReviewActionRow.timestamp)
            ).scalars()
            return list(rows)

    def list_media(self) -> list[MediaIndexRow]:
        with self.session() as session:
            rows = session.execute(select(MediaIndexRow).order_by(MediaIndexRow.start_time))
            return list(rows.scalars())

    def get_media(self, clip_id: UUID) -> MediaIndexRow | None:
        with self.session() as session:
            return session.get(MediaIndexRow, str(clip_id))

    def get_media_by_sha256(self, sha256: str) -> MediaIndexRow | None:
        with self.session() as session:
            return (
                session.execute(select(MediaIndexRow).where(MediaIndexRow.sha256 == sha256))
                .scalars()
                .first()
            )

    def update_camera(
        self,
        camera_id: str,
        label: str | None = None,
        source_type: str | None = None,
        indoor: bool | None = None,
        site_id: str | None = None,
        notes: str | None = None,
    ) -> CameraRow | None:
        """Explicit user edit via `camera update` -- only the fields
        actually passed (non-None) are changed; the rest keep their
        existing value. Returns None if no camera is registered under
        this camera_id yet (the caller should suggest `camera register`
        instead)."""

        with self.session() as session:
            row = session.get(CameraRow, camera_id)
            if row is None:
                return None
            if label is not None:
                row.label = label
            if source_type is not None:
                row.source_type = source_type
            if indoor is not None:
                row.indoor = indoor
            if site_id is not None:
                row.site_id = site_id
            if notes is not None:
                row.notes = notes
            return row

    def update_media_location(self, clip_id: UUID, new_stored_path: str) -> None:
        with self.session() as session:
            row = session.get(MediaIndexRow, str(clip_id))
            if row is not None:
                row.stored_path = new_stored_path

    def update_media_ingest_mode(
        self, clip_id: UUID, new_stored_path: str, new_ingest_mode: str
    ) -> None:
        """Update both `stored_path` and `ingest_mode` in one transaction --
        used by `TriageService.convert_ingest_mode`, where the two must
        always change together (an inconsistent partial update would leave
        the row claiming a mode that doesn't match where the bytes
        actually live)."""

        with self.session() as session:
            row = session.get(MediaIndexRow, str(clip_id))
            if row is not None:
                row.stored_path = new_stored_path
                row.ingest_mode = new_ingest_mode

    # -- triage -------------------------------------------------------------

    def upsert_triage(
        self,
        clip_id: UUID,
        camera_id: str,
        state: str,
        signal_count: int,
        event_ids: list[UUID],
        classified_at: datetime,
        reason: str,
    ) -> None:
        row = TriageRow(
            clip_id=str(clip_id),
            camera_id=camera_id,
            state=state,
            signal_count=signal_count,
            event_ids_csv=",".join(str(e) for e in event_ids),
            classified_at=classified_at,
            reason=reason,
        )
        with self.session() as session:
            session.merge(row)

    def list_triage(self, state: str | None = None) -> list[TriageRow]:
        statement = select(TriageRow).order_by(TriageRow.classified_at)
        if state is not None:
            statement = statement.where(TriageRow.state == state)
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def get_triage(self, clip_id: UUID) -> TriageRow | None:
        with self.session() as session:
            return session.get(TriageRow, str(clip_id))

    # -- face recognition -----------------------------------------------------

    def upsert_face_cluster(self, cluster: FaceCluster) -> None:
        row = FaceClusterRow(
            cluster_id=str(cluster.cluster_id),
            created_at=cluster.created_at,
            updated_at=cluster.updated_at,
            label=cluster.label,
            representative_crops_csv=",".join(cluster.representative_crop_paths),
            observation_count=cluster.observation_count,
            first_seen_at=cluster.first_seen_at,
            last_seen_at=cluster.last_seen_at,
            model_version=cluster.model_version,
            merged_into=str(cluster.merged_into) if cluster.merged_into else None,
        )
        with self.session() as session:
            session.merge(row)

    def insert_face_observation(self, observation: FaceObservation) -> None:
        row = FaceObservationRow(
            observation_id=str(observation.observation_id),
            signal_id=str(observation.signal_id),
            event_id=str(observation.event_id) if observation.event_id else None,
            clip_id=str(observation.clip_id),
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            crop_path=observation.crop_path,
            crop_sha256=observation.crop_sha256,
            detector_confidence=observation.detector_confidence,
            embedding_distance_to_cluster=observation.embedding_distance_to_cluster,
            cluster_id=str(observation.cluster_id) if observation.cluster_id else None,
            detector_version=observation.detector_version,
            duplicate_of_observation_id=(
                str(observation.duplicate_of_observation_id)
                if observation.duplicate_of_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def get_face_observation(self, observation_id: UUID) -> FaceObservationRow | None:
        with self.session() as session:
            return session.get(FaceObservationRow, str(observation_id))

    def list_face_clusters(self) -> list[FaceClusterRow]:
        with self.session() as session:
            rows = session.execute(select(FaceClusterRow).order_by(FaceClusterRow.last_seen_at))
            return list(rows.scalars())

    def get_face_cluster(self, cluster_id: UUID) -> FaceClusterRow | None:
        with self.session() as session:
            return session.get(FaceClusterRow, str(cluster_id))

    def search_face_clusters(self, query: str) -> list[FaceClusterRow]:
        """Substring match on cluster id or label. Case-insensitive."""

        pattern = f"%{query}%"
        with self.session() as session:
            rows = session.execute(
                select(FaceClusterRow)
                .where(
                    FaceClusterRow.cluster_id.ilike(pattern) | FaceClusterRow.label.ilike(pattern)
                )
                .order_by(FaceClusterRow.last_seen_at)
            )
            return list(rows.scalars())

    def list_face_observations(
        self, cluster_id: UUID, include_duplicates: bool = True
    ) -> list[FaceObservationRow]:
        statement = (
            select(FaceObservationRow)
            .where(FaceObservationRow.cluster_id == str(cluster_id))
            .order_by(FaceObservationRow.observed_at)
        )
        if not include_duplicates:
            statement = statement.where(FaceObservationRow.duplicate_of_observation_id.is_(None))
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_all_face_observations(self) -> list[FaceObservationRow]:
        with self.session() as session:
            rows = session.execute(
                select(FaceObservationRow).order_by(FaceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def list_face_observations_for_event(self, event_id: UUID) -> list[FaceObservationRow]:
        """Event-scoped, unlike `list_face_observations` (cluster-scoped) --
        used by the review UI's enrichment panel, which wants every
        observation attached to one event regardless of which cluster it
        ended up in."""

        with self.session() as session:
            rows = session.execute(
                select(FaceObservationRow)
                .where(FaceObservationRow.event_id == str(event_id))
                .order_by(FaceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def mark_face_observation_duplicate(self, observation_id: UUID, duplicate_of: UUID) -> None:
        with self.session() as session:
            row = session.get(FaceObservationRow, str(observation_id))
            if row is not None:
                row.duplicate_of_observation_id = str(duplicate_of)

    def set_face_cluster_label(self, cluster_id: UUID, label: str) -> None:
        with self.session() as session:
            row = session.get(FaceClusterRow, str(cluster_id))
            if row is not None:
                row.label = label

    def set_face_cluster_merge(self, source_cluster_id: UUID, target_cluster_id: UUID) -> None:
        with self.session() as session:
            row = session.get(FaceClusterRow, str(source_cluster_id))
            if row is not None:
                row.merged_into = str(target_cluster_id)

    def set_face_observation_review_status(self, observation_id: UUID, status: str) -> None:
        with self.session() as session:
            row = session.get(FaceObservationRow, str(observation_id))
            if row is not None:
                row.review_status = status

    def set_face_cluster_representative(
        self,
        cluster_id: UUID,
        observation_ids: list[UUID],
        representative_crop_paths: list[str],
    ) -> None:
        with self.session() as session:
            row = session.get(FaceClusterRow, str(cluster_id))
            if row is not None:
                row.representative_observation_ids_csv = ",".join(str(o) for o in observation_ids)
                row.representative_crops_csv = ",".join(representative_crop_paths)

    def set_face_observation_cluster(self, observation_id: UUID, cluster_id: UUID | None) -> None:
        """Reassign (or clear, if `cluster_id` is `None`) which cluster an
        observation belongs to -- the primitive `detach_observation`/
        `move_observation` in `core/recognition.py` build on. The
        observation row itself is never deleted; only its `cluster_id`
        changes."""

        with self.session() as session:
            row = session.get(FaceObservationRow, str(observation_id))
            if row is not None:
                row.cluster_id = str(cluster_id) if cluster_id is not None else None

    def set_face_cluster_observation_count(self, cluster_id: UUID, observation_count: int) -> None:
        with self.session() as session:
            row = session.get(FaceClusterRow, str(cluster_id))
            if row is not None:
                row.observation_count = observation_count

    def mark_face_observation_crop_purged(self, observation_id: UUID) -> None:
        with self.session() as session:
            row = session.get(FaceObservationRow, str(observation_id))
            if row is not None:
                row.crop_purged_at = utc_now()

    def list_face_observations_by_cluster_ids(
        self, cluster_ids: list[UUID]
    ) -> list[FaceObservationRow]:
        if not cluster_ids:
            return []
        ids = [str(c) for c in cluster_ids]
        with self.session() as session:
            rows = session.execute(
                select(FaceObservationRow).where(FaceObservationRow.cluster_id.in_(ids))
            )
            return list(rows.scalars())

    def list_face_observations_eligible_for_purge(self) -> list[FaceObservationRow]:
        """Already reviewed (confirmed or rejected), not yet purged, and --
        for a confirmed observation -- not one of its cluster's chosen
        representatives. A rejected observation has no cluster concept to
        check (it's being purged specifically because it was judged not a
        face at all)."""

        with self.session() as session:
            candidates = session.execute(
                select(FaceObservationRow).where(
                    FaceObservationRow.review_status.in_(("user_confirmed", "user_rejected")),
                    FaceObservationRow.crop_purged_at.is_(None),
                )
            ).scalars()
            representative_ids_by_cluster = {
                row.cluster_id: set(row.representative_observation_ids_csv.split(","))
                for row in session.execute(select(FaceClusterRow)).scalars()
                if row.representative_observation_ids_csv
            }
            return [
                observation
                for observation in candidates
                if observation.review_status == "user_rejected"
                or observation.observation_id
                not in representative_ids_by_cluster.get(observation.cluster_id or "", set())
            ]

    # -- plate recognition ----------------------------------------------------

    def upsert_plate_record(self, record: PlateRecord) -> None:
        row = PlateRecordRow(
            plate_id=str(record.plate_id),
            normalized_text=record.normalized_text,
            created_at=record.created_at,
            updated_at=record.updated_at,
            label=record.label,
            observation_count=record.observation_count,
            first_seen_at=record.first_seen_at,
            last_seen_at=record.last_seen_at,
            example_crops_csv=",".join(ref.path for ref in record.example_crops),
            merged_into=str(record.merged_into) if record.merged_into else None,
        )
        with self.session() as session:
            session.merge(row)

    def get_plate_record_by_text(self, normalized_text: str) -> PlateRecordRow | None:
        with self.session() as session:
            return session.execute(
                select(PlateRecordRow).where(PlateRecordRow.normalized_text == normalized_text)
            ).scalar_one_or_none()

    def get_plate_record(self, plate_id: UUID) -> PlateRecordRow | None:
        with self.session() as session:
            return session.get(PlateRecordRow, str(plate_id))

    def search_plate_records(self, query: str) -> list[PlateRecordRow]:
        """Substring match on normalized plate text, plate id, or label."""

        pattern = f"%{query.upper()}%"
        with self.session() as session:
            rows = session.execute(
                select(PlateRecordRow)
                .where(
                    PlateRecordRow.normalized_text.ilike(pattern)
                    | PlateRecordRow.plate_id.ilike(pattern)
                    | PlateRecordRow.label.ilike(pattern)
                )
                .order_by(PlateRecordRow.last_seen_at)
            )
            return list(rows.scalars())

    def set_plate_record_merge(self, source_plate_id: UUID, target_plate_id: UUID) -> None:
        with self.session() as session:
            row = session.get(PlateRecordRow, str(source_plate_id))
            if row is not None:
                row.merged_into = str(target_plate_id)

    def insert_plate_observation(self, observation: PlateObservation) -> None:
        row = PlateObservationRow(
            observation_id=str(observation.observation_id),
            signal_id=str(observation.signal_id),
            event_id=str(observation.event_id) if observation.event_id else None,
            clip_id=str(observation.clip_id),
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            crop_path=observation.crop_path,
            crop_sha256=observation.crop_sha256,
            raw_ocr_text=observation.raw_ocr_text,
            normalized_text=observation.normalized_text,
            ocr_confidence=observation.ocr_confidence,
            detector_confidence=observation.detector_confidence,
            review_status=observation.review_status,
            user_corrected_text=observation.user_corrected_text,
            detector_version=observation.detector_version,
            duplicate_of_observation_id=(
                str(observation.duplicate_of_observation_id)
                if observation.duplicate_of_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def get_plate_observation(self, observation_id: UUID) -> PlateObservationRow | None:
        with self.session() as session:
            return session.get(PlateObservationRow, str(observation_id))

    def list_plate_records(self) -> list[PlateRecordRow]:
        with self.session() as session:
            rows = session.execute(select(PlateRecordRow).order_by(PlateRecordRow.last_seen_at))
            return list(rows.scalars())

    def list_plate_observations(
        self, normalized_text: str | None = None, review_status: str | None = None
    ) -> list[PlateObservationRow]:
        statement = select(PlateObservationRow).order_by(PlateObservationRow.observed_at)
        if normalized_text is not None:
            statement = statement.where(PlateObservationRow.normalized_text == normalized_text)
        if review_status is not None:
            statement = statement.where(PlateObservationRow.review_status == review_status)
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_plate_observations_for_event(self, event_id: UUID) -> list[PlateObservationRow]:
        """Event-scoped, unlike `list_plate_observations` (filtered by
        text/status) -- used by the review UI's enrichment panel."""

        with self.session() as session:
            rows = session.execute(
                select(PlateObservationRow)
                .where(PlateObservationRow.event_id == str(event_id))
                .order_by(PlateObservationRow.observed_at)
            )
            return list(rows.scalars())

    def confirm_plate_observation(self, observation_id: UUID, corrected_text: str) -> None:
        with self.session() as session:
            row = session.get(PlateObservationRow, str(observation_id))
            if row is not None:
                row.user_corrected_text = corrected_text
                row.review_status = "user_confirmed"

    def mark_plate_observation_duplicate_suppressed(
        self, observation_id: UUID, duplicate_of: UUID
    ) -> None:
        with self.session() as session:
            row = session.get(PlateObservationRow, str(observation_id))
            if row is not None:
                row.review_status = "duplicate_suppressed"
                row.duplicate_of_observation_id = str(duplicate_of)

    def mark_plate_observation_rejected(self, observation_id: UUID) -> None:
        with self.session() as session:
            row = session.get(PlateObservationRow, str(observation_id))
            if row is not None:
                row.review_status = "user_rejected"

    def mark_plate_observation_crop_purged(self, observation_id: UUID) -> None:
        with self.session() as session:
            row = session.get(PlateObservationRow, str(observation_id))
            if row is not None:
                row.crop_purged_at = utc_now()

    def list_plate_observations_eligible_for_purge(self) -> list[PlateObservationRow]:
        """Plates have no cluster/representative-crop concept -- any
        observation already reviewed (confirmed or rejected, i.e. not
        still `needs_review`/`auto_accepted`, and not already marked
        `duplicate_suppressed`, which is an automated bookkeeping state
        with its own separate meaning) and not yet purged is eligible."""

        with self.session() as session:
            rows = session.execute(
                select(PlateObservationRow).where(
                    PlateObservationRow.review_status.in_(("user_confirmed", "user_rejected")),
                    PlateObservationRow.crop_purged_at.is_(None),
                )
            )
            return list(rows.scalars())

    # -- voices -------------------------------------------------------------

    def upsert_voice_cluster(self, cluster: VoiceCluster) -> None:
        row = VoiceClusterRow(
            cluster_id=str(cluster.cluster_id),
            created_at=cluster.created_at,
            updated_at=cluster.updated_at,
            label=cluster.label,
            observation_count=cluster.observation_count,
            first_seen_at=cluster.first_seen_at,
            last_seen_at=cluster.last_seen_at,
            model_version=cluster.model_version,
            merged_into=str(cluster.merged_into) if cluster.merged_into else None,
        )
        with self.session() as session:
            session.merge(row)

    def list_voice_clusters(self) -> list[VoiceClusterRow]:
        with self.session() as session:
            rows = session.execute(select(VoiceClusterRow).order_by(VoiceClusterRow.last_seen_at))
            return list(rows.scalars())

    def get_voice_cluster(self, cluster_id: UUID) -> VoiceClusterRow | None:
        with self.session() as session:
            return session.get(VoiceClusterRow, str(cluster_id))

    def search_voice_clusters(self, query: str) -> list[VoiceClusterRow]:
        pattern = f"%{query}%"
        with self.session() as session:
            rows = session.execute(
                select(VoiceClusterRow)
                .where(
                    VoiceClusterRow.cluster_id.ilike(pattern) | VoiceClusterRow.label.ilike(pattern)
                )
                .order_by(VoiceClusterRow.last_seen_at)
            )
            return list(rows.scalars())

    def set_voice_cluster_label(self, cluster_id: UUID, label: str) -> None:
        with self.session() as session:
            row = session.get(VoiceClusterRow, str(cluster_id))
            if row is not None:
                row.label = label

    def set_voice_cluster_merge(self, source_cluster_id: UUID, target_cluster_id: UUID) -> None:
        with self.session() as session:
            row = session.get(VoiceClusterRow, str(source_cluster_id))
            if row is not None:
                row.merged_into = str(target_cluster_id)

    def insert_voice_observation(self, observation: VoiceObservation) -> None:
        row = VoiceObservationRow(
            observation_id=str(observation.observation_id),
            signal_id=str(observation.signal_id),
            event_id=str(observation.event_id) if observation.event_id else None,
            clip_id=str(observation.clip_id),
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            segment_start_seconds=observation.segment_start_seconds,
            segment_end_seconds=observation.segment_end_seconds,
            voiceprint_json=json.dumps(observation.voiceprint),
            energy_confidence=observation.energy_confidence,
            cluster_id=str(observation.cluster_id) if observation.cluster_id else None,
            detector_version=observation.detector_version,
            duplicate_of_observation_id=(
                str(observation.duplicate_of_observation_id)
                if observation.duplicate_of_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def list_voice_observations(
        self, cluster_id: UUID, include_duplicates: bool = True
    ) -> list[VoiceObservationRow]:
        statement = (
            select(VoiceObservationRow)
            .where(VoiceObservationRow.cluster_id == str(cluster_id))
            .order_by(VoiceObservationRow.observed_at)
        )
        if not include_duplicates:
            statement = statement.where(VoiceObservationRow.duplicate_of_observation_id.is_(None))
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_all_voice_observations(self) -> list[VoiceObservationRow]:
        with self.session() as session:
            rows = session.execute(
                select(VoiceObservationRow).order_by(VoiceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def list_voice_observations_for_event(self, event_id: UUID) -> list[VoiceObservationRow]:
        """Event-scoped, unlike `list_voice_observations` (cluster-scoped) --
        used by the review UI's enrichment panel."""

        with self.session() as session:
            rows = session.execute(
                select(VoiceObservationRow)
                .where(VoiceObservationRow.event_id == str(event_id))
                .order_by(VoiceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def mark_voice_observation_duplicate(self, observation_id: UUID, duplicate_of: UUID) -> None:
        with self.session() as session:
            row = session.get(VoiceObservationRow, str(observation_id))
            if row is not None:
                row.duplicate_of_observation_id = str(duplicate_of)

    # -- vehicle appearance ---------------------------------------------------

    def upsert_vehicle_appearance_cluster(self, cluster: VehicleAppearanceCluster) -> None:
        row = VehicleAppearanceClusterRow(
            cluster_id=str(cluster.cluster_id),
            created_at=cluster.created_at,
            updated_at=cluster.updated_at,
            label=cluster.label,
            representative_crops_csv=",".join(cluster.representative_crop_paths),
            observation_count=cluster.observation_count,
            first_seen_at=cluster.first_seen_at,
            last_seen_at=cluster.last_seen_at,
            model_version=cluster.model_version,
            merged_into=str(cluster.merged_into) if cluster.merged_into else None,
        )
        with self.session() as session:
            session.merge(row)

    def list_vehicle_appearance_clusters(self) -> list[VehicleAppearanceClusterRow]:
        with self.session() as session:
            rows = session.execute(
                select(VehicleAppearanceClusterRow).order_by(
                    VehicleAppearanceClusterRow.last_seen_at
                )
            )
            return list(rows.scalars())

    def get_vehicle_appearance_cluster(
        self, cluster_id: UUID
    ) -> VehicleAppearanceClusterRow | None:
        with self.session() as session:
            return session.get(VehicleAppearanceClusterRow, str(cluster_id))

    def search_vehicle_appearance_clusters(self, query: str) -> list[VehicleAppearanceClusterRow]:
        pattern = f"%{query}%"
        with self.session() as session:
            rows = session.execute(
                select(VehicleAppearanceClusterRow)
                .where(
                    VehicleAppearanceClusterRow.cluster_id.ilike(pattern)
                    | VehicleAppearanceClusterRow.label.ilike(pattern)
                )
                .order_by(VehicleAppearanceClusterRow.last_seen_at)
            )
            return list(rows.scalars())

    def set_vehicle_appearance_cluster_label(self, cluster_id: UUID, label: str) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.label = label

    def set_vehicle_appearance_cluster_merge(
        self, source_cluster_id: UUID, target_cluster_id: UUID
    ) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceClusterRow, str(source_cluster_id))
            if row is not None:
                row.merged_into = str(target_cluster_id)

    def set_vehicle_appearance_observation_review_status(
        self, observation_id: UUID, status: str
    ) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.review_status = status

    def set_vehicle_appearance_cluster_representative(
        self,
        cluster_id: UUID,
        observation_ids: list[UUID],
        representative_crop_paths: list[str],
    ) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.representative_observation_ids_csv = ",".join(str(o) for o in observation_ids)
                row.representative_crops_csv = ",".join(representative_crop_paths)

    def set_vehicle_appearance_observation_cluster(
        self, observation_id: UUID, cluster_id: UUID | None
    ) -> None:
        """Reassign (or clear, if `cluster_id` is `None`) which cluster an
        observation belongs to -- mirrors `set_face_observation_cluster`."""

        with self.session() as session:
            row = session.get(VehicleAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.cluster_id = str(cluster_id) if cluster_id is not None else None

    def set_vehicle_appearance_cluster_observation_count(
        self, cluster_id: UUID, observation_count: int
    ) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.observation_count = observation_count

    def mark_vehicle_appearance_observation_crop_purged(self, observation_id: UUID) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.crop_purged_at = utc_now()

    def list_vehicle_appearance_observations_by_cluster_ids(
        self, cluster_ids: list[UUID]
    ) -> list[VehicleAppearanceObservationRow]:
        if not cluster_ids:
            return []
        ids = [str(c) for c in cluster_ids]
        with self.session() as session:
            rows = session.execute(
                select(VehicleAppearanceObservationRow).where(
                    VehicleAppearanceObservationRow.cluster_id.in_(ids)
                )
            )
            return list(rows.scalars())

    def list_vehicle_appearance_observations_eligible_for_purge(
        self,
    ) -> list[VehicleAppearanceObservationRow]:
        """Mirrors `list_face_observations_eligible_for_purge` exactly."""

        with self.session() as session:
            candidates = session.execute(
                select(VehicleAppearanceObservationRow).where(
                    VehicleAppearanceObservationRow.review_status.in_(
                        ("user_confirmed", "user_rejected")
                    ),
                    VehicleAppearanceObservationRow.crop_purged_at.is_(None),
                )
            ).scalars()
            representative_ids_by_cluster = {
                row.cluster_id: set(row.representative_observation_ids_csv.split(","))
                for row in session.execute(select(VehicleAppearanceClusterRow)).scalars()
                if row.representative_observation_ids_csv
            }
            return [
                observation
                for observation in candidates
                if observation.review_status == "user_rejected"
                or observation.observation_id
                not in representative_ids_by_cluster.get(observation.cluster_id or "", set())
            ]

    def insert_vehicle_appearance_observation(
        self, observation: VehicleAppearanceObservation
    ) -> None:
        row = VehicleAppearanceObservationRow(
            observation_id=str(observation.observation_id),
            signal_id=str(observation.signal_id),
            event_id=str(observation.event_id) if observation.event_id else None,
            clip_id=str(observation.clip_id),
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            crop_path=observation.crop_path,
            crop_sha256=observation.crop_sha256,
            fingerprint_json=json.dumps(observation.fingerprint),
            detector_confidence=observation.detector_confidence,
            embedding_distance_to_cluster=observation.embedding_distance_to_cluster,
            cluster_id=str(observation.cluster_id) if observation.cluster_id else None,
            detector_version=observation.detector_version,
            duplicate_of_observation_id=(
                str(observation.duplicate_of_observation_id)
                if observation.duplicate_of_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def get_vehicle_appearance_observation(
        self, observation_id: UUID
    ) -> VehicleAppearanceObservationRow | None:
        with self.session() as session:
            return session.get(VehicleAppearanceObservationRow, str(observation_id))

    def list_vehicle_appearance_observations(
        self, cluster_id: UUID, include_duplicates: bool = True
    ) -> list[VehicleAppearanceObservationRow]:
        statement = (
            select(VehicleAppearanceObservationRow)
            .where(VehicleAppearanceObservationRow.cluster_id == str(cluster_id))
            .order_by(VehicleAppearanceObservationRow.observed_at)
        )
        if not include_duplicates:
            statement = statement.where(
                VehicleAppearanceObservationRow.duplicate_of_observation_id.is_(None)
            )
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_all_vehicle_appearance_observations(self) -> list[VehicleAppearanceObservationRow]:
        with self.session() as session:
            rows = session.execute(
                select(VehicleAppearanceObservationRow).order_by(
                    VehicleAppearanceObservationRow.observed_at
                )
            )
            return list(rows.scalars())

    def list_vehicle_appearance_observations_for_event(
        self, event_id: UUID
    ) -> list[VehicleAppearanceObservationRow]:
        """Event-scoped, unlike `list_vehicle_appearance_observations`
        (cluster-scoped) -- used by the review UI's enrichment panel."""

        with self.session() as session:
            rows = session.execute(
                select(VehicleAppearanceObservationRow)
                .where(VehicleAppearanceObservationRow.event_id == str(event_id))
                .order_by(VehicleAppearanceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def mark_vehicle_appearance_observation_duplicate(
        self, observation_id: UUID, duplicate_of: UUID
    ) -> None:
        with self.session() as session:
            row = session.get(VehicleAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.duplicate_of_observation_id = str(duplicate_of)

    # -- person appearance ----------------------------------------------------
    # Mirrors every vehicle_appearance method above exactly -- see
    # enrichment/person_appearance.py's module docstring.

    def upsert_person_appearance_cluster(self, cluster: PersonAppearanceCluster) -> None:
        row = PersonAppearanceClusterRow(
            cluster_id=str(cluster.cluster_id),
            created_at=cluster.created_at,
            updated_at=cluster.updated_at,
            label=cluster.label,
            representative_crops_csv=",".join(cluster.representative_crop_paths),
            observation_count=cluster.observation_count,
            first_seen_at=cluster.first_seen_at,
            last_seen_at=cluster.last_seen_at,
            model_version=cluster.model_version,
            merged_into=str(cluster.merged_into) if cluster.merged_into else None,
        )
        with self.session() as session:
            session.merge(row)

    def list_person_appearance_clusters(self) -> list[PersonAppearanceClusterRow]:
        with self.session() as session:
            rows = session.execute(
                select(PersonAppearanceClusterRow).order_by(PersonAppearanceClusterRow.last_seen_at)
            )
            return list(rows.scalars())

    def get_person_appearance_cluster(self, cluster_id: UUID) -> PersonAppearanceClusterRow | None:
        with self.session() as session:
            return session.get(PersonAppearanceClusterRow, str(cluster_id))

    def search_person_appearance_clusters(self, query: str) -> list[PersonAppearanceClusterRow]:
        pattern = f"%{query}%"
        with self.session() as session:
            rows = session.execute(
                select(PersonAppearanceClusterRow)
                .where(
                    PersonAppearanceClusterRow.cluster_id.ilike(pattern)
                    | PersonAppearanceClusterRow.label.ilike(pattern)
                )
                .order_by(PersonAppearanceClusterRow.last_seen_at)
            )
            return list(rows.scalars())

    def set_person_appearance_cluster_label(self, cluster_id: UUID, label: str) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.label = label

    def set_person_appearance_cluster_merge(
        self, source_cluster_id: UUID, target_cluster_id: UUID
    ) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceClusterRow, str(source_cluster_id))
            if row is not None:
                row.merged_into = str(target_cluster_id)

    def set_person_appearance_observation_review_status(
        self, observation_id: UUID, status: str
    ) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.review_status = status

    def set_person_appearance_cluster_representative(
        self,
        cluster_id: UUID,
        observation_ids: list[UUID],
        representative_crop_paths: list[str],
    ) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.representative_observation_ids_csv = ",".join(str(o) for o in observation_ids)
                row.representative_crops_csv = ",".join(representative_crop_paths)

    def set_person_appearance_observation_cluster(
        self, observation_id: UUID, cluster_id: UUID | None
    ) -> None:
        """Reassign (or clear, if `cluster_id` is `None`) which cluster an
        observation belongs to -- mirrors `set_face_observation_cluster`."""

        with self.session() as session:
            row = session.get(PersonAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.cluster_id = str(cluster_id) if cluster_id is not None else None

    def set_person_appearance_cluster_observation_count(
        self, cluster_id: UUID, observation_count: int
    ) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceClusterRow, str(cluster_id))
            if row is not None:
                row.observation_count = observation_count

    def mark_person_appearance_observation_crop_purged(self, observation_id: UUID) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.crop_purged_at = utc_now()

    def list_person_appearance_observations_by_cluster_ids(
        self, cluster_ids: list[UUID]
    ) -> list[PersonAppearanceObservationRow]:
        if not cluster_ids:
            return []
        ids = [str(c) for c in cluster_ids]
        with self.session() as session:
            rows = session.execute(
                select(PersonAppearanceObservationRow).where(
                    PersonAppearanceObservationRow.cluster_id.in_(ids)
                )
            )
            return list(rows.scalars())

    def list_person_appearance_observations_eligible_for_purge(
        self,
    ) -> list[PersonAppearanceObservationRow]:
        """Mirrors `list_face_observations_eligible_for_purge` exactly."""

        with self.session() as session:
            candidates = session.execute(
                select(PersonAppearanceObservationRow).where(
                    PersonAppearanceObservationRow.review_status.in_(
                        ("user_confirmed", "user_rejected")
                    ),
                    PersonAppearanceObservationRow.crop_purged_at.is_(None),
                )
            ).scalars()
            representative_ids_by_cluster = {
                row.cluster_id: set(row.representative_observation_ids_csv.split(","))
                for row in session.execute(select(PersonAppearanceClusterRow)).scalars()
                if row.representative_observation_ids_csv
            }
            return [
                observation
                for observation in candidates
                if observation.review_status == "user_rejected"
                or observation.observation_id
                not in representative_ids_by_cluster.get(observation.cluster_id or "", set())
            ]

    def insert_person_appearance_observation(
        self, observation: PersonAppearanceObservation
    ) -> None:
        row = PersonAppearanceObservationRow(
            observation_id=str(observation.observation_id),
            signal_id=str(observation.signal_id),
            event_id=str(observation.event_id) if observation.event_id else None,
            clip_id=str(observation.clip_id),
            camera_id=observation.camera_id,
            observed_at=observation.observed_at,
            crop_path=observation.crop_path,
            crop_sha256=observation.crop_sha256,
            fingerprint_json=json.dumps(observation.fingerprint),
            detector_confidence=observation.detector_confidence,
            embedding_distance_to_cluster=observation.embedding_distance_to_cluster,
            cluster_id=str(observation.cluster_id) if observation.cluster_id else None,
            detector_version=observation.detector_version,
            duplicate_of_observation_id=(
                str(observation.duplicate_of_observation_id)
                if observation.duplicate_of_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def get_person_appearance_observation(
        self, observation_id: UUID
    ) -> PersonAppearanceObservationRow | None:
        with self.session() as session:
            return session.get(PersonAppearanceObservationRow, str(observation_id))

    def list_person_appearance_observations(
        self, cluster_id: UUID, include_duplicates: bool = True
    ) -> list[PersonAppearanceObservationRow]:
        statement = (
            select(PersonAppearanceObservationRow)
            .where(PersonAppearanceObservationRow.cluster_id == str(cluster_id))
            .order_by(PersonAppearanceObservationRow.observed_at)
        )
        if not include_duplicates:
            statement = statement.where(
                PersonAppearanceObservationRow.duplicate_of_observation_id.is_(None)
            )
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def list_all_person_appearance_observations(self) -> list[PersonAppearanceObservationRow]:
        with self.session() as session:
            rows = session.execute(
                select(PersonAppearanceObservationRow).order_by(
                    PersonAppearanceObservationRow.observed_at
                )
            )
            return list(rows.scalars())

    def list_person_appearance_observations_for_event(
        self, event_id: UUID
    ) -> list[PersonAppearanceObservationRow]:
        """Event-scoped, unlike `list_person_appearance_observations`
        (cluster-scoped) -- used by the review UI's enrichment panel."""

        with self.session() as session:
            rows = session.execute(
                select(PersonAppearanceObservationRow)
                .where(PersonAppearanceObservationRow.event_id == str(event_id))
                .order_by(PersonAppearanceObservationRow.observed_at)
            )
            return list(rows.scalars())

    def mark_person_appearance_observation_duplicate(
        self, observation_id: UUID, duplicate_of: UUID
    ) -> None:
        with self.session() as session:
            row = session.get(PersonAppearanceObservationRow, str(observation_id))
            if row is not None:
                row.duplicate_of_observation_id = str(duplicate_of)

    # -- encounters ---------------------------------------------------------

    def insert_encounter(self, encounter: Encounter) -> None:
        row = EncounterRow(
            encounter_id=str(encounter.encounter_id),
            event_id=str(encounter.event_id),
            clip_id=str(encounter.clip_id),
            camera_id=encounter.camera_id,
            observed_at=encounter.observed_at,
            face_observation_id=(
                str(encounter.face_observation_id) if encounter.face_observation_id else None
            ),
            plate_observation_id=(
                str(encounter.plate_observation_id) if encounter.plate_observation_id else None
            ),
            voice_observation_id=(
                str(encounter.voice_observation_id) if encounter.voice_observation_id else None
            ),
            vehicle_appearance_observation_id=(
                str(encounter.vehicle_appearance_observation_id)
                if encounter.vehicle_appearance_observation_id
                else None
            ),
            person_appearance_observation_id=(
                str(encounter.person_appearance_observation_id)
                if encounter.person_appearance_observation_id
                else None
            ),
        )
        with self.session() as session:
            session.add(row)

    def delete_encounters_for_event(self, event_id: UUID) -> None:
        """`Encounter` rows are a pure, freely-rebuildable derived index
        over the face/plate/voice/vehicle-appearance observation rows
        (themselves append-only, never touched here) -- see
        `enrichment/service.py::_derive_encounters`'s docstring. Clearing
        an event's encounters before re-deriving them is what makes a
        rerun a clean replace instead of an unbounded accumulate; this is
        the one place this project deletes a database row outright,
        deliberately, because nothing about an Encounter is itself
        primary evidence."""

        with self.session() as session:
            session.execute(delete(EncounterRow).where(EncounterRow.event_id == str(event_id)))

    def list_encounters_for_event(self, event_id: UUID) -> list[EncounterRow]:
        with self.session() as session:
            rows = session.execute(
                select(EncounterRow)
                .where(EncounterRow.event_id == str(event_id))
                .order_by(EncounterRow.observed_at)
            )
            return list(rows.scalars())

    def list_encounters_by_face_observation_ids(
        self, observation_ids: list[UUID]
    ) -> list[EncounterRow]:
        if not observation_ids:
            return []
        ids = [str(o) for o in observation_ids]
        with self.session() as session:
            rows = session.execute(
                select(EncounterRow)
                .where(EncounterRow.face_observation_id.in_(ids))
                .order_by(EncounterRow.observed_at)
            )
            return list(rows.scalars())

    def list_encounters_by_plate_observation_ids(
        self, observation_ids: list[UUID]
    ) -> list[EncounterRow]:
        if not observation_ids:
            return []
        ids = [str(o) for o in observation_ids]
        with self.session() as session:
            rows = session.execute(
                select(EncounterRow)
                .where(EncounterRow.plate_observation_id.in_(ids))
                .order_by(EncounterRow.observed_at)
            )
            return list(rows.scalars())

    def list_encounters_by_vehicle_appearance_observation_ids(
        self, observation_ids: list[UUID]
    ) -> list[EncounterRow]:
        if not observation_ids:
            return []
        ids = [str(o) for o in observation_ids]
        with self.session() as session:
            rows = session.execute(
                select(EncounterRow)
                .where(EncounterRow.vehicle_appearance_observation_id.in_(ids))
                .order_by(EncounterRow.observed_at)
            )
            return list(rows.scalars())

    def list_encounters_by_person_appearance_observation_ids(
        self, observation_ids: list[UUID]
    ) -> list[EncounterRow]:
        if not observation_ids:
            return []
        ids = [str(o) for o in observation_ids]
        with self.session() as session:
            rows = session.execute(
                select(EncounterRow)
                .where(EncounterRow.person_appearance_observation_id.in_(ids))
                .order_by(EncounterRow.observed_at)
            )
            return list(rows.scalars())

    # -- merge suggestions ------------------------------------------------

    def insert_merge_suggestion(self, suggestion: MergeSuggestion) -> None:
        row = MergeSuggestionRow(
            suggestion_id=str(suggestion.suggestion_id),
            entity_type=suggestion.entity_type,
            source_id=str(suggestion.source_id),
            target_id=str(suggestion.target_id),
            similarity_score=suggestion.similarity_score,
            basis=suggestion.basis,
            status=suggestion.status,
            created_at=suggestion.created_at,
            resolved_at=suggestion.resolved_at,
            resolved_by=suggestion.resolved_by,
        )
        with self.session() as session:
            session.add(row)

    def get_merge_suggestion(self, suggestion_id: UUID) -> MergeSuggestionRow | None:
        with self.session() as session:
            return session.get(MergeSuggestionRow, str(suggestion_id))

    def list_merge_suggestions(
        self, entity_type: str | None = None, status: str | None = None
    ) -> list[MergeSuggestionRow]:
        statement = select(MergeSuggestionRow).order_by(MergeSuggestionRow.created_at)
        if entity_type is not None:
            statement = statement.where(MergeSuggestionRow.entity_type == entity_type)
        if status is not None:
            statement = statement.where(MergeSuggestionRow.status == status)
        with self.session() as session:
            return list(session.execute(statement).scalars())

    def resolve_merge_suggestion(
        self, suggestion_id: UUID, status: str, resolved_by: str, resolved_at: datetime
    ) -> None:
        with self.session() as session:
            row = session.get(MergeSuggestionRow, str(suggestion_id))
            if row is not None:
                row.status = status
                row.resolved_by = resolved_by
                row.resolved_at = resolved_at
