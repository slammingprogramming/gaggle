from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gaggle.schemas.recognition import FaceCluster, PlateRecord
from gaggle.storage.database import UTCDateTimeColumn
from gaggle.storage.repository import Repository
from gaggle.utils.ids import new_uuid

BASE = datetime(2025, 6, 27, 15, 6, 43, 33333, tzinfo=UTC)


def test_type_decorator_recovers_utc_from_a_naive_sqlite_read() -> None:
    """Direct unit test of the fix's actual logic.

    SQLite silently returns naive datetimes on read regardless of what was
    written (a documented SQLAlchemy+SQLite limitation) -- this is exactly
    what crashed a real `enrich` run reading a previously-stored
    `FaceCluster` back out of the database. `process_result_value` must
    reattach UTC so every value downstream is guaranteed timezone-aware,
    which is what every pydantic timestamp field in this project requires.
    """

    column = UTCDateTimeColumn()
    naive_as_returned_by_sqlite = datetime(2025, 6, 27, 15, 6, 43, 33333)
    recovered = column.process_result_value(naive_as_returned_by_sqlite, dialect=None)

    assert recovered is not None
    assert recovered.tzinfo is not None
    assert recovered == BASE


def test_type_decorator_normalizes_on_write_too() -> None:
    column = UTCDateTimeColumn()
    bound = column.process_bind_param(BASE, dialect=None)
    assert bound is not None
    assert bound.tzinfo is not None


def test_type_decorator_passes_through_none() -> None:
    column = UTCDateTimeColumn()
    assert column.process_bind_param(None, dialect=None) is None
    assert column.process_result_value(None, dialect=None) is None


def test_face_cluster_round_trip_through_real_sqlite_stays_timezone_aware(
    tmp_path: Path,
) -> None:
    """Regression test for the exact reported crash.

    A real user hit ``ValidationError: timestamp must be timezone-aware``
    when ``enrich`` re-read a previously-stored ``FaceCluster`` from
    SQLite and fed its ``created_at``/``first_seen_at`` straight back into
    a new ``FaceCluster`` pydantic instance (the pattern
    ``EnrichmentService`` uses to preserve prior stats on every new
    observation). This test reproduces that exact shape end to end
    through the real database layer, not just the isolated column type.
    """

    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    cluster_id = new_uuid()
    original = FaceCluster(
        cluster_id=cluster_id,
        created_at=BASE,
        updated_at=BASE,
        first_seen_at=BASE,
        last_seen_at=BASE,
        model_version="1.0.0",
    )
    repository.database.upsert_face_cluster(original)

    # This is the exact read-then-reconstruct pattern that crashed.
    prior = repository.database.get_face_cluster(cluster_id)
    assert prior is not None
    assert prior.created_at.tzinfo is not None
    assert prior.first_seen_at.tzinfo is not None

    # Must not raise ValidationError.
    rebuilt = FaceCluster(
        cluster_id=cluster_id,
        created_at=prior.created_at,
        updated_at=datetime.now(tz=UTC),
        first_seen_at=prior.first_seen_at,
        last_seen_at=datetime.now(tz=UTC),
        model_version="1.0.0",
    )
    assert rebuilt.created_at == BASE
    assert rebuilt.first_seen_at == BASE


def test_plate_record_round_trip_through_real_sqlite_stays_timezone_aware(
    tmp_path: Path,
) -> None:
    repository = Repository(tmp_path / "workspace")
    repository.initialize()

    plate_id = new_uuid()
    original = PlateRecord(
        plate_id=plate_id,
        normalized_text="ABC1234",
        created_at=BASE,
        updated_at=BASE,
        first_seen_at=BASE,
        last_seen_at=BASE,
    )
    repository.database.upsert_plate_record(original)

    existing = repository.database.get_plate_record(plate_id)
    assert existing is not None
    assert existing.created_at.tzinfo is not None
    assert existing.first_seen_at.tzinfo is not None

    rebuilt = PlateRecord(
        plate_id=plate_id,
        normalized_text="ABC1234",
        created_at=existing.created_at,
        updated_at=datetime.now(tz=UTC),
        first_seen_at=existing.first_seen_at,
        last_seen_at=datetime.now(tz=UTC),
    )
    assert rebuilt.created_at == BASE
    assert rebuilt.first_seen_at == BASE
