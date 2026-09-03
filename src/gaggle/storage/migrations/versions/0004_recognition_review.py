"""recognition review -- adds human review/crop-purge tracking columns
(see gaggle.schemas.recognition.py's review_status/crop_purged_at/
representative_observation_ids and storage/database.py's matching Row
columns) on top of the 0003_encounter schema.

Every new NOT NULL column here (review_status, representative_observation_ids_csv)
gets an explicit server_default so existing rows in an already-populated
workspace backfill correctly -- autogenerate does not add this by itself,
and SQLite requires a default when adding a NOT NULL column to a
non-empty table. Verified against a real ~450-face-observation/
~400-face-cluster workspace copy before this migration was finalized.

Revision ID: 0004_recognition_review
Revises: 0003_encounter
Create Date: 2026-08-11 15:11:43.665535

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import gaggle.storage.database

# revision identifiers, used by Alembic.
revision: str = "0004_recognition_review"
down_revision: str | None = "0003_encounter"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EMPTY_CSV_DEFAULT = sa.text("''")
_NEEDS_REVIEW_DEFAULT = sa.text("'needs_review'")


def upgrade() -> None:
    with op.batch_alter_table("face_cluster", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "representative_observation_ids_csv",
                sa.String(),
                nullable=False,
                server_default=_EMPTY_CSV_DEFAULT,
            )
        )

    with op.batch_alter_table("face_observation", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "review_status", sa.String(), nullable=False, server_default=_NEEDS_REVIEW_DEFAULT
            )
        )
        batch_op.add_column(
            sa.Column(
                "crop_purged_at",
                gaggle.storage.database.UTCDateTimeColumn(timezone=True),
                nullable=True,
            )
        )
        batch_op.create_index(
            batch_op.f("ix_face_observation_review_status"), ["review_status"], unique=False
        )

    with op.batch_alter_table("plate_observation", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "crop_purged_at",
                gaggle.storage.database.UTCDateTimeColumn(timezone=True),
                nullable=True,
            )
        )

    with op.batch_alter_table("vehicle_appearance_cluster", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "representative_observation_ids_csv",
                sa.String(),
                nullable=False,
                server_default=_EMPTY_CSV_DEFAULT,
            )
        )

    with op.batch_alter_table("vehicle_appearance_observation", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "review_status", sa.String(), nullable=False, server_default=_NEEDS_REVIEW_DEFAULT
            )
        )
        batch_op.add_column(
            sa.Column(
                "crop_purged_at",
                gaggle.storage.database.UTCDateTimeColumn(timezone=True),
                nullable=True,
            )
        )
        batch_op.create_index(
            batch_op.f("ix_vehicle_appearance_observation_review_status"),
            ["review_status"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("vehicle_appearance_observation", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_vehicle_appearance_observation_review_status"))
        batch_op.drop_column("crop_purged_at")
        batch_op.drop_column("review_status")

    with op.batch_alter_table("vehicle_appearance_cluster", schema=None) as batch_op:
        batch_op.drop_column("representative_observation_ids_csv")

    with op.batch_alter_table("plate_observation", schema=None) as batch_op:
        batch_op.drop_column("crop_purged_at")

    with op.batch_alter_table("face_observation", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_face_observation_review_status"))
        batch_op.drop_column("crop_purged_at")
        batch_op.drop_column("review_status")

    with op.batch_alter_table("face_cluster", schema=None) as batch_op:
        batch_op.drop_column("representative_observation_ids_csv")
