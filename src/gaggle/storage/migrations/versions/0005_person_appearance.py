"""person_appearance tables -- adds pedestrian/full-body appearance
re-identification (see gaggle.schemas.recognition.PersonAppearanceObservation/
PersonAppearanceCluster and enrichment/person_appearance.py's module
docstring), structurally a near-verbatim copy of the vehicle_appearance
tables at their *current* shape (baseline columns from 0001 plus the
review_status/crop_purged_at/representative_observation_ids_csv columns
0004 added to vehicle_appearance) -- these are brand-new tables, so
there is no separate later migration needed to catch them up the way
0004 had to for the pre-existing vehicle_appearance tables.

Also adds `person_appearance_observation_id` to the `encounter` table:
_derive_encounters now correlates a 5th modality alongside
face/plate/voice/vehicle_appearance.

Revision ID: 0005_person_appearance
Revises: 0004_recognition_review
Create Date: 2026-08-17 19:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import gaggle.storage.database

# revision identifiers, used by Alembic.
revision: str = "0005_person_appearance"
down_revision: str | None = "0004_recognition_review"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NEEDS_REVIEW_DEFAULT = sa.text("'needs_review'")


def upgrade() -> None:
    op.create_table(
        "person_appearance_cluster",
        sa.Column("cluster_id", sa.String(), nullable=False),
        sa.Column(
            "created_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=False
        ),
        sa.Column(
            "updated_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=False
        ),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("representative_crops_csv", sa.String(), nullable=False),
        sa.Column("observation_count", sa.Integer(), nullable=False),
        sa.Column(
            "first_seen_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=True
        ),
        sa.Column(
            "last_seen_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=True
        ),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("merged_into", sa.String(), nullable=True),
        sa.Column("representative_observation_ids_csv", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("cluster_id"),
    )
    with op.batch_alter_table("person_appearance_cluster", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_person_appearance_cluster_merged_into"), ["merged_into"], unique=False
        )

    op.create_table(
        "person_appearance_observation",
        sa.Column("observation_id", sa.String(), nullable=False),
        sa.Column("signal_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=True),
        sa.Column("clip_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column(
            "observed_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=False
        ),
        sa.Column("crop_path", sa.String(), nullable=False),
        sa.Column("crop_sha256", sa.String(), nullable=False),
        sa.Column("fingerprint_json", sa.String(), nullable=False),
        sa.Column("detector_confidence", sa.Float(), nullable=False),
        sa.Column("embedding_distance_to_cluster", sa.Float(), nullable=True),
        sa.Column("cluster_id", sa.String(), nullable=True),
        sa.Column("detector_version", sa.String(), nullable=False),
        sa.Column("duplicate_of_observation_id", sa.String(), nullable=True),
        sa.Column(
            "review_status", sa.String(), nullable=False, server_default=_NEEDS_REVIEW_DEFAULT
        ),
        sa.Column(
            "crop_purged_at",
            gaggle.storage.database.UTCDateTimeColumn(timezone=True),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("observation_id"),
    )
    with op.batch_alter_table("person_appearance_observation", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_person_appearance_observation_clip_id"), ["clip_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_person_appearance_observation_cluster_id"), ["cluster_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_person_appearance_observation_event_id"), ["event_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_person_appearance_observation_review_status"),
            ["review_status"],
            unique=False,
        )

    with op.batch_alter_table("encounter", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("person_appearance_observation_id", sa.String(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_encounter_person_appearance_observation_id"),
            ["person_appearance_observation_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("encounter", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_encounter_person_appearance_observation_id"))
        batch_op.drop_column("person_appearance_observation_id")

    with op.batch_alter_table("person_appearance_observation", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_person_appearance_observation_review_status"))
        batch_op.drop_index(batch_op.f("ix_person_appearance_observation_event_id"))
        batch_op.drop_index(batch_op.f("ix_person_appearance_observation_cluster_id"))
        batch_op.drop_index(batch_op.f("ix_person_appearance_observation_clip_id"))
    op.drop_table("person_appearance_observation")

    with op.batch_alter_table("person_appearance_cluster", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_person_appearance_cluster_merged_into"))
    op.drop_table("person_appearance_cluster")
