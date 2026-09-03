"""encounter table -- adds the derived cross-modality Encounter records
(see gaggle.schemas.encounter.Encounter and storage/database.py's
EncounterRow) on top of the 0002_camera schema. Never user-written: rows
here are only ever produced by
enrichment/service.py::EnrichmentService._derive_encounters.

Revision ID: 0003_encounter
Revises: 0002_camera
Create Date: 2026-08-10 22:53:10.854598

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import gaggle.storage.database

# revision identifiers, used by Alembic.
revision: str = "0003_encounter"
down_revision: str | None = "0002_camera"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "encounter",
        sa.Column("encounter_id", sa.String(), nullable=False),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("clip_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column(
            "observed_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=False
        ),
        sa.Column("face_observation_id", sa.String(), nullable=True),
        sa.Column("plate_observation_id", sa.String(), nullable=True),
        sa.Column("voice_observation_id", sa.String(), nullable=True),
        sa.Column("vehicle_appearance_observation_id", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("encounter_id"),
    )
    with op.batch_alter_table("encounter", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_encounter_clip_id"), ["clip_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_encounter_event_id"), ["event_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_encounter_face_observation_id"), ["face_observation_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_encounter_plate_observation_id"), ["plate_observation_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_encounter_vehicle_appearance_observation_id"),
            ["vehicle_appearance_observation_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_encounter_voice_observation_id"), ["voice_observation_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("encounter", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_encounter_voice_observation_id"))
        batch_op.drop_index(batch_op.f("ix_encounter_vehicle_appearance_observation_id"))
        batch_op.drop_index(batch_op.f("ix_encounter_plate_observation_id"))
        batch_op.drop_index(batch_op.f("ix_encounter_face_observation_id"))
        batch_op.drop_index(batch_op.f("ix_encounter_event_id"))
        batch_op.drop_index(batch_op.f("ix_encounter_clip_id"))

    op.drop_table("encounter")
