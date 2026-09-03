"""camera table -- adds the optional Camera registry (see
gaggle.schemas.camera.Camera and storage/database.py's CameraRow) on top of
the 0001_baseline schema. Registration is always optional: camera_id stays
a free-form string everywhere else in the schema; this table only adds
opt-in metadata (source_type, indoor/outdoor, site_id for cross-camera
sync scoping) about it.

Revision ID: 0002_camera
Revises: 0001_baseline
Create Date: 2026-08-10 22:32:44.398284

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

import gaggle.storage.database

# revision identifiers, used by Alembic.
revision: str = "0002_camera"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "camera",
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("indoor", sa.Boolean(), nullable=True),
        sa.Column("site_id", sa.String(), nullable=True),
        sa.Column(
            "created_at", gaggle.storage.database.UTCDateTimeColumn(timezone=True), nullable=False
        ),
        sa.Column("notes", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("camera_id"),
    )
    with op.batch_alter_table("camera", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_camera_site_id"), ["site_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("camera", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_camera_site_id"))

    op.drop_table("camera")
