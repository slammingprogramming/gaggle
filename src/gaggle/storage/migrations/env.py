"""Alembic environment script.

Loaded by `alembic.command.upgrade`/`stamp` (see `storage/migrate.py`,
which builds the `Config` programmatically and never depends on finding
`alembic.ini` relative to the current working directory) as well as by the
`alembic` CLI directly for development-time `alembic revision --autogenerate`
(see the repo-root `alembic.ini`, which points `script_location` at this
package-internal directory so migration scripts ship as part of the
installed package rather than living outside it).
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from gaggle.storage.database import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # render_as_batch=True: SQLite has limited native ALTER TABLE
        # support (no DROP COLUMN with constraints, no column type
        # changes, etc. in general) -- batch mode recreates the table
        # under the hood when a migration needs more than a plain ADD
        # COLUMN. Costs nothing for migrations that don't need it.
        context.configure(
            connection=connection, target_metadata=target_metadata, render_as_batch=True
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
