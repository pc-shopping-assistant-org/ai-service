"""Application DDL only. Run saver.setup separately via bootstrap command."""

import asyncio

from alembic import context

from ai_service.config.settings import Settings
from ai_service.infrastructure.persistence.schema import metadata


def database_url() -> str:
    value = Settings().database_url
    if value is None:
        raise RuntimeError("AI_DATABASE_URL must be configured")
    return value.get_secret_value()


def migrate(connection) -> None:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()


async def online() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(database_url(), hide_parameters=True)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(url=database_url(), target_metadata=metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
