"""Lifecycle-owned DB resources; never migrate/setup inside request handlers."""

from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from ai_service.config.settings import Settings


@asynccontextmanager
async def open_stateful_database(settings: Settings):
    # Deferred import lets stateless APIs remain usable with persistence disabled.
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    if settings.database_url is None:
        raise RuntimeError("AI_DATABASE_URL must be configured; no in-memory fallback")
    url = make_url(settings.database_url.get_secret_value())
    if url.drivername != "postgresql+psycopg":
        raise ValueError("AI_DATABASE_URL must use postgresql+psycopg")
    engine = create_async_engine(url, hide_parameters=True)
    try:
        # psycopg's URL excludes SQLAlchemy's driver suffix, with secrets kept
        # inside resource construction, never logged or returned to customers.
        async with AsyncConnectionPool[AsyncConnection[dict[str, Any]]](
            url.set(drivername="postgresql").render_as_string(hide_password=False),
            open=False,
            min_size=1,
            max_size=5,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        ) as pool:
            await pool.wait()
            yield engine, AsyncPostgresSaver(pool)
    finally:
        await engine.dispose()
