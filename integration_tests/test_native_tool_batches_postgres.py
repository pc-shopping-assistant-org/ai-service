"""Native per-call pending writes survive PostgreSQL saver reopen."""

from contextlib import asynccontextmanager

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_native_tool_batches import exercise_batch_recovery


@pytest.mark.asyncio
async def test_batch_recovery_after_pool_reopen_ignores_orphan(postgres_dsn):
    @asynccontextmanager
    async def open_saver():
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            yield saver

    await exercise_batch_recovery(open_saver)
