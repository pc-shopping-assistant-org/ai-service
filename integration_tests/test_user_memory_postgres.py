"""Explicit memory survives adapter restart and stays isolated by JWT owner."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.user_memory import (
    MAX_MEMORIES,
    MemoryInput,
    MemoryLimitReached,
)
from ai_service.infrastructure.persistence.schema import user_memories
from ai_service.infrastructure.persistence.user_memory import PostgresUserMemoryStore


@pytest.mark.asyncio
async def test_memory_restart_ownership_expiry_and_quota(postgres_dsn):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    owner, foreign = uuid4(), uuid4()
    value = MemoryInput(kind="PREFERENCE", content="Prefer NVIDIA")
    store = PostgresUserMemoryStore(engine)
    try:
        await store.put(owner, "gpu-brand", value)
        await engine.dispose()
        reopened = PostgresUserMemoryStore(engine)
        assert (await reopened.list(owner))[0].content == value.content
        assert await reopened.list(foreign) == []
        await reopened.delete(foreign, "gpu-brand")
        assert len(await reopened.list(owner)) == 1
        await reopened.put(
            owner, "gpu-brand", value.model_copy(update={"content": "Prefer AMD"})
        )
        assert (await reopened.list(owner))[0].content == "Prefer AMD"
        for index in range(MAX_MEMORIES - 1):
            await reopened.put(owner, f"preference-{index}", value)
        with pytest.raises(MemoryLimitReached):
            await reopened.put(owner, "overflow", value)
        await reopened.put(owner, "gpu-brand", value)  # replacement at quota is allowed
        async with engine.begin() as conn:
            await conn.execute(
                update(user_memories)
                .where(user_memories.c.account_id == owner)
                .values(
                    updated_at=datetime.now(UTC) - timedelta(days=100),
                    expires_at=datetime.now(UTC) - timedelta(days=1),
                )
            )
        assert await reopened.list(owner) == []
        await reopened.put(owner, "after-expiry", value)
        assert len(await reopened.list(owner)) == 1
        await reopened.delete(owner, "after-expiry")
        assert await reopened.list(owner) == []
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                delete(user_memories).where(
                    user_memories.c.account_id.in_([owner, foreign])
                )
            )
        await engine.dispose()
