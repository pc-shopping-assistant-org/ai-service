"""Real DB gate after `alembic upgrade head`; never silently create missing tables."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import ConversationNotFound
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.schema import (
    agent_runs,
    conversation_messages,
    conversations,
)


@pytest.mark.asyncio
async def test_owned_records_survive_reopen_and_database_enforces_scoped_refs(
    postgres_dsn,
):
    dsn = postgres_dsn
    url = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_async_engine(url, hide_parameters=True)
    account_id, other_owner = uuid4(), uuid4()
    store = PostgresConversationStore(engine)
    first = await store.create(account_id, "Core gate")
    second = await store.create(other_owner, None)
    try:
        await engine.dispose()
        assert (await store.get(account_id, first.id)).revision == 0
        assert [row.id for row in await store.list_owned(account_id)] == [first.id]
        for requested in [first.id, uuid4()]:
            with pytest.raises(ConversationNotFound):
                await store.get(other_owner, requested)
        run_id = uuid4()
        async with engine.begin() as connection:
            await connection.execute(
                insert(agent_runs).values(
                    id=run_id,
                    conversation_id=first.id,
                    request_id=uuid4(),
                    request_hash="0" * 64,
                    request_payload={},
                    kind="CHAT",
                    status="FAILED_RETRYABLE",
                    base_revision=0,
                    attempt_count=0,
                    max_attempts=2,
                    retry_policy_version="gate-v1",
                    retry_deadline=datetime.now(UTC) + timedelta(minutes=5),
                )
            )
        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(
                    insert(conversation_messages).values(
                        id=uuid4(),
                        conversation_id=second.id,
                        run_id=run_id,
                        sequence=1,
                        role="assistant",
                        content="wrong conversation",
                    )
                )
        with pytest.raises(IntegrityError):
            async with engine.begin() as connection:
                await connection.execute(
                    conversations.update()
                    .where(conversations.c.id == first.id)
                    .values(revision=1)
                )
    finally:
        async with engine.begin() as connection:
            await connection.execute(
                delete(conversations).where(
                    conversations.c.id.in_([first.id, second.id])
                )
            )
        await engine.dispose()
