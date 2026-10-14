import asyncio
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.events import (
    InvalidEventCursor,
    PostgresRunEventStore,
    append_run_event,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import (
    agent_run_events,
    agent_runs,
    conversations,
)


@pytest.mark.asyncio
async def test_event_journal_dedup_rollback_reopen_owner_and_cursor(postgres_dsn):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    owner, conversation = uuid4(), None
    store = PostgresConversationStore(engine)
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            conversation = await store.create(owner, "Events fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(engine, saver, request_versions=(2,))
            run = await coordinator.submit(
                owner,
                TurnRequestV2(
                    conversation_id=conversation.id,
                    request_id=uuid4(),
                    expected_revision=0,
                    message="Events",
                ),
            )
            payload = {"event": "START", "run": run.model_dump(mode="json")}
            async with engine.begin() as conn:
                first = await append_run_event(conn, run.run_id, "start", payload)
                assert (
                    await append_run_event(conn, run.run_id, "start", payload) == first
                )
            with pytest.raises(ValueError, match="dedup"):
                async with engine.begin() as conn:
                    await append_run_event(
                        conn,
                        run.run_id,
                        "start",
                        {"event": "DELTA", "delta": "different"},
                    )
            with pytest.raises(RuntimeError):
                async with engine.begin() as conn:
                    await append_run_event(
                        conn,
                        run.run_id,
                        "rolled-back",
                        {"event": "DELTA", "delta": "not committed"},
                    )
                    raise RuntimeError("rollback fixture transaction")
            async with engine.begin() as conn:
                second = await append_run_event(
                    conn,
                    run.run_id,
                    "answer",
                    {"event": "DELTA", "delta": "validated answer"},
                )
        await engine.dispose()
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(
                    insert(agent_run_events).values(
                        id=uuid4(),
                        conversation_id=conversation.id,
                        run_id=run.run_id,
                        sequence=99,
                        dedup_key="invalid-payload",
                        event_type="START",
                        payload={
                            "data": {},
                            "message": "AI_CHAT_STREAM_STARTED",
                            "errors": [],
                        },
                    )
                )
        journal = PostgresRunEventStore(engine)
        rows = await journal.read(owner, conversation.id, run.run_id, None, 100)
        assert [row.sequence for row in rows] == [1, 2]
        assert [row.event_id for row in rows] == [first.event_id, second.event_id]
        assert rows[0].envelope.message.value == "AI_CHAT_STREAM_STARTED"
        assert rows[1].envelope.data.delta == "validated answer"
        assert (
            await journal.read(owner, conversation.id, run.run_id, second.event_id, 100)
            == []
        )
        assert [
            row.event_id
            for row in await journal.read(
                owner, conversation.id, run.run_id, first.event_id, 100
            )
        ] == [second.event_id]
        with pytest.raises(RunAdmissionError, match="AI_CONVERSATION_NOT_FOUND"):
            await journal.read(uuid4(), conversation.id, run.run_id, None, 100)
        with pytest.raises(InvalidEventCursor):
            await journal.read(owner, conversation.id, run.run_id, uuid4(), 100)

        async def concurrent_append():
            async with engine.begin() as conn:
                return await append_run_event(
                    conn,
                    run.run_id,
                    "concurrent",
                    {"event": "DELTA", "delta": "same outcome"},
                )

        left, right = await asyncio.gather(concurrent_append(), concurrent_append())
        assert left == right and left.sequence == 3
        assert (
            len(await journal.read(owner, conversation.id, run.run_id, None, 100)) == 3
        )
        async with engine.connect() as conn:
            original = (
                (
                    await conn.execute(
                        select(agent_runs).where(agent_runs.c.id == run.run_id)
                    )
                )
                .mappings()
                .one()
            )
        assert original["status"] == "PENDING"
    finally:
        if conversation:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
        await engine.dispose()
