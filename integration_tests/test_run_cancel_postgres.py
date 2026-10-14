"""Durable owner cancellation and actual managed child drain with PostgreSQL."""

import asyncio
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.application.use_cases.stateful_conversations import StatefulRunService
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.events import PostgresRunEventStore
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import (
    agent_runs,
    conversation_messages,
    conversations,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "running,local_signal", [(False, True), (True, True), (True, False)]
)
async def test_cancel_owner_idempotency_and_shared_thread_drained_before_release(
    postgres_dsn, running, local_signal
):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, conversation, executor = (
        PostgresConversationStore(engine),
        uuid4(),
        None,
        None,
    )
    started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            conversation = await store.create(owner, "Cancel fixture")
            head = await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(engine, saver, request_versions=(2,))

            async def execution(claim, children):
                async def writer():
                    try:
                        started.set()
                        await asyncio.Event().wait()
                    finally:
                        draining.set()
                        await release.wait()

                children.create_task(writer())
                await asyncio.Event().wait()

            executor = ManagedExecutor(
                coordinator,
                execution,
                worker_id=uuid4(),
                poll_seconds=0.02,
                heartbeat_seconds=0.02,
            )
            service = StatefulRunService(
                coordinator,
                executor.wake,
                executor.cancel_run if local_signal else None,
            )
            request = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="Test cancel",
            )
            submitted = await service.submit(owner, request)
            if running:
                await executor.start()
                await asyncio.wait_for(started.wait(), 5)
            for actor, target in ((uuid4(), submitted.run_id), (owner, uuid4())):
                with pytest.raises(
                    RunAdmissionError, match="AI_CONVERSATION_NOT_FOUND"
                ):
                    await service.cancel(actor, conversation.id, target)
            view = await service.cancel(owner, conversation.id, submitted.run_id)
            if running:
                assert view.status == "RUNNING"
                await asyncio.wait_for(draining.wait(), 5)
                busy = await store.get(owner, conversation.id)
                assert busy.active_run_id == submitted.run_id and busy.revision == 0
                assert busy.accepted_checkpoint_id == head.accepted_checkpoint_id
                current_events = await PostgresRunEventStore(engine).read(
                    owner, conversation.id, submitted.run_id
                )
                assert [item.envelope.data.event for item in current_events] == [
                    "START"
                ]
                assert (
                    await service.cancel(owner, conversation.id, submitted.run_id)
                ).status == "RUNNING"
                release.set()
                for _ in range(200):
                    view = await service.get(owner, conversation.id, submitted.run_id)
                    if view.status == "CANCELLED" or executor.last_error:
                        break
                    await asyncio.sleep(0.02)
                assert view.status == "CANCELLED", executor.last_error
            else:
                assert view.status == "CANCELLED"
                assert await coordinator.claim_pending(uuid4(), 1) == []
            assert (
                await service.cancel(owner, conversation.id, submitted.run_id)
            ).status == "CANCELLED"
            after = await store.get(owner, conversation.id)
            assert after.active_run_id is None and after.revision == 0
            assert after.accepted_checkpoint_id == head.accepted_checkpoint_id
            async with engine.connect() as conn:
                row = (
                    (
                        await conn.execute(
                            select(agent_runs).where(
                                agent_runs.c.id == submitted.run_id
                            )
                        )
                    )
                    .mappings()
                    .one()
                )
                messages = (
                    (
                        await conn.execute(
                            select(conversation_messages).where(
                                conversation_messages.c.conversation_id
                                == conversation.id
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
            assert row["cancel_requested"] and row["error"] is None
            assert bool(row["executor_stop_evidence"]) == running
            assert len(messages) == 1 and messages[0]["role"] == "user"
            events = await PostgresRunEventStore(engine).read(
                owner, conversation.id, submitted.run_id
            )
            assert [item.envelope.data.event for item in events] == ["START", "ERROR"]
            assert events[-1].envelope.data.code.value == "AI_RUN_CANCELLED"
    finally:
        release.set()
        if executor:
            assert await executor.stop(grace_seconds=5)
        if conversation:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
