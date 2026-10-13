"""Main production create/get path with SQL admission and native PostgreSQL root."""

import asyncio
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.schema import conversations


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["CONCURRENT", "CRASH", "LEGACY_NATIVE"])
async def test_application_root_init_exclusion_reconciliation_and_legacy_policy(
    postgres_dsn, mode
):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, conversation, saved_ref = (
        PostgresConversationStore(engine),
        uuid4(),
        None,
        None,
    )
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "Root audit")
            initializer = NativeConversationInitializer(saver)

            class Init:
                calls = 0

                async def initialize(self, identifier):
                    nonlocal saved_ref
                    self.calls += 1
                    await asyncio.sleep(0.02)
                    saved_ref = await initializer.initialize(identifier)
                    if mode == "CRASH":
                        raise RuntimeError("after native commit before app CAS")
                    return saved_ref

            proxy = Init()
            if mode == "CONCURRENT":
                first, second = await asyncio.gather(
                    store.initialize(owner, conversation.id, proxy),
                    store.initialize(owner, conversation.id, proxy),
                )
                assert (
                    first.accepted_checkpoint_id
                    == second.accepted_checkpoint_id
                    == saved_ref.checkpoint_id
                )
                assert proxy.calls == 1
            elif mode == "CRASH":
                with pytest.raises(RuntimeError, match="before app CAS"):
                    await store.initialize(owner, conversation.id, proxy)
                record = await store.get(owner, conversation.id)
                assert record.accepted_checkpoint_id is None and record.revision == 0
                await initializer.graph.aupdate_state(
                    {
                        "configurable": {
                            "thread_id": str(conversation.id),
                            "checkpoint_id": saved_ref.checkpoint_id,
                        },
                        "metadata": {
                            "app_checkpoint_kind": "debug",
                            "app_initialization_id": str(uuid4()),
                        },
                    },
                    {"public_build": {"unaccepted": True}},
                    as_node="initialize_root",
                )
            else:
                await initializer.graph.aupdate_state(
                    {
                        "configurable": {
                            "thread_id": str(conversation.id),
                            "checkpoint_id": "",
                        },
                        "metadata": {
                            "app_graph_version": "legacy",
                            "app_checkpoint_kind": "legacy",
                        },
                    },
                    {
                        "consultation": {"legacy": True},
                        "requirement_state": {},
                        "public_build": None,
                    },
                    as_node="initialize_root",
                )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            initializer = NativeConversationInitializer(saver)
            if mode == "LEGACY_NATIVE":
                with pytest.raises(ValueError, match="migration"):
                    await store.initialize(owner, conversation.id, initializer)
                assert (
                    await store.get(owner, conversation.id)
                ).accepted_checkpoint_id is None
                assert (
                    len(
                        [
                            row
                            async for row in saver.alist(
                                {"configurable": {"thread_id": str(conversation.id)}}
                            )
                        ]
                    )
                    == 1
                )
            else:
                record = await store.initialize(owner, conversation.id, initializer)
                assert record.accepted_checkpoint_id == saved_ref.checkpoint_id
                assert record.revision == 0 and record.active_run_id is None
                roots = [
                    row
                    async for row in saver.alist(
                        {"configurable": {"thread_id": str(conversation.id)}},
                        filter={
                            "app_checkpoint_kind": "initial_root",
                            "app_graph_version": "conversation-root-v1",
                            "app_initialization_id": str(conversation.id),
                        },
                    )
                ]
                assert len(roots) == 1
    finally:
        if conversation:
            async with engine.begin() as connection:
                await connection.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()


@pytest.mark.asyncio
async def test_create_ready_root_and_reload_preserves_unknown_state(postgres_dsn):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    owner = uuid4()
    conversation_id = None
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            store = PostgresConversationStore(engine)
            service = StatefulConversationService(
                store,
                PostgresAcceptedCheckpointReader(saver),
                NativeConversationInitializer(saver),
            )
            created = await service.create(owner, "Main flow")
            conversation_id = created.conversation_id
            assert created.readiness == "READY" and created.revision == 0
            assert created.state.target_budget_vnd is None and created.build is None
            record = await store.get(owner, conversation_id)
            assert record.accepted_checkpoint_id is not None
            # Calling initialize again uses the existing accepted ref; no new turn.
            again = await store.initialize(
                owner, conversation_id, NativeConversationInitializer(saver)
            )
            assert again.accepted_checkpoint_id == record.accepted_checkpoint_id
        await engine.dispose()
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            restored = await StatefulConversationService(
                PostgresConversationStore(engine),
                PostgresAcceptedCheckpointReader(saver),
                NativeConversationInitializer(saver),
            ).get(owner, conversation_id)
            assert restored == created
    finally:
        if conversation_id is not None:
            async with engine.begin() as connection:
                await connection.execute(
                    delete(conversations).where(conversations.c.id == conversation_id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation_id))
        await engine.dispose()
