"""Actual V2 admission/executor/root after saver reopen; model/catalog fixtures."""

import asyncio
from uuid import uuid4

import pytest
from agent_task_proposals import WithTaskProposals
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from test_flexible_graph_postgres import Catalog, Model

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import (
    FLEXIBLE_GRAPH_VERSION,
    build_flexible_graph,
)
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import (
    agent_runs,
    conversation_messages,
    conversations,
)


@pytest.mark.asyncio
async def test_failed_first_v2_run_releases_then_next_run_uses_root_after_reopen(
    postgres_dsn,
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
    service = PCBuildApplicationService()

    class Products(Catalog):
        async def resolve(self, mention):
            return []

    class FirstFailure(Model):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if transcript:
                raise ValueError("Scripted terminal error after durable tool commit")
            return await super().decide(message, transcript, allowed, repair, context)

    try:
        for phase in ("FAIL", "NEXT"):
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.setup()
                if phase == "FAIL":
                    conversation = await store.create(owner, "First-run audit")
                    initialized = await store.initialize(
                        owner, conversation.id, NativeConversationInitializer(saver)
                    )
                    root = CheckpointRefV1(
                        thread_id=conversation.id,
                        checkpoint_id=initialized.accepted_checkpoint_id,
                    )
                else:
                    latest = await saver.aget_tuple(
                        {"configurable": {"thread_id": str(conversation.id)}}
                    )
                    assert (
                        latest.config["configurable"]["checkpoint_id"]
                        != root.checkpoint_id
                    )
                coordinator = PostgresRunCoordinator(
                    engine, saver, request_versions=(2,), flexible_service=service
                )
                reader = PostgresAcceptedCheckpointReader(saver)
                products = Products()
                products.product.name = (
                    "Orphan GPU" if phase == "FAIL" else "Accepted GPU"
                )
                products.product.evidence_ref = (
                    "catalog:orphan" if phase == "FAIL" else "catalog:accepted"
                )
                model = FirstFailure() if phase == "FAIL" else Model()

                async def factory(
                    claim,
                    coordinator=coordinator,
                    root=root,
                    reader=reader,
                    conversation=conversation,
                    model=model,
                    products=products,
                    saver=saver,
                ):
                    request, head = await coordinator.load(claim)
                    assert head.revision == 0 and head.accepted_checkpoint_ref == root
                    accepted = await reader.read(root)
                    assert (
                        not accepted.requirement_state.requirements
                        and accepted.flexible is None
                    )
                    attempts = AttemptBoundary(coordinator.ledger, claim)

                    class Extractor:
                        async def extract_tasks(self, message):
                            return TaskDraftBatch(
                                tasks=[
                                    {
                                        "kind": "ANSWER",
                                        "start": 0,
                                        "end": len(message),
                                        "quote": message,
                                    }
                                ]
                            )

                    context = ToolContext(
                        account_id=owner,
                        conversation_id=conversation.id,
                        run_id=claim.run_id,
                        requirements=accepted.requirement_state,
                        read_catalog_allowed=True,
                        execution_allowed=True,
                    )
                    graph = build_flexible_graph(
                        WithTaskProposals(model, Extractor()),
                        ToolDispatcher(products, service),
                        context,
                        [],
                        attempts,
                        conversation_revision=0,
                        publication=FlexiblePublicationInput.for_request(
                            accepted, request
                        ),
                        task_catalog=products,
                        checkpointer=saver,
                    )
                    return DurableConversationGraph(
                        graph, accepted_versions={FLEXIBLE_GRAPH_VERSION}
                    )

                executor = ManagedExecutor(
                    coordinator,
                    ConversationExecution(coordinator, factory, FLEXIBLE_GRAPH_VERSION),
                    worker_id=uuid4(),
                    poll_seconds=0.02,
                )
                await executor.start()
                request = TurnRequestV2(
                    conversation_id=conversation.id,
                    request_id=uuid4(),
                    expected_revision=0,
                    message=f"GPU {phase}?",
                )
                submitted = await coordinator.submit(owner, request)
                executor.wake()
                expected = "FAILED_TERMINAL" if phase == "FAIL" else "COMPLETED"
                for _ in range(200):
                    view = await coordinator.get(
                        owner, conversation.id, submitted.run_id
                    )
                    if view.status == expected or executor.last_error:
                        break
                    await asyncio.sleep(0.02)
                assert view.status == expected, executor.last_error
                head = await store.get(owner, conversation.id)
                assert head.active_run_id is None and products.calls == 1
                assert (
                    await coordinator.submit(owner, request)
                ).run_id == submitted.run_id
                if phase == "FAIL":
                    assert (
                        head.revision == 0
                        and head.accepted_checkpoint_id == root.checkpoint_id
                    )
                    async with engine.connect() as conn:
                        run = (
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
                    assert run["executor_stop_evidence"]["kind"] == "EXECUTOR_STOP_ACK"
                    assert (
                        run["last_checkpoint_ref"]["checkpoint_id"]
                        != root.checkpoint_id
                    )
                else:
                    assert (
                        head.revision == 1
                        and head.accepted_checkpoint_id != root.checkpoint_id
                    )
                    ref = CheckpointRefV1(
                        thread_id=head.id, checkpoint_id=head.accepted_checkpoint_id
                    )
                    accepted = await reader.read(ref)
                    assert "Accepted GPU" in accepted.flexible.assistant_message
                    assert "Orphan GPU" not in accepted.flexible.assistant_message
                    assert accepted.flexible.answer.evidence_refs == [
                        "catalog:accepted"
                    ]
                    native = await saver.aget_tuple(
                        DurableConversationGraph.config(ref)
                    )
                    assert (
                        native.metadata["app_base_checkpoint_id"] == root.checkpoint_id
                    )
                    assert len(native.checkpoint["channel_values"]["ledger"]) == 1
                    async with engine.connect() as conn:
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
                    assert (
                        len(messages) == 3
                        and sum(row["role"] == "assistant" for row in messages) == 1
                    )
                assert await executor.stop(grace_seconds=5)
                executor = None
    finally:
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
