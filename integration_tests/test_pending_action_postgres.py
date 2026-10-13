"""Whole pending actions after saver reopen use stored request and accepted base."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.contracts import (
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    PendingActionV1,
    TurnRequestV1,
)
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import conversations


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["DISCARD", "REPLACE"])
async def test_pending_action_reopen_publish_and_idempotency(postgres_dsn, action):
    class Extractor:
        def __init__(self):
            self.calls = []

        async def extract(self, message, state):
            self.calls.append(message)
            if message == "Build 30000000 VND gaming 1080p":
                return Draft(
                    items=[
                        DraftItem(
                            subject="SETUP",
                            predicate="budget",
                            operator="LTE",
                            value=QuantityValue(amount=30_000_000, unit="VND"),
                            start=6,
                            end=18,
                        ),
                        DraftItem(
                            target="GOAL",
                            subject="WORKLOAD",
                            predicate="use_case",
                            value=TextValue(text="GAMING_1080P"),
                            start=19,
                            end=len(message),
                        ),
                    ]
                )
            return Draft(
                items=[
                    DraftItem(
                        subject="RAM",
                        predicate="capacity",
                        operator="GTE",
                        value=UnresolvedQuantityValue(amount=32)
                        if message == "RAM 32"
                        else QuantityValue(amount=64, unit="GB"),
                        start=0,
                        end=len(message),
                    )
                ]
            )

    class Catalog:
        async def load(self, draft, state):
            identifier = uuid4()
            return [
                CatalogRowV1(
                    variant_id=identifier,
                    sku="RAM64",
                    product_status="ACTIVE",
                    variant_status="ACTIVE",
                    quantity=1,
                    component=ComponentSpec(
                        id=identifier,
                        name="RAM64",
                        category="RAM",
                        price=1_000_000,
                        capacity_gb=64,
                    ),
                    retrieved_at=datetime.now(UTC),
                    evidence_refs=["test:ram64"],
                )
            ]

    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, extractor = PostgresConversationStore(engine), uuid4(), Extractor()
    conversation = await store.create(owner, "Pending actions")
    pending_id = uuid4()
    executor = None
    try:
        for turn, message in enumerate(
            [
                "Build 30000000 VND gaming 1080p",
                "RAM 32",
                "Bỏ yêu cầu pending" if action == "DISCARD" else "RAM 64GB",
            ]
        ):
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                initializer = NativeConversationInitializer(saver)
                if turn == 0:
                    await store.initialize(owner, conversation.id, initializer)
                coordinator = PostgresRunCoordinator(engine, saver)
                reader = PostgresAcceptedCheckpointReader(saver)
                service = StatefulConversationService(store, reader, initializer)
                request = TurnRequestV1(
                    conversation_id=conversation.id,
                    request_id=pending_id if turn == 1 else uuid4(),
                    expected_revision=turn,
                    message=message,
                    pending_action=PendingActionV1(
                        action=action,
                        pending_message_id=pending_id,
                        constraints_revision=1,
                    )
                    if turn == 2
                    else None,
                )
                admitted = await coordinator.submit(owner, request)

                def factory(claim):
                    return DurableConversationGraph(
                        build_requirement_graph(
                            extractor, Catalog(), checkpointer=saver
                        )
                    )

                executor = ManagedExecutor(
                    coordinator,
                    ConversationExecution(coordinator, factory, "pending-actions-v1"),
                    worker_id=uuid4(),
                    poll_seconds=0.01,
                )
                await executor.start()
                async with asyncio.timeout(5):
                    while True:
                        run = await coordinator.get(
                            owner, conversation.id, admitted.run_id
                        )
                        if run.status.value not in {"PENDING", "RUNNING"}:
                            assert run.status.value == (
                                "WAITING_INPUT" if turn == 1 else "COMPLETED"
                            )
                            break
                        await asyncio.sleep(0.01)
                assert await executor.stop(grace_seconds=5)
                executor = None
                view = await service.get(owner, conversation.id)
                assert view.revision == turn + 1
                assert view.state.target_budget_vnd == 30_000_000
                if turn == 1:
                    assert view.clarification.pending_message_id == pending_id
                if turn == 2:
                    assert view.clarification is None
                    assert await coordinator.submit(owner, request) == run
                    changed = request.model_copy(
                        update={
                            "pending_action": request.pending_action.model_copy(
                                update={
                                    "action": "REPLACE"
                                    if action == "DISCARD"
                                    else "DISCARD"
                                }
                            )
                        }
                    )
                    with pytest.raises(RunAdmissionError, match="AI_REQUEST_CONFLICT"):
                        await coordinator.submit(owner, changed)
                    head = await store.get(owner, conversation.id)
                    snapshot = await reader.read(
                        CheckpointRefV1(
                            thread_id=conversation.id,
                            checkpoint_id=head.accepted_checkpoint_id,
                        )
                    )
                    ram = next(
                        (
                            r
                            for r in snapshot.requirement_state.requirements.values()
                            if r.subject == "RAM"
                        ),
                        None,
                    )
                    if action == "DISCARD":
                        assert ram is None
                        assert extractor.calls == [
                            "Build 30000000 VND gaming 1080p",
                            "RAM 32",
                        ]
                    else:
                        assert ram.value.amount == 64
                        assert ram.evidence.message_id == request.request_id
                    history = await service.messages(owner, conversation.id, 20, None)
                    assert history.items[-1].content.startswith(
                        "Đã bỏ" if action == "DISCARD" else "INFEASIBLE"
                    )
    finally:
        if executor is not None:
            assert await executor.stop(grace_seconds=5)
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation.id))
        async with engine.begin() as connection:
            await connection.execute(
                delete(conversations).where(conversations.c.id == conversation.id)
            )
        await engine.dispose()
