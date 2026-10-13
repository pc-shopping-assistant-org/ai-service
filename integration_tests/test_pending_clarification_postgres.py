"""Pending SKU choices survive closed pools and publish from the accepted head."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.contracts import (
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    ClarificationSelectionV1,
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
from ai_service.infrastructure.persistence.schema import (
    agent_call_attempts,
    conversations,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("free_text", [False, True])
async def test_postgres_pending_choice_reopen_atomic_merge_and_public_projection(
    postgres_dsn,
    free_text,
):
    class Extractor:
        calls = 0

        async def extract(self, message, state):
            self.calls += 1
            assert self.calls == 1
            return Draft(
                items=[
                    DraftItem(
                        subject="SETUP",
                        predicate="budget",
                        operator="LTE",
                        value=QuantityValue(amount=25_000_000, unit="VND"),
                        start=0,
                        end=len(message),
                    ),
                    DraftItem(
                        target="GOAL",
                        subject="WORKLOAD",
                        predicate="use_case",
                        value=TextValue(text="GAMING_1080P"),
                        start=0,
                        end=len(message),
                    ),
                    DraftItem(
                        subject="GPU",
                        predicate="identity",
                        value=TextValue(text="RTX 5070 Ti"),
                        start=0,
                        end=len(message),
                    ),
                ]
            )

    class Catalog:
        async def load(self, draft, state):
            return rows

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        async def ainvoke(self, messages):
            return {
                "selections": [
                    {
                        "item_index": 2,
                        "variant_id": str(rows[1].variant_id),
                        "start": 0,
                        "end": len("Chọn bản thứ hai"),
                        "quote": "Chọn bản thứ hai",
                    }
                ]
            }

    class Provider:
        def create_model(self):
            return Model()

    rows = [
        CatalogRowV1(
            variant_id=(identity := uuid4()),
            sku=f"TUF-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            retrieved_at=datetime.now(UTC),
            evidence_refs=[f"test:{identity}"],
            component=ComponentSpec(
                id=identity, name="RTX 5070 Ti", category="GPU", price=10_000_000
            ),
        )
        for index in range(2)
    ]
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, extractor = PostgresConversationStore(engine), uuid4(), Extractor()
    created = await store.create(owner, "Pending choice")
    source_id = uuid4()
    executor = None
    try:
        for turn in range(2):
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                initializer = NativeConversationInitializer(saver)
                if turn == 0:
                    await store.initialize(owner, created.id, initializer)
                coordinator = PostgresRunCoordinator(engine, saver)
                service = StatefulConversationService(
                    store, PostgresAcceptedCheckpointReader(saver), initializer
                )
                request = TurnRequestV1(
                    conversation_id=created.id,
                    request_id=source_id if turn == 0 else uuid4(),
                    expected_revision=turn,
                    message="Build 25 triệu giữ RTX 5070 Ti"
                    if turn == 0
                    else "Chọn bản thứ hai",
                    clarification_reply=None
                    if turn == 0 or free_text
                    else ClarificationReplyV1(
                        pending_message_id=source_id,
                        selections=[
                            ClarificationSelectionV1(
                                item_index=2, variant_id=rows[1].variant_id
                            )
                        ],
                    ),
                )
                admitted = await coordinator.submit(owner, request)

                def factory(claim, active_saver=saver, active_coordinator=coordinator):
                    return DurableConversationGraph(
                        build_requirement_graph(
                            extractor,
                            Catalog(),
                            checkpointer=active_saver,
                            clarifier=NativeRequirementExtractor(
                                Provider(),
                                AttemptBoundary(active_coordinator.ledger, claim),
                            ),
                        )
                    )

                executor = ManagedExecutor(
                    coordinator,
                    ConversationExecution(coordinator, factory, "pending-v1"),
                    worker_id=uuid4(),
                    poll_seconds=0.01,
                )
                await executor.start()
                expected_status = "WAITING_INPUT" if turn == 0 else "COMPLETED"
                async with asyncio.timeout(5):
                    while True:
                        run = await coordinator.get(owner, created.id, admitted.run_id)
                        if run.status.value not in {"PENDING", "RUNNING"}:
                            assert run.status.value == expected_status
                            break
                        await asyncio.sleep(0.01)
                assert await executor.stop(grace_seconds=5)
                executor = None
                view = await service.get(owner, created.id)
                assert view.revision == turn + 1
                if turn == 0:
                    assert view.clarification.pending_message_id == source_id
                    assert (
                        view.clarification.items[0].options[1].variant_id
                        == rows[1].variant_id
                    )
                    assert (
                        view.state.target_budget_vnd is None
                    )  # Still wholly unmerged.
                else:
                    assert view.clarification is None
                    assert view.state.target_budget_vnd == 25_000_000
                    assert (
                        view.state.pinned_parts["GPU"].variant_id == rows[1].variant_id
                    )
                    history = await service.messages(owner, created.id, 20, None)
                    assert history.items[-1].content.startswith("INFEASIBLE")
                # Each iteration closes the native saver connection before the next.
        assert extractor.calls == 1
        async with engine.connect() as connection:
            count = (
                await connection.execute(
                    select(func.count())
                    .select_from(agent_call_attempts)
                    .where(agent_call_attempts.c.run_id == admitted.run_id)
                )
            ).scalar_one()
            assert count == int(free_text)
    finally:
        if executor is not None:
            assert await executor.stop(grace_seconds=5)
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(created.id))
        async with engine.begin() as connection:
            await connection.execute(
                delete(conversations).where(conversations.c.id == created.id)
            )
        await engine.dispose()
