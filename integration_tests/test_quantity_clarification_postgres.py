"""Accepted pending units survive closed saver pools and managed publication."""

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
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    TurnRequestV1,
    UnitClarificationReplyV1,
    UnitSelectionV1,
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
@pytest.mark.parametrize(
    "reply_kind", ["UNIT_TYPED", "UNIT_FREE", "AMEND_AMOUNT", "DISCARD_RAM"]
)
async def test_quantity_unit_reopen_atomic_merge_and_native_publish(
    postgres_dsn, reply_kind
):
    message = "Build 25000000 VND gaming 1080p, RAM 32"

    class Extractor:
        calls = 0

        async def extract(self, text, state):
            self.calls += 1
            assert self.calls == 1
            return Draft(
                items=[
                    DraftItem(
                        subject="SETUP",
                        predicate="budget",
                        operator="LTE",
                        value=QuantityValue(amount=25_000_000, unit="VND"),
                        start=6,
                        end=18,
                    ),
                    DraftItem(
                        target="GOAL",
                        subject="WORKLOAD",
                        predicate="use_case",
                        value=TextValue(text="GAMING_1080P"),
                        start=19,
                        end=31,
                    ),
                    DraftItem(
                        subject="RAM",
                        predicate="capacity",
                        operator="GTE",
                        value=UnresolvedQuantityValue(amount=32),
                        start=33,
                        end=len(message),
                    ),
                ]
            )

    identifier = uuid4()
    canonical = CatalogRowV1(
        variant_id=identifier,
        sku="RAM32",
        product_status="ACTIVE",
        variant_status="ACTIVE",
        quantity=1,
        retrieved_at=datetime.now(UTC),
        evidence_refs=["test:ram32"],
        component=ComponentSpec(
            id=identifier,
            name="RAM 32GB",
            category="RAM",
            price=1_000_000,
            capacity_gb=32,
        ),
    )

    class Catalog:
        async def load(self, draft, state):
            return [canonical]

    class Model:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema.__name__
            return self

        async def ainvoke(self, messages):
            if reply_kind in {"AMEND_AMOUNT", "DISCARD_RAM"}:
                if self.schema == "UnitChoiceDraft":
                    return {"selections": []}
                assert self.schema == "PendingAmendment"
                if reply_kind == "DISCARD_RAM":
                    return {
                        "discards": [
                            {"item_index": 2, "start": 0, "end": 6, "quote": "Bỏ RAM"}
                        ]
                    }
                return {
                    "replacements": [
                        {
                            "item_index": 2,
                            "item": DraftItem(
                                subject="RAM",
                                predicate="capacity",
                                operator="GTE",
                                value=QuantityValue(amount=64, unit="GB"),
                                start=0,
                                end=4,
                            ).model_dump(mode="json"),
                        }
                    ]
                }
            return {
                "selections": [
                    {"item_index": 2, "unit": "GB", "start": 0, "end": 2, "quote": "GB"}
                ]
            }

    class Provider:
        def create_model(self):
            return Model()

    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, source, extractor = (
        PostgresConversationStore(engine),
        uuid4(),
        uuid4(),
        Extractor(),
    )
    conversation = await store.create(owner, "Unit confirmation")
    executor = None
    try:
        for turn in range(2):
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                initializer = NativeConversationInitializer(saver)
                if turn == 0:
                    await store.initialize(owner, conversation.id, initializer)
                coordinator = PostgresRunCoordinator(engine, saver)
                reader = PostgresAcceptedCheckpointReader(saver)
                service = StatefulConversationService(store, reader, initializer)
                request = TurnRequestV1(
                    conversation_id=conversation.id,
                    request_id=source if turn == 0 else uuid4(),
                    expected_revision=turn,
                    message=message
                    if turn == 0
                    else "64GB"
                    if reply_kind == "AMEND_AMOUNT"
                    else "Bỏ RAM"
                    if reply_kind == "DISCARD_RAM"
                    else "GB nhé",
                    unit_clarification_reply=None
                    if turn == 0 or reply_kind != "UNIT_TYPED"
                    else UnitClarificationReplyV1(
                        pending_message_id=source,
                        constraints_revision=0,
                        selections=[
                            UnitSelectionV1(
                                item_index=2, unit="GB", start=0, end=2, quote="GB"
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
                            unit_clarifier=NativeRequirementExtractor(
                                Provider(),
                                AttemptBoundary(active_coordinator.ledger, claim),
                            ),
                            amender=NativeRequirementExtractor(
                                Provider(),
                                AttemptBoundary(active_coordinator.ledger, claim),
                            ),
                        )
                    )

                executor = ManagedExecutor(
                    coordinator,
                    ConversationExecution(coordinator, factory, "units-v1"),
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
                                "WAITING_INPUT" if turn == 0 else "COMPLETED"
                            )
                            break
                        await asyncio.sleep(0.01)
                assert await executor.stop(grace_seconds=5)
                executor = None
                view = await service.get(owner, conversation.id)
                assert view.revision == turn + 1
                if turn == 0:
                    assert view.state.target_budget_vnd is None
                    assert view.clarification.items[0].quantity_amount == 32
                    assert "GB" in view.clarification.items[0].unit_options
                else:
                    assert view.clarification is None
                    assert view.state.target_budget_vnd == 25_000_000
                    head = await store.get(owner, conversation.id)
                    snapshot = await reader.read(
                        CheckpointRefV1(
                            thread_id=conversation.id,
                            checkpoint_id=head.accepted_checkpoint_id,
                        )
                    )
                    ram = next(
                        (
                            item
                            for item in snapshot.requirement_state.requirements.values()
                            if item.subject == "RAM"
                        ),
                        None,
                    )
                    if reply_kind == "DISCARD_RAM":
                        assert ram is None
                    elif reply_kind == "AMEND_AMOUNT":
                        assert ram.value == QuantityValue(amount=64, unit="GB")
                        assert ram.evidence.message_id == request.request_id
                    else:
                        assert ram.value == QuantityValue(amount=32, unit="GB")
                        assert ram.evidence.message_id == source
                        assert (
                            ram.evidence.supporting_evidence[0].message_id
                            == request.request_id
                        )
                    history = await service.messages(owner, conversation.id, 20, None)
                    assert history.items[-1].content.startswith("INFEASIBLE")
        async with engine.connect() as connection:
            attempts = (
                await connection.execute(
                    select(func.count())
                    .select_from(agent_call_attempts)
                    .where(agent_call_attempts.c.run_id == admitted.run_id)
                )
            ).scalar_one()
            assert (
                attempts
                == {
                    "UNIT_TYPED": 0,
                    "UNIT_FREE": 1,
                    "AMEND_AMOUNT": 2,
                    "DISCARD_RAM": 2,
                }[reply_kind]
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
