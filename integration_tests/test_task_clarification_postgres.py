"""Native accepted pending task survives saver reopen and managed publication."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    ProductEvidence,
    ToolCall,
    ToolContext,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
from ai_service.capabilities.assistant.stateful_contracts import (
    TaskClarificationReplyV1,
    TaskProductSelectionV1,
    TurnRequestV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.graph.react import ReactDecision
from ai_service.infrastructure.graph.requirement_build import (
    ReactComposition,
    build_requirement_graph,
)
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
@pytest.mark.parametrize("reply_kind", ["TYPED", "FREE_TEXT", "AMEND_MENTION"])
async def test_managed_comparison_reopen_keeps_requirements_and_consumes_product_choice(
    postgres_dsn,
    reply_kind,
):
    message = "So RTX 4060 và RTX 4070, budget 25000000 VND"
    source = uuid4()
    owner = uuid4()
    products = [
        ProductEvidence(
            product_id=uuid4(),
            name=name,
            price_vnd=price,
            evidence_ref=f"catalog:{index}",
            retrieved_at=datetime.now(UTC),
        )
        for index, (name, price) in enumerate(
            [
                ("ASUS RTX 4060", 8_000_000),
                ("MSI RTX 4060", 7_000_000),
                ("RTX 4070", 15_000_000),
            ]
        )
    ]

    class Extractor:
        calls = 0

        async def extract(self, user_message, state):
            self.calls += 1
            assert self.calls == 1
            return Draft(
                items=[
                    DraftItem(
                        subject="SETUP",
                        predicate="budget",
                        operator="LTE",
                        value=QuantityValue(amount=25_000_000, unit="VND"),
                        start=message.index("25000000"),
                        end=len(message),
                    )
                ],
                tasks=[
                    {
                        "kind": "COMPARE",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                        "subjects": [
                            {"start": 3, "end": 11, "quote": "RTX 4060"},
                            {"start": 15, "end": 23, "quote": "RTX 4070"},
                        ],
                    }
                ],
            )

    class Hardware:
        async def load(self, draft, state):
            assert draft.items
            return []

    class Catalog:
        async def resolve(self, mention):
            if mention == "MSI RTX 4060":
                return products[1:2]
            if mention == "RTX 4060" and reply_kind == "AMEND_MENTION":
                return []
            return products[:2] if mention == "RTX 4060" else products[2:]

        async def get(self, product_id):
            return next(
                (item for item in products if item.product_id == product_id), None
            )

    class Model:
        async def decide(self, user_message, transcript, allowed, repair, context=None):
            assert context.task_message == message
            task = context.completion_requirements[0]
            assert task.kind == "COMPARE"
            assert task.product_ids == [products[1].product_id, products[2].product_id]
            if not transcript:
                return ReactDecision(
                    calls=[
                        ToolCall(
                            call_id="compare",
                            name="compare_products",
                            arguments={
                                "product_ids": [
                                    str(identifier) for identifier in task.product_ids
                                ]
                            },
                        )
                    ]
                )
            return ReactDecision(
                answer=AnswerCandidate(
                    text="Comparison", evidence_refs=transcript[-1].result.evidence_refs
                )
            )

    class ChoiceModel:
        def with_structured_output(self, schema, **kwargs):
            self.schema = schema.__name__
            return self

        async def ainvoke(self, messages):
            if reply_kind == "AMEND_MENTION":
                assert self.schema == "TaskAmendment"
                text = "MSI RTX 4060"
                return {
                    "edits": [
                        {
                            "task_index": 0,
                            "subject_index": 0,
                            "evidence": {"start": 0, "end": len(text), "quote": text},
                        }
                    ]
                }
            text = "Chọn MSI"
            return {
                "selections": [
                    {
                        "task_index": 0,
                        "subject_index": 0,
                        "product_id": str(products[1].product_id),
                        "evidence": {"start": 0, "end": len(text), "quote": text},
                    }
                ]
            }

    class ChoiceProvider:
        def create_model(self):
            return ChoiceModel()

    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, extractor = PostgresConversationStore(engine), Extractor()
    created = await store.create(owner, "Task clarification")
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
                before = await service.get(owner, created.id)
                reply = None
                if turn:
                    assert before.task_clarification.pending_message_id == source
                    assert before.state.target_budget_vnd == 25_000_000
                    offered = before.task_clarification.items[0]
                    reply = (
                        TaskClarificationReplyV1(
                            pending_message_id=source,
                            constraints_revision=before.task_clarification.constraints_revision,
                            selections=[
                                TaskProductSelectionV1(
                                    task_index=offered.task_index,
                                    subject_index=offered.subject_index,
                                    product_id=offered.options[1].product_id,
                                )
                            ],
                        )
                        if reply_kind != "AMEND_MENTION"
                        else None
                    )
                request = TurnRequestV1(
                    conversation_id=created.id,
                    request_id=source if not turn else uuid4(),
                    expected_revision=turn,
                    message=message
                    if not turn
                    else "MSI RTX 4060"
                    if reply_kind == "AMEND_MENTION"
                    else "Chọn MSI",
                    task_clarification_reply=reply if reply_kind == "TYPED" else None,
                )
                admitted = await coordinator.submit(owner, request)
                # Additive reply fields still participate in normal idempotent admission.
                assert (
                    await coordinator.submit(owner, request)
                ).run_id == admitted.run_id

                def factory(claim, active_saver=saver, active_coordinator=coordinator):
                    trusted = ToolContext(
                        account_id=owner,
                        conversation_id=created.id,
                        run_id=claim.run_id,
                        requirements=CanonicalRequirementState(),
                        execution_allowed=True,
                        read_catalog_allowed=True,
                    )
                    return DurableConversationGraph(
                        build_requirement_graph(
                            extractor,
                            Hardware(),
                            checkpointer=active_saver,
                            react=ReactComposition(
                                Model(),
                                ToolDispatcher(Catalog(), PCBuildApplicationService()),
                                trusted,
                                Catalog(),
                                AttemptBoundary(active_coordinator.ledger, claim),
                                task_clarifier=NativeRequirementExtractor(
                                    ChoiceProvider(),
                                    AttemptBoundary(active_coordinator.ledger, claim),
                                ),
                                task_amender=NativeRequirementExtractor(
                                    ChoiceProvider(),
                                    AttemptBoundary(active_coordinator.ledger, claim),
                                ),
                            ),
                        )
                    )

                executor = ManagedExecutor(
                    coordinator,
                    ConversationExecution(
                        coordinator, factory, "task-clarification-v1"
                    ),
                    worker_id=uuid4(),
                    poll_seconds=0.01,
                )
                await executor.start()
                async with asyncio.timeout(5):
                    while True:
                        run = await coordinator.get(owner, created.id, admitted.run_id)
                        if run.status.value not in {"PENDING", "RUNNING"}:
                            assert run.status.value == (
                                "WAITING_INPUT" if not turn else "COMPLETED"
                            )
                            break
                        await asyncio.sleep(0.01)
                assert await executor.stop(grace_seconds=5)
                executor = None
                view = await service.get(owner, created.id)
                assert view.revision == turn + 1
                assert view.state.target_budget_vnd == 25_000_000
                if not turn:
                    if reply_kind == "AMEND_MENTION":
                        assert view.task_clarification.items[0].reason == "NOT_FOUND"
                        assert view.task_clarification.items[0].options == []
                    else:
                        assert (
                            view.task_clarification.items[0].options[1].product_id
                            == products[1].product_id
                        )
                else:
                    assert (
                        view.task_clarification is None and view.clarification is None
                    )
                    history = await service.messages(owner, created.id, 20, None)
                    assert "MSI RTX 4060" in history.items[-1].content
                    assert "RTX 4070" in history.items[-1].content
                    assert "ASUS RTX 4060" not in history.items[-1].content
        assert extractor.calls == 1
        async with engine.connect() as connection:
            count = (
                await connection.execute(
                    select(func.count())
                    .select_from(agent_call_attempts)
                    .where(agent_call_attempts.c.run_id == admitted.run_id)
                )
            ).scalar_one()
            assert count == 2 + int(reply_kind != "TYPED")
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
