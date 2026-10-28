"""Managed comparison selection preserves original source across publication."""

import asyncio
from uuid import uuid4

import pytest
from agent_task_proposals import WithTaskProposals
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine
from test_flexible_graph_postgres import Budget, Catalog

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
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
from ai_service.infrastructure.persistence.schema import conversations


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["TYPED", "FREE", "AMEND", "REPLACE", "DISCARD"])
async def test_managed_task_choice_publishes_comparison_not_choice_message(
    postgres_dsn,
    mode,
):
    free = mode in {"FREE", "AMEND"}
    amend = mode == "AMEND"
    replace = mode == "REPLACE"
    discard = mode == "DISCARD"

    class Products(Catalog):
        def __init__(self):
            super().__init__()
            self.first = self.product
            self.other = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "Alternative",
                    "evidence_ref": "catalog:other",
                }
            )
            self.second = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "Second",
                    "evidence_ref": "catalog:second",
                }
            )
            self.corrected = self.product.model_copy(
                update={
                    "product_id": uuid4(),
                    "name": "Corrected",
                    "evidence_ref": "catalog:corrected",
                }
            )

        async def resolve(self, mention):
            if mention == "GPU C":
                return [self.corrected]
            return [self.first, self.other] if mention == "GPU A" else [self.second]

        async def get(self, identifier):
            return next(
                (
                    p
                    for p in [self.first, self.other, self.second, self.corrected]
                    if p.product_id == identifier
                ),
                None,
            )

    class Extractor:
        calls = 0

        async def extract_tasks(self, message):
            self.calls += 1
            if message == "Find GPU":
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
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "COMPARE",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                        "subjects": [
                            {"start": 3, "end": 8, "quote": "GPU A"},
                            {"start": 12, "end": 17, "quote": "GPU B"},
                        ],
                    }
                ]
            )

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if discard and message == "Bỏ yêu cầu đang chờ":
                raise AssertionError("Discard must not invoke model")
            if not context.completion_requirements:
                return FlexibleDecision(
                    action={
                        "kind": "ASK_CLARIFICATION",
                        "question": "Which GPU?",
                        "missing_fields": ["AMBIGUOUS"],
                    }
                )
            if not transcript:
                if context.completion_requirements[0].kind == "ANSWER":
                    return FlexibleDecision(
                        action={
                            "kind": "CALL_TOOL",
                            "call": {
                                "call_id": "search",
                                "name": "search_catalog",
                                "arguments": {"query": "GPU"},
                            },
                        }
                    )
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "compare",
                            "name": "compare_products",
                            "arguments": {
                                "product_ids": [
                                    str(p)
                                    for p in context.completion_requirements[
                                        0
                                    ].product_ids
                                ]
                            },
                        },
                    }
                )
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "Untrusted summary",
                        "evidence_refs": transcript[-1].result.evidence_refs,
                    },
                }
            )

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
    products, extractor, service = Products(), Extractor(), PCBuildApplicationService()

    class Choices:
        calls = 0

        async def choose_task(self, message, options):
            from ai_service.capabilities.assistant.react.task_resolution import (
                TaskChoiceDraft,
            )

            self.calls += 1
            if amend:
                return TaskChoiceDraft()
            return TaskChoiceDraft(
                selections=[
                    {
                        "task_index": 0,
                        "subject_index": 0,
                        "product_id": products.first.product_id,
                        "evidence": {"start": 0, "end": len(message), "quote": message},
                    }
                ]
            )

    choices = Choices()

    class Amender:
        calls = 0

        async def amend_task(self, message, pending):
            from ai_service.capabilities.assistant.react.task_amendment import (
                TaskAmendment,
            )

            self.calls += 1
            return TaskAmendment(
                edits=[
                    {
                        "task_index": 0,
                        "subject_index": 0,
                        "evidence": {"start": 0, "end": len(message), "quote": message},
                    }
                ]
            )

    amender = Amender()
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "Comparison selection")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(
                engine, saver, request_versions=(2,), flexible_service=service
            )
            reader = PostgresAcceptedCheckpointReader(saver)

            async def factory(claim):
                request, head = await coordinator.load(claim)
                accepted = await reader.read(head.accepted_checkpoint_ref)
                graph = build_flexible_graph(
                    WithTaskProposals(Model(), extractor),
                    ToolDispatcher(products, service),
                    ToolContext(
                        account_id=owner,
                        conversation_id=claim.conversation_id,
                        run_id=claim.run_id,
                        requirements=accepted.requirement_state,
                        read_catalog_allowed=True,
                        execution_allowed=True,
                    ),
                    [],
                    Budget(),
                    conversation_revision=head.revision,
                    publication=FlexiblePublicationInput.for_request(
                        accepted, request, owner=owner
                    ),
                    task_catalog=None
                    if request.pending_action
                    and request.pending_action.action == "DISCARD"
                    else products,
                    checkpointer=saver,
                    task_clarifier=choices if free else None,
                    task_amender=amender if amend else None,
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

            async def run(request, expected):
                submitted = await coordinator.submit(owner, request)
                executor.wake()
                async with asyncio.timeout(5):
                    while True:
                        view = await coordinator.get(
                            owner, conversation.id, submitted.run_id
                        )
                        if (
                            view.status not in {"PENDING", "RUNNING"}
                            or executor.last_error
                        ):
                            break
                        await asyncio.sleep(0.01)
                assert view.status == expected, executor.last_error
                record = await store.get(owner, conversation.id)
                assert record.active_run_id is None
                return await reader.read(
                    CheckpointRefV1(
                        thread_id=record.id, checkpoint_id=record.accepted_checkpoint_id
                    )
                )

            first = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="So GPU A và GPU B",
            )
            waiting = await run(first, "WAITING_INPUT")
            pending = waiting.flexible.pending_task_resolution
            assert pending is not None and extractor.calls == 1
            followup_id = uuid4()
            completed = await run(
                TurnRequestV2(
                    conversation_id=conversation.id,
                    request_id=followup_id,
                    expected_revision=1,
                    message="Bỏ yêu cầu đang chờ"
                    if discard
                    else "Find GPU"
                    if replace
                    else "GPU C"
                    if amend
                    else "Choose first",
                    pending_action={
                        "target": "TASKS",
                        "action": "DISCARD" if discard else "REPLACE",
                        "pending_message_id": pending.message_id,
                        "constraints_revision": pending.constraints_revision,
                    }
                    if replace or discard
                    else None,
                    task_reply=None
                    if free or replace or discard
                    else {
                        "pending_message_id": pending.message_id,
                        "constraints_revision": pending.constraints_revision,
                        "selections": [
                            {
                                "task_index": 0,
                                "subject_index": 0,
                                "product_id": products.first.product_id,
                            }
                        ],
                    },
                ),
                "COMPLETED",
            )
            if discard:
                assert completed.flexible.status == "DISCARDED"
                assert not completed.flexible.task_outcomes
                assert completed.requirement_state == waiting.requirement_state
                assert completed.public_build == waiting.public_build
                assert extractor.calls == 1
                record = await store.get(owner, conversation.id)
                assert record.revision == 2 and record.active_run_id is None
                fresh = await run(
                    TurnRequestV2(
                        conversation_id=conversation.id,
                        request_id=uuid4(),
                        expected_revision=2,
                        message="Find GPU",
                    ),
                    "COMPLETED",
                )
                assert (
                    fresh.flexible.status == "COMPLETED"
                    and fresh.flexible.task_message == "Find GPU"
                )
                assert (await store.get(owner, conversation.id)).revision == 3
                return
            assert completed.flexible.task_message_id == (
                followup_id if replace else first.request_id
            )
            task = completed.flexible.task_outcomes[0]
            assert (
                task.task.kind == ("ANSWER" if replace else "COMPARE")
                and task.status == "COMPLETED"
            )
            if not replace:
                assert task.task.product_ids == [
                    products.corrected.product_id
                    if amend
                    else products.first.product_id,
                    products.second.product_id,
                ]
            assert completed.requirement_state == waiting.requirement_state
            assert completed.public_build is None and extractor.calls == (
                2 if replace else 1
            )
            assert choices.calls == (1 if free else 0)
            assert amender.calls == (1 if amend else 0)
            assert "Untrusted summary" not in completed.flexible.assistant_message
            assert (await store.get(owner, conversation.id)).revision == 2
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
