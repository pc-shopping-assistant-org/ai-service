"""Explicit-user admission with managed SQL/native execution; LLM/catalog fixtures."""

import asyncio
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from test_flexible_managed_build_postgres import CountingService, Products, hardware

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import (
    ExecutionFailed,
    ManagedExecutor,
)
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
    conversation_messages,
    conversations,
)

MESSAGE = "Build 30000000 VND GAMING_1080P"


def user_draft():
    return Draft(
        items=[
            {
                "subject": "SETUP",
                "predicate": "budget",
                "operator": "LTE",
                "start": len("Build "),
                "end": len("Build 30000000 VND"),
                "value": {"kind": "QUANTITY", "amount": 30_000_000, "unit": "VND"},
            },
            {
                "target": "GOAL",
                "subject": "WORKLOAD",
                "predicate": "use_case",
                "start": len("Build 30000000 VND "),
                "end": len(MESSAGE),
                "value": {"kind": "TEXT", "text": "GAMING_1080P"},
            },
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("read_first", [False, True])
async def test_user_requirements_publish_without_confirmation_and_survive_recovery(
    postgres_dsn, recover, read_first
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
    service, catalog, gate, extractors = (
        CountingService(),
        hardware(),
        asyncio.Event(),
        [],
    )

    class Extractor:
        calls = 0

        def __init__(self, attempts):
            self.attempts = attempts

        async def extract_tasks(self, message):
            async def generate():
                return TaskDraftBatch(
                    tasks=[
                        {
                            "kind": "BUILD",
                            "start": 0,
                            "end": len(message),
                            "quote": message,
                        }
                    ]
                )

            return await self.attempts.invoke(
                AttemptKind.MODEL,
                "tasks:register:v2",
                canonical_hash(message),
                generate,
            )

        async def extract(self, message, state):
            async def generate():
                self.calls += 1
                assert message == MESSAGE and not state.requirements
                return user_draft()

            fingerprint = canonical_hash(
                {"message": message, "state": state.model_dump(mode="json")}
            )
            return await self.attempts.invoke(
                AttemptKind.MODEL,
                "requirements:extract:" + fingerprint,
                fingerprint,
                generate,
            )

    class Model:
        failed = False

        async def decide(self, message, transcript, allowed, repair, context=None):
            if not context.requirements.requirements:
                if read_first and not transcript:
                    return FlexibleDecision(
                        action={
                            "kind": "CALL_TOOL",
                            "call": {
                                "call_id": "before",
                                "name": "search_catalog",
                                "arguments": {"query": "GPU"},
                            },
                        }
                    )
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "draft": user_draft().model_dump(),
                    }
                )
            if not transcript:
                if recover and not self.failed:
                    self.failed = True
                    raise ExecutionFailed(RetryFailure.UPSTREAM_TIMEOUT)
                if recover:
                    await gate.wait()
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "build",
                            "name": "build_pc",
                            "arguments": {},
                        },
                    }
                )
            result = transcript[-1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "BUILT",
                    "answer": {
                        "text": "Every game 4K",
                        "build_ref": result.result.build_ref,
                        "evidence_refs": result.evidence_refs,
                    },
                }
            )

    model = Model()
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "Explicit-user fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(
                engine,
                saver,
                request_versions=(2,),
                flexible_service=service,
                max_attempts=2,
                auto_recovery=recover,
            )
            reader = PostgresAcceptedCheckpointReader(saver)

            async def factory(claim):
                request, head = await coordinator.load(claim)
                accepted = await reader.read(head.accepted_checkpoint_ref)
                attempts = AttemptBoundary(coordinator.ledger, claim)
                extractor = Extractor(attempts)
                extractors.append(extractor)
                context = ToolContext(
                    account_id=owner,
                    conversation_id=claim.conversation_id,
                    run_id=claim.run_id,
                    requirements=accepted.requirement_state,
                    catalog=catalog,
                    read_catalog_allowed=True,
                    execution_allowed=True,
                )
                products = Products()
                flow = build_flexible_graph(
                    model,
                    ToolDispatcher(products, service),
                    context,
                    [],
                    attempts,
                    conversation_revision=head.revision,
                    publication=FlexiblePublicationInput.for_request(accepted, request),
                    task_extractor=extractor,
                    task_catalog=products,
                    user_extractor=extractor,
                    checkpointer=saver,
                )
                return DurableConversationGraph(
                    flow, accepted_versions={FLEXIBLE_GRAPH_VERSION}
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
                message=MESSAGE,
            )
            submitted = await coordinator.submit(owner, request)
            executor.wake()
            if recover:
                for _ in range(200):
                    view = await coordinator.get(
                        owner, conversation.id, submitted.run_id
                    )
                    if view.status == "RECOVERING" or executor.last_error:
                        break
                    await asyncio.sleep(0.02)
                assert view.status == "RECOVERING", executor.last_error
                held = await store.get(owner, conversation.id)
                assert held.revision == 0 and held.active_run_id == submitted.run_id
                original = await reader.read(
                    CheckpointRefV1(
                        thread_id=held.id, checkpoint_id=held.accepted_checkpoint_id
                    )
                )
                assert (
                    not original.requirement_state.requirements
                    and original.public_build is None
                )
                gate.set()
            for _ in range(200):
                view = await coordinator.get(owner, conversation.id, submitted.run_id)
                if view.status == "COMPLETED" or executor.last_error:
                    break
                await asyncio.sleep(0.02)
            assert view.status == "COMPLETED", executor.last_error
            head = await store.get(owner, conversation.id)
            accepted = await reader.read(
                CheckpointRefV1(
                    thread_id=head.id, checkpoint_id=head.accepted_checkpoint_id
                )
            )
            assert head.revision == 1 and head.active_run_id is None
            assert (
                len(accepted.requirement_state.requirements)
                == len(accepted.requirement_state.goals)
                == 1
            )
            assert (
                accepted.flexible.pending_proposal is None
                and accepted.flexible.status == "COMPLETED"
            )
            assert 0 < accepted.public_build.total_spending_vnd <= 30_000_000
            assert (
                service.calls == sum(extractor.calls for extractor in extractors) == 1
            )
            terminal = await saver.aget_tuple(
                DurableConversationGraph.config(
                    CheckpointRefV1(
                        thread_id=head.id, checkpoint_id=head.accepted_checkpoint_id
                    )
                )
            )
            ledger = terminal.checkpoint["channel_values"]["ledger"]
            assert len(ledger) == (2 if read_first else 1)
            if read_first:
                assert ledger[0]["call"]["call_id"] == "before"
                assert ledger[0]["result"]["constraints_revision"] == 0
                assert (
                    ledger[-1]["result"]["constraints_revision"]
                    == accepted.requirement_state.revision
                )
            assert "Every game" not in accepted.flexible.assistant_message
            assert (await coordinator.submit(owner, request)).run_id == submitted.run_id
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
            assert len(messages) == 2
    finally:
        gate.set()
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
