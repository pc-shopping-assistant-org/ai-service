"""Cleanup releases a stopped retry, never a live or unproven execution."""

from datetime import UTC, datetime, timedelta
from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.capabilities.assistant.stateful_contracts import ExecutionStopEvidenceV1
from ai_service.infrastructure.execution.supervisor import ExecutionOutcome
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import (
    agent_runs,
    conversation_messages,
    conversations,
)


class Work(TypedDict):
    committed: bool
    message: str
    message_id: str


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    ["EXPIRED", "EXHAUSTED", "CANCEL", "MISSING_PROOF", "STALE_PROOF", "RUNNING"],
)
async def test_retry_cleanup_requires_stored_current_stop_ack(postgres_dsn, scenario):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, conversation = PostgresConversationStore(engine), uuid4(), None
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            conversation = await store.create(owner, "Retry cleanup fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(
                engine, saver, request_versions=(2,), max_attempts=2
            )
            request = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="GPU?",
            )
            submitted = await coordinator.submit(owner, request)
            claim = (await coordinator.claim_pending(uuid4(), 1))[0]

            async def work(state):
                return {"committed": True}

            async def failed(state):
                raise TimeoutError("fixture timeout")

            builder = StateGraph(Work)
            builder.add_node("work", work)
            builder.add_node("failed", failed)
            builder.add_edge(START, "work")
            builder.add_edge("work", "failed")
            builder.add_edge("failed", END)
            graph = DurableConversationGraph(builder.compile(checkpointer=saver))
            _, head = await coordinator.load(claim)

            async def observed(ref):
                await coordinator.observe_checkpoint(claim, ref, FLEXIBLE_GRAPH_VERSION)

            with pytest.raises(TimeoutError):
                await graph.new_turn(
                    head,
                    claim.run_id,
                    request.execution_input(),
                    FLEXIBLE_GRAPH_VERSION,
                    execution=claim,
                    checkpoint_sink=observed,
                )
            proof = ExecutionStopEvidenceV1(
                kind="EXECUTOR_STOP_ACK",
                execution_id=claim.execution_id,
                worker_instance_id=claim.worker_instance_id,
                generation=claim.generation,
                stopped_at=datetime.now(UTC),
                checkpoint_writes_drained=True,
                proof_ref="fixture:awaited-native-failure",
            )
            await coordinator.stopped(
                claim,
                proof,
                ExecutionOutcome(
                    status="FAILED", failure=RetryFailure.UPSTREAM_TIMEOUT
                ),
            )
            before = await store.get(owner, conversation.id)
            assert before.active_run_id == submitted.run_id
            values = {"retry_deadline": datetime.now(UTC) - timedelta(seconds=1)}
            if scenario == "EXHAUSTED":
                values = {"attempt_count": 2}
            elif scenario == "CANCEL":
                values = {"cancel_requested": True}
            elif scenario == "MISSING_PROOF":
                values["executor_stop_evidence"] = None
            elif scenario == "STALE_PROOF":
                values["executor_stop_evidence"] = proof.model_copy(
                    update={"generation": claim.generation + 1}
                ).model_dump(mode="json")
            elif scenario == "RUNNING":
                values.update(status="RUNNING", executor_stop_evidence=None)
            async with engine.begin() as conn:
                await conn.execute(
                    update(agent_runs)
                    .where(agent_runs.c.id == submitted.run_id)
                    .values(**values)
                )
            assert await coordinator.claim_pending(uuid4(), 1) == []
            after = await store.get(owner, conversation.id)
            released = scenario in {"EXPIRED", "EXHAUSTED", "CANCEL"}
            assert after.active_run_id == (None if released else submitted.run_id)
            assert after.revision == before.revision == 0
            assert after.accepted_checkpoint_id == before.accepted_checkpoint_id
            run = await coordinator.get(owner, conversation.id, submitted.run_id)
            assert run.status == (
                "CANCELLED"
                if scenario == "CANCEL"
                else "FAILED_TERMINAL"
                if released
                else "RUNNING"
                if scenario == "RUNNING"
                else "FAILED_RETRYABLE"
            )
            async with engine.connect() as conn:
                roles = (
                    (
                        await conn.execute(
                            select(conversation_messages.c.role).where(
                                conversation_messages.c.conversation_id
                                == conversation.id
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
            assert roles == ["user"]
            if released:
                assert (
                    await coordinator.submit(owner, request)
                ).run_id == submitted.run_id
                fresh = await coordinator.submit(
                    owner, request.model_copy(update={"request_id": uuid4()})
                )
                assert fresh.status == "PENDING" and fresh.run_id != submitted.run_id
    finally:
        if conversation:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
