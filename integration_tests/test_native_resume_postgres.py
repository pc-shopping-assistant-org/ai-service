"""Native resume across saver reopen; no SQL takeover/publication authority."""

from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from operator import add
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.stateful_contracts import (
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
)
from ai_service.infrastructure.graph.durable import DurableConversationGraph


class ResumeState(TypedDict):
    values: Annotated[list[str], add]


@pytest.mark.asyncio
async def test_exact_native_resume_reopen_does_not_reexecute_committed_node(
    postgres_dsn,
):
    calls, failed, refs = Counter(), False, []
    conversation = uuid4()

    async def optimize(state):
        calls["optimize"] += 1
        return {"values": ["optimized"]}

    async def explain(state):
        calls["explain"] += 1
        if failed:
            raise TimeoutError("scripted explanation timeout")
        return {}

    @asynccontextmanager
    async def open_graph():
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            builder = StateGraph(ResumeState)
            builder.add_node("optimize", optimize)
            builder.add_node("explain", explain)
            builder.add_edge(START, "optimize")
            builder.add_edge("optimize", "explain")
            builder.add_edge("explain", END)
            yield DurableConversationGraph(builder.compile(checkpointer=saver))

    async def observed(ref):
        refs.append(ref)

    previous = ExecutionClaim(
        run_id=uuid4(),
        conversation_id=conversation,
        execution_id=uuid4(),
        worker_instance_id=uuid4(),
        generation=1,
    )
    current = previous.model_copy(
        update={"execution_id": uuid4(), "worker_instance_id": uuid4(), "generation": 2}
    )
    proof = ExecutionStopEvidenceV1(
        kind="EXECUTOR_STOP_ACK",
        execution_id=previous.execution_id,
        worker_instance_id=previous.worker_instance_id,
        generation=1,
        stopped_at=datetime.now(UTC),
        checkpoint_writes_drained=True,
        proof_ref="fixture:joined-execution",
    )
    inputs = {"values": ["new request"]}
    try:
        async with open_graph() as graph:
            head = ConversationHeadV1(conversation_id=conversation, revision=0)
            base = await graph.new_turn(head, uuid4(), {"values": []}, "resume-test-v1")
            head = head.model_copy(
                update={"revision": 1, "accepted_checkpoint_ref": base}
            )
            failed = True
            with pytest.raises(TimeoutError):
                await graph.new_turn(
                    head,
                    previous.run_id,
                    inputs,
                    "resume-test-v1",
                    execution=previous,
                    checkpoint_sink=observed,
                )
            source = refs[-1]
            await graph.graph.aupdate_state(
                graph.config(source), {"values": ["orphan"]}, as_node="optimize"
            )
        failed = False
        async with open_graph() as graph:
            final = await graph.resume(
                head,
                source,
                previous,
                current,
                inputs,
                "resume-test-v1",
                evidence=proof,
                checkpoint_sink=observed,
            )
            snapshot = await graph.read(final)
            assert snapshot.values["values"] == [
                "optimized",
                "new request",
                "optimized",
            ]
            assert calls == {"optimize": 2, "explain": 3}
            assert not snapshot.next and not snapshot.tasks
            assert snapshot.metadata["app_execution_id"] == str(current.execution_id)
            assert snapshot.metadata["app_generation"] == 2
            assert refs[-1] == final
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation))
