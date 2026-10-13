from datetime import UTC, datetime
from operator import add
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.stateful_contracts import (
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
)
from ai_service.infrastructure.graph.durable import DurableConversationGraph


class State(TypedDict):
    turns: Annotated[list[str], add]


@pytest.mark.asyncio
async def test_adapter_returns_own_checkpoint_and_new_turn_ignores_orphan_latest():
    async def finish(state: State):
        return {}

    builder = StateGraph(State)
    builder.add_node("finish", finish)
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    adapter = DurableConversationGraph(builder.compile(checkpointer=InMemorySaver()))
    conversation_id = uuid4()
    initial = ConversationHeadV1(conversation_id=conversation_id, revision=0)
    first = await adapter.new_turn(initial, uuid4(), {"turns": ["accepted"]}, "test-v1")
    # An unaccepted first run must not become the implicit base of a new turn.
    with pytest.raises(ValueError, match="Unaccepted checkpoint"):
        await adapter.new_turn(initial, uuid4(), {"turns": ["wrong"]}, "test-v1")
    accepted = initial.model_copy(
        update={"revision": 1, "accepted_checkpoint_ref": first}
    )
    orphan = await adapter.new_turn(accepted, uuid4(), {"turns": ["orphan"]}, "test-v1")
    final = await adapter.new_turn(accepted, uuid4(), {"turns": ["new"]}, "test-v1")
    assert final != orphan
    assert (await adapter.read(final)).values["turns"] == ["accepted", "new"]
    assert (await adapter.read(first)).values["turns"] == ["accepted"]


@pytest.mark.asyncio
async def test_emitted_refs_are_awaited_before_a_later_node_failure():
    refs = []

    async def committed(state):
        return {"turns": ["committed"]}

    async def fail(state):
        raise TimeoutError("explanation failed")

    builder = StateGraph(State)
    builder.add_node("committed", committed)
    builder.add_node("fail", fail)
    builder.add_edge(START, "committed")
    builder.add_edge("committed", "fail")
    builder.add_edge("fail", END)
    adapter = DurableConversationGraph(builder.compile(checkpointer=InMemorySaver()))

    async def record(ref):
        snapshot = await adapter.read(ref)
        refs.append((ref, snapshot))

    with pytest.raises(TimeoutError):
        await adapter.new_turn(
            ConversationHeadV1(conversation_id=uuid4(), revision=0),
            uuid4(),
            {"turns": []},
            "test-v1",
            checkpoint_sink=record,
        )
    assert refs
    ref, snapshot = refs[-1]
    assert snapshot.values["turns"] == ["committed"]
    assert snapshot.next == ("fail",)
    assert (await adapter.read(ref)).values == snapshot.values


@pytest.mark.asyncio
async def test_resume_exact_stopped_execution_keeps_committed_work_and_rejects_changed_input():
    calls = {"commit": 0, "explain": 0}
    fail = False

    async def commit(state):
        calls["commit"] += 1
        return {"turns": ["optimized"]}

    async def explain(state):
        calls["explain"] += 1
        if fail:
            raise TimeoutError("provider timeout")
        return {}

    builder = StateGraph(State)
    builder.add_node("commit", commit)
    builder.add_node("explain", explain)
    builder.add_edge(START, "commit")
    builder.add_edge("commit", "explain")
    builder.add_edge("explain", END)
    adapter = DurableConversationGraph(builder.compile(checkpointer=InMemorySaver()))
    conversation = uuid4()
    head = ConversationHeadV1(conversation_id=conversation, revision=0)
    base = await adapter.new_turn(head, uuid4(), {"turns": []}, "test-v1")
    head = head.model_copy(update={"revision": 1, "accepted_checkpoint_ref": base})
    previous = ExecutionClaim(
        run_id=uuid4(),
        conversation_id=conversation,
        execution_id=uuid4(),
        worker_instance_id=uuid4(),
        generation=1,
    )
    refs = []

    async def record(ref):
        refs.append(ref)

    fail = True
    inputs = {"turns": ["request"]}
    with pytest.raises(TimeoutError):
        await adapter.new_turn(
            head,
            previous.run_id,
            inputs,
            "test-v1",
            execution=previous,
            checkpoint_sink=record,
        )
    source = refs[-1]
    current = previous.model_copy(
        update={"execution_id": uuid4(), "worker_instance_id": uuid4(), "generation": 2}
    )
    proof = ExecutionStopEvidenceV1(
        kind="EXECUTOR_STOP_ACK",
        execution_id=previous.execution_id,
        worker_instance_id=previous.worker_instance_id,
        generation=previous.generation,
        stopped_at=datetime.now(UTC),
        checkpoint_writes_drained=True,
        proof_ref="fixture:joined-executor",
    )
    with pytest.raises(ValueError, match="context"):
        await adapter.resume(
            head,
            source,
            previous,
            current,
            {"turns": ["changed"]},
            "test-v1",
            evidence=proof,
        )
    assert calls == {"commit": 2, "explain": 2}
    with pytest.raises(ValueError, match="STOP_IDENTITY"):
        await adapter.resume(
            head,
            source,
            previous,
            current,
            inputs,
            "test-v1",
            evidence=proof.model_copy(update={"generation": 2}),
        )
    with pytest.raises(ValueError, match="execution context"):
        await adapter.resume(
            head, source, previous, previous, inputs, "test-v1", evidence=proof
        )
    await adapter.graph.aupdate_state(
        adapter.config(source), {"turns": ["orphan"]}, as_node="commit"
    )
    fail = False
    final = await adapter.resume(
        head, source, previous, current, inputs, "test-v1", evidence=proof
    )
    assert calls == {"commit": 2, "explain": 3}
    snapshot = await adapter.read(final)
    assert snapshot.values["turns"] == ["optimized", "request", "optimized"]
    assert snapshot.metadata["app_generation"] == 2
    assert snapshot.metadata["app_execution_id"] == str(current.execution_id)
