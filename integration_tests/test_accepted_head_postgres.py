"""B1 accepted-head gate (owner reported PASS on the pinned stack 2026-10-08).

Run explicitly against local ai_db via AI_TEST_POSTGRES_DSN after pinning the
LangGraph/checkpointer stack. This file is outside the default unit test tree;
the B1 gate is mandatory, never replaced by mocked saver/SQLite evidence.
"""

from collections import Counter
from contextlib import asynccontextmanager
from operator import add
from typing import Annotated, TypedDict
from uuid import UUID, uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ConversationHeadV1,
)
from ai_service.infrastructure.graph.durable import DurableConversationGraph


class GateState(TypedDict):
    run_id: str
    turn: str
    sentinels: Annotated[list[str], add]
    pause: bool
    publication: str | None


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_orphan", [False, True])
async def test_new_turn_uses_accepted_head_after_orphan_and_pool_restart(
    pending_orphan,
    postgres_dsn,
):
    dsn = postgres_dsn
    calls = Counter()

    async def extract(state: GateState):
        calls[(state["run_id"], "extract")] += 1
        return {"sentinels": [state["turn"]], "publication": None}

    async def prepare(state: GateState):
        assert state["publication"] is None
        if state["pause"]:
            interrupt("orphan awaiting input")
        calls[(state["run_id"], "prepare")] += 1
        return {"publication": state["run_id"]}

    @asynccontextmanager
    async def open_graph():
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            # Test database bootstrap; never call setup in a request handler.
            await saver.setup()
            builder = StateGraph(GateState)
            builder.add_node("extract", extract)
            builder.add_node("prepare", prepare)
            builder.add_edge(START, "extract")
            builder.add_edge("extract", "prepare")
            builder.add_edge("prepare", END)
            yield builder.compile(checkpointer=saver), saver

    config = {"configurable": {"thread_id": str(uuid4())}}
    async with open_graph() as (graph, _):
        await graph.ainvoke(
            {
                "run_id": "R1",
                "turn": "accepted-A",
                "sentinels": [],
                "pause": False,
                "publication": None,
            },
            config,
            durability="sync",
        )
        accepted = await graph.aget_state(config)
        assert accepted.next == ()
        assert accepted.values["publication"] == "R1"
        accepted_id = accepted.config["configurable"]["checkpoint_id"]
        await graph.ainvoke(
            {"run_id": "R2", "turn": "orphan-B", "pause": pending_orphan},
            config,
            durability="sync",
        )
        orphan = await graph.aget_state(config)
        orphan_id = orphan.config["configurable"]["checkpoint_id"]
        assert orphan_id != accepted_id
        assert "orphan-B" in orphan.values["sentinels"]
        assert bool(orphan.next) is pending_orphan
        explicit = await graph.aget_state(accepted.config)
        assert explicit.values["sentinels"] == ["accepted-A"]

    # Reopen saver/connection and compile a new graph; no in-memory checkpoints.
    async with open_graph() as (graph, saver):
        # Selected semantics: explicit accepted base + new input. Preserve
        # this gate when updating the exact locked graph/checkpointer versions.
        new_input = {"run_id": "R3", "turn": "new-R3"}
        run_context = {
            "app_run_id": "R3",
            "app_base_checkpoint_id": accepted_id,
            "app_input_hash": canonical_hash(new_input),
            "app_graph_version": "gate-v1",
        }
        before_calls = calls.copy()
        result = await graph.ainvoke(
            new_input,
            {**accepted.config, "metadata": run_context},
            durability="sync",
        )
        assert result["sentinels"] == ["accepted-A", "new-R3"]
        assert result["publication"] == "R3"
        assert calls[("R1", "extract")] == 1
        assert calls[("R1", "prepare")] == 1
        assert calls[("R3", "extract")] == 1
        assert calls[("R3", "prepare")] == 1
        assert calls - before_calls == Counter(
            {("R3", "extract"): 1, ("R3", "prepare"): 1}
        )
        # Latest lookup is used only to observe this isolated experiment, NOT
        # as an application-head resolver or as a production finalization API.
        terminal = await graph.aget_state(config)
        assert terminal.next == ()
        assert terminal.values["publication"] == "R3"
        cursor = terminal
        seen = set()
        input_boundaries = 0
        while cursor.config["configurable"]["checkpoint_id"] != accepted_id:
            ref = cursor.config["configurable"]
            assert ref["checkpoint_id"] != orphan_id
            assert ref["checkpoint_id"] not in seen
            seen.add(ref["checkpoint_id"])
            assert ref["thread_id"] == config["configurable"]["thread_id"]
            assert ref.get("checkpoint_ns", "") == ""
            assert cursor.metadata is not None
            for name, value in run_context.items():
                assert cursor.metadata.get(name) == value
            assert cursor.parent_config is not None
            assert "orphan-B" not in cursor.values["sentinels"]
            if cursor.metadata["source"] == "input":
                # The START task has not applied the new input to state values
                # yet. Prove run ownership via metadata + native input channel,
                # not via inherited values.run_id/publication at this boundary.
                input_boundaries += 1
                assert cursor.values == accepted.values
                assert cursor.next == (START,)
                assert (
                    cursor.parent_config["configurable"]["checkpoint_id"] == accepted_id
                )
                raw = await saver.aget_tuple(cursor.config)
                assert raw is not None
                assert raw.checkpoint["channel_values"][START] == new_input
            else:
                assert cursor.metadata["source"] == "loop"
                assert cursor.values["run_id"] == "R3"
            cursor = await graph.aget_state(cursor.parent_config)
        assert input_boundaries == 1

        # New P2 adapter extension: still needs owner native rerun. It must
        # return its exact emitted checkpoint, never the orphan/latest base.
        adapter = DurableConversationGraph(graph)
        conversation_id = UUID(config["configurable"]["thread_id"])
        head = ConversationHeadV1(
            conversation_id=conversation_id,
            revision=1,
            accepted_checkpoint_ref=CheckpointRefV1(
                thread_id=conversation_id,
                checkpoint_id=accepted_id,
            ),
        )
        run_id = uuid4()
        ref = await adapter.new_turn(
            head,
            run_id,
            {
                "run_id": str(run_id),
                "turn": "adapter-R4",
                "pause": False,
            },
            "gate-v1",
        )
        own = await adapter.read(ref)
        assert own.values["sentinels"] == ["accepted-A", "adapter-R4"]
        assert own.values["publication"] == str(run_id)
        assert ref.checkpoint_id not in (accepted_id, orphan_id)
