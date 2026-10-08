"""B1 candidate gate, NOT a selected/verified production invocation strategy.

Run explicitly against local ai_db via AI_TEST_POSTGRES_DSN after pinning the
LangGraph/checkpointer stack. This file is outside the default unit test tree;
the B1 gate is mandatory, never replaced by mocked saver/SQLite evidence.
"""

import os
from collections import Counter
from contextlib import asynccontextmanager
from operator import add
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


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
):
    dsn = os.environ.get("AI_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.fail(
            "AI_TEST_POSTGRES_DSN must point to local development ai_db"
        )
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
            yield builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": str(uuid4())}}
    async with open_graph() as graph:
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
    async with open_graph() as graph:
        # Candidate A: explicit base + new input. No production adapter assumes
        # these semantics until this gate passes on exact locked versions.
        result = await graph.ainvoke(
            {"run_id": "R3", "turn": "new-R3"}, accepted.config, durability="sync"
        )
        assert result["sentinels"] == ["accepted-A", "new-R3"]
        assert result["publication"] == "R3"
        assert calls[("R1", "extract")] == 1
        assert calls[("R1", "prepare")] == 1
        assert calls[("R3", "extract")] == 1
        assert calls[("R3", "prepare")] == 1
        # Latest lookup is used only to observe this isolated experiment, NOT
        # as an application-head resolver or as a production finalization API.
        terminal = await graph.aget_state(config)
        assert terminal.next == ()
        assert terminal.values["publication"] == "R3"
        cursor = terminal
        while cursor.config["configurable"]["checkpoint_id"] != accepted_id:
            assert cursor.config["configurable"]["checkpoint_id"] != orphan_id
            assert cursor.values["run_id"] == "R3"
            assert cursor.parent_config is not None
            cursor = await graph.aget_state(cursor.parent_config)
