"""ISSUE-092 native-root experiment, NOT a production readiness/CAS contract.

Memory exercises pinned native semantics without I/O. PostgreSQL is the required
durability gate: run explicitly with AI_TEST_POSTGRES_DSN. Recompiling with the
same memory saver is not evidence of process durability. No P0 DTO is changed.
"""

import asyncio
from collections import Counter
from contextlib import asynccontextmanager
from operator import add
from typing import Annotated, TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph


class RootState(TypedDict):
    run_id: str | None
    consultation: str
    public_build: dict | None
    turns: Annotated[list[str], add]
    fail: bool


@pytest.fixture(params=["memory", "postgres"])
def root_backend(request):
    # Do not request the mandatory DB fixture for the independently runnable
    # memory variant. Missing DB settings FAIL the postgres gate, never skip it.
    return (
        request.param,
        request.getfixturevalue("postgres_dsn") if request.param == "postgres" else None,
    )


def graph_factory(backend, calls):
    kind, dsn = backend
    memory = InMemorySaver()

    async def initialize_root(state):
        calls["initializer_executed"] += 1
        raise AssertionError("aupdate_state must not execute the initializer")

    async def extract(state):
        calls[(state["run_id"], "extract")] += 1
        return {"consultation": state["run_id"], "public_build": None}

    async def finish(state):
        calls[(state["run_id"], "finish")] += 1
        if state["fail"]:
            raise RuntimeError("terminal first-run failure")
        return {"public_build": {"run_id": state["run_id"]}}

    def compile_graph(saver):
        builder = StateGraph(RootState)
        # Isolated root-only node ends immediately: it schedules no turn/tool.
        builder.add_node("initialize_root", initialize_root)
        builder.add_edge("initialize_root", END)
        builder.add_node("extract", extract)
        builder.add_node("finish", finish)
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "finish")
        builder.add_edge("finish", END)
        return builder.compile(checkpointer=saver)

    @asynccontextmanager
    async def open_graph():
        if kind == "memory":
            yield compile_graph(memory), memory
        else:
            async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
                await saver.setup()
                yield compile_graph(saver), saver

    return open_graph


async def create_root(graph, thread):
    initialization = {
        "app_initialization_id": str(uuid4()),
        "app_checkpoint_kind": "initial_root",
        "app_graph_version": "root-experiment-v1",
    }
    exact = await graph.aupdate_state(
        {**thread, "metadata": initialization},
        {
            "run_id": None,
            "consultation": "UNKNOWN",
            "public_build": None,
            "turns": [],
            "fail": False,
        },
        as_node="initialize_root",
    )
    root = await graph.aget_state(exact)
    assert root.next == () and root.tasks == ()
    assert root.values["consultation"] == "UNKNOWN"
    assert root.values["public_build"] is None and root.values["turns"] == []
    assert root.metadata["source"] == "update"
    for name, value in initialization.items():
        assert root.metadata[name] == value
    return exact, initialization


@pytest.mark.asyncio
async def test_terminal_r1_orphan_then_r2_uses_exact_initial_root_after_reopen(root_backend):
    calls = Counter()
    open_graph = graph_factory(root_backend, calls)
    thread = {"configurable": {"thread_id": str(uuid4())}}
    async with open_graph() as (graph, _):
        root_ref, _ = await create_root(graph, thread)
        assert not calls  # Root initialization executed no model/tool/turn node.
        root_id = root_ref["configurable"]["checkpoint_id"]
        # Native config intentionally bypasses the unchanged P0 revision-zero
        # DTO. This is experimental evidence, not an accepted application head.
        worker = asyncio.create_task(graph.ainvoke(
            {"run_id": "R1", "turns": ["orphan-R1"], "fail": True},
            {**root_ref, "metadata": {"app_run_id": "R1"}},
            durability="sync",
        ))
        with pytest.raises(RuntimeError, match="terminal first-run failure"):
            await worker
        assert worker.done()  # Local graph worker stopped before the new turn.
        orphan = await graph.aget_state(thread)
        orphan_id = orphan.config["configurable"]["checkpoint_id"]
        assert orphan_id != root_id
        assert orphan.values["turns"] == ["orphan-R1"]
        assert calls == Counter({("R1", "extract"): 1, ("R1", "finish"): 1})

    async with open_graph() as (graph, _):
        restored = await graph.aget_state(root_ref)
        assert restored.next == () and restored.tasks == ()
        assert restored.values["turns"] == []
        before = calls.copy()
        result = await graph.ainvoke(
            {"run_id": "R2", "turns": ["new-R2"], "fail": False},
            {**root_ref, "metadata": {"app_run_id": "R2", "app_base_checkpoint_id": root_id}},
            durability="sync",
        )
        assert result["turns"] == ["new-R2"]
        assert result["public_build"] == {"run_id": "R2"}
        assert calls - before == Counter({("R2", "extract"): 1, ("R2", "finish"): 1})
        # Latest is only an observation of this isolated gate, never selection
        # of a production head. Walk native parents to prove exact ancestry.
        cursor = await graph.aget_state(thread)
        assert cursor.next == () and cursor.tasks == ()
        seen = set()
        while cursor.config["configurable"]["checkpoint_id"] != root_id:
            checkpoint_id = cursor.config["configurable"]["checkpoint_id"]
            assert checkpoint_id != orphan_id and checkpoint_id not in seen
            seen.add(checkpoint_id)
            assert "orphan-R1" not in cursor.values.get("turns", [])
            assert cursor.parent_config is not None
            cursor = await graph.aget_state(cursor.parent_config)


@pytest.mark.asyncio
async def test_initializer_crash_before_app_cas_recovers_by_server_metadata_not_latest(root_backend):
    calls = Counter()
    open_graph = graph_factory(root_backend, calls)
    thread = {"configurable": {"thread_id": str(uuid4())}}
    async with open_graph() as (graph, _):
        root_ref, initialization = await create_root(graph, thread)
        root_id = root_ref["configurable"]["checkpoint_id"]
        # Crash boundary: checkpoint committed, application CAS never called.
        # An unrelated debug checkpoint makes latest deliberately unsafe.
        debug_ref = await graph.aupdate_state(
            {**root_ref, "metadata": {
                "app_initialization_id": str(uuid4()), "app_checkpoint_kind": "debug",
            }},
            {"consultation": "UNACCEPTED_DEBUG"},
            as_node="initialize_root",
        )
        assert debug_ref["configurable"]["checkpoint_id"] != root_id
        assert not calls

    async with open_graph() as (graph, saver):
        latest = await graph.aget_state(thread)
        assert latest.config["configurable"]["checkpoint_id"] != root_id
        # Trusted initializer identity was persisted before execution. Recover
        # candidates via native saver metadata; never assume latest is root.
        matches = [item async for item in saver.alist(thread, filter=initialization)]
        assert len(matches) == 1
        recovered = matches[0].config
        assert recovered["configurable"]["checkpoint_id"] == root_id
        snapshot = await graph.aget_state(recovered)
        assert snapshot.next == () and snapshot.tasks == ()
        assert snapshot.values["consultation"] == "UNKNOWN"
        assert snapshot.values["public_build"] is None
        wrong = {**initialization, "app_initialization_id": str(uuid4())}
        assert [item async for item in saver.alist(thread, filter=wrong)] == []
        # Recovery only identifies an exact candidate. Production CAS/readiness,
        # concurrent initializer exclusion and migrations remain ISSUE-092.
        assert not calls
