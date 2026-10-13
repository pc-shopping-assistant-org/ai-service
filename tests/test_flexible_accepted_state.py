from uuid import uuid4

import pytest
from test_flexible_graph import graph
from test_flexible_publication import base_snapshot
from test_react_tools import Catalog, context
from test_waiting_publication import Proposer

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.waiting_publication import (
    prepare_waiting_publication,
)


@pytest.mark.asyncio
async def test_persisted_waiting_state_preserves_original_task_and_pending_hash():
    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    terminal = await graph(Proposer(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Build PC"}
    )
    waiting = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        tasks,
        source_message="Build PC",
        conversation_revision=0,
    )
    published = FlexiblePublishedState(
        conversation_id=ctx.conversation_id,
        task_message="Build PC",
        task_message_id=uuid4(),
        status="WAITING_INPUT",
        waiting_message=waiting.assistant_message,
        waiting_tasks=waiting.task_outcomes,
        pending_proposal=waiting.pending_proposal,
    )
    snapshot = AcceptedConversationSnapshot.model_validate(
        waiting.snapshot.model_dump() | {"flexible": published.model_dump()}
    )
    restored = AcceptedConversationSnapshot.model_validate_json(
        snapshot.model_dump_json()
    )
    assert restored.flexible.task_message == "Build PC"
    assert (
        restored.flexible.pending_proposal.proposal_hash
        == waiting.pending_proposal.proposal_hash
    )
    assert restored.flexible.task_outcomes[0].status == "UNRESOLVED"
    assert restored.requirement_state.requirements == {}


@pytest.mark.asyncio
async def test_persisted_pending_cannot_be_bound_to_changed_requirement_snapshot():
    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    terminal = await graph(Proposer(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Build PC"}
    )
    waiting = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        tasks,
        source_message="Build PC",
        conversation_revision=0,
    )
    published = FlexiblePublishedState(
        conversation_id=ctx.conversation_id,
        task_message="Build PC",
        task_message_id=uuid4(),
        status="WAITING_INPUT",
        waiting_message=waiting.assistant_message,
        waiting_tasks=waiting.task_outcomes,
        pending_proposal=waiting.pending_proposal,
    )
    data = waiting.snapshot.model_dump(mode="json") | {
        "flexible": published.model_dump(mode="json")
    }
    data["requirement_state"]["unresolved"] = ["other state"]
    with pytest.raises(ValueError):
        AcceptedConversationSnapshot.model_validate(data)


def test_waiting_cannot_persist_completed_tasks_or_absent_artifact():
    task = CompletionRequirement(kind="BUILD", constraints_revision=0)
    with pytest.raises(ValueError):
        FlexiblePublishedState(
            conversation_id=uuid4(),
            task_message="Build PC",
            task_message_id=uuid4(),
            status="WAITING_INPUT",
            waiting_message="Missing budget",
            waiting_tasks=[{"task": task, "status": "COMPLETED"}],
        )


@pytest.mark.asyncio
async def test_reader_uses_exact_v2_snapshot_and_rejects_foreign_newer_record():
    from typing import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
    from ai_service.infrastructure.persistence.accepted_state import (
        PostgresAcceptedCheckpointReader,
    )

    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    terminal = await graph(Proposer(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Build PC"}
    )
    waiting = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        tasks,
        source_message="Build PC",
        conversation_revision=0,
    )
    published = FlexiblePublishedState(
        conversation_id=ctx.conversation_id,
        task_message="Build PC",
        task_message_id=uuid4(),
        status="WAITING_INPUT",
        waiting_message=waiting.assistant_message,
        waiting_tasks=waiting.task_outcomes,
        pending_proposal=waiting.pending_proposal,
    )
    snapshot = AcceptedConversationSnapshot.model_validate(
        waiting.snapshot.model_dump() | {"flexible": published.model_dump()}
    )

    class State(TypedDict):
        accepted_snapshot: dict

    async def finish(state):
        return {}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("finish", finish)
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    flow = builder.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    await flow.ainvoke({"accepted_snapshot": snapshot.model_dump(mode="json")}, config)
    saved = await flow.aget_state(config)
    ref = CheckpointRefV1(
        thread_id=ctx.conversation_id,
        checkpoint_id=saved.config["configurable"]["checkpoint_id"],
    )
    wrong = snapshot.model_dump(mode="json")
    wrong["flexible"]["conversation_id"] = str(uuid4())
    wrong["flexible"]["pending_proposal"] = None
    wrong["flexible"]["clarification"] = {
        "kind": "ASK_CLARIFICATION",
        "question": "Budget?",
        "missing_fields": ["BUDGET_REQUIRED"],
    }
    await flow.ainvoke({"accepted_snapshot": wrong}, config)
    reader = PostgresAcceptedCheckpointReader(saver)
    assert (await reader.read(ref)).flexible.task_message == "Build PC"
    bad = await flow.aget_state(config)
    with pytest.raises(ValueError):
        await reader.read(
            CheckpointRefV1(
                thread_id=ctx.conversation_id,
                checkpoint_id=bad.config["configurable"]["checkpoint_id"],
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_snapshot_encoder_attaches_task_source_to_preparation(waiting):
    from test_flexible_graph import ReadModel

    from ai_service.capabilities.assistant.react.publication import (
        prepare_answer_publication,
    )
    from ai_service.capabilities.assistant.react.snapshot_publication import (
        accepted_snapshot_for,
    )
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )

    ctx = context()
    tasks = (
        [CompletionRequirement(kind="BUILD", constraints_revision=0)]
        if waiting
        else [CompletionRequirement(kind="ANSWER")]
    )
    source_id = uuid4()
    terminal = await graph(
        Proposer() if waiting else ReadModel(), Catalog(), ctx, required=tasks
    ).ainvoke({"message": "Original task"})
    if waiting:
        prepared = prepare_waiting_publication(
            terminal,
            base_snapshot(ctx),
            ctx,
            tasks,
            source_message="Original task",
            conversation_revision=0,
        )
    else:
        prepared = prepare_answer_publication(
            terminal, base_snapshot(ctx), ctx, tasks, PCBuildApplicationService()
        )
    result = accepted_snapshot_for(
        prepared,
        conversation_id=ctx.conversation_id,
        task_message_id=source_id,
        task_message="Original task",
    )
    restored = AcceptedConversationSnapshot.model_validate_json(
        result.model_dump_json()
    )
    assert restored.flexible.task_message_id == source_id
    assert restored.flexible.task_message == "Original task"
    assert restored.flexible.status == ("WAITING_INPUT" if waiting else "COMPLETED")
    assert restored.flexible.assistant_message
