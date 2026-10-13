"""Native snapshot serialization/read gate, not managed publication proof."""

from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)


class SnapshotState(TypedDict):
    accepted_snapshot: dict


def flow(saver):
    builder = StateGraph(SnapshotState)
    builder.add_node("finish", lambda state: {})
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_v2_accepted_snapshot_survives_reopen_and_ignores_newer_orphan(
    postgres_dsn,
):
    conversation_id, source_id = uuid4(), uuid4()
    snapshot = AcceptedConversationSnapshot(
        consultation=ConsultationStateV1(),
        requirement_state=CanonicalRequirementState(),
        flexible=FlexiblePublishedState(
            conversation_id=conversation_id,
            task_message_id=source_id,
            task_message="Build PC",
            status="WAITING_INPUT",
            waiting_message="Budget?",
            waiting_tasks=[
                {
                    "task": {"kind": "BUILD", "constraints_revision": 0},
                    "status": "UNRESOLVED",
                }
            ],
            clarification={
                "kind": "ASK_CLARIFICATION",
                "question": "Budget?",
                "missing_fields": ["BUDGET_REQUIRED"],
            },
        ),
    )
    config = {"configurable": {"thread_id": str(conversation_id)}}
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = flow(saver)
            await graph.ainvoke(
                {"accepted_snapshot": snapshot.model_dump(mode="json")},
                config,
                durability="sync",
            )
            saved = await graph.aget_state(config)
            ref = CheckpointRefV1(
                thread_id=conversation_id,
                checkpoint_id=saved.config["configurable"]["checkpoint_id"],
            )
            orphan = snapshot.model_dump(mode="json")
            orphan["flexible"]["task_message"] = "Different task"
            await graph.ainvoke(
                {"accepted_snapshot": orphan}, config, durability="sync"
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            restored = await PostgresAcceptedCheckpointReader(saver).read(ref)
            assert restored == snapshot
            assert restored.flexible.task_message_id == source_id
            assert restored.flexible.task_outcomes[0].status == "UNRESOLVED"
            assert restored.requirement_state.requirements == {}
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation_id))


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_prepare_node_recovery_does_not_repeat_agent_or_tool(
    postgres_dsn, monkeypatch, waiting
):
    from test_flexible_graph_postgres import Budget, Catalog, Model

    from ai_service.capabilities.assistant.react.contracts import (
        CompletionRequirement,
        ToolContext,
    )
    from ai_service.capabilities.assistant.react.snapshot_publication import (
        FlexiblePublicationInput,
    )
    from ai_service.capabilities.assistant.react.tools import ToolDispatcher
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )
    from ai_service.infrastructure.graph.flexible import build_flexible_graph

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    tasks = (
        [CompletionRequirement(kind="BUILD", constraints_revision=0)]
        if waiting
        else [CompletionRequirement(kind="ANSWER")]
    )
    binding = FlexiblePublicationInput(
        accepted=AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            requirement_state=ctx.requirements,
        ),
        task_message_id=uuid4(),
        task_message="Original task",
        request_message="Original task",
    )
    catalog = Catalog()

    class CountingModel(Model):
        calls = 0

        async def decide(self, *args, **kwargs):
            self.calls += 1
            return await super().decide(*args, **kwargs)

    model = CountingModel(propose=waiting)
    original_prepare = FlexiblePublicationInput.prepare
    fail = True

    def prepare(self, *args, **kwargs):
        if fail:
            raise TimeoutError("preparation failed after terminal validation")
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(FlexiblePublicationInput, "prepare", prepare)
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}

    def compile_graph(saver):
        return build_flexible_graph(
            model,
            ToolDispatcher(catalog, PCBuildApplicationService()),
            ctx,
            tasks,
            Budget(),
            conversation_revision=0,
            publication=binding,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = compile_graph(saver)
            with pytest.raises(TimeoutError):
                await graph.ainvoke(
                    {"message": "Original task"}, config, durability="sync"
                )
            saved = await graph.aget_state(config)
            assert saved.next == ("prepare_publish",)
            assert saved.values["accepted_snapshot"] is None
            calls, tool_calls = model.calls, catalog.calls
        fail = False
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            graph = compile_graph(saver)
            result = await graph.ainvoke(None, saved.config, durability="sync")
            terminal = await graph.aget_state(config)
            ref = CheckpointRefV1(
                thread_id=ctx.conversation_id,
                checkpoint_id=terminal.config["configurable"]["checkpoint_id"],
            )
            snapshot = await PostgresAcceptedCheckpointReader(saver).read(ref)
            assert snapshot.model_dump(mode="json") == result["accepted_snapshot"]
            assert snapshot.flexible.status == (
                "WAITING_INPUT" if waiting else "COMPLETED"
            )
            assert snapshot.flexible.task_message_id == binding.task_message_id
            assert (model.calls, catalog.calls) == (calls, tool_calls)
            assert not terminal.next
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
