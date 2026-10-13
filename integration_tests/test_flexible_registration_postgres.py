"""Task registration recovery with native PostgreSQL; no managed-CAS claim."""

from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_flexible_graph_postgres import Budget, Catalog, Model

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import ToolContext
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("unresolved", [False, True])
async def test_registered_tasks_survive_reopen_without_extraction_or_lookup_repeat(
    postgres_dsn, unresolved
):
    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    message = "So GPU A và GPU B" if unresolved else "GPU?"
    task = {
        "kind": "COMPARE" if unresolved else "ANSWER",
        "start": 0,
        "end": len(message),
        "quote": message,
    }
    if unresolved:
        task["subjects"] = [
            {"start": 3, "end": 8, "quote": "GPU A"},
            {"start": 12, "end": 17, "quote": "GPU B"},
        ]

    class Extractor:
        calls = 0

        async def extract_tasks(self, message):
            self.calls += 1
            return TaskDraftBatch(tasks=[task])

    class Products(Catalog):
        lookups = 0

        async def resolve(self, mention):
            self.lookups += 1
            return []

    class ResumeModel(Model):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if unresolved:
                return FlexibleDecision(
                    action={
                        "kind": "ASK_CLARIFICATION",
                        "question": "Which product?",
                        "missing_fields": ["NOT_FOUND"],
                    }
                )
            return await super().decide(message, transcript, allowed, repair, context)

    extractor, products = Extractor(), Products()
    binding = FlexiblePublicationInput(
        accepted=AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            requirement_state=ctx.requirements,
        ),
        task_message_id=uuid4(),
        task_message=message,
        request_message=message,
    )

    def compile_flow(model, saver):
        return build_flexible_graph(
            model,
            ToolDispatcher(products, PCBuildApplicationService()),
            ctx,
            [],
            Budget(),
            conversation_revision=0,
            publication=binding,
            task_extractor=extractor,
            task_catalog=products,
            checkpointer=saver,
        )

    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = compile_flow(Model(fail_after_tool=True), saver)
            with pytest.raises(TimeoutError):
                await graph.ainvoke({"message": message}, config, durability="sync")
            saved = await graph.aget_state(config)
            assert saved.next == ("agent",)
            assert saved.values["task_draft"]["tasks"][0]["kind"] == task["kind"]
            counts = (extractor.calls, products.lookups, products.calls)
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            graph = compile_flow(ResumeModel(), saver)
            await graph.ainvoke(None, saved.config, durability="sync")
            terminal = await graph.aget_state(config)
            ref = CheckpointRefV1(
                thread_id=ctx.conversation_id,
                checkpoint_id=terminal.config["configurable"]["checkpoint_id"],
            )
            snapshot = await PostgresAcceptedCheckpointReader(saver).read(ref)
            assert not terminal.next
            assert (extractor.calls, products.lookups, products.calls) == counts
            assert extractor.calls == 1
            assert snapshot.flexible.task_message_id == binding.task_message_id
            if unresolved:
                assert (
                    snapshot.flexible.pending_task_resolution.tasks[0].kind == "COMPARE"
                )
                assert snapshot.flexible.status == "WAITING_INPUT"
                assert not snapshot.flexible.task_outcomes
            else:
                assert snapshot.flexible.task_outcomes[0].task.kind == "ANSWER"
                assert snapshot.flexible.status == "COMPLETED"
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
