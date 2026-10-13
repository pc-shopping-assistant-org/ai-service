from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_flexible_graph import ReadModel
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_task_resolution import compare_task

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


class Extractor:
    def __init__(self, tasks):
        self.tasks, self.calls = tasks, 0

    async def extract_tasks(self, message):
        self.calls += 1
        return TaskDraftBatch(tasks=self.tasks)


class Products(Catalog):
    async def resolve(self, mention):
        return []


def compile_flow(model, ctx, extractor, products, saver=None):
    return build_flexible_graph(
        model,
        ToolDispatcher(products, PCBuildApplicationService()),
        ctx,
        [],
        Budget(),
        conversation_revision=0,
        task_extractor=extractor,
        task_catalog=products,
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(ctx),
            task_message_id=uuid4(),
            task_message="GPU?",
            request_message="GPU?",
        ),
        checkpointer=saver,
    )


@pytest.mark.asyncio
async def test_registration_is_checkpointed_before_agent_and_drives_completion():
    ctx, saver, products = context(), InMemorySaver(), Products()
    extractor = Extractor([{"kind": "ANSWER", "start": 0, "end": 4, "quote": "GPU?"}])
    graph = compile_flow(ReadModel(fail_once=True), ctx, extractor, products, saver)
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    with pytest.raises(TimeoutError):
        await graph.ainvoke({"message": "GPU?"}, config)
    saved = await graph.aget_state(config)
    assert saved.values["task_draft"]["tasks"][0]["kind"] == "ANSWER"
    assert saved.values["task_resolution"]["requirements"][0]["kind"] == "ANSWER"
    result = await graph.ainvoke(None, saved.config)
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == "COMPLETED"
    assert snapshot.flexible.task_outcomes[0].task.kind == "ANSWER"
    assert extractor.calls == 1
    assert products.calls == 1


@pytest.mark.asyncio
async def test_pending_subjects_cannot_be_completed_by_generic_answer():
    ctx, products, task = context(), Products(), compare_task()

    class IncorrectThenAsk(ReadModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        async def decide(self, message, transcript, allowed, repair, context=None):
            self.calls += 1
            if self.calls == 1:
                return FlexibleDecision(
                    action={
                        "kind": "FINAL_ANSWER",
                        "answer": {"text": "Comparison complete"},
                    }
                )
            assert repair.code == "TASK_INCOMPLETE"
            assert repair.missing_fields == ["NOT_FOUND"]
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Which products?",
                    "missing_fields": ["NOT_FOUND"],
                }
            )

    graph = build_flexible_graph(
        IncorrectThenAsk(),
        ToolDispatcher(products, PCBuildApplicationService()),
        ctx,
        [],
        Budget(),
        conversation_revision=0,
        task_extractor=Extractor([task]),
        task_catalog=products,
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(ctx),
            task_message_id=uuid4(),
            task_message=task.quote,
            request_message=task.quote,
        ),
    )
    result = await graph.ainvoke({"message": task.quote})
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == "WAITING_INPUT"
    assert snapshot.flexible.answer is None
    assert result["repairs"] == 1


@pytest.mark.asyncio
async def test_unresolved_compare_can_read_then_ask_without_becoming_answer():
    ctx, products = context(), Products()
    task = compare_task()
    binding = FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=uuid4(),
        task_message=task.quote,
        request_message=task.quote,
    )

    class ReaderThenAsk(ReadModel):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return await super().decide(
                    message, transcript, allowed, repair, context
                )
            assert context.requested_tasks.tasks[0].kind == "COMPARE"
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Which products?",
                    "missing_fields": ["NOT_FOUND"],
                }
            )

    graph = build_flexible_graph(
        ReaderThenAsk(),
        ToolDispatcher(products, PCBuildApplicationService()),
        ctx,
        [],
        Budget(),
        conversation_revision=0,
        publication=binding,
        task_extractor=Extractor([task]),
        task_catalog=products,
    )
    terminal = await graph.ainvoke({"message": task.quote})
    snapshot = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    assert snapshot.flexible.status == "WAITING_INPUT"
    assert snapshot.flexible.pending_task_resolution.tasks[0].kind == "COMPARE"
    assert not snapshot.flexible.task_outcomes
    assert snapshot.flexible.answer is None
    assert products.calls == 1
