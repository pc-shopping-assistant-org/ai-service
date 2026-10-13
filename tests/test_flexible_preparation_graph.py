from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_flexible_graph import ReadModel
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_waiting_publication import Proposer

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("waiting", [False, True])
async def test_terminal_checkpoint_contains_revalidated_snapshot(waiting):
    ctx, saver = context(), InMemorySaver()
    task_id = uuid4()
    tasks = (
        [CompletionRequirement(kind="BUILD", constraints_revision=0)]
        if waiting
        else [CompletionRequirement(kind="ANSWER")]
    )
    binding = FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=task_id,
        task_message="Original task",
        request_message="Original task",
    )
    graph = build_flexible_graph(
        Proposer() if waiting else ReadModel(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        tasks,
        Budget(),
        conversation_revision=0,
        publication=binding,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    result = await graph.ainvoke({"message": "Original task"}, config)
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == ("WAITING_INPUT" if waiting else "COMPLETED")
    assert snapshot.flexible.task_message_id == task_id
    assert snapshot.flexible.task_message == "Original task"
    saved = await graph.aget_state(config)
    assert not saved.next
    assert saved.values["accepted_snapshot"] == result["accepted_snapshot"]
    assert snapshot.requirement_state == ctx.requirements
    assert binding.accepted.flexible is None


@pytest.mark.asyncio
async def test_stored_request_mismatch_rejects_before_model_dispatch():
    ctx, model = context(), ReadModel()
    binding = FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=uuid4(),
        task_message="GPU",
        request_message="GPU",
    )
    graph = build_flexible_graph(
        model,
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
        publication=binding,
    )
    with pytest.raises(ValueError):
        await graph.ainvoke({"message": "Different request"})
    assert not model.transcripts


@pytest.mark.asyncio
async def test_confirmation_preparation_keeps_original_task_not_ok_reply():
    from ai_service.capabilities.assistant.react.actions import FlexibleDecision
    from ai_service.capabilities.assistant.requirements.proposals import (
        ProposalConfirmation,
    )

    ctx, source_id = context(), uuid4()
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    first_binding = FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=source_id,
        task_message="Build PC",
        request_message="Build PC",
    )

    def build(model, binding, trusted, revision):
        return build_flexible_graph(
            model,
            ToolDispatcher(Catalog(), PCBuildApplicationService()),
            trusted,
            tasks,
            Budget(),
            conversation_revision=revision,
            publication=binding,
        )

    first = await build(Proposer(), first_binding, ctx, 0).ainvoke(
        {"message": "Build PC"}
    )
    accepted = AcceptedConversationSnapshot.model_validate(first["accepted_snapshot"])
    proposal = accepted.flexible.pending_proposal
    reply = ProposalConfirmation(
        proposal_id=proposal.proposal_id,
        proposal_hash=proposal.proposal_hash,
        target_revision=proposal.target_revision,
        base_requirements_hash=proposal.base_requirements_hash,
    )
    binding = FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=source_id,
        task_message="Build PC",
        request_message="OK",
        confirmation=reply,
    )

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Budget?",
                    "missing_fields": ["BUDGET_REQUIRED"],
                }
            )

    next_ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    terminal = await build(Ask(), binding, next_ctx, 1).ainvoke({"message": "OK"})
    snapshot = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    assert snapshot.flexible.task_message == "Build PC"
    assert snapshot.flexible.task_message_id == source_id
    assert snapshot.flexible.pending_proposal is None
    assert snapshot.flexible.clarification is not None
    assert snapshot.flexible.task_outcomes[0].status == "UNRESOLVED"
    assert len(snapshot.requirement_state.requirements) == 1
    assert not accepted.requirement_state.requirements
    bad_binding = binding.model_copy(update={"task_message": "OK"})
    with pytest.raises(ValueError):
        build(Ask(), bad_binding, next_ctx, 1)
