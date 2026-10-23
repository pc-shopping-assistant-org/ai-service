from uuid import uuid4

import pytest
from test_flexible_publication import base_snapshot
from test_flexible_unit_continuation import pending_amount
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.resolution_reply import (
    FreeResolutionReplyV2,
    derive_resolution_reply,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitChoiceDraft,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.parametrize(
    "field", ["run_id", "pending_message_id", "request_message_hash"]
)
def test_free_choice_receipt_cannot_cross_run_source_or_request(field):
    _, _, _, pending = pending_amount()
    run = uuid4()
    receipt = FreeResolutionReplyV2(
        run_id=run,
        pending_message_id=pending.draft.message_id,
        request_message_hash=canonical_hash("GB"),
        unit_choice=UnitChoiceDraft(),
    )
    receipt = receipt.model_copy(
        update={field: "0" * 64 if field.endswith("hash") else uuid4()}
    )
    with pytest.raises(ValueError, match="source mismatch"):
        derive_resolution_reply(receipt, "GB", pending, run)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_response", [None, "Câu trả lời chen ngang trước đó."])
async def test_unclear_free_reply_preserves_pending_source_and_does_not_call_agent(
    prior_response,
):
    class NoAgent:
        async def decide(self, *args):
            raise AssertionError("Unclear unit reply must keep waiting")

    class Choices:
        async def choose_unit(self, message, options):
            return UnitChoiceDraft()

    state, owner, conversation, pending = pending_amount()
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Which unit?",
        waiting_tasks=[TaskOutcome(task=tasks[0], status="UNRESOLVED")],
        pending_resolution=pending,
        response={"text": prior_response} if prior_response else None,
    )
    binding = FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=accepted.flexible.task_message_id,
        task_message="Build PC",
        request_message="Chưa biết",
    )
    service = PCBuildApplicationService()
    graph = build_flexible_graph(
        NoAgent(),
        ToolDispatcher(Catalog(), service),
        ctx,
        tasks,
        Budget(),
        conversation_revision=1,
        publication=binding,
        unit_clarifier=Choices(),
    )
    result = await graph.ainvoke({"message": "Chưa biết"})
    assert result["phase"] == "WAITING_INPUT"
    assert result["accepted_snapshot"]["flexible"].get("response") is None
    assert result["accepted_snapshot"]["flexible"][
        "pending_resolution"
    ] == pending.model_dump(mode="json")
    assert result["accepted_snapshot"]["requirement_state"] == state.model_dump(
        mode="json"
    )
    result["free_resolution_reply"]["run_id"] = str(uuid4())
    with pytest.raises(ValueError, match="source mismatch"):
        binding.prepare(result, ctx, tasks, service, 1)
