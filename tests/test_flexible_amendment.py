from uuid import uuid4

import pytest
from test_flexible_publication import base_snapshot
from test_flexible_unit_continuation import pending_amount
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.amendment import PendingAmendment
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitChoiceDraft,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_target", [False, True])
async def test_amendment_is_pending_consent_and_rederived_from_original_source(
    invalid_target,
):
    class NoAgent:
        async def decide(self, *args):
            raise AssertionError("Changed draft must wait for confirmation")

    class Choices:
        async def choose_unit(self, *args):
            return UnitChoiceDraft()

    class Amender:
        async def amend(self, message, pending):
            return PendingAmendment(
                replacements=[
                    {
                        "item_index": 0,
                        "item": {
                            "subject": "RAM" if invalid_target else "STORAGE",
                            "predicate": "capacity",
                            "start": 0,
                            "end": 3,
                            "value": {"kind": "QUANTITY", "amount": 2, "unit": "TB"},
                        },
                    }
                ]
            )

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
    )
    binding = FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=accepted.flexible.task_message_id,
        task_message="Build PC",
        request_message="2TB",
    )
    service = PCBuildApplicationService()
    flow = build_flexible_graph(
        NoAgent(),
        ToolDispatcher(Catalog(), service),
        ctx,
        tasks,
        Budget(),
        conversation_revision=1,
        publication=binding,
        unit_clarifier=Choices(),
        amender=Amender(),
    )
    if invalid_target:
        with pytest.raises(ValueError, match="target identity"):
            await flow.ainvoke({"message": "2TB"})
        return
    values = await flow.ainvoke({"message": "2TB"})
    proposal = values["accepted_snapshot"]["flexible"]["pending_proposal"]
    assert proposal["patch"]["set_requirements"][0]["value"]["amount"] == 2000
    assert values["working_requirements"] == state.model_dump(mode="json")
    assert values["accepted_snapshot"]["flexible"]["task_message"] == "Build PC"
    assert "2TB" in proposal["source_messages"].values()
    original_run = values["free_amendment"]["run_id"]
    values["free_amendment"]["run_id"] = str(uuid4())
    with pytest.raises(ValueError, match="receipt source mismatch"):
        binding.prepare(values, ctx, tasks, service, 1)
    values["free_amendment"]["run_id"] = original_run
    values["pending_proposal"]["patch"]["set_requirements"][0]["value"]["amount"] = 999
    with pytest.raises(ValueError):
        binding.prepare(values, ctx, tasks, service, 1)
