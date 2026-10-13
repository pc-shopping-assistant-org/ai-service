import pytest
from test_flexible_graph import ReadModel
from test_flexible_publication import base_snapshot
from test_flexible_reference_continuation import pending_reference
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
from ai_service.capabilities.assistant.requirements.contracts import requirements_hash
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    continue_reference,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    ChoiceDraft,
    resolve_draft,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("free", [False, True])
async def test_noop_choice_continues_original_task_without_new_consent(free):
    state, owner, conversation, rows, pending, reply = pending_reference()
    resolved = resolve_draft(
        pending.draft.resolution.draft,
        message_id=pending.draft.message_id,
        message=pending.draft.message,
        state=state,
        catalog=rows,
        selections={0: rows[0].variant_id},
    )
    state = merge_requirements(
        state, resolved.patch, {pending.draft.message_id: pending.draft.message}
    )
    pending.draft.resolution.draft.items[0].target_id = next(iter(state.requirements))
    pending.draft.resolution = resolve_draft(
        pending.draft.resolution.draft,
        message_id=pending.draft.message_id,
        message=pending.draft.message,
        state=state,
        catalog=rows,
    )
    pending.base_requirements_hash = requirements_hash(state)
    pending.draft.constraints_revision = state.revision
    noop = continue_reference(
        state,
        pending,
        reply,
        rows,
        owner=owner,
        conversation_id=conversation,
        conversation_revision=1,
    )
    assert noop.admission.disposition == "ACCEPT" and noop.admission.pending is None
    assert noop.admission.requirements == state and not noop.admission.content_changed
    ctx = context(state)
    ctx.account_id, ctx.conversation_id, ctx.catalog = owner, conversation, rows
    tasks = [CompletionRequirement(kind="ANSWER")]
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=pending.draft.message_id,
        task_message="Find GPU",
        status="WAITING_INPUT",
        waiting_message="Choose GPU",
        waiting_tasks=[TaskOutcome(task=tasks[0], status="UNRESOLVED")],
        pending_resolution=pending,
    )
    binding = FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=accepted.flexible.task_message_id,
        task_message="Find GPU",
        request_message="Choose first",
        reference_reply=None if free else reply,
    )

    class Choices:
        async def choose(self, message, options):
            return ChoiceDraft(
                selections=[
                    {
                        "item_index": 0,
                        "variant_id": rows[0].variant_id,
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

    products, service = Catalog(), PCBuildApplicationService()
    flow = build_flexible_graph(
        ReadModel(),
        ToolDispatcher(products, service),
        ctx,
        tasks,
        Budget(),
        conversation_revision=1,
        publication=binding,
        clarifier=Choices() if free else None,
    )
    result = await flow.ainvoke({"message": "Choose first"})
    published = result["accepted_snapshot"]
    assert result["phase"] == "READY_TO_PUBLISH"
    assert published["requirement_state"] == state.model_dump(mode="json")
    assert published["flexible"]["status"] == "COMPLETED"
    assert published["flexible"]["task_message"] == "Find GPU"
    assert published["flexible"]["pending_resolution"] is None
    assert products.calls == 1
    changed = reply.model_copy(
        update={
            "selections": [
                reply.selections[0].model_copy(
                    update={"variant_id": rows[1].variant_id}
                )
            ]
        }
    )
    changed_binding = binding.model_copy(update={"reference_reply": changed})
    with pytest.raises(ValueError, match="retain pending consent"):
        changed_binding.prepare(
            dict(result) | {"free_resolution_reply": None}, ctx, tasks, service, 1
        )
