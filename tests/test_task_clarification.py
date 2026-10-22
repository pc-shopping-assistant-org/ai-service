from uuid import uuid4

import pytest
from test_task_resolution import Catalog, compare_task, product

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    TaskClarificationReplyV1,
    TaskProductSelectionV1,
)


@pytest.mark.asyncio
async def test_task_choice_budget_denial_cannot_dispatch_model():
    from test_native_requirement_extractor import Budget as ChoiceBudget
    from test_native_requirement_extractor import Provider

    from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor

    first, alternative, second = (
        product("ASUS RTX 4060"),
        product("MSI RTX 4060"),
        product("RTX 4070"),
    )
    resolution = await resolve_tasks(
        [compare_task()],
        compare_task().quote,
        CanonicalRequirementState(),
        Catalog({"RTX 4060": [first, alternative], "RTX 4070": [second]}),
    )
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=compare_task().quote,
        constraints_revision=0,
        tasks=[compare_task()],
        resolution=resolution,
    )
    provider = Provider({"selections": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeTaskExtractor(provider, ChoiceBudget(denied=True)).choose_task(
            "Chọn MSI", public_task_options(pending)
        )
    assert provider.model.messages == []


@pytest.mark.asyncio
async def test_offered_product_choice_resolves_original_comparison_and_keeps_unique_binding():
    first, alternative, second = (
        product("ASUS RTX 4060"),
        product("MSI RTX 4060"),
        product("RTX 4070"),
    )
    catalog = Catalog({"RTX 4060": [first, alternative], "RTX 4070": [second]})
    state = CanonicalRequirementState(revision=3)
    initial = await resolve_tasks([compare_task()], compare_task().quote, state, catalog)
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=compare_task().quote,
        constraints_revision=3,
        tasks=[compare_task()],
        resolution=initial,
    )
    projection = public_task_options(pending)
    AcceptedConversationSnapshot(
        consultation=ConsultationStateV1(),
        pending_tasks=pending,
        public_task_clarification=projection,
        requirement_state=state,
    )
    with pytest.raises(ValueError, match="revision binding"):
        AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            pending_tasks=pending,
            public_task_clarification=projection,
            requirement_state=CanonicalRequirementState(revision=2),
        )
    with pytest.raises(ValueError, match="projection binding"):
        AcceptedConversationSnapshot(
            consultation=ConsultationStateV1(),
            pending_tasks=pending,
            public_task_clarification=None,
            requirement_state=state,
        )
    reply = TaskClarificationReplyV1(
        pending_message_id=pending.message_id,
        constraints_revision=3,
        selections=[
            TaskProductSelectionV1(
                task_index=0, subject_index=0, product_id=alternative.product_id
            )
        ],
    )
    selected = selections_for_pending_task(pending, reply, state)
    assert any(item.product_id == second.product_id for item in selected)
    resolved = await resolve_tasks(
        pending.tasks, pending.message, state, catalog, selections=selected
    )
    assert not resolved.pending
    assert resolved.requirements[0].product_ids == [
        alternative.product_id,
        second.product_id,
    ]

    catalog.matches["RTX 4060"] = [first]
    unavailable = await resolve_tasks(
        pending.tasks, pending.message, state, catalog, selections=selected
    )
    assert unavailable.requirements == []
    assert unavailable.pending[0].reason == "SELECTED_PRODUCT_UNAVAILABLE"
    with pytest.raises(ValueError, match="candidate"):
        selections_for_pending_task(
            pending,
            reply.model_copy(
                update={
                    "selections": [
                        TaskProductSelectionV1(
                            task_index=0, subject_index=0, product_id=uuid4()
                        )
                    ]
                }
            ),
            state,
        )
    with pytest.raises(ValueError, match="source/revision"):
        selections_for_pending_task(
            pending, reply.model_copy(update={"constraints_revision": 2}), state
        )
