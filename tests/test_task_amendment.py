from uuid import uuid4

import pytest
from test_task_resolution import Catalog, compare_task, product

from ai_service.capabilities.assistant.react.task_amendment import (
    TaskAmendment,
    TaskMentionEdit,
    amend_pending_task,
)
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskEvidence
from ai_service.capabilities.assistant.stateful_contracts import (
    TaskClarificationReplyV1,
    TaskProductSelectionV1,
)


def edit():
    return TaskAmendment(
        edits=[
            TaskMentionEdit(
                task_index=0,
                subject_index=0,
                evidence=TaskEvidence(start=0, end=8, quote="RTX 4080"),
            )
        ]
    )


@pytest.mark.asyncio
async def test_edited_mention_can_remain_ambiguous_then_take_a_choice_after_roundtrip():
    second, first_option, second_option = (
        product("RTX 4070"),
        product("ASUS RTX 4080"),
        product("MSI RTX 4080"),
    )
    state = CanonicalRequirementState(revision=3)
    task = compare_task()
    catalog = Catalog(
        {
            "RTX 4060": [],
            "RTX 4070": [second],
            "RTX 4080": [first_option, second_option],
        }
    )
    resolution = await resolve_tasks([task], task.quote, state, catalog)
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=task.quote,
        constraints_revision=3,
        tasks=[task],
        resolution=resolution,
    )
    edited = amend_pending_task(
        pending, edit(), state, message_id=uuid4(), message="RTX 4080"
    )
    edited.resolution = await resolve_tasks(
        edited.tasks,
        edited.message,
        state,
        catalog,
        selections=edited.resolution.bindings,
        mention_overrides=edited.mention_overrides,
    )
    reopened = PendingTaskDraft.model_validate_json(edited.model_dump_json())
    assert public_task_options(reopened).items[0].mention == "RTX 4080"
    reply = TaskClarificationReplyV1(
        pending_message_id=reopened.message_id,
        constraints_revision=3,
        selections=[
            TaskProductSelectionV1(
                task_index=0, subject_index=0, product_id=second_option.product_id
            )
        ],
    )
    selected = selections_for_pending_task(reopened, reply, state)
    final = await resolve_tasks(
        reopened.tasks,
        reopened.message,
        state,
        catalog,
        selections=selected,
        mention_overrides=reopened.mention_overrides,
    )
    assert final.requirements[0].product_ids == [
        second_option.product_id,
        second.product_id,
    ]


@pytest.mark.asyncio
async def test_edited_missing_mention_resolves_without_changing_other_product_or_task():
    second, corrected = product("RTX 4070"), product("RTX 4080")
    catalog = Catalog({"RTX 4060": [], "RTX 4070": [second], "RTX 4080": [corrected]})
    state = CanonicalRequirementState(revision=3)
    task = compare_task()
    initial = await resolve_tasks([task], task.quote, state, catalog)
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=task.quote,
        constraints_revision=3,
        tasks=[task],
        resolution=initial,
    )
    updated = amend_pending_task(
        pending, edit(), state, message_id=uuid4(), message="RTX 4080"
    )
    resolved = await resolve_tasks(
        updated.tasks,
        updated.message,
        state,
        catalog,
        selections=updated.resolution.bindings,
        mention_overrides=updated.mention_overrides,
    )
    assert resolved.requirements[0].kind == "COMPARE"
    assert resolved.requirements[0].product_ids == [
        corrected.product_id,
        second.product_id,
    ]
    assert updated.tasks == pending.tasks and not pending.mention_overrides
    assert public_task_options(updated).items[0].mention == "RTX 4080"
    with pytest.raises(ValueError, match="evidence"):
        amend_pending_task(
            pending, edit(), state, message_id=uuid4(), message="RTX 4090"
        )


@pytest.mark.asyncio
async def test_task_amendment_adapter_is_budgeted_and_cannot_supply_catalog_ids():
    from test_native_requirement_extractor import Budget as ChoiceBudget
    from test_native_requirement_extractor import Provider

    from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor

    state = CanonicalRequirementState()
    second = product("RTX 4070")
    resolution = await resolve_tasks(
        [compare_task()],
        compare_task().quote,
        state,
        Catalog({"RTX 4060": [], "RTX 4070": [second]}),
    )
    pending = PendingTaskDraft(
        message_id=uuid4(),
        message=compare_task().quote,
        constraints_revision=0,
        tasks=[compare_task()],
        resolution=resolution,
    )
    provider, budget = Provider(edit().model_dump(mode="json")), ChoiceBudget()
    result = await NativeTaskExtractor(provider, budget).amend_task(
        "RTX 4080", pending
    )
    assert result.edits[0].evidence.quote == "RTX 4080"
    assert budget.reservations[0][0] == "MODEL"
    denied = Provider({"edits": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeTaskExtractor(denied, ChoiceBudget(denied=True)).amend_task(
            "RTX 4080", pending
        )
    assert denied.model.messages == []
    from pydantic import ValidationError

    forged = edit().model_dump(mode="json")
    forged["edits"][0]["product_id"] = str(uuid4())
    with pytest.raises(ValidationError, match="Extra inputs"):
        TaskAmendment.model_validate(forged)
