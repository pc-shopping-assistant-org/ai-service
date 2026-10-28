from uuid import uuid4

import pytest
from test_native_requirement_extractor import Budget, Provider
from test_quantity_clarification import MESSAGE, draft

from ai_service.capabilities.assistant.requirements.amendment import (
    ItemDiscard,
    ItemReplacement,
    PendingAmendment,
    amend_pending,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    DraftItem,
    PendingDraft,
    resolve_draft,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)


def pending_quantity():
    source = uuid4()
    state = CanonicalRequirementState()
    resolution = resolve_draft(
        draft(), message_id=source, message=MESSAGE, state=state, catalog=[]
    )
    return state, PendingDraft(
        message_id=source,
        message=MESSAGE,
        constraints_revision=0,
        resolution=resolution,
    )


def replace_ram():
    return PendingAmendment(
        replacements=[
            ItemReplacement(
                item_index=2,
                item=DraftItem(
                    subject="RAM",
                    predicate="capacity",
                    operator="GTE",
                    value=QuantityValue(amount=64, unit="GB"),
                    start=0,
                    end=4,
                ),
            )
        ]
    )


def test_drop_unmerged_item_does_not_clear_original_budget():
    state, pending = pending_quantity()
    followup = uuid4()
    changed = amend_pending(
        pending,
        PendingAmendment(
            discards=[ItemDiscard(item_index=2, start=0, end=6, quote="Bỏ RAM")]
        ),
        state,
        message_id=followup,
        message="Bỏ RAM",
    )
    resolved = resolve_draft(
        changed.resolution.draft,
        message_id=pending.message_id,
        message=pending.message,
        state=state,
        catalog=[],
        item_sources=changed.item_sources,
        discarded_items=changed.discarded_items,
    )
    merged = merge_requirements(state, resolved.patch, changed.source_messages())
    assert len(merged.requirements) == 1
    assert next(iter(merged.requirements.values())).value.amount == 25_000_000
    assert pending.discarded_items == []


def test_amendment_preserves_original_completion_task_source():
    from ai_service.capabilities.assistant.requirements.tasks import TaskDraft

    state, pending = pending_quantity()
    pending.resolution.draft = Draft(
        items=pending.resolution.draft.items,
        tasks=[TaskDraft(kind="BUILD", start=0, end=5, quote="Build")],
    )
    changed = amend_pending(
        pending, replace_ram(), state, message_id=uuid4(), message="64GB"
    )
    resolved = resolve_draft(
        changed.resolution.draft,
        message_id=changed.message_id,
        message=changed.message,
        state=state,
        catalog=[],
        item_sources=changed.item_sources,
        discarded_items=changed.discarded_items,
    )
    assert resolved.draft.tasks == pending.resolution.draft.tasks
    assert resolved.draft.tasks[0].quote == "Build"


def test_invalid_amendment_bundle_is_atomic_and_does_not_change_target():
    state, pending = pending_quantity()
    original = pending.model_dump(mode="json")
    invalid = replace_ram().model_copy(deep=True)
    invalid.replacements[0].item = DraftItem(
        subject="GPU",
        predicate="capacity",
        value=QuantityValue(amount=64, unit="GB"),
        start=0,
        end=4,
    )
    with pytest.raises(ValueError, match="target identity"):
        amend_pending(pending, invalid, state, message_id=uuid4(), message="64GB")
    assert pending.model_dump(mode="json") == original


@pytest.mark.asyncio
async def test_native_amendment_reserves_model_before_dispatch():
    _, pending = pending_quantity()
    budget = Budget()
    provider = Provider(
        {
            "replacements": [
                {
                    "item_index": 2,
                    "item": {
                        **replace_ram()
                        .replacements[0]
                        .item.model_dump(exclude={"start", "end"}),
                        "quote": "64GB",
                    },
                }
            ],
        }
    )
    amendment = await NativeRequirementExtractor(provider, budget).amend(
        "64GB", pending
    )
    assert amendment.replacements[0].item.value.amount == 64
    assert budget.reservations[0][0] == "MODEL"
    assert MESSAGE in provider.model.messages[-1][1]
    denied = Provider({"replacements": [], "discards": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeRequirementExtractor(denied, Budget(denied=True)).amend(
            "64GB", pending
        )
    assert denied.model.messages == []


def test_amend_amount_retains_budget_and_binds_original_and_new_sources():
    original, followup = uuid4(), uuid4()
    state = CanonicalRequirementState()
    initial = resolve_draft(
        draft(), message_id=original, message=MESSAGE, state=state, catalog=[]
    )
    pending = PendingDraft(
        message_id=original, message=MESSAGE, constraints_revision=0, resolution=initial
    )
    modified = amend_pending(
        pending, replace_ram(), state, message_id=followup, message="64GB"
    )
    resolved = resolve_draft(
        modified.resolution.draft,
        message_id=original,
        message=MESSAGE,
        state=state,
        catalog=[],
        item_sources=modified.item_sources,
        discarded_items=modified.discarded_items,
    )
    merged = merge_requirements(
        state, resolved.patch, {original: MESSAGE, followup: "64GB"}
    )
    budget, ram = [
        next(item for item in merged.requirements.values() if item.subject == subject)
        for subject in ["SETUP", "RAM"]
    ]
    assert budget.value.amount == 25_000_000 and budget.evidence.message_id == original
    assert (
        ram.value == QuantityValue(amount=64, unit="GB")
        and ram.evidence.message_id == followup
    )
    assert merged.revision == 1
    with pytest.raises(ValueError, match="evidence"):
        merge_requirements(state, resolved.patch, {original: MESSAGE})
