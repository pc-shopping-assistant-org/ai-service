from uuid import uuid4

import pytest
from test_requirement_pipeline import state as original_state

from ai_service.capabilities.assistant.react.actions import ProposeRequirementsAction
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.requirements.user_admission import (
    apply_user_receipt,
    assess_user_draft,
)


def draft(amount=32):
    return Draft(
        items=[
            {
                "subject": "RAM",
                "predicate": "capacity",
                "start": 0,
                "end": 8,
                "value": {"kind": "QUANTITY", "amount": amount, "unit": "GB"},
            }
        ]
    )


def test_matching_user_draft_merges_a_copy_with_user_source():
    state, run_id = CanonicalRequirementState(), uuid4()
    action = ProposeRequirementsAction(draft=draft())
    receipt = assess_user_draft(
        state, action, draft(), run_id=run_id, message="RAM 32GB", catalog=[]
    )
    assert receipt is not None
    restored = type(receipt).model_validate_json(receipt.model_dump_json())
    result = apply_user_receipt(
        state, restored, run_id=run_id, message="RAM 32GB", catalog=[]
    )
    requirement = next(iter(result.requirements.values()))
    assert requirement.value.amount == 32 and requirement.evidence.quote == "RAM 32GB"
    assert not state.requirements and result.revision == 1
    assert (
        apply_user_receipt(
            state, restored, run_id=run_id, message="RAM 32GB", catalog=[]
        )
        == result
    )


def test_explicit_clear_preserves_unmentioned_budget_and_goals():
    state, first_run = original_state(), uuid4()
    receipt = assess_user_draft(
        state,
        ProposeRequirementsAction(draft=draft()),
        draft(),
        run_id=first_run,
        message="RAM 32GB",
        catalog=[],
    )
    with_ram = apply_user_receipt(
        state, receipt, run_id=first_run, message="RAM 32GB", catalog=[]
    )
    ram_id = next(
        identifier
        for identifier, item in with_ram.requirements.items()
        if item.subject == "RAM"
    )
    removal = Draft(
        items=[
            {
                "operation": "CLEAR",
                "target_id": ram_id,
                "subject": "RAM",
                "predicate": "capacity",
                "start": 0,
                "end": len("Bỏ RAM"),
            }
        ]
    )
    run_id = uuid4()
    receipt = assess_user_draft(
        with_ram,
        ProposeRequirementsAction(draft=removal),
        removal,
        run_id=run_id,
        message="Bỏ RAM",
        catalog=[],
    )
    cleared = apply_user_receipt(
        with_ram, receipt, run_id=run_id, message="Bỏ RAM", catalog=[]
    )
    assert cleared.requirements == state.requirements and cleared.goals == state.goals
    assert ram_id in with_ram.requirements and ram_id not in cleared.requirements


@pytest.mark.parametrize(
    "problem", ["agent_text", "different_value", "omitted_item", "ambiguous_unit"]
)
def test_source_quote_alone_cannot_grant_user_merge_authority(problem):
    state, proposed, extracted = CanonicalRequirementState(), draft(), draft()
    source_text = None
    message = "RAM 32GB"
    if problem == "agent_text":
        source_text = "RAM 32GB"
    elif problem == "different_value":
        proposed = draft(16)
    elif problem == "omitted_item":
        extracted.items.append(extracted.items[0].model_copy(deep=True))
    else:
        message = "RAM 32"
        payload = proposed.model_dump()
        payload["items"][0]["value"] = {"kind": "UNRESOLVED_QUANTITY", "amount": 32}
        payload["items"][0]["end"] = len(message)
        proposed = Draft.model_validate(payload)
        extracted = proposed.model_copy(deep=True)
    action = ProposeRequirementsAction(draft=proposed, source_text=source_text)
    receipt = assess_user_draft(
        state, action, extracted, run_id=uuid4(), message=message, catalog=[]
    )
    assert receipt is None and not state.requirements


@pytest.mark.parametrize("problem", ["run", "message", "base"])
def test_user_receipt_cannot_be_replayed_against_another_turn_or_state(problem):
    state, run_id = CanonicalRequirementState(), uuid4()
    receipt = assess_user_draft(
        state,
        ProposeRequirementsAction(draft=draft()),
        draft(),
        run_id=run_id,
        message="RAM 32GB",
        catalog=[],
    )
    message = "RAM 32GB"
    if problem == "run":
        run_id = uuid4()
    elif problem == "message":
        message = "RAM 64GB"
    else:
        state.unresolved.append("New requirement")
    with pytest.raises(ValueError):
        apply_user_receipt(state, receipt, run_id=run_id, message=message, catalog=[])
