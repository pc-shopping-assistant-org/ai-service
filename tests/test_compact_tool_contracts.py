"""Model wire stays small; domain admission and publication stay authoritative."""

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from test_explicit_user_admission import draft
from test_native_react_model import Provider
from test_react_tools import context

from ai_service.capabilities.assistant.react.actions import ProposeRequirementsAction
from ai_service.capabilities.assistant.react.proposal_resolution import (
    resolve_requirement_proposal,
)
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.user_admission import (
    apply_user_receipt,
)
from ai_service.infrastructure.providers.react_model import (
    NativeFlexibleModel,
    build_react_request,
)


@pytest.mark.asyncio
async def test_flat_requirement_tool_binds_quote_and_quantity_without_nested_value():
    args = {
        "items": [
            {
                "subject": "RAM",
                "predicate": "capacity",
                "value": 32,
                "unit": "GB",
                "quote": "RAM 32GB",
            }
        ]
    }
    response = AIMessage(
        content="",
        tool_calls=[
            {"id": "requirement", "name": "propose_requirements", "args": args}
        ],
    )
    decision = await NativeFlexibleModel(Provider(response)).decide(
        "RAM 32GB", [], (), None, context()
    )
    assert decision is not None
    item = decision.action.draft.items[0]
    assert item.value.amount == 32 and item.value.unit == "GB"
    assert (item.start, item.end) == (0, 8)


def test_fresh_chat_exposes_no_business_clarification_or_schema_metadata():
    _, tools = build_react_request("Xin chào", [], (), None, context())
    names = {tool["name"] for tool in tools}
    assert names == {"propose_requirements"}
    proposal = tools[0]["parameters"]
    assert "items" in proposal["properties"] and "draft" not in proposal["properties"]
    assert '"title":' not in json.dumps(proposal)
    assert '"default":' not in json.dumps(proposal)


@pytest.mark.asyncio
async def test_source_bound_user_proposal_merges_without_another_extractor():
    ctx = context()
    action = ProposeRequirementsAction(draft=draft())
    receipt = await resolve_requirement_proposal(action, ctx, "RAM 32GB", 0)
    assert receipt.policy_version == "source-bound-draft-v2"
    assert receipt.extracted is None
    changed = apply_user_receipt(
        ctx.requirements, receipt, run_id=ctx.run_id, message="RAM 32GB", catalog=[]
    )
    assert next(iter(changed.requirements.values())).value.amount == 32
    assert not ctx.requirements.requirements
    with pytest.raises(ValueError):
        apply_user_receipt(
            ctx.requirements, receipt, run_id=uuid4(), message="RAM 32GB", catalog=[]
        )


@pytest.mark.asyncio
async def test_agent_suggestion_still_requires_bound_confirmation():
    result = await resolve_requirement_proposal(
        ProposeRequirementsAction(draft=draft(), source_text="RAM 32GB"),
        context(),
        "Tư vấn giúp tôi",
        0,
    )
    assert result.disposition == "NEEDS_CONFIRMATION" and result.pending is not None


@pytest.mark.asyncio
async def test_quote_does_not_authorize_a_different_quantity():
    result = await resolve_requirement_proposal(
        ProposeRequirementsAction(draft=draft(16)), context(), "RAM 32GB", 0
    )
    assert not hasattr(result, "policy_version")
    assert result.draft.resolution.patch is None


@pytest.mark.asyncio
async def test_two_source_bound_proposals_in_one_run_preserve_both_requirements():
    ctx = context()
    message = "RAM 32GB; SSD 32GB"
    first = await resolve_requirement_proposal(
        ProposeRequirementsAction(draft=draft()), ctx, message, 0
    )
    ctx.requirements = apply_user_receipt(
        ctx.requirements, first, run_id=ctx.run_id, message=message, catalog=[]
    )
    second = RequirementDraft(
        items=[
            {
                "subject": "STORAGE",
                "predicate": "capacity",
                "start": 10,
                "end": 18,
                "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
            }
        ]
    )
    receipt = await resolve_requirement_proposal(
        ProposeRequirementsAction(draft=second), ctx, message, 0
    )
    receipt = type(receipt).model_validate_json(receipt.model_dump_json())
    changed = apply_user_receipt(
        ctx.requirements, receipt, run_id=ctx.run_id, message=message, catalog=[]
    )
    assert {item.subject for item in changed.requirements.values()} == {
        "RAM",
        "STORAGE",
    }
    assert len(ctx.requirements.requirements) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["SET", "CLEAR"])
async def test_source_admission_cannot_target_another_requirement(operation):
    ctx = context()
    receipt = await resolve_requirement_proposal(
        ProposeRequirementsAction(draft=draft()), ctx, "RAM 32GB", 0
    )
    ctx.requirements = apply_user_receipt(
        ctx.requirements, receipt, run_id=ctx.run_id, message="RAM 32GB", catalog=[]
    )
    identity = next(iter(ctx.requirements.requirements))
    replacement = RequirementDraft(
        items=[
            {
                "target_id": identity,
                "operation": operation,
                "subject": "STORAGE",
                "predicate": "capacity",
                "start": 0,
                "end": 8,
                "value": {"kind": "QUANTITY", "amount": 64, "unit": "GB"}
                if operation == "SET"
                else None,
            }
        ]
    )
    with pytest.raises(ValueError, match="target"):
        await resolve_requirement_proposal(
            ProposeRequirementsAction(draft=replacement), ctx, "SSD 64GB", 0
        )
