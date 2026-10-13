from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
    continue_reference,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    PendingDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.stateful_contracts import ClarificationReplyV1


def pending_reference():
    state, owner, conversation, source = (
        CanonicalRequirementState(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    catalog = [
        CatalogRowV1(
            variant_id=(identifier := uuid4()),
            sku=f"GPU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=ComponentSpec(
                id=identifier,
                name=f"RTX 5070 Ti {index}",
                category="GPU",
                price=20_000_000,
            ),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["catalog:gpu"],
        )
        for index in range(2)
    ]
    message = "RTX 5070 Ti"
    draft = Draft(
        items=[
            {
                "subject": "GPU",
                "predicate": "identity",
                "start": 0,
                "end": len(message),
                "value": {"kind": "TEXT", "text": message},
            }
        ]
    )
    resolution = resolve_draft(
        draft, message_id=source, message=message, state=state, catalog=catalog
    )
    pending = PendingResolution(
        owner=owner,
        conversation_id=conversation,
        target_revision=1,
        base_requirements_hash=requirements_hash(state),
        draft=PendingDraft(
            message_id=source,
            message=message,
            constraints_revision=0,
            resolution=resolution,
        ),
    )
    reply = ClarificationReplyV1(
        pending_message_id=source,
        selections=[{"item_index": 0, "variant_id": catalog[0].variant_id}],
    )
    return state, owner, conversation, catalog, pending, reply


def test_offered_sku_becomes_pending_confirmation_not_immediate_merge():
    state, owner, conversation, catalog, pending, reply = pending_reference()
    result = continue_reference(
        state,
        pending,
        reply,
        catalog,
        owner=owner,
        conversation_id=conversation,
        conversation_revision=1,
    )
    assert result.admission.disposition == "NEEDS_CONFIRMATION"
    assert result.admission.pending.target_revision == 2
    assert result.admission.requirements == state and result.pending is None
    assert (
        result.admission.pending.patch.set_requirements[0].value.variant_id
        == catalog[0].variant_id
    )


@pytest.mark.parametrize(
    "problem",
    [
        "owner",
        "conversation",
        "source",
        "candidate",
        "not_published",
        "missing_row",
        "content",
    ],
)
def test_invalid_or_unavailable_choice_cannot_resolve_pending(problem):
    state, owner, conversation, catalog, pending, reply = pending_reference()
    revision = 1
    if problem == "owner":
        owner = uuid4()
    if problem == "conversation":
        conversation = uuid4()
    if problem == "source":
        reply.pending_message_id = uuid4()
    if problem == "candidate":
        reply.selections[0].variant_id = uuid4()
    if problem == "not_published":
        revision = 0
    if problem == "missing_row":
        catalog = []
    if problem == "content":
        state.unresolved.append("new requirement")
    with pytest.raises(ValueError):
        continue_reference(
            state,
            pending,
            reply,
            catalog,
            owner=owner,
            conversation_id=conversation,
            conversation_revision=revision,
        )
    assert state.requirements == {}


@pytest.mark.asyncio
async def test_graph_continues_sku_choice_without_calling_model_or_merging():
    from test_react_graph import Budget
    from test_react_tools import Catalog, context

    from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
    from ai_service.capabilities.assistant.react.tools import ToolDispatcher
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )
    from ai_service.infrastructure.graph.flexible import build_flexible_graph

    class NoModel:
        async def decide(self, *args):
            raise AssertionError("Typed SKU continuation must not call the model")

    state, owner, conversation, catalog, pending, reply = pending_reference()
    ctx = context(state, catalog)
    ctx.account_id, ctx.conversation_id = owner, conversation
    flow = build_flexible_graph(
        NoModel(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=1,
    )
    result = await flow.ainvoke(
        {
            "message": "Chọn cái đầu",
            "pending_draft": pending.model_dump(mode="json"),
            "reference_reply": reply.model_dump(mode="json"),
        }
    )
    assert result["phase"] == "WAITING_INPUT" and result["pending_draft"] is None
    assert result["pending_proposal"]["target_revision"] == 2
    assert result["working_requirements"]["requirements"] == {}
