from uuid import uuid4

import pytest

from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
    continue_unit,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    PendingDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    UnitClarificationReplyV1,
)


def pending_amount():
    state, owner, conversation, source = (
        CanonicalRequirementState(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    message = "SSD 1"
    draft = Draft(
        items=[
            {
                "subject": "STORAGE",
                "predicate": "capacity",
                "start": 0,
                "end": 5,
                "value": {"kind": "UNRESOLVED_QUANTITY", "amount": 1},
            }
        ]
    )
    resolution = resolve_draft(
        draft, message_id=source, message=message, state=state, catalog=[]
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
    return state, owner, conversation, pending


def reply(pending, *, quote="TB", start=5, end=7):
    return UnitClarificationReplyV1(
        pending_message_id=pending.draft.message_id,
        constraints_revision=0,
        selections=[
            {"item_index": 0, "unit": "TB", "quote": quote, "start": start, "end": end}
        ],
    )


def test_unit_keeps_original_amount_and_full_reply_source_without_merging():
    state, owner, conversation, pending = pending_amount()
    source = uuid4()
    result = continue_unit(
        state,
        pending,
        reply(pending),
        [],
        message_id=source,
        message="Dùng TB nhé",
        owner=owner,
        conversation_id=conversation,
        conversation_revision=1,
    )
    item = result.admission.pending.patch.set_requirements[0]
    assert item.value.amount == 1000 and item.value.unit == "GB"
    assert result.admission.pending.source_messages[source] == "Dùng TB nhé"
    assert item.evidence.quote == "SSD 1"
    assert item.evidence.supporting_evidence[0].quote == "TB"
    assert item.evidence.supporting_evidence[0].start == 5
    assert result.admission.requirements == state and state.requirements == {}


@pytest.mark.parametrize(
    "message,selection",
    [
        ("Dùng GB nhé", {"quote": "TB", "start": 5, "end": 7}),
        ("2TB", {"quote": "TB", "start": 1, "end": 3}),
        ("Không biết", {"quote": "TB", "start": 0, "end": 2}),
    ],
)
def test_forged_unit_or_changed_amount_is_rejected(message, selection):
    state, owner, conversation, pending = pending_amount()
    with pytest.raises(ValueError):
        continue_unit(
            state,
            pending,
            reply(pending, **selection),
            [],
            message_id=uuid4(),
            message=message,
            owner=owner,
            conversation_id=conversation,
            conversation_revision=1,
        )
    assert state.requirements == {}


@pytest.mark.asyncio
async def test_typed_unit_reply_ends_at_confirmation_without_model_call():
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
            raise AssertionError("Unit continuation should not call model")

    state, owner, conversation, pending = pending_amount()
    ctx = context(state)
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
            "message": "Dùng TB nhé",
            "pending_draft": pending.model_dump(mode="json"),
            "unit_reply": reply(pending).model_dump(mode="json"),
        }
    )
    assert result["phase"] == "WAITING_INPUT" and result["pending_draft"] is None
    assert (
        result["pending_proposal"]["patch"]["set_requirements"][0]["value"]["amount"]
        == 1000
    )
    assert result["working_requirements"]["requirements"] == {}


def test_unit_then_sku_continuation_preserves_original_draft_and_full_unit_message():
    from test_flexible_reference_continuation import pending_reference

    from ai_service.capabilities.assistant.requirements.pending_resolution import (
        continue_reference,
    )
    from ai_service.capabilities.assistant.stateful_contracts import (
        ClarificationReplyV1,
    )

    state, owner, conversation, pending = pending_amount()
    catalog = pending_reference()[3]
    message = "SSD 1; RTX 5070 Ti"
    draft = pending.draft.resolution.draft.model_copy(deep=True)
    from ai_service.capabilities.assistant.requirements.resolver import DraftItem

    draft.items.append(
        DraftItem(
            subject="GPU",
            predicate="identity",
            start=7,
            end=len(message),
            value={"kind": "TEXT", "text": "RTX 5070 Ti"},
        )
    )
    pending.draft.message = message
    pending.draft.resolution = resolve_draft(
        draft,
        message_id=pending.draft.message_id,
        message=message,
        state=state,
        catalog=catalog,
    )
    source = uuid4()
    partial = continue_unit(
        state,
        pending,
        reply(pending),
        catalog,
        message_id=source,
        message="Dùng TB nhé",
        owner=owner,
        conversation_id=conversation,
        conversation_revision=1,
    )
    assert partial.pending is not None and partial.admission is None
    restored = PendingResolution.model_validate_json(partial.pending.model_dump_json())
    selected = ClarificationReplyV1(
        pending_message_id=pending.draft.message_id,
        selections=[{"item_index": 1, "variant_id": catalog[0].variant_id}],
    )
    resolved = continue_reference(
        state,
        restored,
        selected,
        catalog,
        owner=owner,
        conversation_id=conversation,
        conversation_revision=2,
    )
    proposal = resolved.admission.pending
    assert proposal.source_messages[source] == "Dùng TB nhé"
    assert len(proposal.patch.set_requirements) == 2
    assert proposal.patch.set_requirements[0].value.amount == 1000
    assert proposal.patch.set_requirements[1].value.variant_id == catalog[0].variant_id
