from uuid import uuid4

import pytest
from test_native_requirement_extractor import Budget, Provider

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    ClarificationItemV1,
    PublicClarificationV1,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    Evidence,
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    confirmations_for_pending,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    DraftItem,
    PendingDraft,
    RequirementDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.stateful_contracts import (
    UnitClarificationReplyV1,
    UnitSelectionV1,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)

MESSAGE = "Build 25000000 VND gaming 1080p, RAM 32"


def draft():
    return RequirementDraft(
        items=[
            DraftItem(
                subject="SETUP",
                predicate="budget",
                operator="LTE",
                value=QuantityValue(amount=25_000_000, unit="VND"),
                start=6,
                end=18,
            ),
            DraftItem(
                target="GOAL",
                subject="WORKLOAD",
                predicate="use_case",
                value=TextValue(text="GAMING_1080P"),
                start=19,
                end=31,
            ),
            DraftItem(
                subject="RAM",
                predicate="capacity",
                operator="GTE",
                value=UnresolvedQuantityValue(amount=32),
                start=33,
                end=len(MESSAGE),
            ),
        ]
    )


def test_unit_selection_rejects_invalid_span_at_request_validation():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="span"):
        UnitSelectionV1(item_index=2, unit="GB", start=4, end=3, quote="GB")


@pytest.mark.parametrize(
    "amount, quote, unit, expected",
    [
        (25, "triệu", "VND", QuantityValue(amount=25_000_000, unit="VND")),
        (1, "TB", "TB", QuantityValue(amount=1000, unit="GB")),
    ],
)
def test_confirmed_units_use_server_normalization(amount, quote, unit, expected):
    from ai_service.capabilities.assistant.requirements.units import confirmed_quantity

    assert confirmed_quantity(amount, quote, unit) == expected


def test_unit_reply_cannot_silently_ignore_a_changed_amount():
    source, followup = uuid4(), uuid4()
    state = CanonicalRequirementState()
    initial = resolve_draft(
        draft(), message_id=source, message=MESSAGE, state=state, catalog=[]
    )
    pending = PendingDraft(
        message_id=source, message=MESSAGE, constraints_revision=0, resolution=initial
    )
    reply = UnitClarificationReplyV1(
        pending_message_id=source,
        constraints_revision=0,
        selections=[UnitSelectionV1(item_index=2, unit="GB", start=3, end=5, quote="GB")],
    )
    with pytest.raises(ValueError, match="amount"):
        confirmations_for_pending(
            pending, reply, state, message_id=followup, message="64 GB"
        )


@pytest.mark.asyncio
async def test_unit_choice_budget_denial_does_not_dispatch_model():
    provider = Provider({"selections": []})
    chooser = NativeRequirementExtractor(provider, Budget(denied=True))
    options = PublicClarificationV1(
        pending_message_id=uuid4(),
        constraints_revision=0,
        items=[
            ClarificationItemV1(
                item_index=2,
                reason="QUANTITY_UNIT_REQUIRED",
                quantity_amount=32,
                unit_options=["GB", "TB"],
            )
        ],
    )
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await chooser.choose_unit("GB", options)
    assert provider.model.messages == []


def test_unit_confirmation_merges_original_draft_once_and_binds_both_sources():
    source, followup = uuid4(), uuid4()
    state = CanonicalRequirementState()
    initial = resolve_draft(
        draft(), message_id=source, message=MESSAGE, state=state, catalog=[]
    )
    assert (
        initial.patch is None and initial.pending[0].reason == "QUANTITY_UNIT_REQUIRED"
    )
    pending = PendingDraft(
        message_id=source, message=MESSAGE, constraints_revision=0, resolution=initial
    )
    reply = UnitClarificationReplyV1(
        pending_message_id=source,
        constraints_revision=0,
        selections=[UnitSelectionV1(item_index=2, unit="GB", start=0, end=2, quote="GB")],
    )
    confirmed = confirmations_for_pending(
        pending, reply, state, message_id=followup, message="GB"
    )
    resolved = resolve_draft(
        draft(),
        message_id=source,
        message=MESSAGE,
        state=state,
        catalog=[],
        quantity_confirmations=confirmed,
    )
    merged = merge_requirements(
        state, resolved.patch, {source: MESSAGE, followup: "GB"}
    )
    assert merged.revision == 1
    assert {item.predicate for item in merged.requirements.values()} == {
        "budget",
        "capacity",
    }
    ram = next(item for item in merged.requirements.values() if item.subject == "RAM")
    assert ram.value == QuantityValue(amount=32, unit="GB")
    assert (
        ram.evidence.message_id == source
        and ram.evidence.supporting_evidence[0].message_id == followup
    )
    with pytest.raises(ValueError, match="evidence"):
        merge_requirements(state, resolved.patch, {source: MESSAGE})
    with pytest.raises(ValueError, match="evidence"):
        confirmations_for_pending(
            pending, reply, state, message_id=followup, message="TB"
        )
    with pytest.raises(ValueError, match="source/revision"):
        confirmations_for_pending(
            pending,
            reply.model_copy(update={"constraints_revision": 1}),
            state,
            message_id=followup,
            message="GB",
        )


def test_absent_supporting_evidence_keeps_existing_hash():
    source = uuid4()
    payload = {"message_id": str(source), "start": 0, "end": 2, "quote": "GB"}
    assert canonical_hash(Evidence.model_validate(payload)) == canonical_hash(payload)
