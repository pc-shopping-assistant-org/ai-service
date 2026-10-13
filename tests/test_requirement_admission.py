from uuid import uuid4

import pytest
from test_flexible_contracts import pending_proposal

from ai_service.capabilities.assistant.requirements.admission import (
    confirm_agent_patch,
    propose_agent_patch,
)
from ai_service.capabilities.assistant.requirements.contracts import RequirementPatch


def propose(original, owner, conversation, pending):
    return propose_agent_patch(
        original,
        pending.patch,
        pending.source_messages,
        owner=owner,
        conversation_id=conversation,
        target_revision=13,
    )


def test_agent_patch_only_creates_pending_and_does_not_mutate_accepted_input():
    original, owner, conversation, pending, _ = pending_proposal()
    before = original.model_dump(mode="json")
    result = propose(original, owner, conversation, pending)
    assert result.disposition == "NEEDS_CONFIRMATION"
    assert result.requirements.model_dump(mode="json") == before
    assert not result.content_changed
    assert result.pending.patch == pending.patch
    assert original.model_dump(mode="json") == before


def test_exact_confirmation_merges_atomically_consumes_pending_and_signals_change():
    original, owner, conversation, pending, reply = pending_proposal()
    before = original.model_dump(mode="json")
    result = confirm_agent_patch(
        original,
        pending,
        reply,
        owner=owner,
        conversation_id=conversation,
        conversation_revision=13,
    )
    identifier = pending.patch.set_requirements[0].id
    assert result.disposition == "ACCEPT" and result.pending is None
    assert result.content_changed
    assert result.requirements.requirements[identifier].value.amount == 16
    assert original.model_dump(mode="json") == before
    with pytest.raises(ValueError, match="No pending"):
        confirm_agent_patch(
            result.requirements,
            result.pending,
            reply,
            owner=owner,
            conversation_id=conversation,
            conversation_revision=13,
        )


def test_pending_slot_cannot_be_silently_replaced_even_with_valid_new_patch():
    original, owner, conversation, pending, _ = pending_proposal()
    with pytest.raises(ValueError, match="already pending"):
        propose_agent_patch(
            original,
            pending.patch,
            pending.source_messages,
            owner=owner,
            conversation_id=conversation,
            target_revision=14,
            pending=pending,
        )


def test_noop_preserves_hash_and_needs_no_new_confirmation():
    original, owner, conversation, pending, _ = pending_proposal()
    result = propose_agent_patch(
        original,
        RequirementPatch(),
        {},
        owner=owner,
        conversation_id=conversation,
        target_revision=13,
    )
    assert result.disposition == "ACCEPT"
    assert result.pending is None and not result.content_changed
    # A no-op does not dismiss the already-published pending proposal.
    result = propose_agent_patch(
        original,
        RequirementPatch(),
        {},
        owner=owner,
        conversation_id=conversation,
        target_revision=14,
        pending=pending,
    )
    assert result.pending == pending


def test_invalid_confirmation_or_forged_source_never_partially_changes_state():
    original, owner, conversation, pending, reply = pending_proposal()
    before = original.model_dump(mode="json")
    with pytest.raises(ValueError):
        confirm_agent_patch(
            original,
            pending,
            reply,
            owner=uuid4(),
            conversation_id=conversation,
            conversation_revision=13,
        )
    with pytest.raises(ValueError, match="evidence"):
        propose_agent_patch(
            original,
            pending.patch,
            {next(iter(pending.source_messages)): "fake"},
            owner=owner,
            conversation_id=conversation,
            target_revision=13,
        )
    assert original.model_dump(mode="json") == before


def test_repeated_identical_requirement_keeps_resolved_binding_and_content_hash():
    from ai_service.capabilities.assistant.requirements.contracts import Binding

    original, owner, conversation, _, _ = pending_proposal()
    item = next(iter(original.requirements.values()))
    original.resolved_bindings[item.id] = Binding(
        requirement_id=item.id, variant_id=uuid4(), evidence_refs=["catalog:known"]
    )
    result = propose_agent_patch(
        original,
        RequirementPatch(set_requirements=[item]),
        {item.evidence.message_id: item.evidence.quote},
        owner=owner,
        conversation_id=conversation,
        target_revision=13,
    )
    assert result.disposition == "ACCEPT" and not result.content_changed
    assert result.requirements.resolved_bindings == original.resolved_bindings
