from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_requirement_pipeline import state

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.pending_action import (
    validate_pending_action,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    DraftItem,
    PendingDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.units import UnresolvedQuantityValue
from ai_service.capabilities.assistant.stateful_contracts import (
    PendingActionV1,
    TurnRequestV1,
)


def pending(original):
    message, source = "RAM 32", uuid4()
    draft = Draft(
        items=[
            DraftItem(
                subject="RAM",
                predicate="capacity",
                operator="GTE",
                value=UnresolvedQuantityValue(amount=32),
                start=0,
                end=len(message),
            )
        ]
    )
    return PendingDraft(
        message_id=source,
        message=message,
        constraints_revision=original.revision,
        resolution=resolve_draft(
            draft, message_id=source, message=message, state=original, catalog=[]
        ),
    )


def test_pending_action_requires_current_source_revision_and_single_reply_kind():
    original = state()
    artifact = pending(original)
    action = PendingActionV1(
        action="DISCARD",
        pending_message_id=artifact.message_id,
        constraints_revision=original.revision,
    )
    proof = validate_pending_action(action, original, artifact, None)
    assert proof.pending_hash == canonical_hash(artifact)
    for invalid in [
        action.model_copy(update={"pending_message_id": uuid4()}),
        action.model_copy(update={"constraints_revision": original.revision + 1}),
    ]:
        with pytest.raises(ValueError):
            validate_pending_action(invalid, original, artifact, None)
    with pytest.raises(ValueError, match="pending"):
        validate_pending_action(action, original, None, None)
    with pytest.raises(ValidationError, match="pending action"):
        TurnRequestV1(
            conversation_id=uuid4(),
            request_id=uuid4(),
            expected_revision=2,
            message="Bỏ",
            pending_action=action,
            unit_clarification_reply={
                "pending_message_id": artifact.message_id,
                "constraints_revision": 0,
                "selections": [
                    {"item_index": 0, "unit": "GB", "start": 0, "end": 2, "quote": "GB"}
                ],
            },
        )
