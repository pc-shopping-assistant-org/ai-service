from copy import deepcopy
from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_flexible_publication import base_snapshot
from test_flexible_reference_continuation import pending_reference
from test_react_tools import context

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    FlexibleDecision,
    ProposeRequirementsAction,
    decode_flexible_decision,
)
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    PendingDraft,
    RequirementDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.user_admission import (
    apply_user_receipt,
    decode_user_requirement_receipt,
)


def legacy_draft(message: str, *, with_task: bool) -> Draft:
    start = message.index("RAM 32GB")
    return Draft(
        items=[
            {
                "subject": "RAM",
                "predicate": "capacity",
                "start": start,
                "end": start + len("RAM 32GB"),
                "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
            }
        ],
        tasks=[
            {
                "kind": "BUILD",
                "start": 0,
                "end": len(message),
                "quote": message,
            }
        ]
        if with_task
        else [],
    )


def test_old_checkpoint_decision_is_validated_then_projected_to_v2():
    message = "Build PC RAM 32GB"
    original = legacy_draft(message, with_task=False)
    legacy_payload = {
        "schema_version": 2,
        "action": {
            "kind": "PROPOSE_REQUIREMENTS",
            "source_text": None,
            "draft": original.model_dump(mode="json"),
        },
    }

    restored = decode_flexible_decision(legacy_payload)

    assert isinstance(restored.action, ProposeRequirementsAction)
    assert restored.action.draft == RequirementDraft(items=original.items)
    assert restored.action.source_text is None
    assert decode_flexible_decision(restored.model_dump(mode="json")) == restored


def test_old_checkpoint_decision_with_task_authority_or_extra_fields_is_rejected():
    message = "Build PC RAM 32GB"
    with_task = legacy_draft(message, with_task=True)
    legacy_payload = {
        "schema_version": 2,
        "action": {
            "kind": "PROPOSE_REQUIREMENTS",
            "draft": with_task.model_dump(mode="json"),
        },
    }
    with pytest.raises(ValidationError):
        decode_flexible_decision(legacy_payload)
    with pytest.raises(ValidationError):
        decode_flexible_decision(legacy_payload | {"untrusted": True})


def test_old_user_receipt_reopens_without_changing_requirement_consent():
    message = "Build PC RAM 32GB"
    state, run_id = CanonicalRequirementState(), uuid4()
    extracted = legacy_draft(message, with_task=True)
    proposed = legacy_draft(message, with_task=False)
    old_receipt = {
        "policy_version": "explicit-draft-match-v1",
        "run_id": str(run_id),
        "message_hash": canonical_hash(message),
        "base_requirements_hash": requirements_hash(state),
        "proposed": {
            "kind": "PROPOSE_REQUIREMENTS",
            "source_text": None,
            "draft": proposed.model_dump(mode="json"),
        },
        "extracted": extracted.model_dump(mode="json"),
    }
    original_items_hash = canonical_hash([item.model_dump(mode="json") for item in extracted.items])

    restored = decode_user_requirement_receipt(old_receipt, message=message)
    merged = apply_user_receipt(
        state, restored, run_id=run_id, message=message, catalog=[]
    )

    assert restored.extracted == RequirementDraft(items=extracted.items)
    assert canonical_hash(
        [item.model_dump(mode="json") for item in restored.extracted.items]
    ) == original_items_hash
    assert next(iter(merged.requirements.values())).value.amount == 32


def test_old_user_receipt_validates_task_evidence_before_dropping_legacy_field():
    message = "Build PC RAM 32GB"
    state, run_id = CanonicalRequirementState(), uuid4()
    draft = legacy_draft(message, with_task=True)
    old_receipt = {
        "policy_version": "explicit-draft-match-v1",
        "run_id": str(run_id),
        "message_hash": canonical_hash(message),
        "base_requirements_hash": requirements_hash(state),
        "proposed": {
            "kind": "PROPOSE_REQUIREMENTS",
            "draft": legacy_draft(message, with_task=False).model_dump(mode="json"),
        },
        "extracted": draft.model_dump(mode="json"),
    }
    old_receipt = deepcopy(old_receipt)
    old_receipt["extracted"]["tasks"][0]["quote"] = "not the source"

    with pytest.raises(ValueError, match="Task evidence"):
        decode_user_requirement_receipt(old_receipt, message=message)


def test_current_decision_schema_still_rejects_combined_requirement_draft():
    with pytest.raises(ValidationError):
        FlexibleDecision(
            action={
                "kind": "PROPOSE_REQUIREMENTS",
                "draft": {
                    "schema_version": 1,
                    "items": [],
                    "tasks": [],
                },
            }
        )


def test_accepted_snapshot_preserves_legacy_pending_draft_and_task_evidence():
    state, owner, conversation, catalog, original, _ = pending_reference()
    message = original.draft.message
    legacy = Draft(
        items=original.draft.resolution.draft.items,
        tasks=[
            {
                "kind": "BUILD",
                "start": 0,
                "end": len(message),
                "quote": message,
            }
        ],
    )
    resolution = resolve_draft(
        legacy,
        message_id=original.draft.message_id,
        message=message,
        state=state,
        catalog=catalog,
    )
    pending = PendingResolution(
        owner=owner,
        conversation_id=conversation,
        target_revision=original.target_revision,
        base_requirements_hash=original.base_requirements_hash,
        draft=PendingDraft(
            message_id=original.draft.message_id,
            message=message,
            constraints_revision=original.draft.constraints_revision,
            resolution=resolution,
        ),
    )
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    snapshot = base_snapshot(ctx)
    snapshot.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Chọn đúng GPU trước khi build.",
        waiting_tasks=[
            TaskOutcome(
                task=CompletionRequirement(kind="BUILD", constraints_revision=0),
                status="UNRESOLVED",
            )
        ],
        pending_resolution=pending,
    )
    original_json = snapshot.model_dump_json()

    restored = AcceptedConversationSnapshot.model_validate_json(original_json)

    assert restored.model_dump_json() == original_json
    restored_draft = restored.flexible.pending_resolution.draft.resolution.draft
    assert isinstance(restored_draft, Draft)
    assert restored_draft.tasks[0].quote == message
