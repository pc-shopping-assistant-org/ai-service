from uuid import uuid4

import pytest
from pydantic import ValidationError
from test_requirement_pipeline import state

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.search_evidence import BuildSearchEvidence
from ai_service.capabilities.assistant.requirements.contracts import (
    Evidence,
    QuantityValue,
    Requirement,
    RequirementPatch,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
    confirm_proposal,
    issue_proposal,
)


def test_content_hash_ignores_legacy_counter_but_detects_changed_constraints():
    original = state()
    changed_counter = original.model_copy(update={"revision": 99})
    assert requirements_hash(original) == requirements_hash(changed_counter)
    changed_content = original.model_copy(deep=True)
    next(iter(changed_content.requirements.values())).value.amount = 40_000_000
    assert requirements_hash(original) != requirements_hash(changed_content)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "CALL_TOOL",
            "call": {
                "call_id": "s1",
                "name": "search_web",
                "arguments": {"query": "GPU"},
            },
        },
        {
            "kind": "PROPOSE_REQUIREMENTS",
            "draft": {
                "items": [
                    {
                        "subject": "RAM",
                        "predicate": "capacity",
                        "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
                        "start": 0,
                        "end": 8,
                    }
                ]
            },
        },
        {
            "kind": "ASK_CLARIFICATION",
            "question": "Ngân sách bao nhiêu?",
            "missing_fields": ["budget"],
        },
        {
            "kind": "FINAL_ANSWER",
            "answer": {"text": "Typed candidate"},
            "outcome": "INSUFFICIENT_EVIDENCE",
        },
    ],
)
def test_four_actions_roundtrip_without_authority_fields(payload):
    decision = FlexibleDecision(action=payload)
    assert FlexibleDecision.model_validate_json(decision.model_dump_json()) == decision
    with pytest.raises(ValidationError):
        FlexibleDecision(action=payload | {"account_id": str(uuid4())})
    with pytest.raises(ValidationError):
        FlexibleDecision(action=payload, answer={"text": "Second action"})


def pending_proposal():
    original, owner, conversation = state(), uuid4(), uuid4()
    source = uuid4()
    message = "RAM 16GB"
    requirement = Requirement(
        id=uuid4(),
        subject="RAM",
        predicate="capacity",
        value=QuantityValue(amount=16, unit="GB"),
        evidence=Evidence(message_id=source, start=0, end=len(message), quote=message),
    )
    proposal = issue_proposal(
        original,
        RequirementPatch(set_requirements=[requirement]),
        {source: message},
        owner=owner,
        conversation_id=conversation,
        target_revision=13,
    )
    confirmation = ProposalConfirmation(
        proposal_id=proposal.proposal_id,
        proposal_hash=proposal.proposal_hash,
        target_revision=13,
        base_requirements_hash=requirements_hash(original),
    )
    return original, owner, conversation, proposal, confirmation


def test_confirmation_binds_immutable_payload_and_does_not_merge_state():
    original, owner, conversation, proposal, confirmation = pending_proposal()
    before = original.model_dump(mode="json")
    restored = PendingRequirementProposal.model_validate_json(
        proposal.model_dump_json()
    )
    patch = confirm_proposal(
        restored,
        confirmation,
        original,
        owner=owner,
        conversation_id=conversation,
        conversation_revision=14,
    )
    assert patch == proposal.patch
    assert original.model_dump(mode="json") == before
    altered = proposal.model_dump()
    altered["patch"]["set_requirements"][0]["value"]["amount"] = 64
    with pytest.raises(ValidationError, match="hash"):
        PendingRequirementProposal.model_validate(altered)


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposal_id", uuid4()),
        ("proposal_hash", "0" * 64),
        ("target_revision", 12),
        ("base_requirements_hash", "0" * 64),
    ],
)
def test_wrong_confirmation_binding_rejected(field, value):
    original, owner, conversation, proposal, confirmation = pending_proposal()
    invalid = confirmation.model_copy(update={field: value})
    with pytest.raises(ValueError):
        confirm_proposal(
            proposal,
            invalid,
            original,
            owner=owner,
            conversation_id=conversation,
            conversation_revision=13,
        )


def test_confirmation_rejects_foreign_owner_and_changed_base_state():
    original, owner, conversation, proposal, confirmation = pending_proposal()
    with pytest.raises(ValueError, match="owner"):
        confirm_proposal(
            proposal,
            confirmation,
            original,
            owner=uuid4(),
            conversation_id=conversation,
            conversation_revision=13,
        )
    changed = original.model_copy(deep=True)
    next(iter(changed.requirements.values())).value.amount = 1
    with pytest.raises(ValueError, match="state"):
        confirm_proposal(
            proposal,
            confirmation,
            changed,
            owner=owner,
            conversation_id=conversation,
            conversation_revision=13,
        )


def search_evidence():
    return {
        "outcome": "INFEASIBLE",
        "scope": "canonical candidate snapshot only",
        "requirements_hash": requirements_hash(state()),
        "optimizer_input_hash": canonical_hash({"input": 1}),
        "catalog_snapshot_hash": canonical_hash([]),
        "algorithm_version": "exact-v1",
        "pruning_version": "lossless-v1",
        "policy_version": "policy-v1",
        "search_complete": True,
        "coverage_complete": True,
        "pruning_sound": True,
        "termination": "EXHAUSTED",
    }


@pytest.mark.parametrize(
    "field", ["search_complete", "coverage_complete", "pruning_sound"]
)
def test_incomplete_search_cannot_be_relabelled_proven_infeasible(field):
    payload = search_evidence() | {field: False}
    with pytest.raises(ValidationError, match="proof"):
        BuildSearchEvidence.model_validate(payload)
    incomplete = BuildSearchEvidence.model_validate(
        payload | {"outcome": "SEARCH_INCOMPLETE"}
    )
    assert incomplete.outcome == "SEARCH_INCOMPLETE"


def test_scoped_exhaustion_roundtrip_and_no_feasible_shortcut():
    proof = BuildSearchEvidence.model_validate(search_evidence())
    assert BuildSearchEvidence.model_validate_json(proof.model_dump_json()) == proof
    with pytest.raises(ValidationError, match="proof"):
        BuildSearchEvidence.model_validate(
            search_evidence() | {"termination": "TIME_LIMIT"}
        )
    with pytest.raises(ValidationError):
        BuildSearchEvidence.model_validate(
            search_evidence() | {"outcome": "NO_FEASIBLE_BUILD"}
        )


def test_heuristic_found_build_is_feasible_not_incomplete():
    found = search_evidence() | {
        "outcome": "FEASIBLE",
        "build_ref": uuid4(),
        "termination": "FOUND_RESULT",
        "search_complete": False,
        "coverage_complete": False,
        "pruning_sound": False,
    }
    assert BuildSearchEvidence.model_validate(found).outcome == "FEASIBLE"
    with pytest.raises(ValidationError):
        BuildSearchEvidence.model_validate(
            found | {"outcome": "SEARCH_INCOMPLETE", "build_ref": None}
        )


def test_action_cannot_change_task_and_versions_are_explicit():
    with pytest.raises(ValidationError):
        FlexibleDecision(
            schema_version=2.0,
            action={
                "kind": "ASK_CLARIFICATION",
                "question": "Budget?",
                "missing_fields": ["budget"],
            },
        )
    with pytest.raises(ValidationError, match="tasks"):
        FlexibleDecision(
            action={
                "kind": "PROPOSE_REQUIREMENTS",
                "draft": {
                    "items": [
                        {
                            "subject": "RAM",
                            "predicate": "capacity",
                            "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
                            "start": 0,
                            "end": 8,
                        }
                    ],
                    "tasks": [
                        {"kind": "ANSWER", "start": 0, "end": 8, "quote": "RAM 32GB"}
                    ],
                },
            }
        )
    _, _, _, proposal, confirmation = pending_proposal()
    with pytest.raises(ValidationError):
        PendingRequirementProposal.model_validate(
            proposal.model_dump() | {"schema_version": 1}
        )
    with pytest.raises(ValidationError):
        ProposalConfirmation.model_validate(
            confirmation.model_dump() | {"schema_version": 2.0}
        )
