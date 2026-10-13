from uuid import uuid4

import pytest

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.public_contracts import PublicBuildV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    Evidence,
    QuantityValue,
    Requirement,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.publication import public_projection


def test_stale_public_build_projection_survives_contract_revalidation():
    previous = PublicBuildV1(
        objective="BALANCED",
        parts=[
            {
                "slot": "GPU",
                "variant_id": uuid4(),
                "name": "Saved GPU",
                "spending_vnd": 8_000_000,
            }
        ],
        total_spending_vnd=8_000_000,
        stale=False,
    )
    consultation, build = public_projection(
        CanonicalRequirementState(revision=2), None, previous
    )
    restored = ConsultationStateV1.model_validate(consultation.model_dump(mode="json"))
    assert restored.build_stale and build.stale
    assert restored.constraints_revision == 2
    accepted = AcceptedConversationSnapshot(consultation=restored, public_build=build)
    data = accepted.model_dump(mode="json")
    data["public_build"]["parts"][0]["spending_vnd"] = 1
    data["public_build"]["total_spending_vnd"] = 1
    with pytest.raises(ValueError, match="snapshot binding"):
        AcceptedConversationSnapshot.model_validate(data)


def test_public_projection_uses_both_goal_and_requirement_constraints():
    evidence = Evidence(message_id=uuid4(), start=0, end=1, quote="x")
    budget = Requirement(
        id=uuid4(),
        subject="SETUP",
        predicate="budget",
        operator="LTE",
        value=QuantityValue(amount=25_000_000, unit="VND"),
        evidence=evidence,
    )
    workload = Requirement(
        id=uuid4(),
        subject="WORKLOAD",
        predicate="use_case",
        value=TextValue(text="GAMING_1080P"),
        evidence=evidence,
    )
    state = CanonicalRequirementState(
        goals={budget.id: budget},
        requirements={workload.id: workload},
    )
    consultation, build = public_projection(state, None, None)
    assert consultation.constraints.target_budget_vnd.value == 25_000_000
    assert consultation.constraints.use_case.value == "GAMING_1080P"
    assert build is None
