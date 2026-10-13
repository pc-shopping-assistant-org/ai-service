from uuid import uuid4

import pytest
from test_react_tools import context

from ai_service.capabilities.assistant.public_contracts import PublicBuildV1
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    SavedBuildEvidence,
    ToolContext,
)
from ai_service.capabilities.assistant.react.tools import validate_answer
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1


def saved_context(stale=False):
    trusted = context()
    snapshot = PublicBuildV1(
        objective="BALANCED",
        parts=[
            {
                "slot": "GPU",
                "variant_id": uuid4(),
                "name": "Historical GPU",
                "spending_vnd": 8_000_000,
            }
        ],
        total_spending_vnd=8_000_000,
        stale=stale,
    )
    saved = SavedBuildEvidence(
        source_ref=CheckpointRefV1(
            thread_id=trusted.conversation_id, checkpoint_id="accepted"
        ),
        snapshot=snapshot,
    )
    return ToolContext.model_validate(trusted.model_dump() | {"saved_build": saved})


@pytest.mark.parametrize("stale", [False, True])
def test_saved_build_explanation_uses_snapshot_not_model_prices(stale):
    trusted = saved_context(stale)
    ref = trusted.saved_build.evidence_ref
    answer = validate_answer(
        AnswerCandidate(
            text="Historical GPU costs 1 VND and runs every game smoothly",
            evidence_refs=[ref],
        ),
        [],
        trusted,
        [CompletionRequirement(kind="EXPLAIN_SAVED_BUILD", saved_build_ref=ref)],
    )
    assert answer.status == "COMPLETED"
    assert "8,000,000 VND" in answer.text
    assert "Historical GPU" in answer.text
    assert "costs 1 VND" not in answer.text
    assert "every game" not in answer.text
    assert "không phải báo giá hiện tại" in answer.text
    assert ("STALE" in answer.text) == stale


def test_saved_build_cannot_satisfy_new_build_task():
    trusted = saved_context()
    with pytest.raises(ValueError, match="Build completion"):
        validate_answer(
            AnswerCandidate(
                text="Build done", evidence_refs=[trusted.saved_build.evidence_ref]
            ),
            [],
            trusted,
            [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        )


def test_foreign_conversation_snapshot_is_rejected_at_context_boundary():
    trusted = saved_context()
    data = trusted.model_dump(mode="json")
    data["saved_build"]["source_ref"]["thread_id"] = str(uuid4())
    with pytest.raises(ValueError, match="conversation"):
        ToolContext.model_validate(data)


def test_explanation_requires_the_exact_authorized_snapshot_ref():
    trusted = saved_context()
    with pytest.raises(ValueError, match="Saved build completion"):
        validate_answer(
            AnswerCandidate(
                text="Saved setup", evidence_refs=[trusted.saved_build.evidence_ref]
            ),
            [],
            trusted,
            [
                CompletionRequirement(
                    kind="EXPLAIN_SAVED_BUILD",
                    saved_build_ref="saved:another-checkpoint",
                )
            ],
        )
