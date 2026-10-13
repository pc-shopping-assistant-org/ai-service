from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError
from test_requirement_pipeline import state

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import PublicBuildV1
from ai_service.capabilities.assistant.requirements.contracts import QuantityValue
from ai_service.capabilities.assistant.requirements.pending_action import (
    validate_pending_action,
)
from ai_service.capabilities.assistant.requirements.publication import public_projection
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
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph


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


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["DISCARD", "REPLACE"])
async def test_whole_pending_action_preserves_accepted_state_and_build(action):
    original = state()
    artifact = pending(original)
    build = PublicBuildV1(
        objective="BALANCED",
        parts=[
            {
                "slot": "GPU",
                "variant_id": uuid4(),
                "name": "Accepted GPU",
                "spending_vnd": 8_000_000,
            }
        ],
        total_spending_vnd=8_000_000,
        stale=False,
    )
    consultation, _ = public_projection(original, None, build)
    consultation.build_stale = False
    consultation.published_build_hash = canonical_hash(build)

    class Extractor:
        calls = 0

        async def extract(self, message, current):
            self.calls += 1
            assert message == "RAM 64GB"
            assert current == original
            return Draft(
                items=[
                    DraftItem(
                        subject="RAM",
                        predicate="capacity",
                        operator="GTE",
                        value=QuantityValue(amount=64, unit="GB"),
                        start=0,
                        end=len(message),
                    )
                ]
            )

    class Catalog:
        calls = 0

        async def load(self, draft, current):
            self.calls += 1
            return []

    extractor, catalog = Extractor(), Catalog()
    graph = build_requirement_graph(extractor, catalog, checkpointer=InMemorySaver())
    source = uuid4()
    result = await graph.ainvoke(
        {
            "message": "Bỏ yêu cầu pending" if action == "DISCARD" else "RAM 64GB",
            "message_id": str(source),
            "requirement_state": original.model_dump(mode="json"),
            "consultation": consultation.model_dump(mode="json"),
            "public_build": build.model_dump(mode="json"),
            "pending_clarification": artifact.model_dump(mode="json"),
            "pending_action": PendingActionV1(
                action=action,
                pending_message_id=artifact.message_id,
                constraints_revision=original.revision,
            ).model_dump(mode="json"),
        },
        {"configurable": {"thread_id": str(uuid4())}},
    )
    assert result["phase"] == "READY_TO_PUBLISH"
    assert result["pending_clarification"] is None
    assert result["public_clarification"] is None
    canonical = type(original).model_validate(result["requirement_state"])
    assert all(
        item in canonical.requirements.values()
        for item in original.requirements.values()
    )
    if action == "DISCARD":
        assert result["outcome"] == "PENDING_DISCARDED"
        assert canonical == original
        assert result["public_build"] == build.model_dump(mode="json")
        assert result["consultation"] == consultation.model_dump(mode="json")
        assert extractor.calls == catalog.calls == 0
    else:
        ram = next(
            item for item in canonical.requirements.values() if item.subject == "RAM"
        )
        assert ram.value.amount == 64
        assert ram.evidence.message_id == source
        assert result["public_build"]["stale"] is True
        assert extractor.calls == catalog.calls == 1
