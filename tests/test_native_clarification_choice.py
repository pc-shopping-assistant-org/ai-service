from uuid import uuid4

import pytest
from test_native_requirement_extractor import Budget, Provider

from ai_service.capabilities.assistant.public_contracts import (
    ClarificationItemV1,
    ClarificationOptionV1,
    PublicClarificationV1,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)


def options(variant_id):
    return PublicClarificationV1(
        pending_message_id=uuid4(),
        constraints_revision=0,
        items=[
            ClarificationItemV1(
                item_index=2,
                reason="AMBIGUOUS_VARIANT",
                options=[
                    ClarificationOptionV1(
                        variant_id=variant_id, sku="TUF-2", name="RTX 5070 Ti"
                    )
                ],
            )
        ],
    )


@pytest.mark.asyncio
async def test_native_choice_is_typed_budgeted_and_source_bound():
    identifier = uuid4()
    message = "Tôi chọn bản thứ hai"
    provider = Provider(
        {
            "selections": [
                {
                    "item_index": 2,
                    "variant_id": str(identifier),
                    "start": 0,
                    "end": len(message),
                    "quote": message,
                }
            ]
        }
    )
    budget = Budget()
    chosen = await NativeRequirementExtractor(provider, budget).choose(
        message, options(identifier)
    )
    assert chosen.selections[0].variant_id == identifier
    assert budget.reservations[0][0] == "MODEL"
    assert "TUF-2" in provider.model.messages[-1][1]


@pytest.mark.asyncio
async def test_choice_budget_denial_cannot_call_provider():
    provider = Provider({"selections": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeRequirementExtractor(provider, Budget(denied=True)).choose(
            "chọn", options(uuid4())
        )
    assert provider.model.messages == []
