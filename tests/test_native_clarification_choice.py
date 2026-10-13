from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_native_requirement_extractor import Budget, Provider
from test_pending_clarification import Catalog, Extractor

from ai_service.capabilities.assistant.public_contracts import (
    ClarificationItemV1,
    ClarificationOptionV1,
    PublicClarificationV1,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["unknown_id", "forged_span", "empty"])
async def test_graph_choice_revalidates_before_whole_draft_merge(bad):
    catalog = Catalog()
    identifier = uuid4() if bad == "unknown_id" else catalog.rows[1].variant_id
    message = "Chọn bản thứ hai"
    provider = Provider(
        {
            "selections": []
            if bad == "empty"
            else [
                {
                    "item_index": 2,
                    "variant_id": str(identifier),
                    "start": 0,
                    "end": len(message),
                    "quote": "Không có trong câu user"
                    if bad == "forged_span"
                    else message,
                }
            ]
        }
    )
    chooser = NativeRequirementExtractor(provider, Budget())
    graph = build_requirement_graph(
        Extractor(), catalog, clarifier=chooser, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    first = await graph.ainvoke(
        {
            "message": "Build 25 triệu giữ RTX 5070 Ti",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    if bad != "empty":
        with pytest.raises(ValueError, match="clarification"):
            await graph.ainvoke(
                {"message": message, "message_id": str(uuid4())}, config
            )
    else:
        result = await graph.ainvoke(
            {"message": message, "message_id": str(uuid4())}, config
        )
        assert result["outcome"] == "CLARIFICATION"
        assert result["requirement_state"] == first["requirement_state"]


@pytest.mark.asyncio
async def test_free_text_choice_reuses_original_budget_and_pin():
    catalog = Catalog()
    message = "Chọn bản thứ hai"
    chooser = NativeRequirementExtractor(
        Provider(
            {
                "selections": [
                    {
                        "item_index": 2,
                        "variant_id": str(catalog.rows[1].variant_id),
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            }
        ),
        Budget(),
    )
    graph = build_requirement_graph(
        Extractor(), catalog, clarifier=chooser, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    await graph.ainvoke(
        {
            "message": "Build 25 triệu giữ RTX 5070 Ti",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    result = await graph.ainvoke(
        {"message": message, "message_id": str(uuid4())}, config
    )
    state = CanonicalRequirementState.model_validate(result["requirement_state"])
    assert {r.predicate for r in state.requirements.values()} == {"budget", "identity"}
    assert result["pending_clarification"] is None
    assert result["outcome"] == "INFEASIBLE"


@pytest.mark.asyncio
async def test_resume_after_catalog_failure_does_not_repeat_completed_choice_call():
    class FailsOnceCatalog(Catalog):
        calls = 0

        async def load(self, draft, state):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("catalog temporarily unavailable")
            return await super().load(draft, state)

    catalog, budget = FailsOnceCatalog(), Budget()
    message = "Chọn bản thứ hai"
    chooser = NativeRequirementExtractor(
        Provider(
            {
                "selections": [
                    {
                        "item_index": 2,
                        "variant_id": str(catalog.rows[1].variant_id),
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            }
        ),
        budget,
    )
    graph = build_requirement_graph(
        Extractor(), catalog, clarifier=chooser, checkpointer=InMemorySaver()
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    await graph.ainvoke(
        {
            "message": "Build 25 triệu giữ RTX 5070 Ti",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    with pytest.raises(RuntimeError, match="catalog temporarily"):
        await graph.ainvoke({"message": message, "message_id": str(uuid4())}, config)
    assert (await graph.aget_state(config)).next == ("resolve",)
    restored = await graph.ainvoke(None, config)
    assert restored["outcome"] == "INFEASIBLE"
    assert len(budget.reservations) == 1
