from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph


class Extractor:
    calls = 0

    async def extract(self, message, state):
        self.calls += 1
        assert self.calls == 1, "Pending selection must not re-extract/drop the draft"
        return Draft(
            items=[
                DraftItem(
                    subject="SETUP",
                    predicate="budget",
                    operator="LTE",
                    value=QuantityValue(amount=25_000_000, unit="VND"),
                    start=0,
                    end=len(message),
                ),
                DraftItem(
                    target="GOAL",
                    subject="WORKLOAD",
                    predicate="use_case",
                    value=TextValue(text="GAMING_1080P"),
                    start=0,
                    end=len(message),
                ),
                DraftItem(
                    subject="GPU",
                    predicate="identity",
                    value=TextValue(text="RTX 5070 Ti"),
                    start=0,
                    end=len(message),
                ),
            ]
        )


class Catalog:
    def __init__(self):
        self.rows = [
            CatalogRowV1(
                variant_id=(identifier := uuid4()),
                sku=f"TUF-{index}",
                product_status="ACTIVE",
                variant_status="ACTIVE",
                quantity=1,
                component=ComponentSpec(
                    id=identifier, name="RTX 5070 Ti", category="GPU", price=10_000_000
                ),
                retrieved_at=datetime.now(UTC),
                evidence_refs=[f"test:{identifier}"],
            )
            for index in range(2)
        ]

    async def load(self, draft, state):
        return self.rows


@pytest.mark.asyncio
async def test_pending_sku_choice_survives_reopen_and_merges_entire_original_draft():
    saver, catalog, extractor = InMemorySaver(), Catalog(), Extractor()
    graph = build_requirement_graph(extractor, catalog, checkpointer=saver)
    source_id = uuid4()
    config = {"configurable": {"thread_id": str(uuid4())}}
    first = await graph.ainvoke(
        {
            "message": "Build 25 triệu gaming 1080p, giữ RTX 5070 Ti",
            "message_id": str(source_id),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    assert first["outcome"] == "CLARIFICATION"
    assert first["requirement_state"]["requirements"] == {}
    assert "TUF-1" in first["answer"]
    reopened = build_requirement_graph(extractor, catalog, checkpointer=saver)
    second = await reopened.ainvoke(
        {
            "message": "Tôi chọn bản thứ hai",
            "message_id": str(uuid4()),
            "clarification_reply": {
                "pending_message_id": str(source_id),
                "selections": [
                    {"item_index": 2, "variant_id": str(catalog.rows[1].variant_id)}
                ],
            },
        },
        config,
    )
    state = CanonicalRequirementState.model_validate(second["requirement_state"])
    assert {r.predicate for r in state.requirements.values()} == {"budget", "identity"}
    pin = next(r for r in state.requirements.values() if r.predicate == "identity")
    assert pin.value.variant_id == catalog.rows[1].variant_id
    assert second["outcome"] == "INFEASIBLE"  # GPU-only catalog, not lost requirements.
    assert second["pending_clarification"] is None
    assert extractor.calls == 1


@pytest.mark.asyncio
async def test_pending_choice_rejects_stale_source_or_non_candidate_without_merge():
    for stale in (True, False):
        catalog, extractor = Catalog(), Extractor()
        graph = build_requirement_graph(
            extractor, catalog, checkpointer=InMemorySaver()
        )
        source_id = uuid4()
        config = {"configurable": {"thread_id": str(uuid4())}}
        await graph.ainvoke(
            {
                "message": "Build 25 triệu giữ RTX 5070 Ti",
                "message_id": str(source_id),
                "requirement_state": CanonicalRequirementState().model_dump(
                    mode="json"
                ),
            },
            config,
        )
        with pytest.raises(ValueError, match="clarification"):
            await graph.ainvoke(
                {
                    "message": "Chọn",
                    "message_id": str(uuid4()),
                    "clarification_reply": {
                        "pending_message_id": str(uuid4() if stale else source_id),
                        "selections": [
                            {
                                "item_index": 2,
                                "variant_id": str(
                                    catalog.rows[0].variant_id if stale else uuid4()
                                ),
                            }
                        ],
                    },
                },
                config,
            )


@pytest.mark.asyncio
async def test_missing_selection_preserves_pending_budget_and_candidates():
    catalog, extractor = Catalog(), Extractor()
    graph = build_requirement_graph(extractor, catalog, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}
    first = await graph.ainvoke(
        {
            "message": "Build 25 triệu giữ RTX 5070 Ti",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    second = await graph.ainvoke(
        {"message": "bản thứ hai", "message_id": str(uuid4())}, config
    )
    assert second["outcome"] == "CLARIFICATION"
    assert second["pending_clarification"] == first["pending_clarification"]
    assert "TUF-1" in second["answer"]
    assert extractor.calls == 1
