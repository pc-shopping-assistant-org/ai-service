from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_requirement_pipeline import rows

from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft, DraftItem
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph


class Extractor:
    async def extract(self, message, state):
        return Draft(
            items=[
                DraftItem(
                    subject="SETUP",
                    predicate="budget",
                    operator="LTE",
                    value=QuantityValue(amount=30_000_000, unit="VND"),
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
                    subject="RAM",
                    predicate="capacity",
                    operator="GTE",
                    value=QuantityValue(amount=32, unit="GB"),
                    start=0,
                    end=len(message),
                ),
            ]
        )


class Catalog:
    def __init__(self, catalog):
        self.catalog = catalog

    async def load(self, draft, state):
        return self.catalog


def graph_input():
    return {
        "message": "Build 30 triệu gaming, RAM ít nhất 32GB",
        "message_id": str(uuid4()),
        "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
    }


@pytest.mark.asyncio
async def test_requirement_graph_runs_optimizer_and_only_emits_verified_summary(
    realistic_catalog,
):
    graph = build_requirement_graph(Extractor(), Catalog(rows(realistic_catalog)))
    result = await graph.ainvoke(graph_input())
    assert result["outcome"] == "BUILT"
    assert result["satisfaction"]["errors"] == []
    assert "32" in result["answer"]
    assert "mọi game" not in result["answer"]


@pytest.mark.asyncio
async def test_resume_after_explanation_failure_reuses_optimizer_checkpoint(
    realistic_catalog,
):
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )

    class CountingService(PCBuildApplicationService):
        calls = 0

        def build_pc(self, *args, **kwargs):
            self.calls += 1
            return super().build_pc(*args, **kwargs)

    class FailingOnceExplanation:
        calls = 0

        async def explain(self, receipt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("explanation unavailable")
            return "Đây là diễn giải, không phải nguồn fact."

    service = CountingService()
    explanation = FailingOnceExplanation()
    saver = InMemorySaver()
    graph = build_requirement_graph(
        Extractor(),
        Catalog(rows(realistic_catalog)),
        service=service,
        explainer=explanation,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(RuntimeError, match="explanation unavailable"):
        await graph.ainvoke(graph_input(), config, durability="sync")
    assert service.calls == 1
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("explain",)
    recovered = await graph.ainvoke(None, snapshot.config, durability="sync")
    assert recovered["outcome"] == "BUILT"
    assert service.calls == 1 and explanation.calls == 2
    # Unverified free prose is not promoted into the factual customer summary.
    assert recovered["answer"] != "Đây là diễn giải, không phải nguồn fact."


@pytest.mark.asyncio
async def test_size_bound_rejects_before_external_extraction(realistic_catalog):
    class NeverCalled:
        async def extract(self, message, state):
            raise AssertionError("provider must not receive oversized state")

    graph = build_requirement_graph(
        NeverCalled(), Catalog(rows(realistic_catalog)), max_state_bytes=1
    )
    with pytest.raises(ValueError, match="configured bound"):
        await graph.ainvoke(graph_input())


@pytest.mark.asyncio
async def test_followup_catalog_load_receives_accumulated_requirements(
    realistic_catalog,
):
    from test_requirement_pipeline import requirement, state

    from ai_service.capabilities.assistant.requirements.contracts import ReferenceValue

    canonical = rows(realistic_catalog)
    gpu = next(row for row in canonical if row.component.category == "GPU")
    pin = requirement("GPU", "identity", ReferenceValue(variant_id=gpu.variant_id))
    previous = state(pin)
    received = []

    class NoNewRequirements:
        async def extract(self, message, current):
            return Draft(items=[])

    class FollowupCatalog:
        async def load(self, draft, current):
            received.append(current)
            return canonical

    graph = build_requirement_graph(NoNewRequirements(), FollowupCatalog())
    payload = graph_input() | {"requirement_state": previous.model_dump(mode="json")}
    result = await graph.ainvoke(payload)
    assert received == [previous] and result["outcome"] == "BUILT"
    assert result["receipt"]["build"]["parts"]["GPU"]["id"] == str(gpu.variant_id)
