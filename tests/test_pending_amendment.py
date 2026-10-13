from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_native_requirement_extractor import Budget, Provider
from test_quantity_clarification import MESSAGE, draft
from test_requirement_pipeline import rows

from ai_service.capabilities.assistant.requirements.amendment import (
    ItemDiscard,
    ItemReplacement,
    PendingAmendment,
    amend_pending,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.resolver import (
    DraftItem,
    PendingDraft,
    resolve_draft,
)
from ai_service.infrastructure.graph.requirement_build import build_requirement_graph
from ai_service.infrastructure.providers.requirement_extractor import (
    NativeRequirementExtractor,
)


def pending_quantity():
    source = uuid4()
    state = CanonicalRequirementState()
    resolution = resolve_draft(
        draft(), message_id=source, message=MESSAGE, state=state, catalog=[]
    )
    return state, PendingDraft(
        message_id=source,
        message=MESSAGE,
        constraints_revision=0,
        resolution=resolution,
    )


def test_drop_unmerged_item_does_not_clear_original_budget():
    state, pending = pending_quantity()
    followup = uuid4()
    changed = amend_pending(
        pending,
        PendingAmendment(
            discards=[ItemDiscard(item_index=2, start=0, end=6, quote="Bỏ RAM")]
        ),
        state,
        message_id=followup,
        message="Bỏ RAM",
    )
    resolved = resolve_draft(
        changed.resolution.draft,
        message_id=pending.message_id,
        message=pending.message,
        state=state,
        catalog=[],
        item_sources=changed.item_sources,
        discarded_items=changed.discarded_items,
    )
    merged = merge_requirements(state, resolved.patch, changed.source_messages())
    assert len(merged.requirements) == 1
    assert next(iter(merged.requirements.values())).value.amount == 25_000_000
    assert pending.discarded_items == []


def test_amendment_preserves_original_completion_task_source():
    from ai_service.capabilities.assistant.requirements.tasks import TaskDraft

    state, pending = pending_quantity()
    pending.resolution.draft.tasks = [
        TaskDraft(kind="BUILD", start=0, end=5, quote="Build")
    ]
    changed = amend_pending(
        pending, replace_ram(), state, message_id=uuid4(), message="64GB"
    )
    resolved = resolve_draft(
        changed.resolution.draft,
        message_id=changed.message_id,
        message=changed.message,
        state=state,
        catalog=[],
        item_sources=changed.item_sources,
        discarded_items=changed.discarded_items,
    )
    assert resolved.draft.tasks == pending.resolution.draft.tasks
    assert resolved.draft.tasks[0].quote == "Build"


def test_invalid_amendment_bundle_is_atomic_and_does_not_change_target():
    state, pending = pending_quantity()
    original = pending.model_dump(mode="json")
    invalid = replace_ram().model_copy(deep=True)
    invalid.replacements[0].item = DraftItem(
        subject="GPU",
        predicate="capacity",
        value=QuantityValue(amount=64, unit="GB"),
        start=0,
        end=4,
    )
    with pytest.raises(ValueError, match="target identity"):
        amend_pending(pending, invalid, state, message_id=uuid4(), message="64GB")
    assert pending.model_dump(mode="json") == original


@pytest.mark.asyncio
async def test_native_amendment_reserves_model_before_dispatch():
    _, pending = pending_quantity()
    budget = Budget()
    provider = Provider(replace_ram().model_dump(mode="json"))
    amendment = await NativeRequirementExtractor(provider, budget).amend(
        "64GB", pending
    )
    assert amendment.replacements[0].item.value.amount == 64
    assert budget.reservations[0][0] == "MODEL"
    assert MESSAGE in provider.model.messages[-1][1]
    denied = Provider({"replacements": [], "discards": []})
    with pytest.raises(ValueError, match="CALL_BUDGET_EXHAUSTED"):
        await NativeRequirementExtractor(denied, Budget(denied=True)).amend(
            "64GB", pending
        )
    assert denied.model.messages == []


@pytest.mark.asyncio
async def test_sku_mention_amendment_uses_fresh_catalog_and_preserves_budget():
    from test_pending_clarification import Catalog, Extractor

    from ai_service.capabilities.assistant.requirements.contracts import ReferenceValue

    catalog, extractor = Catalog(), Extractor()

    class Amender:
        async def amend(self, message, pending):
            return PendingAmendment(
                replacements=[
                    ItemReplacement(
                        item_index=2,
                        item=DraftItem(
                            subject="GPU",
                            predicate="identity",
                            value=TextValue(text="TUF-1"),
                            start=4,
                            end=9,
                        ),
                    )
                ]
            )

    graph = build_requirement_graph(
        extractor, catalog, checkpointer=InMemorySaver(), amender=Amender()
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    await graph.ainvoke(
        {
            "message": "Build 25 triệu gaming 1080p, giữ RTX 5070 Ti",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    result = await graph.ainvoke(
        {"message": "Giữ TUF-1", "message_id": str(uuid4())}, config
    )
    state = CanonicalRequirementState.model_validate(result["requirement_state"])
    assert result["outcome"] == "INFEASIBLE" and result["pending_clarification"] is None
    gpu = next(item for item in state.requirements.values() if item.subject == "GPU")
    assert gpu.value == ReferenceValue(variant_id=catalog.rows[1].variant_id)
    assert (
        result["consultation"]["constraints"]["target_budget_vnd"]["value"]
        == 25_000_000
    )


def replace_ram():
    return PendingAmendment(
        replacements=[
            ItemReplacement(
                item_index=2,
                item=DraftItem(
                    subject="RAM",
                    predicate="capacity",
                    operator="GTE",
                    value=QuantityValue(amount=64, unit="GB"),
                    start=0,
                    end=4,
                ),
            )
        ]
    )


def test_amend_amount_retains_budget_and_binds_original_and_new_sources():
    original, followup = uuid4(), uuid4()
    state = CanonicalRequirementState()
    initial = resolve_draft(
        draft(), message_id=original, message=MESSAGE, state=state, catalog=[]
    )
    pending = PendingDraft(
        message_id=original, message=MESSAGE, constraints_revision=0, resolution=initial
    )
    modified = amend_pending(
        pending, replace_ram(), state, message_id=followup, message="64GB"
    )
    resolved = resolve_draft(
        modified.resolution.draft,
        message_id=original,
        message=MESSAGE,
        state=state,
        catalog=[],
        item_sources=modified.item_sources,
        discarded_items=modified.discarded_items,
    )
    merged = merge_requirements(
        state, resolved.patch, {original: MESSAGE, followup: "64GB"}
    )
    budget, ram = [
        next(item for item in merged.requirements.values() if item.subject == subject)
        for subject in ["SETUP", "RAM"]
    ]
    assert budget.value.amount == 25_000_000 and budget.evidence.message_id == original
    assert (
        ram.value == QuantityValue(amount=64, unit="GB")
        and ram.evidence.message_id == followup
    )
    assert merged.revision == 1
    with pytest.raises(ValueError, match="evidence"):
        merge_requirements(state, resolved.patch, {original: MESSAGE})


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_graph_amends_without_reextracting_original_and_reuses_checkpoint(
    realistic_catalog, resume
):
    class Extractor:
        calls = 0

        async def extract(self, message, state):
            self.calls += 1
            assert self.calls == 1
            return draft()

    class Amender:
        calls = 0

        async def amend(self, message, pending):
            self.calls += 1
            assert message == "64GB"
            return replace_ram()

    class Catalog:
        calls = 0

        async def load(self, draft, state):
            self.calls += 1
            if resume and self.calls == 2:
                raise RuntimeError("catalog after amendment unavailable")
            return rows(realistic_catalog)

    saver, extractor, amender = InMemorySaver(), Extractor(), Amender()
    config = {"configurable": {"thread_id": str(uuid4())}}
    graph = build_requirement_graph(
        extractor, Catalog(), checkpointer=saver, amender=amender
    )
    first = await graph.ainvoke(
        {
            "message": MESSAGE,
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        },
        config,
    )
    assert first["outcome"] == "CLARIFICATION"
    if resume:
        with pytest.raises(RuntimeError, match="catalog after amendment"):
            await graph.ainvoke(
                {"message": "64GB", "message_id": str(uuid4())},
                config,
                durability="sync",
            )
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("resolve",)
        final = await graph.ainvoke(None, snapshot.config, durability="sync")
    else:
        final = await graph.ainvoke(
            {"message": "64GB", "message_id": str(uuid4())}, config
        )
    assert (
        final["consultation"]["constraints"]["target_budget_vnd"]["value"] == 25_000_000
    )
    canonical = CanonicalRequirementState.model_validate(final["requirement_state"])
    ram = next(
        item for item in canonical.requirements.values() if item.subject == "RAM"
    )
    assert ram.value == QuantityValue(amount=64, unit="GB")
    assert not final["pending_clarification"]
    assert amender.calls == 1
