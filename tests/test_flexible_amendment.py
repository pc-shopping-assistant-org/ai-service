from uuid import uuid4

import pytest
from test_flexible_publication import base_snapshot
from test_flexible_unit_continuation import pending_amount
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.amendment import PendingAmendment
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitChoiceDraft,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_target", [False, True])
async def test_amendment_is_pending_consent_and_rederived_from_original_source(
    invalid_target,
):
    class NoAgent:
        async def decide(self, *args):
            raise AssertionError("Changed draft must wait for confirmation")

    class Choices:
        async def choose_unit(self, *args):
            return UnitChoiceDraft()

    class Amender:
        async def amend(self, message, pending):
            return PendingAmendment(
                replacements=[
                    {
                        "item_index": 0,
                        "item": {
                            "subject": "RAM" if invalid_target else "STORAGE",
                            "predicate": "capacity",
                            "start": 0,
                            "end": 3,
                            "value": {"kind": "QUANTITY", "amount": 2, "unit": "TB"},
                        },
                    }
                ]
            )

    state, owner, conversation, pending = pending_amount()
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Which unit?",
        waiting_tasks=[TaskOutcome(task=tasks[0], status="UNRESOLVED")],
        pending_resolution=pending,
    )
    binding = FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=accepted.flexible.task_message_id,
        task_message="Build PC",
        request_message="2TB",
    )
    service = PCBuildApplicationService()
    flow = build_flexible_graph(
        NoAgent(),
        ToolDispatcher(Catalog(), service),
        ctx,
        tasks,
        Budget(),
        conversation_revision=1,
        publication=binding,
        unit_clarifier=Choices(),
        amender=Amender(),
    )
    if invalid_target:
        with pytest.raises(ValueError, match="target identity"):
            await flow.ainvoke({"message": "2TB"})
        return
    values = await flow.ainvoke({"message": "2TB"})
    proposal = values["accepted_snapshot"]["flexible"]["pending_proposal"]
    assert proposal["patch"]["set_requirements"][0]["value"]["amount"] == 2000
    assert values["working_requirements"] == state.model_dump(mode="json")
    assert values["accepted_snapshot"]["flexible"]["task_message"] == "Build PC"
    assert "2TB" in proposal["source_messages"].values()
    original_run = values["free_amendment"]["run_id"]
    values["free_amendment"]["run_id"] = str(uuid4())
    with pytest.raises(ValueError, match="receipt source mismatch"):
        binding.prepare(values, ctx, tasks, service, 1)
    values["free_amendment"]["run_id"] = original_run
    values["pending_proposal"]["patch"]["set_requirements"][0]["value"]["amount"] = 999
    with pytest.raises(ValueError):
        binding.prepare(values, ctx, tasks, service, 1)


@pytest.mark.asyncio
async def test_reference_amendment_resolves_catalog_sku_without_merging_existing_state():
    from datetime import UTC, datetime

    from test_requirement_pipeline import state as requirements_state

    from ai_service.application.ports.hardware import ComponentSpec
    from ai_service.capabilities.assistant.provenance import (
        CatalogRowV1,
        canonical_hash,
    )
    from ai_service.capabilities.assistant.react.resolution_reply import (
        FreeAmendmentV2,
        apply_amendment_receipt,
    )
    from ai_service.capabilities.assistant.requirements.amendment import (
        ItemReplacement,
    )
    from ai_service.capabilities.assistant.requirements.contracts import (
        TextValue,
        requirements_hash,
    )
    from ai_service.capabilities.assistant.requirements.pending_resolution import (
        PendingResolution,
    )
    from ai_service.capabilities.assistant.requirements.resolver import (
        DraftItem,
        PendingDraft,
        RequirementDraft,
        resolve_draft,
    )

    state = requirements_state()
    owner, conversation, source_id = uuid4(), uuid4(), uuid4()
    catalog = [
        CatalogRowV1(
            variant_id=(identifier := uuid4()),
            sku=f"GPU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=ComponentSpec(
                id=identifier,
                name=f"RTX 5070 Ti {index}",
                category="GPU",
                price=20_000_000,
            ),
            retrieved_at=datetime.now(UTC),
            evidence_refs=[f"catalog:{identifier}"],
        )
        for index in range(2)
    ]
    source_message = "RTX 5070 Ti"
    initial = resolve_draft(
        RequirementDraft(
            items=[
                DraftItem(
                    subject="GPU",
                    predicate="identity",
                    value=TextValue(text=source_message),
                    start=0,
                    end=len(source_message),
                )
            ]
        ),
        message_id=source_id,
        message=source_message,
        state=state,
        catalog=catalog,
    )
    pending = PendingResolution(
        owner=owner,
        conversation_id=conversation,
        target_revision=1,
        base_requirements_hash=requirements_hash(state),
        draft=PendingDraft(
            message_id=source_id,
            message=source_message,
            constraints_revision=state.revision,
            resolution=initial,
        ),
    )
    ctx = context(state, catalog)
    ctx.account_id, ctx.conversation_id = owner, conversation
    message = "GPU-1"
    amendment = PendingAmendment(
        replacements=[
            ItemReplacement(
                item_index=0,
                item=DraftItem(
                    subject="GPU",
                    predicate="identity",
                    value=TextValue(text=message),
                    start=0,
                    end=len(message),
                ),
            )
        ]
    )
    receipt = FreeAmendmentV2(
        run_id=ctx.run_id,
        pending_message_id=source_id,
        request_message_hash=canonical_hash(message),
        amendment=amendment,
    )

    result = apply_amendment_receipt(
        receipt, ctx, pending, message, conversation_revision=1
    )

    assert result.admission.disposition == "NEEDS_CONFIRMATION"
    assert result.admission.requirements == state
    assert next(iter(state.requirements.values())).predicate == "budget"
    assert (
        result.admission.pending.patch.set_requirements[0].value.variant_id
        == catalog[1].variant_id
    )
