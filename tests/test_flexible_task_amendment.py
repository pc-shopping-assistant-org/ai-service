from uuid import uuid4

import pytest
from test_flexible_task_selection import Ask, Compare, pending_comparison
from test_react_graph import Budget
from test_task_resolution import product

from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.task_amendment import TaskAmendment
from ai_service.capabilities.assistant.react.task_resolution import TaskChoiceDraft
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "COMPLETE",
        "PARTIAL",
        "INVALID_TARGET",
        "CROSS_RUN",
        "DROP_OTHER",
        "FORGED_QUOTE",
    ],
)
async def test_task_amendment_only_changes_explicit_subject(scenario):
    ctx, products, extractor, accepted = await pending_comparison()
    ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    corrected = product("ASUS 4080")
    products.matches["RTX 4080"] = [corrected]
    if scenario == "PARTIAL":
        products.matches["RTX 4080"].append(product("MSI 4080"))
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=1,
        message="RTX 4080",
    )

    class Choices:
        async def choose_task(self, *args):
            return TaskChoiceDraft()

    class Amender:
        async def amend_task(self, message, pending):
            return TaskAmendment(
                edits=[
                    {
                        "task_index": 0,
                        "subject_index": 9 if scenario == "INVALID_TARGET" else 0,
                        "evidence": {"start": 0, "end": len(message), "quote": message},
                    }
                ]
            )

    service = PCBuildApplicationService()
    model = Ask() if scenario == "PARTIAL" else Compare()
    flow = build_flexible_graph(
        model,
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=1,
        publication=FlexiblePublicationInput.for_request(accepted, request),
        task_catalog=products,
        task_clarifier=Choices(),
        task_amender=Amender(),
    )
    if scenario == "INVALID_TARGET":
        with pytest.raises(ValueError, match="target"):
            await flow.ainvoke(request.execution_input())
        return
    values = await flow.ainvoke(request.execution_input())
    if scenario == "CROSS_RUN":
        values["free_task_amendment"]["run_id"] = str(uuid4())
    if scenario == "DROP_OTHER":
        values["task_resolution"]["bindings"] = [
            item
            for item in values["task_resolution"]["bindings"]
            if item["subject_index"] != 1
        ]
    if scenario == "FORGED_QUOTE":
        values["free_task_amendment"]["amendment"]["edits"][0]["evidence"]["quote"] = (
            "RTX 4090"
        )

    def validate():
        return revalidate_flexible_publication(
            values,
            accepted,
            request,
            owner=ctx.account_id,
            run_id=ctx.run_id,
            base_ref=CheckpointRefV1(
                thread_id=ctx.conversation_id, checkpoint_id="fixture-base"
            ),
            service=service,
        )

    if scenario in {"CROSS_RUN", "DROP_OTHER", "FORGED_QUOTE"}:
        with pytest.raises(ValueError):
            validate()
        return
    published = validate()
    assert extractor.calls == 1
    assert published.requirement_state == accepted.requirement_state
    assert published.flexible.task_message == accepted.flexible.task_message
    if scenario == "PARTIAL":
        pending = published.flexible.pending_task_resolution
        assert pending.mention_overrides[0].evidence.quote == "RTX 4080"
        assert any(
            item.product_id == products.second.product_id
            for item in pending.resolution.bindings
        )
    else:
        assert published.flexible.task_outcomes[0].task.product_ids == [
            corrected.product_id,
            products.second.product_id,
        ]
