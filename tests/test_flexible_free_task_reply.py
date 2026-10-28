from uuid import uuid4

import pytest
from test_flexible_task_selection import Ask, Compare, pending_comparison
from test_react_graph import Budget

from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
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
    ["VALID", "EMPTY", "UNAVAILABLE", "FORGED", "CROSS_RUN", "DROPPED_BINDING"],
)
async def test_free_task_selection_keeps_original_source_and_offered_identities(
    scenario,
):
    ctx, products, extractor, accepted = await pending_comparison()
    ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=1,
        message="Chọn ASUS",
    )
    service = PCBuildApplicationService()

    class Choices:
        calls = 0

        async def choose_task(self, message, options):
            self.calls += 1
            if scenario == "EMPTY":
                return TaskChoiceDraft()
            return TaskChoiceDraft(
                selections=[
                    {
                        "task_index": 0,
                        "subject_index": 0,
                        "product_id": uuid4()
                        if scenario == "FORGED"
                        else products.first.product_id,
                        "evidence": {"start": 0, "end": len(message), "quote": message},
                    }
                ]
            )

    if scenario == "UNAVAILABLE":
        products.matches["RTX 4060"] = [products.alternative]
    model = (
        Ask("SELECTED_PRODUCT_UNAVAILABLE")
        if scenario == "UNAVAILABLE"
        else Ask()
        if scenario == "EMPTY"
        else Compare()
    )
    choices = Choices()
    flow = build_flexible_graph(
        model,
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=1,
        publication=FlexiblePublicationInput.for_request(accepted, request),
        task_catalog=products,
        task_clarifier=choices,
    )
    if scenario == "FORGED":
        with pytest.raises(ValueError, match="candidate"):
            await flow.ainvoke(request.execution_input())
        assert model.calls == 0
        return
    values = await flow.ainvoke(request.execution_input())
    assert choices.calls == 1 and extractor.calls == 1
    if scenario == "CROSS_RUN":
        values["free_task_reply"]["run_id"] = str(uuid4())
    if scenario == "DROPPED_BINDING":
        values["task_resolution"]["bindings"] = [
            item
            for item in values["task_resolution"]["bindings"]
            if item["subject_index"] != 1
        ]

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

    if scenario in {"CROSS_RUN", "DROPPED_BINDING"}:
        with pytest.raises(ValueError):
            validate()
        return
    published = validate()
    assert published.flexible.task_message_id == accepted.flexible.task_message_id
    assert published.flexible.task_message == accepted.flexible.task_message
    assert published.requirement_state == accepted.requirement_state
    assert published.public_build == accepted.public_build
    if scenario in {"EMPTY", "UNAVAILABLE"}:
        assert published.flexible.status == "WAITING_INPUT"
        if scenario == "UNAVAILABLE":
            assert any(
                item.product_id == products.first.product_id
                for item in published.flexible.pending_task_resolution.resolution.bindings
            )
    else:
        assert published.flexible.status == "COMPLETED"
        assert published.flexible.task_outcomes[0].task.product_ids == [
            products.first.product_id,
            products.second.product_id,
        ]
