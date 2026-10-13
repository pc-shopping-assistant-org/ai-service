from uuid import uuid4

import pytest
from test_flexible_graph import ReadModel
from test_flexible_publication import base_snapshot
from test_flexible_registration_graph import Extractor, Products
from test_react_graph import Budget
from test_react_tools import context

from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", [None, "owner", "run", "tasks", "text", "request", "accepted"]
)
async def test_publication_is_rederived_from_authorized_base_not_producer_snapshot(
    corruption,
):
    ctx, service = context(), PCBuildApplicationService()
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=0,
        message="GPU?",
    )
    accepted = base_snapshot(ctx)
    products = Products()
    graph = build_flexible_graph(
        ReadModel(),
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=0,
        publication=FlexiblePublicationInput.for_request(accepted, request),
        task_extractor=Extractor(
            [{"kind": "ANSWER", "start": 0, "end": 4, "quote": "GPU?"}]
        ),
        task_catalog=products,
    )
    terminal = await graph.ainvoke(request.execution_input())
    if corruption == "owner":
        terminal["publication_context"]["context"]["account_id"] = str(uuid4())
    if corruption == "run":
        terminal["publication_context"]["context"]["run_id"] = str(uuid4())
    if corruption == "tasks":
        terminal["task_draft"]["tasks"][0]["kind"] = "BUILD"
    if corruption == "text":
        terminal["accepted_snapshot"]["flexible"]["answer"]["text"] = (
            "Unverified number"
        )
    if corruption == "request":
        request.message = "Different stored message"
    if corruption == "accepted":
        accepted.requirement_state.unresolved = ["changed base"]

    def prepare():
        return revalidate_flexible_publication(
            terminal,
            accepted,
            request,
            owner=ctx.account_id,
            run_id=ctx.run_id,
            base_ref=CheckpointRefV1(
                thread_id=ctx.conversation_id, checkpoint_id="base"
            ),
            service=service,
        )

    if corruption:
        with pytest.raises(ValueError):
            prepare()
    else:
        result = prepare()
        assert result.flexible.status == "COMPLETED"
        assert "Canonical GPU" in result.flexible.answer.text
