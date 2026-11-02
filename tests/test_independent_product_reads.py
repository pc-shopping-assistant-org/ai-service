from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from test_flexible_publication import base_snapshot
from test_flexible_reference_continuation import pending_reference
from test_flexible_unit_continuation import pending_amount
from test_native_react_model import Provider
from test_react_graph import Budget
from test_react_tools import context
from test_saved_build_explanation import saved_context

from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ProductEvidence,
)
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.public_views import (
    public_resolution_options,
)
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.amendment import PendingAmendment
from ai_service.capabilities.assistant.requirements.resolver import ChoiceDraft
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.providers.react_model import NativeFlexibleModel


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize(
    "waiting,attached", [(False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("selection", ["calls", "explanation_refs", "sole_result"])
@pytest.mark.parametrize("pending_kind", ["UNIT", "REFERENCE"])
async def test_product_read_completes_without_build_dependencies(
    count, waiting, attached, selection, pending_kind
):
    from datetime import UTC, datetime

    original, owner, conversation, pending = pending_amount()
    hardware = []
    if pending_kind == "REFERENCE":
        original, owner, conversation, hardware, pending, _ = pending_reference()
    ctx = context(original)
    ctx.catalog_ready = False
    ctx.account_id, ctx.conversation_id = owner, conversation
    historical = saved_context().saved_build
    ctx.saved_build = historical.model_copy(
        update={
            "source_ref": CheckpointRefV1(
                thread_id=conversation, checkpoint_id="accepted"
            )
        }
    )
    accepted = base_snapshot(ctx)
    if waiting:
        accepted.flexible = FlexiblePublishedState(
            conversation_id=conversation,
            task_message_id=pending.draft.message_id,
            task_message=pending.draft.message,
            status="WAITING_INPUT",
            waiting_message="Which unit?",
            pending_resolution=pending,
            resolution_options=public_resolution_options(pending.draft, hardware),
            waiting_tasks=[
                TaskOutcome(
                    task=CompletionRequirement(kind="BUILD", constraints_revision=0),
                    status="UNRESOLVED",
                )
            ],
        )
    products = [
        ProductEvidence(
            product_id=uuid4(),
            name=f"Product {i}",
            price_vnd=1000000 * (i + 1),
            evidence_ref=f"catalog:read:{i}",
            retrieved_at=datetime.now(UTC),
        )
        for i in range(count)
    ]

    class Catalog:
        async def get(self, identifier):
            return next((p for p in products if p.product_id == identifier), None)

    class NoOptimizer(PCBuildApplicationService):
        def build_pc(self, *args, **kwargs):
            raise AssertionError("Product reads cannot invoke optimizer")

    class NoHardware:
        async def load(self, *args):
            raise AssertionError("Product reads cannot scan hardware")

    class SideQuestion:
        async def choose(self, message, options):
            return ChoiceDraft()

        async def amend(self, message, pending):
            return PendingAmendment()

    provider = Provider(None)
    calls = 0

    async def respond(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "read",
                        "name": "get_product_details"
                        if count == 1
                        else "compare_products",
                        "args": {"product_id": str(products[0].product_id)}
                        if count == 1
                        else {"product_ids": [str(p.product_id) for p in products]},
                    }
                ],
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "answer",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "text": "Untrusted price 1 VND",
                                **(
                                    {}
                                    if selection == "sole_result"
                                    else (
                                        {"calls": ["read"]}
                                        if selection == "calls"
                                        else {
                                            "explanation_refs": [
                                                p.evidence_ref for p in products
                                            ]
                                        }
                                    )
                                ),
                                "explanation": "Nhận xét dựa trên các sản phẩm đã chọn.",
                                "task_index": 0 if attached else None,
                            }
                        ]
                    },
                }
            ],
        )

    provider.model.ainvoke = respond
    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=1,
        message="Explain product" if count == 1 else "Compare products",
        selected_product_ids=[p.product_id for p in products] if attached else [],
    )
    binding = FlexiblePublicationInput.for_request(accepted, request, owner=owner)
    assert binding.task_message == (
        request.message if attached else accepted.flexible.task_message
    )
    task = CompletionRequirement(
        kind="ANSWER" if count == 1 else "COMPARE",
        product_ids=[p.product_id for p in products],
    )
    graph = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(Catalog(), NoOptimizer()),
        ctx,
        [task] if attached else [row.task for row in accepted.flexible.task_outcomes],
        Budget(),
        conversation_revision=1,
        publication=binding,
        catalog_loader=NoHardware(),
        clarifier=SideQuestion(),
        amender=SideQuestion(),
    )
    terminal = await graph.ainvoke({"message": request.message})
    assert terminal["phase"] == "READY_TO_PUBLISH"
    if attached:
        assert terminal["answer"]["status"] == "COMPLETED"
    answer = terminal["answer"] if attached else terminal["response"]
    assert all(p.name in answer["text"] for p in products)
    assert "Untrusted price" not in answer["text"]
    snapshot = terminal["accepted_snapshot"]
    assert snapshot["public_build"] == accepted.public_build.model_dump(mode="json")
    assert snapshot["requirement_state"] == accepted.requirement_state.model_dump(
        mode="json"
    )
    if waiting:
        assert snapshot["flexible"][
            "pending_resolution"
        ] == accepted.flexible.pending_resolution.model_dump(mode="json")
        assert snapshot["flexible"]["task_message_id"] == str(
            accepted.flexible.task_message_id
        )
    assert calls == 2
    assert (
        revalidate_flexible_publication(
            terminal,
            accepted,
            request,
            owner=owner,
            run_id=ctx.run_id,
            base_ref=CheckpointRefV1(thread_id=conversation, checkpoint_id="accepted"),
            service=PCBuildApplicationService(),
        ).model_dump(mode="json")
        == snapshot
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [True, False])
async def test_pending_reply_loads_hardware_only_for_sku_resolution(reference):
    original, owner, conversation, hardware, pending, reply = pending_reference()
    if not reference:
        original, owner, conversation, pending = pending_amount()
        hardware = []
    ctx = context(original)
    ctx.catalog_ready = False
    ctx.account_id, ctx.conversation_id = owner, conversation
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=pending.draft.message_id,
        task_message=pending.draft.message,
        status="WAITING_INPUT",
        waiting_message="Choose a GPU",
        pending_resolution=pending,
        resolution_options=public_resolution_options(pending.draft, hardware),
        waiting_tasks=[
            TaskOutcome(
                task=CompletionRequirement(kind="BUILD", constraints_revision=0),
                status="UNRESOLVED",
            )
        ],
    )
    chosen = False
    loads = 0

    class Choices:
        async def choose(self, message, options):
            nonlocal chosen
            chosen = True
            if not reference:
                return ChoiceDraft()
            return ChoiceDraft(
                selections=[
                    {
                        "item_index": 0,
                        "variant_id": reply.selections[0].variant_id,
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

        async def amend(self, message, pending):
            return PendingAmendment(
                replacements=[
                    {
                        "item_index": 0,
                        "item": {
                            "subject": "STORAGE",
                            "predicate": "capacity",
                            "start": 0,
                            "end": len(message),
                            "value": {"kind": "QUANTITY", "amount": 1024, "unit": "GB"},
                        },
                    }
                ]
            )

    class Hardware:
        async def load(self, *args):
            nonlocal loads
            assert chosen
            assert reference, "Scalar amendments do not need hardware"
            loads += 1
            return hardware

    class NoModel:
        async def decide(self, *args):
            raise AssertionError(
                "A resolved choice needs consent, not another agent call"
            )

    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=1,
        message="Choose first GPU" if reference else "1024GB",
    )
    graph = build_flexible_graph(
        NoModel(),
        ToolDispatcher(None, PCBuildApplicationService()),
        ctx,
        [row.task for row in accepted.flexible.task_outcomes],
        Budget(),
        conversation_revision=1,
        publication=FlexiblePublicationInput.for_request(
            accepted, request, owner=owner
        ),
        catalog_loader=Hardware(),
        clarifier=Choices(),
        amender=Choices(),
    )
    terminal = await graph.ainvoke({"message": request.message})
    assert terminal["phase"] == "WAITING_INPUT"
    assert loads == (1 if reference else 0)
    snapshot = revalidate_flexible_publication(
        terminal,
        accepted,
        request,
        owner=owner,
        run_id=ctx.run_id,
        base_ref=CheckpointRefV1(thread_id=conversation, checkpoint_id="accepted"),
        service=PCBuildApplicationService(),
    )
    assert snapshot.requirement_state == original
    assert snapshot.flexible.pending_proposal is not None
    assert snapshot.flexible.pending_resolution is None
