from uuid import uuid4

import pytest
from test_flexible_publication import base_snapshot
from test_flexible_registration_graph import Extractor
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_task_resolution import compare_task, product

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


class Products(Catalog):
    def __init__(self):
        super().__init__()
        self.first, self.alternative, self.second = (
            product("ASUS 4060"),
            product("MSI 4060"),
            product("MSI 4070"),
        )
        self.matches = {
            "RTX 4060": [self.first, self.alternative],
            "RTX 4070": [self.second],
        }

    async def resolve(self, mention):
        return self.matches.get(mention, [])

    async def get(self, identifier):
        return next(
            (
                p
                for rows in self.matches.values()
                for p in rows
                if p.product_id == identifier
            ),
            None,
        )


class Ask:
    def __init__(self, reason="AMBIGUOUS"):
        self.calls, self.reason = 0, reason

    async def decide(self, *args):
        self.calls += 1
        return FlexibleDecision(
            action={
                "kind": "ASK_CLARIFICATION",
                "question": "Which product?",
                "missing_fields": [self.reason],
            }
        )


class Compare:
    calls = 0

    async def decide(self, message, transcript, allowed, repair, context=None):
        self.calls += 1
        if not transcript:
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": "compare",
                        "name": "compare_products",
                        "arguments": {
                            "product_ids": [
                                str(p)
                                for p in context.completion_requirements[0].product_ids
                            ],
                        },
                    },
                }
            )
        return FlexibleDecision(
            action={
                "kind": "FINAL_ANSWER",
                "answer": {
                    "text": "Ignored prose",
                    "evidence_refs": transcript[-1].result.evidence_refs,
                },
            }
        )


def compile_flow(model, products, ctx, binding, extractor, revision):
    return build_flexible_graph(
        model,
        ToolDispatcher(products, PCBuildApplicationService()),
        ctx,
        [],
        Budget(),
        conversation_revision=revision,
        publication=binding,
        task_extractor=extractor,
        task_catalog=products,
    )


async def pending_comparison():
    ctx, products, task = context(), Products(), compare_task()
    extractor = Extractor([task])
    binding = FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=uuid4(),
        task_message=task.quote,
        request_message=task.quote,
    )
    terminal = await compile_flow(Ask(), products, ctx, binding, extractor, 0).ainvoke(
        {"message": task.quote}
    )
    accepted = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    return ctx, products, extractor, accepted


def selected_binding(accepted, products):
    return FlexiblePublicationInput(
        accepted=accepted,
        task_message_id=accepted.flexible.task_message_id,
        task_message=accepted.flexible.task_message,
        request_message="Chọn ASUS",
        task_reply={
            "pending_message_id": accepted.flexible.task_message_id,
            "constraints_revision": accepted.requirement_state.revision,
            "selections": [
                {
                    "task_index": 0,
                    "subject_index": 0,
                    "product_id": products.first.product_id,
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_offered_choice_completes_original_comparison_without_reextracting():
    ctx, products, extractor, accepted = await pending_comparison()
    ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    terminal = await compile_flow(
        Compare(), products, ctx, selected_binding(accepted, products), extractor, 1
    ).ainvoke({"message": "Chọn ASUS"})
    result = AcceptedConversationSnapshot.model_validate(terminal["accepted_snapshot"])
    assert result.flexible.status == "COMPLETED"
    assert result.flexible.task_message == "So RTX 4060 và RTX 4070"
    assert result.flexible.task_message_id == accepted.flexible.task_message_id
    assert result.flexible.task_outcomes[0].task.product_ids == [
        products.first.product_id,
        products.second.product_id,
    ]
    assert result.flexible.pending_task_resolution is None
    assert extractor.calls == 1
    assert accepted.flexible.status == "WAITING_INPUT"


@pytest.mark.asyncio
async def test_stored_v2_request_binds_original_task_source_not_choice_message():
    from ai_service.capabilities.assistant.react.turn_contracts import (
        TurnRequestV2,
        parse_turn_request,
        turn_request_payload,
    )

    ctx, products, _, accepted = await pending_comparison()
    selected = selected_binding(accepted, products)
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=1,
        message=selected.request_message,
        task_reply=selected.task_reply,
    )
    stored = parse_turn_request(turn_request_payload(request))
    bound = FlexiblePublicationInput.for_request(accepted, stored)
    assert bound == selected
    assert bound.task_message_id != request.request_id
    assert bound.task_message == "So RTX 4060 và RTX 4070"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["source", "revision", "candidate", "subject"])
async def test_stale_or_unoffered_choice_rejects_before_agent(bad):
    ctx, products, extractor, accepted = await pending_comparison()
    binding = selected_binding(accepted, products)
    if bad == "source":
        binding.task_reply.pending_message_id = uuid4()
    if bad == "revision":
        binding.task_reply.constraints_revision += 1
    if bad == "candidate":
        binding.task_reply.selections[0].product_id = products.second.product_id
    if bad == "subject":
        binding.task_reply.selections[0].subject_index = 1
    model = Compare()
    with pytest.raises(ValueError):
        await compile_flow(model, products, ctx, binding, extractor, 1).ainvoke(
            {"message": "Chọn ASUS"}
        )
    assert model.calls == 0


@pytest.mark.asyncio
async def test_selected_unavailable_product_never_switches_to_remaining_match():
    ctx, products, extractor, accepted = await pending_comparison()
    binding = selected_binding(accepted, products)
    products.matches["RTX 4060"] = [products.alternative]
    terminal = await compile_flow(
        Ask("SELECTED_PRODUCT_UNAVAILABLE"),
        products,
        ctx.model_copy(update={"run_id": uuid4()}),
        binding,
        extractor,
        1,
    ).ainvoke({"message": "Chọn ASUS"})
    result = AcceptedConversationSnapshot.model_validate(terminal["accepted_snapshot"])
    assert result.flexible.status == "WAITING_INPUT"
    pending = result.flexible.pending_task_resolution
    assert pending.resolution.pending[0].reason == "SELECTED_PRODUCT_UNAVAILABLE"
    assert pending.resolution.bindings[0].product_id == products.first.product_id
    assert not result.flexible.task_outcomes
    assert extractor.calls == 1
