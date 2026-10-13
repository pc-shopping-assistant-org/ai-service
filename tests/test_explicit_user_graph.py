from uuid import uuid4

import pytest
from test_explicit_user_admission import draft
from test_flexible_contracts import pending_proposal
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.requirements.contracts import requirements_hash
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_confirmation_then_explicit_edit_replays_in_order_without_mutating_head(
    registered,
):
    canonical, owner, conversation, pending, confirmation = pending_proposal()
    ctx, products, service = context(canonical), Catalog(), PCBuildApplicationService()
    ctx.account_id, ctx.conversation_id = owner, conversation
    tasks = [CompletionRequirement(kind="ANSWER")]
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Tư vấn RAM",
        status="WAITING_INPUT",
        waiting_message="Xác nhận RAM 16GB?",
        waiting_tasks=[TaskOutcome(task=tasks[0], status="UNRESOLVED")],
        pending_proposal=pending,
    )
    before = accepted.model_dump(mode="json")
    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=13,
        message="RAM 32GB",
        confirmation=confirmation,
    )
    binding = FlexiblePublicationInput.for_request(accepted, request)
    observed_hashes = []
    edit = draft()
    edit.items[0].target_id = pending.patch.set_requirements[0].id

    class Extractor:
        async def extract_tasks(self, message):
            assert message == accepted.flexible.task_message
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "ANSWER",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

        async def extract(self, message, state):
            assert message == request.message
            ram = next(
                item for item in state.requirements.values() if item.subject == "RAM"
            )
            assert ram.value.amount == 16
            observed_hashes.append(requirements_hash(state))
            return edit

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert accepted.model_dump(mode="json") == before
            observed_hashes.append(requirements_hash(context.requirements))
            ram = next(
                item
                for item in context.requirements.requirements.values()
                if item.subject == "RAM"
            )
            if ram.value.amount == 16:
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "draft": edit.model_dump(),
                    }
                )
            assert ram.value.amount == 32
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "fresh",
                            "name": "search_catalog",
                            "arguments": {"query": "RAM"},
                        },
                    }
                )
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "Untrusted",
                        "evidence_refs": transcript[-1].result.evidence_refs,
                    },
                }
            )

    graph = build_flexible_graph(
        Model(),
        ToolDispatcher(products, service),
        ctx,
        [] if registered else tasks,
        Budget(),
        conversation_revision=13,
        publication=binding,
        user_extractor=Extractor(),
        task_extractor=Extractor() if registered else None,
        task_catalog=products if registered else None,
    )
    values = await graph.ainvoke(request.execution_input())
    published = binding.prepare(values, ctx, tasks, service, 13)
    assert accepted.model_dump(mode="json") == before
    assert len({requirements_hash(canonical), *observed_hashes}) == 3
    ram = next(
        item
        for item in published.requirement_state.requirements.values()
        if item.subject == "RAM"
    )
    assert ram.value.amount == 32 and len(values["user_receipts"]) == 1
    assert published.flexible.status == "COMPLETED"
    with pytest.raises(ValueError):
        binding.prepare(dict(values) | {"user_receipts": []}, ctx, tasks, service, 13)
    if registered:

        def revalidate(candidate):
            return revalidate_flexible_publication(
                candidate,
                accepted,
                request,
                owner=owner,
                run_id=ctx.run_id,
                base_ref=CheckpointRefV1(
                    thread_id=conversation, checkpoint_id="fixture-base"
                ),
                service=service,
            )

        assert revalidate(values) == published
        forged = dict(values) | {"task_resolution": dict(values["task_resolution"])}
        forged["task_resolution"]["requirements"] = [
            dict(values["task_resolution"]["requirements"][0])
            | {"product_ids": [str(uuid4())]}
        ]
        with pytest.raises(
            ValueError, match="Registration cannot replace accepted continuation tasks"
        ):
            revalidate(forged)
        assert accepted.model_dump(mode="json") == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["MATCH", "READ_FIRST", "WAITING", "MISMATCH", "AGENT_TEXT"]
)
async def test_user_admission_updates_working_state_and_rederives_before_publish(mode):
    ctx, products, service = context(), Catalog(), PCBuildApplicationService()
    products.product.name = "RAM 32GB"
    accepted = base_snapshot(ctx)
    before = accepted.model_dump(mode="json")
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=0,
        message="RAM 32GB",
    )
    binding = FlexiblePublicationInput.for_request(accepted, request)

    class Extractor:
        calls = 0

        async def extract_tasks(self, message):
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "BUILD" if mode == "WAITING" else "ANSWER",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

        async def extract(self, message, state):
            self.calls += 1
            assert message == request.message and not state.requirements
            return (
                draft()
                if mode in {"MATCH", "READ_FIRST", "WAITING"}
                else Draft(items=[])
            )

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not context.requirements.requirements:
                if mode == "READ_FIRST" and not transcript:
                    return FlexibleDecision(
                        action={
                            "kind": "CALL_TOOL",
                            "call": {
                                "call_id": "before",
                                "name": "search_catalog",
                                "arguments": {"query": "RAM"},
                            },
                        }
                    )
                if mode == "READ_FIRST":
                    products.product.price_vnd = 9_000_000
                    products.product.evidence_ref = "catalog:ram:new"
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "draft": draft().model_dump(),
                        "source_text": message if mode == "AGENT_TEXT" else None,
                    }
                )
            if mode == "WAITING":
                return FlexibleDecision(
                    action={
                        "kind": "ASK_CLARIFICATION",
                        "question": "Ngân sách?",
                        "missing_fields": ["BUDGET_REQUIRED"],
                    }
                )
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "ram",
                            "name": "search_catalog",
                            "arguments": {"query": "RAM"},
                        },
                    }
                )
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "Untrusted",
                        "evidence_refs": transcript[-1].result.evidence_refs,
                    },
                }
            )

    extractor = Extractor()
    graph = build_flexible_graph(
        Model(),
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=0,
        publication=binding,
        task_extractor=extractor,
        task_catalog=products,
        user_extractor=extractor,
    )
    values = await graph.ainvoke(request.execution_input())
    result = revalidate_flexible_publication(
        values,
        accepted,
        request,
        owner=ctx.account_id,
        run_id=ctx.run_id,
        base_ref=CheckpointRefV1(
            thread_id=ctx.conversation_id, checkpoint_id="fixture-root"
        ),
        service=service,
    )
    assert accepted.model_dump(mode="json") == before
    if mode in {"MATCH", "READ_FIRST", "WAITING"}:
        assert result.flexible.status == (
            "COMPLETED" if mode in {"MATCH", "READ_FIRST"} else "WAITING_INPUT"
        )
        if mode == "WAITING":
            assert (
                result.flexible.pending_proposal is None
                and result.flexible.clarification is not None
            )
        assert len(result.requirement_state.requirements) == 1
        assert len(values["user_receipts"]) == 1 and extractor.calls == 1
        if mode == "READ_FIRST":
            assert len(values["ledger"]) == 2 and products.calls == 2
            assert (
                values["ledger"][0]["result"]["result"]["products"][0]["price_vnd"]
                == 8_000_000
            )
            assert "9,000,000 VND" in result.flexible.assistant_message
            assert "8,000,000 VND" not in result.flexible.assistant_message
            stale = dict(values) | {
                "decision": {
                    "action": {
                        "kind": "FINAL_ANSWER",
                        "answer": {
                            "text": "Old price",
                            "evidence_refs": ["catalog:gpu:1"],
                        },
                    }
                }
            }
            with pytest.raises(ValueError):
                revalidate_flexible_publication(
                    stale,
                    accepted,
                    request,
                    owner=ctx.account_id,
                    run_id=ctx.run_id,
                    base_ref=CheckpointRefV1(
                        thread_id=ctx.conversation_id, checkpoint_id="fixture-root"
                    ),
                    service=service,
                )
        forged = dict(values) | {"user_receipts": []}
        with pytest.raises(ValueError):
            revalidate_flexible_publication(
                forged,
                accepted,
                request,
                owner=ctx.account_id,
                run_id=ctx.run_id,
                base_ref=CheckpointRefV1(
                    thread_id=ctx.conversation_id, checkpoint_id="fixture-root"
                ),
                service=service,
            )
    else:
        assert (
            result.flexible.status == "WAITING_INPUT"
            and result.flexible.pending_proposal is not None
        )
        assert not result.requirement_state.requirements
        assert extractor.calls == (0 if mode == "AGENT_TEXT" else 1)
