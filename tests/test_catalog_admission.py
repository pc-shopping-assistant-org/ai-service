"""Requirement admission can progress before optimizer candidates are needed."""

from copy import deepcopy
from uuid import uuid4

import pytest
from agent_task_proposals import WithTaskProposals
from langgraph.checkpoint.memory import InMemorySaver
from test_flexible_publication import base_snapshot
from test_flexible_reference_continuation import pending_reference
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_requirement_pipeline import requirement, rows, state
from test_setup_preferences import accessory
from test_waiting_publication import Proposer

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.actions import (
    AskClarificationAction,
    FlexibleDecision,
)
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolName,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    QuantityValue,
    ReferenceValue,
    TextValue,
)
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


class OfflineHardware:
    async def load(self, draft, requirements):
        raise BackendUnavailableError("Hardware backend unavailable")


def unloaded_context():
    return context().model_copy(update={"catalog_ready": False}, deep=True)


def binding(ctx, message):
    return FlexiblePublicationInput(
        accepted=base_snapshot(ctx),
        task_message_id=uuid4(),
        task_message=message,
        request_message=message,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_missing_build_goals_clarify_without_hardware_dependency(registered):
    ctx, products, message = unloaded_context(), Catalog(), "Build PC"
    source = binding(ctx, message)

    class Ask:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert ToolName.BUILD_PC not in allowed
            return FlexibleDecision(action=AskClarificationAction())

    class Tasks:
        async def extract_tasks(self, message):
            return TaskDraftBatch(
                tasks=[
                    {
                        "kind": "BUILD",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ]
            )

    graph = build_flexible_graph(
        WithTaskProposals(Ask(), Tasks()) if registered else Ask(),
        ToolDispatcher(products, PCBuildApplicationService()),
        ctx,
        []
        if registered
        else [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=0,
        publication=source,
        task_catalog=products if registered else None,
        catalog_loader=OfflineHardware(),
    )
    result = await graph.ainvoke({"message": message})
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == "WAITING_INPUT"
    assert snapshot.flexible.task_outcomes[0].task.kind == "BUILD"
    assert snapshot.flexible.clarification.missing_fields == [
        "BUDGET_REQUIRED",
        "USE_CASE_REQUIRED",
    ]
    assert snapshot.requirement_state == source.accepted.requirement_state
    assert result["catalog_context"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("unresolved", [False, True])
async def test_scalar_requirement_can_wait_for_confirmation_or_unit_without_catalog(
    unresolved,
):
    ctx, message = unloaded_context(), "Tư vấn RAM"
    source = binding(ctx, message)
    graph = build_flexible_graph(
        Proposer(unresolved),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=0,
        publication=source,
        catalog_loader=OfflineHardware(),
    )
    result = await graph.ainvoke({"message": message})
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == "WAITING_INPUT"
    assert bool(snapshot.flexible.pending_resolution) == unresolved
    assert bool(snapshot.flexible.pending_proposal) != unresolved
    assert snapshot.requirement_state == source.accepted.requirement_state
    assert result["catalog_context"] is None


@pytest.mark.asyncio
async def test_unloaded_catalog_does_not_prove_missing_component_metadata():
    ram = requirement("RAM", "capacity", QuantityValue(amount=32, unit="GB"))
    ctx = unloaded_context().model_copy(
        update={
            "requirements": CanonicalRequirementState(requirements={ram.id: ram}),
        }
    )
    message, model_calls = "Build PC RAM 32GB", []

    class Ask:
        async def decide(self, message, transcript, allowed, repair, context=None):
            model_calls.append(True)
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Cần thêm thông tin",
                    "missing_fields": ["METADATA_INSUFFICIENT"],
                }
            )

    source, service = binding(ctx, message), PCBuildApplicationService()
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    graph = build_flexible_graph(
        Ask(),
        ToolDispatcher(Catalog(), service),
        ctx,
        tasks,
        Budget(),
        conversation_revision=0,
        publication=source,
        catalog_loader=OfflineHardware(),
    )
    result = await graph.ainvoke({"message": message})
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.clarification.missing_fields == [
        "BUDGET_REQUIRED",
        "USE_CASE_REQUIRED",
    ]
    assert len(model_calls) == 1
    assert "ngân sách" in snapshot.flexible.assistant_message.lower()
    assert "BUDGET_REQUIRED" not in snapshot.flexible.assistant_message
    forged = deepcopy(result)
    forged["decision"]["action"]["missing_fields"] = ["METADATA_INSUFFICIENT"]
    forged["clarification"]["missing_fields"] = ["METADATA_INSUFFICIENT"]
    with pytest.raises(ValueError, match="actual missing prerequisites"):
        source.prepare(forged, ctx, tasks, service, 0)


@pytest.mark.asyncio
async def test_budget_reply_publishes_only_remaining_question_in_one_model_call():
    ctx, products = unloaded_context(), Catalog()
    service, tasks = (
        PCBuildApplicationService(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    source = binding(ctx, "Build PC")

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(action=AskClarificationAction())

    async def run(model, trusted, publication, revision):
        graph = build_flexible_graph(
            model,
            ToolDispatcher(products, service),
            trusted,
            tasks,
            Budget(),
            conversation_revision=revision,
            publication=publication,
            catalog_loader=OfflineHardware(),
        )
        result = await graph.ainvoke({"message": publication.request_message})
        assert result["model_calls"] == 1
        return AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])

    initial = await run(Ask(), ctx, source, 0)
    assert initial.flexible.status == "WAITING_INPUT"
    assert initial.flexible.clarification.missing_fields == [
        "BUDGET_REQUIRED",
        "USE_CASE_REQUIRED",
    ]
    reply = "30 triệu"

    class BudgetReply:
        async def decide(self, message, *args):
            return FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "draft": {
                        "items": [
                            {
                                "target": "GOAL",
                                "subject": "SETUP",
                                "predicate": "budget",
                                "operator": "LTE",
                                "start": 0,
                                "end": len(message),
                                "value": {
                                    "kind": "QUANTITY",
                                    "amount": 30_000_000,
                                    "unit": "VND",
                                },
                            }
                        ]
                    },
                }
            )

    continuation = source.model_copy(
        update={"accepted": initial, "request_message": reply}
    )
    next_ctx = ctx.model_copy(
        update={"run_id": uuid4(), "requirements": initial.requirement_state}
    )
    updated = await run(BudgetReply(), next_ctx, continuation, 1)
    assert updated.flexible.status == "WAITING_INPUT"
    assert updated.flexible.clarification.missing_fields == ["USE_CASE_REQUIRED"]
    assert "ngân sách" not in updated.flexible.assistant_message.lower()
    assert (
        next(iter(updated.requirement_state.goals.values())).value.amount == 30_000_000
    )
    assert initial.requirement_state == source.accepted.requirement_state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "PIN_UNAVAILABLE",
        "ACCESSORY_BUDGET_EXCEEDED",
        "ACCESSORY_UNAVAILABLE_OR_METADATA_MISSING",
        "METADATA_INSUFFICIENT",
    ],
)
async def test_catalog_blockers_publish_canonical_questions(realistic_catalog, reason):
    catalog = rows(realistic_catalog)
    if reason == "PIN_UNAVAILABLE":
        gpu = next(row for row in catalog if row.component.category == "GPU")
        canonical = state(
            requirement("GPU", "identity", ReferenceValue(variant_id=gpu.variant_id))
        )
        gpu.quantity = 0
        expected = f"{gpu.component.name} hiện hết hàng"
    elif reason in {
        "ACCESSORY_BUDGET_EXCEEDED",
        "ACCESSORY_UNAVAILABLE_OR_METADATA_MISSING",
    }:
        canonical = state(
            requirement("SETUP", "scope", TextValue(text="FULL_SETUP")),
            requirement(
                "SETUP",
                "accessory_budget",
                QuantityValue(amount=1_500_000, unit="VND"),
                operator="LTE",
            ),
            requirement("MONITOR", "selection", TextValue(text="RECOMMEND")),
        )
        if reason == "ACCESSORY_BUDGET_EXCEEDED":
            catalog.append(accessory(catalog, kind="MONITOR", price=2_500_000))
            expected = "vượt mức chi đã đặt"
        else:
            expected = "chưa xác nhận được phụ kiện"
    else:
        canonical = state(
            requirement(
                "STORAGE",
                "capacity",
                QuantityValue(amount=1_000, unit="GB"),
                operator="GTE",
            )
        )
        catalog = [row for row in catalog if row.component.category != "STORAGE"]
        expected = "dung lượng ổ lưu trữ"

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(action=AskClarificationAction())

    ctx = context().model_copy(update={"requirements": canonical, "catalog": catalog})
    source = binding(ctx, "Tiếp tục build PC")
    graph = build_flexible_graph(
        Ask(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=0,
        publication=source,
    )
    terminal = await graph.ainvoke({"message": source.request_message})
    published = AcceptedConversationSnapshot.model_validate(
        terminal["accepted_snapshot"]
    )
    assert published.flexible.status == "WAITING_INPUT"
    assert published.flexible.clarification.missing_fields == [reason]
    assert expected in published.flexible.assistant_message
    assert published.requirement_state == canonical


@pytest.mark.asyncio
@pytest.mark.parametrize("resume", [False, True])
async def test_admitted_build_goals_load_candidates_after_merge_and_publish(
    realistic_catalog,
    resume,
):
    ctx, message = unloaded_context(), "Build 30 triệu gaming"
    source = binding(ctx, message)
    before = source.accepted.model_dump(mode="json")
    draft = RequirementDraft(
        items=[
            {
                "target": "GOAL",
                "subject": "SETUP",
                "predicate": "budget",
                "operator": "LTE",
                "start": message.index("30"),
                "end": message.index(" gaming"),
                "value": {"kind": "QUANTITY", "amount": 30_000_000, "unit": "VND"},
            },
            {
                "target": "GOAL",
                "subject": "WORKLOAD",
                "predicate": "use_case",
                "start": message.index("gaming"),
                "end": len(message),
                "value": {"kind": "TEXT", "text": "GAMING_1080P"},
            },
        ]
    )
    loaded = []

    class Hardware:
        async def load(self, draft, requirements):
            assert requirements.goals, (
                "Candidates requested before user goals were admitted"
            )
            loaded.append(requirements.model_dump(mode="json"))
            if resume and len(loaded) == 1:
                raise TimeoutError("Catalog snapshot temporarily unavailable")
            return rows(realistic_catalog)

    class Extractor:
        calls = 0

        async def extract(self, message, requirements):
            self.calls += 1
            return draft

    class Builder:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not context.requirements.goals:
                return FlexibleDecision(
                    action={"kind": "PROPOSE_REQUIREMENTS", "draft": draft.model_dump()}
                )
            assert ToolName.BUILD_PC not in allowed
            assert transcript
            outcome = transcript[-1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "BUILT",
                    "answer": {
                        "text": "Cấu hình theo yêu cầu",
                        "build_ref": outcome.result.build_ref,
                        "evidence_refs": outcome.evidence_refs,
                    },
                }
            )

    extractor = Extractor()
    graph = build_flexible_graph(
        Builder(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=0,
        publication=source,
        catalog_loader=Hardware(),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}
    if resume:
        with pytest.raises(TimeoutError):
            await graph.ainvoke({"message": message}, config)
        checkpoint = await graph.aget_state(config)
        assert checkpoint.next == ("prepare_agent",)
        assert checkpoint.values["user_receipts"]
        result = await graph.ainvoke(None, checkpoint.config)
    else:
        result = await graph.ainvoke({"message": message}, config)
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert snapshot.flexible.status == "COMPLETED" and snapshot.public_build is not None
    assert len(loaded) == (2 if resume else 1)
    assert all(item == result["working_requirements"] for item in loaded)
    assert extractor.calls == 0
    assert source.accepted.model_dump(mode="json") == before


@pytest.mark.asyncio
async def test_part_reference_still_loads_catalog_before_asking_for_variant():
    _, _, _, catalog, pending, _ = pending_reference()
    ctx, message = unloaded_context(), pending.draft.message
    source, loaded = binding(ctx, message), []

    class Hardware:
        async def load(self, draft, requirements):
            loaded.append(True)
            return catalog

    class Pin:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": message,
                    "draft": RequirementDraft(
                        items=pending.draft.resolution.draft.items
                    ).model_dump(),
                }
            )

    graph = build_flexible_graph(
        Pin(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        ctx,
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
        Budget(),
        conversation_revision=0,
        publication=source,
        catalog_loader=Hardware(),
    )
    result = await graph.ainvoke({"message": message})
    snapshot = AcceptedConversationSnapshot.model_validate(result["accepted_snapshot"])
    assert loaded == [True]
    assert snapshot.flexible.pending_resolution.draft.resolution.pending[
        0
    ].candidate_ids == [row.variant_id for row in catalog]
