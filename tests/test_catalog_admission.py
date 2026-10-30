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
from test_requirement_pipeline import requirement, rows
from test_waiting_publication import Proposer

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
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
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách và nhu cầu sử dụng của bạn là gì?",
                    "missing_fields": ["BUDGET_REQUIRED", "USE_CASE_REQUIRED"],
                }
            )

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
            if len(model_calls) == 1:
                missing = ["METADATA_INSUFFICIENT"]
            else:
                assert set(repair.missing_fields) == {
                    "BUDGET_REQUIRED",
                    "USE_CASE_REQUIRED",
                }
                missing = ["BUDGET_REQUIRED", "USE_CASE_REQUIRED"]
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Cần thêm thông tin",
                    "missing_fields": missing,
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
    assert len(model_calls) == 2
    forged = deepcopy(result)
    forged["decision"]["action"]["missing_fields"] = ["METADATA_INSUFFICIENT"]
    forged["clarification"]["missing_fields"] = ["METADATA_INSUFFICIENT"]
    with pytest.raises(ValueError, match="actual missing prerequisites"):
        source.prepare(forged, ctx, tasks, service, 0)


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
            if not transcript:
                assert ToolName.BUILD_PC in allowed
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "build",
                            "name": "build_pc",
                            "arguments": {},
                        },
                    }
                )
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
