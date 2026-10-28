from uuid import uuid4

import pytest
from agent_task_proposals import WithTaskProposals
from test_flexible_contracts import pending_proposal
from test_flexible_graph import ReadModel
from test_flexible_publication import base_snapshot
from test_flexible_registration_graph import Extractor
from test_flexible_task_selection import pending_comparison
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
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.parametrize("action", ["REPLACE", "DISCARD"])
@pytest.mark.parametrize("bad", [None, "base", "unpublished"])
def test_requirement_pending_action_binds_content_and_published_revision(action, bad):
    state, owner, conversation, pending = pending_amount()
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=pending.draft.message_id,
        task_message=pending.draft.message,
        status="WAITING_INPUT",
        waiting_message="Which unit?",
        waiting_tasks=[
            TaskOutcome(
                task=CompletionRequirement(kind="BUILD", constraints_revision=0),
                status="UNRESOLVED",
            )
        ],
        pending_resolution=pending,
    )
    if bad == "base":
        accepted.flexible.pending_resolution.base_requirements_hash = "0" * 64
    if bad == "unpublished":
        accepted.flexible.pending_resolution.target_revision = 2
    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=1,
        message="Find GPU",
        pending_action={
            "target": "REQUIREMENTS",
            "action": action,
            "pending_message_id": pending.draft.message_id,
            "constraints_revision": state.revision,
        },
    )
    before = accepted.model_dump(mode="json")
    if bad:
        with pytest.raises(ValueError, match="content|publication"):
            FlexiblePublicationInput.for_request(accepted, request, owner=owner)
    else:
        bound = FlexiblePublicationInput.for_request(accepted, request, owner=owner)
        assert bound.discard_pending == (action == "DISCARD")
        assert bound.accepted.requirement_state == state
    assert accepted.model_dump(mode="json") == before


@pytest.mark.asyncio
async def test_explicit_task_replacement_keeps_canonical_snapshot_and_uses_new_source():
    ctx, products, _, accepted = await pending_comparison()
    before = accepted.model_dump(mode="json")
    pending = accepted.flexible.pending_task_resolution
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=1,
        message="Find GPU",
        pending_action={
            "target": "TASKS",
            "action": "REPLACE",
            "pending_message_id": pending.message_id,
            "constraints_revision": pending.constraints_revision,
        },
    )
    bound = FlexiblePublicationInput.for_request(
        accepted, request, owner=ctx.account_id
    )
    assert bound.accepted.flexible is None
    assert bound.accepted.requirement_state == accepted.requirement_state
    assert bound.accepted.public_build == accepted.public_build
    assert (
        bound.task_message == "Find GPU" and bound.task_message_id == request.request_id
    )
    assert accepted.model_dump(mode="json") == before
    with pytest.raises(ValueError):
        FlexiblePublicationInput.for_request(accepted, request)
    ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    service = PCBuildApplicationService()
    extractor = Extractor(
        [{"kind": "ANSWER", "start": 0, "end": 8, "quote": "Find GPU"}]
    )
    flow = build_flexible_graph(
        WithTaskProposals(ReadModel(), extractor),
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=1,
        publication=bound,
        task_catalog=products,
    )
    values = await flow.ainvoke(request.execution_input())
    published = revalidate_flexible_publication(
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
    assert published.flexible.status == "COMPLETED"
    assert published.flexible.task_message_id == request.request_id
    assert published.flexible.task_outcomes[0].task.kind == "ANSWER"
    assert published.requirement_state == accepted.requirement_state


@pytest.mark.asyncio
async def test_requirement_pending_discard_is_application_only_and_preserves_snapshot():
    state, owner, conversation, pending = pending_amount()
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Which unit?",
        waiting_tasks=[
            TaskOutcome(
                task=CompletionRequirement(kind="BUILD", constraints_revision=0),
                status="UNRESOLVED",
            )
        ],
        pending_resolution=pending,
    )
    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=1,
        message="Bỏ yêu cầu RAM đang chờ",
        pending_action={
            "target": "REQUIREMENTS",
            "action": "DISCARD",
            "pending_message_id": pending.draft.message_id,
            "constraints_revision": state.revision,
        },
    )
    bound = FlexiblePublicationInput.for_request(accepted, request, owner=owner)

    class NoModel:
        async def decide(self, *args):
            raise AssertionError("Discard is an application command")

    service = PCBuildApplicationService()
    flow = build_flexible_graph(
        NoModel(),
        ToolDispatcher(Catalog(), service),
        ctx,
        [],
        Budget(),
        conversation_revision=1,
        publication=bound,
    )
    values = await flow.ainvoke(request.execution_input())
    published = revalidate_flexible_publication(
        values,
        accepted,
        request,
        owner=owner,
        run_id=ctx.run_id,
        base_ref=CheckpointRefV1(
            thread_id=conversation, checkpoint_id="fixture-base"
        ),
        service=service,
    )

    assert published.flexible.status == "DISCARDED"
    assert published.flexible.pending_resolution is None
    assert published.requirement_state == state
    assert published.public_build == accepted.public_build
    assert values["model_calls"] == 0 and values["tool_calls"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["TASKS", "CLARIFICATION"])
async def test_bound_discard_acknowledges_without_model_or_tool_or_task_completion(
    target,
):
    ctx, products, _, accepted = await pending_comparison()
    ctx = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    pending = accepted.flexible.pending_task_resolution
    if target == "CLARIFICATION":
        accepted.flexible = FlexiblePublishedState(
            conversation_id=ctx.conversation_id,
            task_message_id=pending.message_id,
            task_message="Build PC",
            status="WAITING_INPUT",
            waiting_message="Budget?",
            waiting_tasks=[
                TaskOutcome(
                    task=CompletionRequirement(
                        kind="BUILD", constraints_revision=ctx.requirements.revision
                    ),
                    status="UNRESOLVED",
                )
            ],
            clarification={
                "kind": "ASK_CLARIFICATION",
                "question": "Budget?",
                "missing_fields": ["budget_vnd"],
            },
        )
    request = TurnRequestV2(
        conversation_id=ctx.conversation_id,
        request_id=uuid4(),
        expected_revision=1,
        message="Bỏ yêu cầu đang chờ",
        pending_action={
            "target": target,
            "action": "DISCARD",
            "pending_message_id": pending.message_id,
            "constraints_revision": pending.constraints_revision,
        },
    )
    bound = FlexiblePublicationInput.for_request(
        accepted, request, owner=ctx.account_id
    )

    class NoModel:
        async def decide(self, *args):
            raise AssertionError("Discard is an application command")

    service = PCBuildApplicationService()
    flow = build_flexible_graph(
        NoModel(),
        ToolDispatcher(products, service),
        ctx,
        [],
        Budget(),
        conversation_revision=1,
        publication=bound,
    )
    values = await flow.ainvoke(request.execution_input())
    published = revalidate_flexible_publication(
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
    assert published.flexible.status == "DISCARDED"
    assert published.flexible.task_message_id == accepted.flexible.task_message_id
    assert not any(
        task.status == "COMPLETED" for task in published.flexible.task_outcomes
    )
    assert published.flexible.pending_task_resolution is None
    assert published.requirement_state == accepted.requirement_state
    assert published.public_build == accepted.public_build
    assert values["model_calls"] == 0 and values["tool_calls"] == 0


@pytest.mark.parametrize("bad", [None, "owner", "id", "hash", "revision", "base"])
@pytest.mark.parametrize("action", ["REPLACE", "DISCARD"])
def test_proposal_replacement_is_bound_and_does_not_merge_old_patch(bad, action):
    state, owner, conversation, proposal, confirmation = pending_proposal()
    ctx = context(state)
    ctx.account_id, ctx.conversation_id = owner, conversation
    accepted = base_snapshot(ctx)
    accepted.flexible = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Confirm",
        waiting_tasks=[
            TaskOutcome(
                task=CompletionRequirement(
                    kind="BUILD", constraints_revision=state.revision
                ),
                status="UNRESOLVED",
            )
        ],
        pending_proposal=proposal,
    )
    payload = confirmation.model_dump() | {"target": "PROPOSAL", "action": action}
    if bad == "id":
        payload["proposal_id"] = uuid4()
    if bad == "hash":
        payload["proposal_hash"] = "0" * 64
    if bad == "revision":
        payload["target_revision"] -= 1
    if bad == "base":
        payload["base_requirements_hash"] = "0" * 64
    request = TurnRequestV2(
        conversation_id=conversation,
        request_id=uuid4(),
        expected_revision=13,
        message="Find GPU",
        pending_action=payload,
    )
    if bad:
        with pytest.raises(ValueError):
            FlexiblePublicationInput.for_request(
                accepted, request, owner=uuid4() if bad == "owner" else owner
            )
        return
    bound = FlexiblePublicationInput.for_request(accepted, request, owner=owner)
    if action == "REPLACE":
        assert bound.accepted.flexible is None
    else:
        assert (
            bound.discard_pending
            and bound.accepted.flexible.pending_proposal == proposal
        )
    assert bound.accepted.requirement_state == state
    assert accepted.flexible.pending_proposal == proposal


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["source", "revision", "target", "mixed"])
async def test_task_replacement_rejects_unbound_or_mixed_commands(bad):
    ctx, _, _, accepted = await pending_comparison()
    pending = accepted.flexible.pending_task_resolution
    payload = {
        "target": "REQUIREMENTS" if bad == "target" else "TASKS",
        "action": "REPLACE",
        "pending_message_id": uuid4() if bad == "source" else pending.message_id,
        "constraints_revision": pending.constraints_revision + 1
        if bad == "revision"
        else pending.constraints_revision,
    }
    with pytest.raises(ValueError):
        request = TurnRequestV2(
            conversation_id=ctx.conversation_id,
            request_id=uuid4(),
            expected_revision=1,
            message="Find GPU",
            pending_action=payload,
            task_reply={
                "pending_message_id": pending.message_id,
                "constraints_revision": pending.constraints_revision,
                "selections": [
                    {
                        "task_index": 0,
                        "subject_index": 0,
                        "product_id": pending.resolution.pending[0]
                        .candidates[0]
                        .product_id,
                    }
                ],
            }
            if bad == "mixed"
            else None,
        )
        FlexiblePublicationInput.for_request(accepted, request, owner=ctx.account_id)
