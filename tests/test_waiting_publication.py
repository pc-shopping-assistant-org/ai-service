from uuid import uuid4

import pytest
from test_flexible_graph import graph
from test_flexible_publication import base_snapshot
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.waiting_publication import (
    prepare_waiting_publication,
)


class Proposer:
    def __init__(self, unresolved=False):
        self.unresolved = unresolved

    async def decide(self, *args):
        return FlexibleDecision(
            action={
                "kind": "PROPOSE_REQUIREMENTS",
                "source_text": "RAM 32" if self.unresolved else "RAM 32GB",
                "draft": {
                    "items": [
                        {
                            "subject": "RAM",
                            "predicate": "capacity",
                            "start": 0,
                            "end": 6 if self.unresolved else 8,
                            "value": {"kind": "UNRESOLVED_QUANTITY", "amount": 32}
                            if self.unresolved
                            else {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
                        }
                    ]
                },
            }
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("unresolved", [False, True])
async def test_new_pending_artifact_preserves_snapshot_and_does_not_complete_task(
    unresolved,
):
    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    base = base_snapshot(ctx)
    terminal = await graph(
        Proposer(unresolved), Catalog(), ctx, required=tasks
    ).ainvoke({"message": "Tư vấn RAM"})
    prepared = prepare_waiting_publication(
        terminal, base, ctx, tasks, source_message="Tư vấn RAM", conversation_revision=0
    )
    assert prepared.status == "WAITING_INPUT"
    assert (
        prepared.snapshot == base and prepared.task_outcomes[0].status == "UNRESOLVED"
    )
    assert (prepared.pending_resolution is not None) == unresolved
    assert (prepared.pending_proposal is not None) != unresolved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["owner", "target", "hash", "request", "working", "patch"]
)
async def test_new_proposal_must_match_rederived_source_payload_and_next_publication(
    bad,
):
    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    terminal = await graph(Proposer(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Tư vấn RAM"}
    )
    source_message = "Tư vấn RAM"
    if bad == "owner":
        ctx.account_id = uuid4()
    if bad == "target":
        terminal["pending_proposal"]["target_revision"] = 2
    if bad == "hash":
        terminal["pending_proposal"]["proposal_hash"] = "a" * 64
    if bad == "request":
        source_message = "Another stored request"
    if bad == "working":
        terminal["working_requirements"]["unresolved"] = ["new"]
    if bad == "patch":
        terminal["pending_proposal"]["patch"]["set_requirements"][0]["value"][
            "amount"
        ] = 16
    if bad in {"patch", "target"}:
        payload = terminal["pending_proposal"]
        payload["proposal_hash"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "proposal_hash"}
        )
    with pytest.raises(ValueError):
        prepare_waiting_publication(
            terminal,
            base_snapshot(ctx),
            ctx,
            tasks,
            source_message=source_message,
            conversation_revision=0,
        )


@pytest.mark.asyncio
async def test_clarification_must_ask_for_actual_missing_prerequisite():
    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách?",
                    "missing_fields": ["BUDGET_REQUIRED"],
                }
            )

    ctx, tasks = (
        context(),
        [CompletionRequirement(kind="BUILD", constraints_revision=0)],
    )
    terminal = await graph(Ask(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Build PC"}
    )
    prepared = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        tasks,
        source_message="Build PC",
        conversation_revision=0,
    )
    assert prepared.clarification.missing_fields == ["BUDGET_REQUIRED"]
    terminal["clarification"]["missing_fields"] = ["MADE_UP"]
    with pytest.raises(ValueError):
        prepare_waiting_publication(
            terminal,
            base_snapshot(ctx),
            ctx,
            tasks,
            source_message="Build PC",
            conversation_revision=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("unit", [False, True])
async def test_waiting_continuation_is_rederived_from_accepted_artifact_and_stored_reply(
    unit,
):
    from test_flexible_reference_continuation import pending_reference
    from test_flexible_unit_continuation import pending_amount, reply

    if unit:
        canonical, owner, conversation, pending = pending_amount()
        catalog, command, message = [], reply(pending), "Dùng TB nhé"
        channel = "unit_reply"
    else:
        canonical, owner, conversation, catalog, pending, command = pending_reference()
        message, channel = "Chọn thứ nhất", "reference_reply"
    ctx = context(canonical, catalog)
    ctx.account_id, ctx.conversation_id = owner, conversation
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]

    class NoModel:
        async def decide(self, *args):
            raise AssertionError("Typed waiting continuation must not call model")

    terminal = await graph(
        NoModel(), Catalog(), ctx, revision=1, required=tasks
    ).ainvoke(
        {
            "message": message,
            "pending_draft": pending.model_dump(mode="json"),
            channel: command.model_dump(mode="json"),
        }
    )
    arguments = {"unit_reply" if unit else "reference_reply": command}
    prepared = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        tasks,
        source_message=message,
        conversation_revision=1,
        accepted_resolution=pending,
        **arguments,
    )
    assert prepared.pending_proposal.target_revision == 2
    assert prepared.snapshot.requirement_state == canonical
    assert prepared.task_outcomes[0].status == "UNRESOLVED"
    with pytest.raises(ValueError):
        prepare_waiting_publication(
            terminal,
            base_snapshot(ctx),
            ctx,
            tasks,
            source_message=message,
            conversation_revision=1,
        )  # No accepted command binding.
