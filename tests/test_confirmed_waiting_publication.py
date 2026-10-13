from uuid import uuid4

import pytest
from test_flexible_graph import graph
from test_flexible_publication import base_snapshot
from test_flexible_unit_continuation import pending_amount, reply
from test_react_tools import Catalog, context
from test_saved_build_explanation import saved_context

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.waiting_publication import (
    prepare_waiting_publication,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    continue_unit,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    ProposalConfirmation,
)


async def confirmed_wait():
    canonical, owner, conversation, pending = pending_amount()
    transition = continue_unit(
        canonical,
        pending,
        reply(pending),
        [],
        message_id=uuid4(),
        message="Dùng TB nhé",
        owner=owner,
        conversation_id=conversation,
        conversation_revision=1,
    )
    proposal = transition.admission.pending
    confirmation = ProposalConfirmation(
        proposal_id=proposal.proposal_id,
        proposal_hash=proposal.proposal_hash,
        target_revision=2,
        base_requirements_hash=proposal.base_requirements_hash,
    )
    ctx = context(canonical)
    ctx.account_id, ctx.conversation_id = owner, conversation
    base = base_snapshot(ctx)
    base.public_build = saved_context().saved_build.snapshot
    base.consultation.published_build_hash = canonical_hash(base.public_build)

    class Ask:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách?",
                    "missing_fields": ["BUDGET_REQUIRED"],
                }
            )

    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    terminal = await graph(Ask(), Catalog(), ctx, revision=2, required=tasks).ainvoke(
        {
            "message": "OK",
            "pending_proposal": proposal.model_dump(mode="json"),
            "confirmation": confirmation.model_dump(mode="json"),
        }
    )
    return base, ctx, proposal, confirmation, tasks, terminal


@pytest.mark.asyncio
async def test_confirmed_requirement_persists_while_next_question_keeps_build_unresolved():
    base, ctx, proposal, confirmation, tasks, terminal = await confirmed_wait()
    before = base.model_dump(mode="json")
    prepared = prepare_waiting_publication(
        terminal,
        base,
        ctx,
        tasks,
        source_message="OK",
        conversation_revision=2,
        accepted_proposal=proposal,
        confirmation=confirmation,
    )
    value = next(iter(prepared.snapshot.requirement_state.requirements.values())).value
    assert value.amount == 1000
    assert (
        prepared.snapshot.public_build.stale
        and prepared.snapshot.public_build.total_spending_vnd == 8_000_000
    )
    assert prepared.task_outcomes[0].status == "UNRESOLVED"
    assert prepared.task_outcomes[0].task.constraints_revision == 1
    assert base.model_dump(mode="json") == before


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["missing", "id", "owner", "extra_change"])
async def test_waiting_question_does_not_authorize_unconfirmed_state_change(bad):
    base, ctx, proposal, confirmation, tasks, terminal = await confirmed_wait()
    if bad == "missing":
        confirmation = None
    if bad == "id":
        confirmation.proposal_id = uuid4()
    if bad == "owner":
        ctx.account_id = uuid4()
    if bad == "extra_change":
        terminal["working_requirements"]["unresolved"] = ["unauthorized"]
    with pytest.raises(ValueError):
        prepare_waiting_publication(
            terminal,
            base,
            ctx,
            tasks,
            source_message="OK",
            conversation_revision=2,
            accepted_proposal=proposal,
            confirmation=confirmation,
        )
