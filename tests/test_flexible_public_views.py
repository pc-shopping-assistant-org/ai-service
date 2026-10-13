from uuid import uuid4

import pytest
from test_flexible_task_selection import pending_comparison
from test_flexible_unit_continuation import pending_amount

from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.public_views import to_public_flexible
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)


@pytest.mark.asyncio
async def test_task_options_are_public_projection_without_internal_catalog_evidence():
    ctx, products, _, accepted = await pending_comparison()
    public = to_public_flexible(accepted.flexible, accepted.requirement_state)
    assert public.status == "WAITING_INPUT"
    assert public.task_clarification.items[0].mention == "RTX 4060"
    assert (
        public.task_clarification.items[0].options[0].product_id
        == products.first.product_id
    )
    raw = public.model_dump_json()
    assert str(ctx.account_id) not in raw
    assert "retrieved_at" not in raw
    assert "checkpoint" not in raw
    assert "evidence_ref" not in raw


def test_unit_resolution_projection_retains_original_amount_and_offered_source():
    canonical, owner, conversation, pending = pending_amount()
    state = FlexiblePublishedState(
        conversation_id=conversation,
        task_message_id=uuid4(),
        task_message="Build PC",
        status="WAITING_INPUT",
        waiting_message="Which unit?",
        waiting_tasks=[
            {
                "task": CompletionRequirement(
                    kind="BUILD", constraints_revision=canonical.revision
                ),
                "status": "UNRESOLVED",
            }
        ],
        pending_resolution=pending,
    )
    public = to_public_flexible(state, canonical)
    assert public.clarification.pending_message_id == pending.draft.message_id
    assert public.clarification.items[0].quantity_amount == 1
    assert "TB" in public.clarification.items[0].unit_options
    raw = public.model_dump_json()
    assert str(owner) not in raw
    assert "source_messages" not in raw
    assert "working_requirements" not in raw


@pytest.mark.asyncio
async def test_proposal_exposes_immutable_binding_and_readable_changes_not_source_evidence():
    from test_flexible_graph import graph
    from test_flexible_publication import base_snapshot
    from test_react_tools import Catalog, context
    from test_waiting_publication import Proposer

    from ai_service.capabilities.assistant.react.snapshot_publication import (
        accepted_snapshot_for,
    )
    from ai_service.capabilities.assistant.react.waiting_publication import (
        prepare_waiting_publication,
    )

    ctx, task = context(), CompletionRequirement(kind="BUILD", constraints_revision=0)
    terminal = await graph(Proposer(), Catalog(), ctx, required=[task]).ainvoke(
        {"message": "Build PC"}
    )
    prepared = prepare_waiting_publication(
        terminal,
        base_snapshot(ctx),
        ctx,
        [task],
        source_message="Build PC",
        conversation_revision=0,
    )
    snapshot = accepted_snapshot_for(
        prepared,
        conversation_id=ctx.conversation_id,
        task_message_id=uuid4(),
        task_message="Build PC",
    )
    public = to_public_flexible(snapshot.flexible, snapshot.requirement_state)
    assert public.proposal.proposal_hash == prepared.pending_proposal.proposal_hash
    assert public.proposal.target_revision == 1
    assert public.proposal.changes[0].subject == "RAM"
    assert public.proposal.changes[0].value.amount == 32
    raw = public.model_dump_json()
    assert str(ctx.account_id) not in raw
    assert "source_messages" not in raw
    assert "supporting_evidence" not in raw
