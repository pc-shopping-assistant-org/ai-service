from uuid import uuid4

import pytest
from test_flexible_contracts import pending_proposal
from test_flexible_graph import ReadModel, graph
from test_flexible_publication import base_snapshot
from test_react_tools import Catalog, context
from test_saved_build_explanation import saved_context

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.publication import (
    prepare_answer_publication,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


async def confirmed_terminal():
    canonical, owner, conversation, pending, confirmation = pending_proposal()
    ctx = context(canonical)
    ctx.account_id, ctx.conversation_id = owner, conversation
    saved = saved_context().saved_build.snapshot
    base = base_snapshot(ctx)
    base.public_build = saved
    base.consultation.published_build_hash = canonical_hash(saved)
    tasks = [CompletionRequirement(kind="ANSWER")]
    terminal = await graph(
        ReadModel(), Catalog(), ctx, revision=13, required=tasks
    ).ainvoke(
        {
            "message": "OK",
            "pending_proposal": pending.model_dump(mode="json"),
            "confirmation": confirmation.model_dump(mode="json"),
        }
    )
    return base, ctx, pending, confirmation, tasks, terminal


@pytest.mark.asyncio
async def test_exact_accepted_confirmation_prepares_changed_state_and_stale_history():
    base, ctx, pending, confirmation, tasks, terminal = await confirmed_terminal()
    before = base.model_dump(mode="json")
    prepared = prepare_answer_publication(
        terminal,
        base,
        ctx,
        tasks,
        PCBuildApplicationService(),
        accepted_proposal=pending,
        confirmation=confirmation,
        conversation_revision=13,
    )
    identifier = pending.patch.set_requirements[0].id
    assert (
        prepared.snapshot.requirement_state.requirements[identifier].value.amount == 16
    )
    assert (
        prepared.snapshot.public_build.stale
        and prepared.snapshot.consultation.build_stale
    )
    assert prepared.snapshot.public_build.total_spending_vnd == 8_000_000
    assert base.model_dump(mode="json") == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["missing", "id", "owner", "revision", "working", "not_consumed"]
)
async def test_changed_state_requires_exact_command_bound_to_accepted_proposal(bad):
    base, ctx, pending, confirmation, tasks, terminal = await confirmed_terminal()
    revision = 13
    if bad == "missing":
        confirmation = None
    if bad == "id":
        confirmation.proposal_id = uuid4()
    if bad == "owner":
        ctx.account_id = uuid4()
    if bad == "revision":
        revision = 12
    if bad == "working":
        terminal["working_requirements"]["unresolved"] = ["unauthorized"]
    if bad == "not_consumed":
        terminal["pending_proposal"] = pending.model_dump(mode="json")
    with pytest.raises(ValueError):
        prepare_answer_publication(
            terminal,
            base,
            ctx,
            tasks,
            PCBuildApplicationService(),
            accepted_proposal=pending,
            confirmation=confirmation,
            conversation_revision=revision,
        )


@pytest.mark.parametrize("owned", [False, True])
def test_partial_binding_projection_does_not_need_a_fake_optimizer_receipt(
    realistic_catalog, owned
):
    from test_requirement_pipeline import rows

    from ai_service.capabilities.assistant.requirements.compiler import (
        compile_requirements,
    )
    from ai_service.capabilities.assistant.requirements.contracts import (
        CanonicalRequirementState,
        Evidence,
        Requirement,
        Resource,
    )
    from ai_service.capabilities.assistant.requirements.publication import (
        public_projection,
    )

    catalog = rows(realistic_catalog)
    row = next(item for item in catalog if item.component.category == "GPU")
    evidence = Evidence(message_id=uuid4(), start=0, end=1, quote="x")
    binding = Requirement(
        id=uuid4(),
        subject="GPU",
        predicate="use_resource" if owned else "identity",
        value={"kind": "REFERENCE", "variant_id": row.variant_id},
        evidence=evidence,
    )
    resource = Resource(id=uuid4(), variant_id=row.variant_id, evidence=evidence)
    canonical = CanonicalRequirementState(
        requirements={binding.id: binding},
        available_resources={resource.id: resource} if owned else {},
    )
    compilation = compile_requirements(canonical, catalog)
    assert not compilation.ready  # Budget/use_case intentionally absent.
    consultation, build = public_projection(
        canonical, None, None, compilation=compilation
    )
    selected = consultation.owned_parts if owned else consultation.pinned_parts
    assert selected["GPU"].variant_id == row.variant_id and build is None
