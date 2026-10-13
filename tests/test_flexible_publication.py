import pytest
from test_flexible_graph import ReadModel, graph
from test_react_tools import Catalog, context
from test_saved_build_explanation import saved_context

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.publication import (
    prepare_answer_publication,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


def base_snapshot(ctx):
    build = ctx.saved_build.snapshot if ctx.saved_build else None
    return AcceptedConversationSnapshot(
        consultation=ConsultationStateV1(
            published_build_hash=canonical_hash(build) if build else None,
            build_stale=build.stale if build else False,
        ),
        public_build=build,
        requirement_state=ctx.requirements,
    )


@pytest.mark.asyncio
async def test_prepare_read_answer_preserves_accepted_snapshot_and_history():
    ctx = saved_context()
    base = base_snapshot(ctx)
    before = base.model_dump(mode="json")
    tasks = [CompletionRequirement(kind="ANSWER")]
    state = await graph(ReadModel(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "GPU"}
    )
    prepared = prepare_answer_publication(
        state, base, ctx, tasks, PCBuildApplicationService()
    )
    assert prepared.snapshot.model_dump(mode="json") == before
    assert "Canonical GPU" in prepared.answer.text
    assert prepared.snapshot.public_build.total_spending_vnd == 8_000_000
    assert base.model_dump(mode="json") == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    ["text", "completion", "evidence", "working_state", "pending", "phase"],
)
async def test_preparation_rejects_inconsistent_terminal_state(corruption):
    ctx, tasks = context(), [CompletionRequirement(kind="ANSWER")]
    state = await graph(ReadModel(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "GPU"}
    )
    if corruption == "text":
        state["answer"]["text"] = "All games at 4K"
    if corruption == "completion":
        state["answer"]["task_outcomes"][0]["status"] = "UNRESOLVED"
    if corruption == "evidence":
        state["decision"]["action"]["answer"]["evidence_refs"] = ["made-up"]
    if corruption == "working_state":
        state["working_requirements"]["unresolved"] = ["new requirement"]
    if corruption == "pending":
        state["pending_proposal"] = {"invalid": True}
    if corruption == "phase":
        state["phase"] = "TOOL_PENDING"
    with pytest.raises(ValueError):
        prepare_answer_publication(
            state, base_snapshot(ctx), ctx, tasks, PCBuildApplicationService()
        )


@pytest.mark.asyncio
async def test_prepare_verified_build_maps_receipt_to_public_snapshot(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    from ai_service.capabilities.assistant.react.actions import FlexibleDecision

    class Builder:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
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
            result = transcript[-1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "BUILT",
                    "answer": {
                        "text": "Untrusted",
                        "build_ref": result.result.build_ref,
                        "evidence_refs": result.evidence_refs,
                    },
                }
            )

    ctx = context(state(), rows(realistic_catalog))
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    base = base_snapshot(ctx)
    terminal = await graph(Builder(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "Build PC"}
    )
    prepared = prepare_answer_publication(
        terminal, base, ctx, tasks, PCBuildApplicationService()
    )
    assert (
        prepared.snapshot.public_build.total_spending_vnd
        == terminal["ledger"][0]["result"]["result"]["receipt"]["build"]["total_price"]
    )
    assert prepared.snapshot.consultation.published_build_hash == canonical_hash(
        prepared.snapshot.public_build
    )
    assert base.public_build is None


@pytest.mark.asyncio
async def test_current_run_identity_is_required_for_publication_evidence():
    from uuid import uuid4

    ctx, tasks = context(), [CompletionRequirement(kind="ANSWER")]
    terminal = await graph(ReadModel(), Catalog(), ctx, required=tasks).ainvoke(
        {"message": "GPU"}
    )
    foreign_run = ctx.model_copy(update={"run_id": uuid4()}, deep=True)
    with pytest.raises(ValueError):
        prepare_answer_publication(
            terminal,
            base_snapshot(ctx),
            foreign_run,
            tasks,
            PCBuildApplicationService(),
        )
