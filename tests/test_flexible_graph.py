from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


class ReadModel:
    def __init__(self, fail_once=False):
        self.fail_once = fail_once
        self.transcripts = []

    async def decide(self, message, transcript, allowed, repair, context=None):
        self.transcripts.append(transcript)
        if not transcript:
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": f"search-{len(self.transcripts)}",
                        "name": "search_catalog",
                        "arguments": {"query": "GPU"},
                    },
                }
            )
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("after tool commit")
        return FlexibleDecision(
            action={
                "kind": "FINAL_ANSWER",
                "answer": {
                    "text": "untrusted prose",
                    "evidence_refs": transcript[-1].result.evidence_refs,
                },
            }
        )


def graph(model, catalog, trusted, *, revision=0, required=None, saver=None):
    return build_flexible_graph(
        model,
        ToolDispatcher(catalog, PCBuildApplicationService()),
        trusted,
        required or [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=revision,
        checkpointer=saver,
    )


@pytest.mark.asyncio
async def test_read_before_budget_or_requirement_extraction_and_grounded_answer():
    catalog, trusted, model = Catalog(), context(), ReadModel()
    result = await graph(model, catalog, trusted).ainvoke({"message": "Xem GPU"})
    assert result["phase"] == "READY_TO_PUBLISH"
    assert trusted.requirements.requirements == {}
    assert len(result["ledger"]) == 1 and catalog.calls == 1
    assert "Canonical GPU" in result["answer"]["text"]
    assert "untrusted prose" not in result["answer"]["text"]


@pytest.mark.asyncio
async def test_proposal_waits_then_exact_confirmation_changes_only_working_state():
    class Proposer:
        async def decide(self, *args):
            return FlexibleDecision(
                action={
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": "RAM 32GB",
                    "draft": {
                        "items": [
                            {
                                "subject": "RAM",
                                "predicate": "capacity",
                                "start": 0,
                                "end": 8,
                                "value": {
                                    "kind": "QUANTITY",
                                    "amount": 32,
                                    "unit": "GB",
                                },
                            }
                        ]
                    },
                }
            )

    catalog, trusted = Catalog(), context()
    first = await graph(Proposer(), catalog, trusted).ainvoke({"message": "Tư vấn RAM"})
    assert (
        first["phase"] == "WAITING_INPUT"
        and first["working_requirements"]["requirements"] == {}
    )
    pending = PendingRequirementProposal.model_validate(first["pending_proposal"])
    assert pending.target_revision == 1
    reply = ProposalConfirmation(
        proposal_id=pending.proposal_id,
        proposal_hash=pending.proposal_hash,
        target_revision=1,
        base_requirements_hash=pending.base_requirements_hash,
    )
    second = await graph(ReadModel(), catalog, trusted, revision=1).ainvoke(
        first | {"message": "OK", "confirmation": reply.model_dump(mode="json")}
    )
    assert second["pending_proposal"] is None
    assert len(second["working_requirements"]["requirements"]) == 1
    assert trusted.requirements.requirements == {}  # No accepted mutation.


@pytest.mark.asyncio
async def test_native_memory_resume_does_not_repeat_committed_read():
    catalog, trusted, model = Catalog(), context(), ReadModel(fail_once=True)
    flow = graph(model, catalog, trusted, saver=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(TimeoutError):
        await flow.ainvoke({"message": "GPU"}, config, durability="sync")
    snapshot = await flow.aget_state(config)
    result = await flow.ainvoke(None, snapshot.config, durability="sync")
    assert result["phase"] == "READY_TO_PUBLISH" and catalog.calls == 1


@pytest.mark.asyncio
async def test_advice_cannot_complete_build_or_claim_infeasibility_without_proof():
    class UnsupportedFinal:
        def __init__(self):
            self.feedback = []

        async def decide(self, message, transcript, allowed, repair, context=None):
            self.feedback.append(repair)
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "INFEASIBLE",
                    "answer": {"text": "Không có nghiệm"},
                }
            )

    trusted, model = context(), UnsupportedFinal()
    with pytest.raises(ValueError, match="VALIDATION_EXHAUSTED"):
        await graph(
            model,
            Catalog(),
            trusted,
            required=[CompletionRequirement(kind="BUILD", constraints_revision=0)],
        ).ainvoke({"message": "Build PC"})
    assert model.feedback[1].code == "SEARCH_PROOF_REQUIRED"


@pytest.mark.asyncio
async def test_confirmed_merge_keeps_old_ledger_but_hides_stale_pairs_from_model():
    from test_flexible_contracts import pending_proposal

    from ai_service.capabilities.assistant.react.contracts import ToolCall
    from ai_service.infrastructure.graph.react import ToolRecord

    canonical, owner, conversation, pending, reply = pending_proposal()
    trusted = context(canonical)
    trusted.account_id, trusted.conversation_id = owner, conversation
    catalog = Catalog()
    old_call = ToolCall(
        call_id="read-old", name="search_catalog", arguments={"query": "GPU"}
    )
    old_result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        old_call, trusted
    )
    old_record = ToolRecord(call=old_call, result=old_result).model_dump(mode="json")
    model = ReadModel()
    result = await graph(model, catalog, trusted, revision=13).ainvoke(
        {
            "message": "OK",
            "working_requirements": canonical.model_dump(mode="json"),
            "pending_proposal": pending.model_dump(mode="json"),
            "confirmation": reply.model_dump(mode="json"),
            "ledger": [old_record],
        }
    )
    assert model.transcripts[0] == []
    assert len(result["ledger"]) == 2 and result["ledger"][0] == old_record
    assert model.transcripts[-1][0].call.call_id != "read-old"


@pytest.mark.asyncio
@pytest.mark.parametrize("lazy", [False, True])
async def test_flexible_build_uses_real_optimizer_and_trusted_task(
    realistic_catalog, lazy
):
    from test_requirement_pipeline import rows, state

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
                        "text": "Every game at 4K",
                        "build_ref": result.result.build_ref,
                        "evidence_refs": result.evidence_refs,
                    },
                }
            )

    trusted = context(state(), rows(realistic_catalog))
    loaded = []

    class Loader:
        async def load(self, draft, requirements):
            loaded.append(True)
            return trusted.catalog

    runtime = (
        trusted.model_copy(update={"catalog": [], "catalog_ready": False})
        if lazy
        else trusted
    )
    result = await build_flexible_graph(
        Builder(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        runtime,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        Budget(),
        conversation_revision=0,
        catalog_loader=Loader() if lazy else None,
    ).ainvoke({"message": "Build PC"})
    assert loaded == ([True] if lazy else [])
    assert result["phase"] == "READY_TO_PUBLISH"
    assert result["ledger"][0]["result"]["result"]["receipt"]["outcome"] == "BUILT"
    assert "Every game" not in result["answer"]["text"]
    assert result["answer"]["build_ref"] is not None


@pytest.mark.asyncio
async def test_read_then_dynamic_clarification_retains_tool_ledger_and_build_task():
    class ReadThenAsk(ReadModel):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return await super().decide(
                    message, transcript, allowed, repair, context
                )
            assert context.completion_requirements[0].kind == "BUILD"
            return FlexibleDecision(
                action={
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách?",
                    "missing_fields": ["BUDGET_REQUIRED"],
                }
            )

    result = await graph(
        ReadThenAsk(),
        Catalog(),
        context(),
        required=[CompletionRequirement(kind="BUILD", constraints_revision=0)],
    ).ainvoke({"message": "Build PC có GPU"})
    assert result["phase"] == "WAITING_INPUT" and result["answer"] == {}
    assert len(result["ledger"]) == 1
    assert result["clarification"]["missing_fields"] == ["BUDGET_REQUIRED"]


@pytest.mark.asyncio
async def test_unresolved_proposal_does_not_replace_existing_confirmation_artifact():
    from test_flexible_contracts import pending_proposal

    canonical, owner, conversation, pending, _ = pending_proposal()
    trusted = context(canonical)
    trusted.account_id, trusted.conversation_id = owner, conversation

    class Replacement(ReadModel):
        attempted = False

        async def decide(self, message, transcript, allowed, repair, context=None):
            if not self.attempted:
                self.attempted = True
                return FlexibleDecision(
                    action={
                        "kind": "PROPOSE_REQUIREMENTS",
                        "source_text": "GPU unknown",
                        "draft": {
                            "items": [
                                {
                                    "subject": "GPU",
                                    "predicate": "identity",
                                    "start": 0,
                                    "end": 11,
                                    "value": {"kind": "TEXT", "text": "GPU unknown"},
                                }
                            ]
                        },
                    }
                )
            if not transcript:
                assert repair.code == "PROPOSAL_CONFIRMATION_REQUIRED"
            return await super().decide(message, transcript, allowed, repair, context)

    result = await graph(Replacement(), Catalog(), trusted, revision=13).ainvoke(
        {"message": "Tư vấn", "pending_proposal": pending.model_dump(mode="json")}
    )
    assert result["pending_proposal"] == pending.model_dump(mode="json")
    assert result.get("pending_draft") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_empty_or_failed_search_can_end_turn_without_completing_build(failed):
    from ai_service.application.errors import BackendUnavailableError

    class MissingCatalog(Catalog):
        async def search(self, query, limit):
            self.calls += 1
            if failed:
                raise BackendUnavailableError("offline")
            return []

    class Negative(ReadModel):
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return await super().decide(
                    message, transcript, allowed, repair, context
                )
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "DEPENDENCY_UNAVAILABLE"
                    if failed
                    else "INSUFFICIENT_EVIDENCE",
                    "supporting_tool_call_ids": [transcript[-1].call.call_id],
                    "answer": {"text": "No feasible configuration"},
                }
            )

    result = await graph(
        Negative(),
        MissingCatalog(),
        context(),
        required=[CompletionRequirement(kind="BUILD", constraints_revision=0)],
    ).ainvoke({"message": "Build PC"})
    assert result["phase"] == "READY_TO_PUBLISH"
    assert result["answer"]["schema_version"] == 2
    assert result["answer"]["status"] == "UNRESOLVED"
    assert result["answer"]["task_outcomes"][0]["task"]["kind"] == "BUILD"
    assert result["answer"]["task_outcomes"][0]["status"] == "UNRESOLVED"
    assert "No feasible" not in result["answer"]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("incomplete", [False, True])
async def test_optimizer_search_report_controls_final_graph_outcome(
    realistic_catalog, incomplete
):
    from test_requirement_pipeline import rows, state

    canonical = state()
    if not incomplete:
        next(iter(canonical.requirements.values())).value.amount = 1
    parts = [
        part.model_copy(update={"performance_score": None})
        if incomplete and part.category == "CPU"
        else part
        for part in realistic_catalog
    ]
    trusted = context(canonical, rows(parts))

    class SearchModel:
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
            report = transcript[-1].result.result.search_evidence
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": report.outcome,
                    "supporting_tool_call_ids": ["build"],
                    "answer": {"text": "Nothing exists anywhere"},
                }
            )

    result = await graph(
        SearchModel(),
        Catalog(),
        trusted,
        required=[CompletionRequirement(kind="BUILD", constraints_revision=0)],
    ).ainvoke({"message": "Build PC"})
    assert result["phase"] == "READY_TO_PUBLISH"
    assert result["answer"]["outcome"] == (
        "SEARCH_INCOMPLETE" if incomplete else "INFEASIBLE"
    )
    assert result["answer"]["status"] == ("UNRESOLVED" if incomplete else "COMPLETED")
    assert "anywhere" not in result["answer"]["text"]


@pytest.mark.asyncio
async def test_new_run_does_not_reuse_old_price_without_catalog_revalidation():
    catalog, trusted = Catalog(), context()
    first = await graph(ReadModel(), catalog, trusted).ainvoke({"message": "GPU"})
    old = first["ledger"][0]
    catalog.product.price_vnd = 9_000_000
    new_context = trusted.model_copy(update={"run_id": uuid4()}, deep=True)
    second = await graph(ReadModel(), catalog, new_context, revision=1).ainvoke(
        first | {"message": "Giá GPU hiện tại?"}
    )
    assert catalog.calls == 2
    assert second["ledger"][0] == old
    assert "9,000,000 VND" in second["answer"]["text"]
    assert "8,000,000 VND" not in second["answer"]["text"]
