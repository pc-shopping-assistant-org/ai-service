import pytest
from test_react_tools import context

from ai_service.application.ports.web_search import ExtractedWebFact
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import FinalAnswerAction
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolCall,
    ToolResult,
)
from ai_service.capabilities.assistant.react.outcomes import (
    validate_final_answer,
    validate_negative_answer,
)
from ai_service.capabilities.assistant.react.tools import call_hash
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


def fact_result(ctx, status):
    fact = {
        "model": "GPU A",
        "attribute": "dimensions",
        "value": [329, 140, 62.5] if status == "SUPPORTED" else None,
        "unit": "mm",
        "fact_status": status,
        "source_url": "https://example.com/specs",
        "document_hash": "a" * 64,
        "parser_version": "specs-v1",
        "locator": "table:dimensions",
        "quote": "Dimensions not specified" if status == "UNKNOWN" else "Dimensions",
    }
    ref = "fact:" + canonical_hash(ExtractedWebFact.model_validate(fact))
    call = ToolCall(
        call_id="page",
        name="read_web_page",
        arguments={"url": fact["source_url"]},
    )
    return ToolResult(
        tool_call_id=call.call_id,
        tool=call.name,
        input_hash=call_hash(call, ctx),
        constraints_revision=ctx.requirements.revision,
        status="COMPLETED",
        evidence_refs=[ref],
        result={
            "tool": "read_web_page",
            "url": fact["source_url"],
            "final_url": fact["source_url"],
            "status": "READABLE",
            "extraction_status": "STRUCTURED",
            "document_hash": fact["document_hash"],
            "facts": [{"evidence_ref": ref, "fact": fact}],
        },
    )


@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT"])
def test_typed_missing_fact_finishes_unresolved_with_canonical_source(status):
    ctx = context()
    page = fact_result(ctx, status)
    action = FinalAnswerAction(
        outcome="INSUFFICIENT_EVIDENCE",
        supporting_tool_call_ids=["page"],
        answer={"text": "999 mm fits any case", "evidence_refs": page.evidence_refs},
    )
    tasks = [
        CompletionRequirement(kind="ANSWER"),
        CompletionRequirement(kind="BUILD", constraints_revision=0),
    ]
    answer = validate_final_answer(
        action, [page], ctx, tasks, PCBuildApplicationService()
    )
    assert answer.status == "UNRESOLVED" and answer.reason == "FACT_" + status
    assert all(task.status == "UNRESOLVED" for task in answer.task_outcomes)
    assert answer.evidence_refs == page.evidence_refs
    assert "GPU A" in answer.text and status in answer.text
    assert "https://example.com/specs" in answer.text
    assert "999" not in answer.text and "fits" not in answer.text


@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT"])
def test_unknown_or_conflicting_fact_cannot_complete_a_positive_answer(status):
    ctx = context()
    page = fact_result(ctx, status)
    action = FinalAnswerAction(
        answer={"text": "Solved", "evidence_refs": page.evidence_refs}
    )
    with pytest.raises(ValueError, match="insufficient|unresolved"):
        validate_final_answer(
            action,
            [page],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
            PCBuildApplicationService(),
        )


@pytest.mark.parametrize("bad", ["supported", "invented", "wrong_call", "no_fact"])
def test_missing_fact_claim_requires_exact_unknown_fact_not_readable_prose(bad):
    ctx = context()
    page = fact_result(ctx, "SUPPORTED" if bad == "supported" else "UNKNOWN")
    action = FinalAnswerAction(
        outcome="INSUFFICIENT_EVIDENCE",
        supporting_tool_call_ids=["invented" if bad == "wrong_call" else "page"],
        answer={
            "text": "Missing",
            "evidence_refs": ["fact:invented"]
            if bad == "invented"
            else []
            if bad == "no_fact"
            else page.evidence_refs,
        },
    )
    with pytest.raises(ValueError):
        validate_negative_answer(
            action, [page], ctx, [CompletionRequirement(kind="ANSWER")]
        )


def result(ctx, *, failed=False, empty=True):
    call = ToolCall(call_id="search", name="search_catalog", arguments={"query": "GPU"})
    return ToolResult(
        tool_call_id=call.call_id,
        tool=call.name,
        input_hash=call_hash(call, ctx),
        constraints_revision=ctx.requirements.revision,
        status="FAILED" if failed else "COMPLETED",
        error="CATALOG_UNAVAILABLE" if failed else None,
        result=None
        if failed
        else {
            "tool": "search_catalog",
            "products": []
            if empty
            else [
                {
                    "product_id": ctx.account_id,
                    "name": "GPU",
                    "retrieved_at": "2026-10-12T00:00:00Z",
                    "evidence_ref": "catalog:gpu",
                }
            ],
        },
        evidence_refs=[] if failed or empty else ["catalog:gpu"],
    )


@pytest.mark.parametrize(
    "failed,outcome,reason",
    [
        (True, "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_UNAVAILABLE"),
        (False, "INSUFFICIENT_EVIDENCE", "NO_RESULTS"),
    ],
)
def test_actual_negative_tool_outcome_does_not_complete_build(failed, outcome, reason):
    ctx = context()
    action = FinalAnswerAction(
        outcome=outcome,
        supporting_tool_call_ids=["search"],
        answer={"text": "There is no feasible build"},
    )
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    answer = validate_negative_answer(action, [result(ctx, failed=failed)], ctx, tasks)
    assert answer.status == "UNRESOLVED"
    assert answer.task_outcomes[0].status == "UNRESOLVED"
    assert answer.task_outcomes[0].task == tasks[0]
    assert answer.reason == reason and answer.build_ref is None
    assert "no feasible" not in answer.text


@pytest.mark.parametrize("ids", [[], ["invented"], ["search", "search"]])
def test_negative_claim_requires_exact_unique_canonical_calls(ids):
    ctx = context()
    action = FinalAnswerAction(
        outcome="INSUFFICIENT_EVIDENCE",
        supporting_tool_call_ids=ids,
        answer={"text": "Not enough data"},
    )
    with pytest.raises(ValueError):
        validate_negative_answer(
            action, [result(ctx)], ctx, [CompletionRequirement(kind="ANSWER")]
        )


def test_valid_search_results_are_not_proof_of_no_results_or_dependency_failure():
    ctx = context()
    for outcome in (
        "INSUFFICIENT_EVIDENCE",
        "DEPENDENCY_UNAVAILABLE",
        "INFEASIBLE",
        "SEARCH_INCOMPLETE",
    ):
        action = FinalAnswerAction(
            outcome=outcome,
            supporting_tool_call_ids=["search"],
            answer={"text": "No results"},
        )
        with pytest.raises(ValueError):
            validate_negative_answer(
                action,
                [result(ctx, empty=False)],
                ctx,
                [CompletionRequirement(kind="ANSWER")],
            )


def test_unreadable_page_supports_source_unavailable_not_infeasibility():
    ctx = context()
    call = ToolCall(
        call_id="page",
        name="read_web_page",
        arguments={"url": "https://example.com/specs"},
    )
    page = ToolResult(
        tool_call_id=call.call_id,
        tool=call.name,
        input_hash=call_hash(call, ctx),
        constraints_revision=0,
        status="COMPLETED",
        result={
            "tool": "read_web_page",
            "url": "https://example.com/specs",
            "final_url": "https://example.com/specs",
            "status": "UNREADABLE",
        },
    )
    action = FinalAnswerAction(
        outcome="INSUFFICIENT_EVIDENCE",
        supporting_tool_call_ids=["page"],
        answer={"text": "There is no measurement"},
    )
    answer = validate_negative_answer(
        action, [page], ctx, [CompletionRequirement(kind="ANSWER")]
    )
    assert answer.reason == "SOURCE_UNAVAILABLE" and answer.status == "UNRESOLVED"
    assert "measurement" not in answer.text


def test_bad_tool_arguments_are_not_dependency_outage_evidence():
    ctx = context()
    rejected = ToolResult.model_validate(
        result(ctx, failed=True).model_dump()
        | {"status": "REJECTED", "error": "BAD_ARGUMENTS"}
    )
    action = FinalAnswerAction(
        outcome="DEPENDENCY_UNAVAILABLE",
        supporting_tool_call_ids=["search"],
        answer={"text": "Offline"},
    )
    with pytest.raises(ValueError):
        validate_negative_answer(
            action, [rejected], ctx, [CompletionRequirement(kind="ANSWER")]
        )
