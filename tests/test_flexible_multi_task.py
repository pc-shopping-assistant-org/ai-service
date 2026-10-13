from copy import deepcopy

import pytest
from test_flexible_outcomes import fact_result
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_requirement_pipeline import rows, state

from ai_service.application.ports.web_search import WebPageReadResult
from ai_service.capabilities.assistant.react.actions import (
    FinalAnswerAction,
    FlexibleDecision,
)
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolCall,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
from ai_service.capabilities.assistant.react.public_views import to_public_flexible
from ai_service.capabilities.assistant.react.publication import (
    prepare_answer_publication,
)
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer
from ai_service.infrastructure.graph.flexible import build_flexible_graph


def candidate(build, page):
    return {
        "outcome": "MULTI_TASK",
        "answer": {"text": "Untrusted aggregate: every game at 4K"},
        "task_answers": [
            {
                "task_index": 0,
                "outcome": "BUILT",
                "answer": {
                    "text": "All games at 4K",
                    "build_ref": build.result.build_ref,
                    "evidence_refs": build.evidence_refs,
                },
            },
            {
                "task_index": 1,
                "outcome": "INSUFFICIENT_EVIDENCE",
                "supporting_tool_call_ids": [page.tool_call_id],
                "answer": {"text": "999 mm", "evidence_refs": page.evidence_refs},
            },
        ],
    }


@pytest.mark.asyncio
async def test_targeted_repair_corrects_only_failed_task_without_reoptimizing(
    realistic_catalog,
):
    ctx = context(state(), rows(realistic_catalog))
    page = fact_result(ctx, "UNKNOWN").result

    class Reader:
        calls = 0

        async def read(self, url):
            self.calls += 1
            return WebPageReadResult(
                url=url,
                final_url=url,
                status="READABLE",
                extraction_status="STRUCTURED",
                document_hash=page.document_hash,
                facts=[entry.fact for entry in page.facts],
            )

    class Service(PCBuildApplicationService):
        calls = 0

        def build_pc(self, *args, **kwargs):
            self.calls += 1
            return super().build_pc(*args, **kwargs)

    class Model:
        feedback = None

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
            if len(transcript) == 1:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "page",
                            "name": "read_web_page",
                            "arguments": {"url": page.url},
                        },
                    }
                )
            payload = candidate(transcript[0].result, transcript[1].result)
            if repair is None:
                payload["task_answers"][1]["outcome"] = "ANSWERED"
            else:
                self.feedback = repair
                assert repair.task_indices == [1]
            return FlexibleDecision(action={"kind": "FINAL_ANSWER", **payload})

    service, reader, model = Service(), Reader(), Model()
    tasks = [
        CompletionRequirement(
            kind="BUILD", constraints_revision=ctx.requirements.revision
        ),
        CompletionRequirement(kind="ANSWER"),
    ]
    graph = build_flexible_graph(
        model,
        ToolDispatcher(Catalog(), service, reader=reader),
        ctx,
        tasks,
        Budget(),
        conversation_revision=0,
    )
    result = await graph.ainvoke({"message": "Build PC và kích thước GPU"})
    assert model.feedback.code == "TASK_INCOMPLETE"
    assert service.calls == reader.calls == 1
    assert [task["status"] for task in result["answer"]["task_outcomes"]] == [
        "COMPLETED",
        "UNRESOLVED",
    ]


@pytest.mark.asyncio
async def test_partial_build_and_unknown_fact_preserve_verified_build(
    realistic_catalog,
):
    ctx = context(state(), rows(realistic_catalog))
    service = PCBuildApplicationService()
    call = ToolCall(call_id="build", name="build_pc", arguments={})
    build = await ToolDispatcher(Catalog(), service).execute(call, ctx)
    page = fact_result(ctx, "UNKNOWN")
    tasks = [
        CompletionRequirement(
            kind="BUILD", constraints_revision=ctx.requirements.revision
        ),
        CompletionRequirement(kind="ANSWER"),
    ]
    action = FinalAnswerAction(**candidate(build, page))
    answer = validate_final_answer(action, [build, page], ctx, tasks, service)
    assert answer.status == "UNRESOLVED" and answer.outcome == "MULTI_TASK"
    assert [row.status for row in answer.task_outcomes] == ["COMPLETED", "UNRESOLVED"]
    assert answer.task_outcomes[1].reason == "FACT_UNKNOWN"
    assert answer.build_ref == build.result.build_ref
    assert "4K" not in answer.text and "999" not in answer.text
    assert "UNKNOWN" in answer.text
    page_call = ToolCall(
        call_id="page",
        name="read_web_page",
        arguments={"url": "https://example.com/specs"},
    )
    values = {
        "phase": "READY_TO_PUBLISH",
        "decision": {"action": action.model_dump(mode="json")},
        "working_requirements": ctx.requirements.model_dump(mode="json"),
        "answer": answer.model_dump(mode="json"),
        "ledger": [
            ToolRecord(call=call, result=build, origin_run_id=ctx.run_id).model_dump(
                mode="json"
            ),
            ToolRecord(
                call=page_call, result=page, origin_run_id=ctx.run_id
            ).model_dump(mode="json"),
        ],
    }
    prepared = prepare_answer_publication(
        values, base_snapshot(ctx), ctx, tasks, service
    )
    assert prepared.snapshot.public_build is not None
    assert (
        prepared.snapshot.public_build.total_spending_vnd
        == build.result.receipt.build.total_price
    )
    published = FlexiblePublishedState(
        conversation_id=ctx.conversation_id,
        task_message_id=ctx.run_id,
        task_message="Build PC và thông số GPU",
        status=answer.status,
        answer=answer,
    )
    view = to_public_flexible(published, ctx.requirements)
    assert [task.status for task in view.task_outcomes] == ["COMPLETED", "UNRESOLVED"]
    assert view.task_outcomes[1].reason == "FACT_UNKNOWN"
    assert view.task_outcomes[0].outcome == "BUILT"
    forged = deepcopy(values)
    forged["answer"]["task_outcomes"][1]["status"] = "COMPLETED"
    with pytest.raises(ValueError):
        prepare_answer_publication(forged, base_snapshot(ctx), ctx, tasks, service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", ["drop", "duplicate", "index", "swap", "invented", "aggregate"]
)
async def test_multi_task_cannot_drop_or_complete_unproven_tasks(
    realistic_catalog, bad
):
    ctx = context(state(), rows(realistic_catalog))
    service = PCBuildApplicationService()
    build = await ToolDispatcher(Catalog(), service).execute(
        ToolCall(call_id="build", name="build_pc", arguments={}), ctx
    )
    page = fact_result(ctx, "UNKNOWN")
    payload = candidate(build, page)
    if bad == "drop":
        payload["task_answers"].pop()
    elif bad == "duplicate":
        payload["task_answers"][1]["task_index"] = 0
    elif bad == "index":
        payload["task_answers"][1]["task_index"] = 2
    elif bad == "swap":
        payload["task_answers"].reverse()
        for index, row in enumerate(payload["task_answers"]):
            row["task_index"] = index
    elif bad == "invented":
        payload["task_answers"][1]["outcome"] = "ANSWERED"
    else:
        payload["answer"]["evidence_refs"] = build.evidence_refs
    tasks = [
        CompletionRequirement(
            kind="BUILD", constraints_revision=ctx.requirements.revision
        ),
        CompletionRequirement(kind="ANSWER"),
    ]
    with pytest.raises(ValueError):
        validate_final_answer(
            FinalAnswerAction(**payload), [build, page], ctx, tasks, service
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("search", ["INFEASIBLE", "SEARCH_INCOMPLETE"])
async def test_scoped_search_result_does_not_erase_completed_catalog_task(
    realistic_catalog, search
):
    canonical = state()
    if search == "INFEASIBLE":
        next(iter(canonical.requirements.values())).value.amount = 1
    else:
        realistic_catalog = [
            part.model_copy(update={"performance_score": None})
            if part.category == "CPU"
            else part
            for part in realistic_catalog
        ]
    ctx = context(canonical, rows(realistic_catalog))
    service = PCBuildApplicationService(DeterministicPCOptimizer(enable_pruning=False))
    dispatcher = ToolDispatcher(Catalog(), service)
    build = await dispatcher.execute(
        ToolCall(call_id="build", name="build_pc", arguments={}), ctx
    )
    catalog = await dispatcher.execute(
        ToolCall(call_id="search", name="search_catalog", arguments={"query": "GPU"}),
        ctx,
    )
    action = FinalAnswerAction(
        outcome="MULTI_TASK",
        answer={"text": "Untrusted scope claim"},
        task_answers=[
            {
                "task_index": 1,
                "outcome": "ANSWERED",
                "answer": {
                    "text": "Invented price",
                    "evidence_refs": catalog.evidence_refs,
                },
            },
            {
                "task_index": 0,
                "outcome": search,
                "supporting_tool_call_ids": ["build"],
                "answer": {"text": "No PC exists anywhere"},
            },
        ],
    )
    tasks = [
        CompletionRequirement(
            kind="BUILD", constraints_revision=ctx.requirements.revision
        ),
        CompletionRequirement(kind="ANSWER"),
    ]
    answer = validate_final_answer(action, [build, catalog], ctx, tasks, service)
    assert answer.status == ("COMPLETED" if search == "INFEASIBLE" else "UNRESOLVED")
    assert [task.status for task in answer.task_outcomes] == [
        "COMPLETED" if search == "INFEASIBLE" else "UNRESOLVED",
        "COMPLETED",
    ]
    assert answer.search_evidence.outcome == search and answer.build_ref is None
    assert "8,000,000 VND" in answer.text and "anywhere" not in answer.text
