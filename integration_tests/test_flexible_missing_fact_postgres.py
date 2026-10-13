"""Native missing-fact recovery and pure publication, not managed SQL CAS proof."""

from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from test_flexible_graph_postgres import Budget, Catalog

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.application.ports.web_search import WebPageReadResult
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolContext,
)
from ai_service.capabilities.assistant.react.publication import (
    prepare_answer_publication,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT"])
@pytest.mark.parametrize("multi", [False, True])
async def test_missing_fact_native_reopen_does_not_refetch_or_complete_task(
    postgres_dsn, status, multi
):
    url = "https://example.com/specs"

    class Reader:
        calls = 0

        async def read(self, requested):
            assert requested == url
            self.calls += 1
            return WebPageReadResult(
                url=url,
                final_url=url,
                status="READABLE",
                extraction_status="STRUCTURED",
                document_hash="a" * 64,
                facts=[
                    {
                        "model": "GPU A",
                        "value": None,
                        "fact_status": status,
                        "source_url": url,
                        "document_hash": "a" * 64,
                        "parser_version": "specs-v1",
                        "locator": "table:dimensions",
                        "quote": "Dimensions not specified",
                    }
                ],
            )

    class Model:
        def __init__(self, fail=False):
            self.fail = fail

        async def decide(self, message, transcript, allowed, repair, context=None):
            if multi and not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "catalog",
                            "name": "search_catalog",
                            "arguments": {"query": "GPU"},
                        },
                    }
                )
            if not transcript or multi and len(transcript) == 1:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "page",
                            "name": "read_web_page",
                            "arguments": {"url": url},
                        },
                    }
                )
            if self.fail:
                raise TimeoutError("after committed page")
            record = transcript[-1]
            if multi:
                return FlexibleDecision(
                    action={
                        "kind": "FINAL_ANSWER",
                        "outcome": "MULTI_TASK",
                        "answer": {"text": "Untrusted combined summary"},
                        "task_answers": [
                            {
                                "task_index": 0,
                                "outcome": "ANSWERED",
                                "answer": {
                                    "text": "Invented price",
                                    "evidence_refs": transcript[0].result.evidence_refs,
                                },
                            },
                            {
                                "task_index": 1,
                                "outcome": "INSUFFICIENT_EVIDENCE",
                                "supporting_tool_call_ids": [record.call.call_id],
                                "answer": {
                                    "text": "999 mm",
                                    "evidence_refs": record.result.evidence_refs,
                                },
                            },
                        ],
                    }
                )
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "INSUFFICIENT_EVIDENCE",
                    "supporting_tool_call_ids": [record.call.call_id],
                    "answer": {
                        "text": "999 mm fits every case",
                        "evidence_refs": record.result.evidence_refs,
                    },
                }
            )

    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    accepted = AcceptedConversationSnapshot(
        consultation=ConsultationStateV1(),
        requirement_state=ctx.requirements,
    )
    tasks = [
        CompletionRequirement(kind="ANSWER"),
        CompletionRequirement(kind="ANSWER")
        if multi
        else CompletionRequirement(kind="BUILD", constraints_revision=0),
    ]
    reader, service = Reader(), PCBuildApplicationService()
    catalog = Catalog()
    dispatcher = ToolDispatcher(catalog, service, reader=reader)
    config = {"configurable": {"thread_id": str(ctx.conversation_id)}}

    def compile_flow(model, saver):
        return build_flexible_graph(
            model,
            dispatcher,
            ctx,
            tasks,
            Budget(),
            conversation_revision=0,
            checkpointer=saver,
        )

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            with pytest.raises(TimeoutError, match="after committed"):
                await compile_flow(Model(fail=True), saver).ainvoke(
                    {"message": "Kích thước GPU?"}, config, durability="sync"
                )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            graph = compile_flow(Model(), saver)
            snapshot = await graph.aget_state(config)
            assert snapshot.next == ("agent",)
            values = await graph.ainvoke(None, snapshot.config, durability="sync")
            prepared = prepare_answer_publication(values, accepted, ctx, tasks, service)
            assert reader.calls == 1
            assert prepared.answer.status == "UNRESOLVED"
            if multi:
                assert prepared.answer.outcome == "MULTI_TASK" and catalog.calls == 1
                assert [task.status for task in prepared.answer.task_outcomes] == [
                    "COMPLETED",
                    "UNRESOLVED",
                ]
                assert prepared.answer.task_outcomes[1].reason == "FACT_" + status
                assert (
                    "8,000,000 VND" in prepared.answer.text
                    and "Invented price" not in prepared.answer.text
                )
            else:
                assert prepared.answer.reason == "FACT_" + status
                assert all(
                    task.status == "UNRESOLVED"
                    for task in prepared.answer.task_outcomes
                )
            assert status in prepared.answer.text and url in prepared.answer.text
            assert "999" not in prepared.answer.text
            assert prepared.snapshot == accepted
            forged = values | {"answer": values["answer"] | {"text": "999 mm"}}
            with pytest.raises(ValueError, match="Terminal answer"):
                prepare_answer_publication(forged, accepted, ctx, tasks, service)
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(ctx.conversation_id))
