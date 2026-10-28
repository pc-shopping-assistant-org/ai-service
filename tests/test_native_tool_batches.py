"""Native batches execute independent reads and return a result for every call."""

import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from test_native_react_model import Provider, native_result
from test_react_graph import Budget
from test_react_tools import Catalog, context

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ReactLimits,
    ToolCall,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.providers.react_model import (
    NativeFlexibleModel,
    build_react_request,
)


def two_searches():
    return AIMessage(
        content="",
        tool_calls=[
            {"id": "search-a", "name": "search_catalog", "args": {"query": "GPU A"}},
            {"id": "search-b", "name": "search_catalog", "args": {"query": "GPU B"}},
        ],
    )


@pytest.mark.asyncio
async def test_adapter_accepts_native_calls_array_without_registration():
    provider = Provider(two_searches())
    decision = await NativeFlexibleModel(provider).decide(
        "Tìm hai GPU", [], (ToolName.SEARCH_CATALOG,), None, context()
    )
    assert [call.call_id for call in decision.action.calls] == ["search-a", "search-b"]
    assert "register_tasks" not in {tool["name"] for tool in provider.model.tools}
    assert provider.model.options["parallel_tool_calls"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_second", [False, True])
async def test_search_batch_runs_together_and_retains_partial_results(fail_second):
    class Products(Catalog):
        active = 0
        peak = 0

        async def search(self, query, limit):
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            if fail_second and query == "GPU B":
                raise BackendUnavailableError("fixture offline")
            return [
                self.product.model_copy(
                    update={
                        "product_id": uuid5(NAMESPACE_URL, query),
                        "name": query,
                        "evidence_ref": f"catalog:{query}",
                    }
                )
            ]

    products, provider, requests = Products(), Provider(None), []

    async def invoke(messages):
        requests.append(messages)
        if len(requests) == 1:
            return two_searches()
        native = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        replies = [m for m in messages if isinstance(m, ToolMessage)]
        assert len(native) == 1 and len(native[0].tool_calls) == 2
        assert [m.tool_call_id for m in replies] == ["search-a", "search-b"]
        payloads = [json.loads(m.content) for m in replies]
        assert payloads[1]["status"] == ("FAILED" if fail_second else "COMPLETED")
        refs = [ref for payload in payloads for ref in payload["evidence_refs"]]
        return native_result(AnswerCandidate(text="Catalog", evidence_refs=refs))

    provider.model.ainvoke = invoke
    graph = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(products, PCBuildApplicationService()),
        context(),
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
    )
    result = await graph.ainvoke({"message": "Tìm GPU A và GPU B"})
    assert len(requests) == 2
    assert products.calls == products.peak == 2
    assert result["tool_calls"] == len(result["ledger"]) == 2
    assert result["answer"]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_invalid_call_does_not_drop_valid_calls_or_break_provider_pairs():
    products, provider, requests = Catalog(), Provider(None), []

    async def invoke(messages):
        requests.append(messages)
        if len(requests) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": "valid", "name": "search_catalog", "args": {"query": "GPU"}},
                    {"id": "bad", "name": "search_catalog", "args": {"limit": 999}},
                    {"id": "unknown", "name": "not_a_tool", "args": {}},
                ],
            )
        replies = [m for m in messages if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in replies] == ["valid", "bad", "unknown"]
        values = [json.loads(m.content) for m in replies]
        assert [item["status"] for item in values] == [
            "COMPLETED",
            "REJECTED",
            "REJECTED",
        ]
        assert [item["error"] for item in values[1:]] == [
            "BAD_ARGUMENTS",
            "UNKNOWN_TOOL",
        ]
        return native_result(
            AnswerCandidate(text="Found", evidence_refs=values[0]["evidence_refs"])
        )

    provider.model.ainvoke = invoke
    graph = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(products, PCBuildApplicationService()),
        context(),
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
    )
    saved = await graph.ainvoke({"message": "GPU"})
    assert products.calls == 1 and saved["tool_calls"] == 3


@pytest.mark.asyncio
async def test_batch_limit_repairs_before_dispatch_and_run_budget_counts_each_call():
    products, provider, requests = Catalog(), Provider(None), []

    async def invoke(messages):
        requests.append(messages)
        if len(requests) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"id": str(i), "name": "search_catalog", "args": {"query": "GPU"}}
                    for i in range(5)
                ],
            )
        if len(requests) == 2:
            assert "max_tool_calls_per_batch" in messages[0].content
        return AIMessage(
            content="",
            tool_calls=[
                {"id": "good", "name": "search_catalog", "args": {"query": "GPU"}}
            ],
        )

    provider.model.ainvoke = invoke
    graph = build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(products, PCBuildApplicationService()),
        context(),
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
        limits=ReactLimits(max_tool_calls=1),
    )
    with pytest.raises(ValueError, match="REACT_TOOL_LIMIT"):
        await graph.ainvoke({"message": "GPU"})
    assert products.calls == 1


@pytest.mark.asyncio
async def test_context_window_is_bounded_and_canonical_results_are_unchanged():
    products, trusted = Catalog(), context()
    products.product.specifications = {
        "description": "Nội dung nguồn. " * 4000,
        "capacity_gb": 32,
    }
    records = []
    for i in range(4):
        call = ToolCall(
            call_id=f"s{i}", name="search_catalog", arguments={"query": "GPU"}
        )
        result = await ToolDispatcher(products, PCBuildApplicationService()).execute(
            call, trusted
        )
        records.append(
            ToolRecord(
                call=call, result=result, origin_run_id=trusted.run_id, model_step=i
            )
        )
    original = [record.model_dump_json() for record in records]
    messages, schemas = build_react_request(
        "GPU",
        records,
        (),
        None,
        trusted,
        limits=ReactLimits(max_tool_context_bytes=1200, max_model_input_bytes=18000),
    )
    pairs = [m for m in messages if isinstance(m, ToolMessage)]
    for reply in pairs:
        value = json.loads(reply.content)
        assert len(reply.content.encode()) <= 1200
        assert value["projection"]["truncated"] is True
        assert value["result"]["products"][0]["price_vnd"] == products.product.price_vnd
        assert value["result"]["products"][0]["specifications"]["capacity_gb"] == 32
    assert [m.tool_calls[0]["id"] for m in messages if isinstance(m, AIMessage)] == [
        m.tool_call_id for m in pairs
    ]
    assert [record.model_dump_json() for record in records] == original
    assert (
        len(
            json.dumps(
                {
                    "messages": [m.model_dump(mode="json") for m in messages],
                    "tools": schemas,
                },
                ensure_ascii=False,
            ).encode()
        )
        <= 18000
    )


async def exercise_batch_recovery(open_saver):
    """Same assertions for memory and a closed/reopened PostgreSQL saver."""
    calls, requests, refs = Counter(), [], []
    trusted, products, provider = context(), Catalog(), Provider(None)
    original_search = products.search

    async def search(query, limit):
        calls[query] += 1
        if query == "GPU B" and calls[query] == 1:
            await asyncio.sleep(0.02)
            raise TimeoutError("worker interrupted during the second call")
        return await original_search(query, limit)

    products.search = search

    async def invoke(messages):
        requests.append(messages)
        if len(requests) == 1:
            return two_searches()
        results = [
            json.loads(m.content) for m in messages if isinstance(m, ToolMessage)
        ]
        return native_result(
            AnswerCandidate(
                text="Results",
                evidence_refs=list(
                    {ref for item in results for ref in item["evidence_refs"]}
                ),
            )
        )

    provider.model.ainvoke = invoke

    def graph_for(saver):
        return DurableConversationGraph(
            build_flexible_graph(
                NativeFlexibleModel(provider),
                ToolDispatcher(products, PCBuildApplicationService()),
                trusted,
                [CompletionRequirement(kind="ANSWER")],
                Budget(),
                conversation_revision=0,
                checkpointer=saver,
            )
        )

    previous = ExecutionClaim(
        run_id=trusted.run_id,
        conversation_id=trusted.conversation_id,
        execution_id=uuid4(),
        worker_instance_id=uuid4(),
        generation=1,
    )
    current = previous.model_copy(
        update={"execution_id": uuid4(), "worker_instance_id": uuid4(), "generation": 2}
    )
    proof = ExecutionStopEvidenceV1(
        kind="EXECUTOR_STOP_ACK",
        execution_id=previous.execution_id,
        worker_instance_id=previous.worker_instance_id,
        generation=1,
        stopped_at=datetime.now(UTC),
        checkpoint_writes_drained=True,
        proof_ref="fixture:joined-and-drained",
    )
    inputs = {"message": "Find GPU A and GPU B"}

    async def emitted(ref):
        refs.append(ref)

    async with open_saver() as saver:
        graph = graph_for(saver)
        config = await graph.graph.aupdate_state(
            {"configurable": {"thread_id": str(trusted.conversation_id)}},
            {"phase": "READY_TO_PUBLISH"},
            as_node="validate_action",
        )
        base = CheckpointRefV1(
            thread_id=trusted.conversation_id,
            checkpoint_id=config["configurable"]["checkpoint_id"],
        )
        head = ConversationHeadV1(
            conversation_id=trusted.conversation_id,
            revision=1,
            accepted_checkpoint_ref=base,
        )
        with pytest.raises(TimeoutError):
            await graph.new_turn(
                head,
                previous.run_id,
                inputs,
                "batch-test-v1",
                execution=previous,
                checkpoint_sink=emitted,
            )
        source = refs[-1]
        # A newer branch must not become the recovery source.
        await graph.graph.aupdate_state(
            graph.config(source),
            {"message": "orphan", "phase": "STARTED"},
            as_node="begin_turn",
        )
    async with open_saver() as saver:
        graph = graph_for(saver)
        final = await graph.resume(
            head, source, previous, current, inputs, "batch-test-v1", evidence=proof
        )
        saved = await graph.read(final)
        assert calls == {"GPU A": 1, "GPU B": 2}
        assert len(requests) == 2
        assert saved.values["message"] == inputs["message"]
        assert len(saved.values["ledger"]) == 2
        assert saved.values["answer"]["status"] == "COMPLETED"
        await saver.adelete_thread(str(trusted.conversation_id))


@pytest.mark.asyncio
async def test_partial_native_batch_resume_reuses_success_and_ignores_newer_orphan():
    saver = InMemorySaver()

    @asynccontextmanager
    async def open_saver():
        yield saver

    await exercise_batch_recovery(open_saver)
