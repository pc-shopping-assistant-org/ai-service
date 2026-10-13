"""Native checkpoint recovery of curated source facts; no live model/network claim."""

from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
    ToolContext,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.react import ReactDecision, build_react_graph
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

URL = "https://www.asus.com/motherboards-components/graphics-cards/tuf-gaming/tuf-rtx5070ti-o16g-gaming/techspec/"


class NoCatalog:
    async def search(self, query, limit):
        raise AssertionError("Web research must not use catalog fixture")

    async def get(self, product_id):
        raise AssertionError("Web research must not use catalog fixture")


class Budget:
    async def invoke(self, kind, key, input_hash, dispatch):
        return await dispatch()


class Model:
    def __init__(self, fail_after_read=False):
        self.fail_after_read = fail_after_read

    async def decide(self, message, transcript, allowed, repair, context=None):
        if not transcript:
            return ReactDecision(
                calls=[
                    ToolCall(
                        call_id="page", name="read_web_page", arguments={"url": URL}
                    )
                ]
            )
        if self.fail_after_read:
            raise RuntimeError("explanation failed after committed tool")
        fact = transcript[0].result.result.facts[0]
        assert fact.fact.value == (329.0, 140.0, 62.5)
        return ReactDecision(
            answer=AnswerCandidate(
                text="Untrusted 999 mm", evidence_refs=[fact.evidence_ref]
            )
        )


@pytest.mark.asyncio
async def test_curated_facts_survive_postgres_reopen_without_refetch(
    postgres_dsn, monkeypatch
):
    html = (
        Path(__file__)
        .parents[1]
        .joinpath("tests/fixtures/asus_gpu_techspec.html")
        .read_text()
    )
    original = httpx.AsyncClient
    calls = []

    def fetch(request):
        calls.append(str(request.url))
        return httpx.Response(200, text=html)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(fetch), **kwargs),
    )
    conversation_id = uuid4()
    config = {"configurable": {"thread_id": str(conversation_id)}}
    ctx = ToolContext(
        account_id=uuid4(),
        conversation_id=conversation_id,
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        catalog=[],
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    dispatcher = ToolDispatcher(
        NoCatalog(), PCBuildApplicationService(), reader=TrafilaturaReader()
    )
    required = [CompletionRequirement(kind="ANSWER")]
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = build_react_graph(
                Model(fail_after_read=True),
                dispatcher,
                ctx,
                required,
                Budget(),
                checkpointer=saver,
            )
            with pytest.raises(RuntimeError, match="explanation failed"):
                await graph.ainvoke(
                    {"message": "Kích thước VGA ASUS này?", "ledger": []},
                    config,
                    durability="sync",
                )
            pending = await graph.aget_state(config)
            assert pending.next == ("agent",)
            assert (
                pending.values["ledger"][0]["result"]["result"]["extraction_status"]
                == "STRUCTURED"
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            graph = build_react_graph(
                Model(), dispatcher, ctx, required, Budget(), checkpointer=saver
            )
            result = await graph.ainvoke(None, config, durability="sync")
            assert result["phase"] == "READY_TO_PUBLISH"
            assert "329 × 140 × 62.5 mm" in result["answer"]["text"]
            assert "999" not in result["answer"]["text"]
            assert URL in result["answer"]["text"]
            assert calls == [URL]
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation_id))
