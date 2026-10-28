"""Common turns enter ReAct directly without task registration/extraction."""

import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from test_flexible_publication import base_snapshot
from test_flexible_registration_graph import Products
from test_native_react_model import Provider, native_result
from test_react_graph import Budget
from test_react_tools import context

from ai_service.capabilities.assistant.react.contracts import AnswerCandidate, ToolName
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.providers.react_model import NativeFlexibleModel


@pytest.mark.asyncio
async def test_native_action_has_no_registration_schema():
    trusted = context()
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "search", "name": "search_catalog", "args": {"query": "GPU"}},
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU?", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision.registration is None
    assert "register_tasks" not in {tool["name"] for tool in provider.model.tools}
    assert decision.action.calls[0].name == "search_catalog"
    assert provider.model.options["parallel_tool_calls"] is True


def flow_for(provider, message="GPU?"):
    trusted, products = context(), Products()
    return build_flexible_graph(
        NativeFlexibleModel(provider),
        ToolDispatcher(products, PCBuildApplicationService()),
        trusted,
        [],
        Budget(),
        conversation_revision=0,
        task_catalog=products,
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(trusted),
            task_message_id=uuid4(),
            task_message=message,
            request_message=message,
        ),
    )


@pytest.mark.asyncio
async def test_greeting_requires_one_model_call_and_no_registration():
    provider = Provider(AIMessage(content="Chào bạn!"))
    graph = flow_for(provider, "Chào bạn")
    result = await graph.ainvoke({"message": "Chào bạn"})
    assert result["response"]["text"] == "Chào bạn!"
    assert result["model_calls"] == 1
    assert not result["task_draft"]["tasks"]


@pytest.mark.asyncio
async def test_catalog_request_executes_without_a_task_model_step():
    provider = Provider(None)
    calls = []

    async def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "search",
                        "name": "search_catalog",
                        "args": {"query": "GPU"},
                    },
                ],
            )
        native = [m for m in messages if isinstance(m, AIMessage) and m.tool_calls]
        assert [c["name"] for c in native[0].tool_calls] == [
            "search_catalog",
        ]
        replies = [m for m in messages if isinstance(m, ToolMessage)]
        assert {m.tool_call_id for m in replies} == {"search"}
        result = json.loads(
            next(m.content for m in replies if m.name == "search_catalog")
        )
        return native_result(
            AnswerCandidate(text="Catalog", evidence_refs=result["evidence_refs"])
        )

    provider.model.ainvoke = invoke
    graph = flow_for(provider)
    result = await graph.ainvoke({"message": "GPU?"})
    assert len(calls) == result["model_calls"] == 2
    assert len(result["ledger"]) == 1
    assert result["answer"]["task_outcomes"][0]["status"] == "COMPLETED"
