import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.factory import build_model_provider
from ai_service.infrastructure.providers.qwen import QwenChatClient, QwenModelProvider
from ai_service.infrastructure.providers.qwen_protocol import prepare_qwen_request


def test_qwen_requires_own_endpoint_and_never_uses_cloud_keys():
    with pytest.raises(ValueError, match="QWEN_BASE_URL"):
        build_model_provider(Settings(_env_file=None, provider="qwen"))
    provider = build_model_provider(
        Settings(
            _env_file=None,
            provider="qwen",
            qwen_base_url="http://localhost:8001/v1",
        )
    )
    assert isinstance(provider, QwenModelProvider) and provider.name == "qwen"


def test_structured_schema_is_added_once_without_mutating_canonical_input():
    request = {
        "messages": [
            {"role": "system", "content": "Extract"},
            {"role": "user", "content": "RAM 32GB"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Draft", "schema": Draft.model_json_schema()},
        },
    }
    before = json.dumps(request)
    prepared = prepare_qwen_request(request)
    assert "response_format" not in prepared
    assert "Draft" in prepared["messages"][0]["content"]
    assert json.dumps(request) == before


@pytest.mark.asyncio
async def test_native_tools_and_structured_heads_route_without_network():
    class Auxiliary(BaseModel):
        selected: bool

    requests = []

    async def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "s1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"GPU"}',
                        },
                    }
                ],
            }
        else:
            message = {
                "role": "assistant",
                "content": '{"items":[],"tasks":[]}'
                if body["model"] == "gearpc-adapter"
                else '{"selected":true}',
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"total_tokens": 20},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = QwenChatClient("http://qwen.test/v1", client=http)
        provider = QwenModelProvider(
            Settings(
                _env_file=None,
                qwen_base_url="http://qwen.test/v1",
                qwen_model_name="gearpc-adapter",
            ),
            client=client,
        )
        model = provider.create_model()
        tool = await model.bind_tools(
            [{"name": "search_web", "parameters": {"type": "object"}}]
        ).ainvoke([HumanMessage("GPU")])
        assert tool.tool_calls[0]["name"] == "search_web"
        assert requests[-1]["parallel_tool_calls"] is False
        assert requests[-1]["chat_template_kwargs"] == {"enable_thinking": False}
        assert await model.with_structured_output(Draft, method="json_schema").ainvoke(
            [("human", "RAM")]
        ) == Draft(items=[])
        await model.with_structured_output(Auxiliary, method="json_schema").ainvoke(
            [("human", "choose")]
        )
        assert requests[-1]["model"] == "Qwen/Qwen3-8B"
        assert requests[-2]["model"] == "gearpc-adapter"
        await provider.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            provider.create_model()


@pytest.mark.asyncio
async def test_truncation_is_failure_and_no_automatic_retry():
    attempts = []

    async def respond(request):
        attempts.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "length",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = QwenChatClient("http://qwen.test/v1", client=http)
        with pytest.raises(ValueError, match="TRUNCATED"):
            await client.complete(
                {"messages": [{"role": "user", "content": "hello"}]},
                model="Qwen/Qwen3-8B",
            )
    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_qwen_plugs_into_existing_native_react_adapter():
    from ai_service.capabilities.assistant.react.contracts import ToolName
    from ai_service.infrastructure.providers.react_model import NativeFlexibleModel

    async def respond(request):
        payload = json.loads(request.content)
        assert [tool["function"]["name"] for tool in payload["tools"]] == ["search_web"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "s1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_web",
                                        "arguments": '{"query":"GPU"}',
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        provider = QwenModelProvider(
            Settings(_env_file=None, qwen_base_url="http://qwen.test/v1"),
            client=QwenChatClient("http://qwen.test/v1", client=http),
        )
        decision = await NativeFlexibleModel(provider).decide(
            "GPU", [], (ToolName.SEARCH_WEB,), None
        )
        assert decision.action.kind == "CALL_TOOL"
        assert decision.action.call.arguments == {"query": "GPU"}
