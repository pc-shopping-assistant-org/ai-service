import pytest
from langchain_core.messages import AIMessageChunk
from test_native_react_model import Model, Provider

from ai_service.infrastructure.providers.model_stream import public_text
from ai_service.infrastructure.providers.react_model import NativeFlexibleModel


@pytest.mark.asyncio
async def test_model_emits_public_text_before_provider_finishes():
    observed = []

    class StreamingModel(Model):
        async def ainvoke(self, messages):
            raise AssertionError("Streaming must consume provider chunks")

        async def astream(self, messages):
            yield AIMessageChunk(content="Xin chào")
            assert observed == [("Xin chào", True)]
            yield AIMessageChunk(content=" bạn.")

    async def sink(text, replace):
        observed.append((text, replace))

    provider = Provider(None)
    provider.model = StreamingModel(None)
    decision = await NativeFlexibleModel(provider, text_sink=sink).decide(
        "Chào", [], (), None
    )
    assert decision.action.text == "Xin chào bạn."
    assert "".join(text for text, _ in observed) == "Xin chào bạn."


@pytest.mark.asyncio
async def test_tool_arguments_and_reasoning_never_leak_into_draft_text():
    observed = []

    class StreamingModel(Model):
        async def astream(self, messages):
            yield AIMessageChunk(content=[{"type": "thinking", "thinking": "PRIVATE"}])
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "search_catalog",
                        "args": '{"query":"GPU"}',
                        "id": "call1",
                        "index": 0,
                    }
                ],
            )

    async def sink(text, replace):
        observed.append(text)

    from ai_service.capabilities.assistant.react.contracts import ToolName

    provider = Provider(None)
    provider.model = StreamingModel(None)
    decision = await NativeFlexibleModel(provider, text_sink=sink).decide(
        "GPU", [], (ToolName.SEARCH_CATALOG,), None
    )
    assert decision.action.calls[0].name == "search_catalog"
    assert observed == []


def test_partial_submit_result_streams_explanation_without_private_references():
    chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "submit_result",
                "id": "answer",
                "index": 0,
                "args": '{"results":[{"refs":["PRIVATE-REF"],"text":"Summary","explanation":"Đang giải thích',
            }
        ],
    )
    assert public_text(chunk) == "Đang giải thích"
