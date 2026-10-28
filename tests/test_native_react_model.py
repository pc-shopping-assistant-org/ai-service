import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.actions import (
    FinalAnswerAction,
    FlexibleDecision,
)
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    ToolCall,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.providers.react_model import NativeFlexibleModel


def final(candidate: AnswerCandidate) -> str:
    return FlexibleDecision(
        action=FinalAnswerAction(answer=candidate)
    ).model_dump_json()


def native_result(candidate: AnswerCandidate) -> AIMessage:
    selection = candidate.model_dump(
        mode="json",
        exclude={"schema_version", "clarification_fields", "explanation"},
        exclude_none=True,
    )
    if candidate.explanation is not None:
        selection["explanation"] = candidate.explanation.text
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "submit",
                "name": "submit_result",
                "args": {"results": [{"task_index": 0, **selection}]},
            }
        ],
    )


class Model:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def bind_tools(self, tools, **kwargs):
        self.tools, self.options = tools, kwargs
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


class Provider:
    def __init__(self, response, name="openai"):
        self.model = Model(response)
        self.name = name

    def create_model(self):
        return self.model


@pytest.mark.asyncio
async def test_native_model_receives_frozen_saved_context_without_execution_authority():
    from test_saved_build_explanation import saved_context

    trusted = saved_context(stale=True)
    ref = trusted.saved_build.evidence_ref
    provider = Provider(
        native_result(AnswerCandidate(text="Saved build", evidence_refs=[ref]))
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Giải thích cấu hình cũ", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    instruction = provider.model.messages[0].content
    assert ref in instruction and "8000000" in instruction
    assert str(trusted.account_id) not in instruction
    assert "historical" in instruction
    assert decision.action.answer.evidence_refs == [ref]


@pytest.mark.asyncio
async def test_native_tool_response_and_state_aware_schemas():
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "search", "name": "search_web", "args": {"query": "GPU review"}},
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU review", [], (ToolName.SEARCH_WEB,), None
    )
    assert decision.action.calls[0].name == "search_web"
    tool_names = {tool["name"] for tool in provider.model.tools}
    assert "search_web" in tool_names
    assert "search_catalog" not in tool_names
    assert provider.model.options == {"parallel_tool_calls": True}
    assert isinstance(provider.model.messages[0], SystemMessage)
    assert isinstance(provider.model.messages[1], HumanMessage)


@pytest.mark.asyncio
async def test_native_transcript_projects_matched_pairs_and_parses_answer():
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    call = ToolCall(
        call_id="catalog", name="search_catalog", arguments={"query": "GPU"}
    )
    trusted = context()
    result = await dispatcher.execute(call, trusted)
    candidate = AnswerCandidate(
        text="Canonical GPU", evidence_refs=result.evidence_refs
    )
    provider = Provider(native_result(candidate), name="gemini")
    decision = await NativeFlexibleModel(provider).decide(
        "GPU",
        [ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)],
        (ToolName.SEARCH_CATALOG,),
        None,
        trusted,
    )
    assert decision.action.answer == candidate
    assert provider.model.options == {}
    assistant, tool = provider.model.messages[2:]
    assert isinstance(assistant, AIMessage) and isinstance(tool, ToolMessage)
    assert assistant.tool_calls[0]["id"] == tool.tool_call_id == call.call_id
    assert json.loads(tool.content)["evidence_refs"] == result.evidence_refs


@pytest.mark.asyncio
async def test_legacy_json_final_is_repairable_not_published():
    decision = await NativeFlexibleModel(
        Provider(AIMessage(content=final(AnswerCandidate(text="Legacy envelope"))))
    ).decide(
        "GPU",
        [],
        (ToolName.SEARCH_WEB,),
        None,
    )
    assert decision is None


@pytest.mark.asyncio
async def test_native_multi_call_is_preserved_as_a_batch():
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "one", "name": "search_web", "args": {"query": "GPU"}},
                {
                    "id": "two",
                    "name": "read_web_page",
                    "args": {"url": "https://example.com"},
                },
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU",
        [],
        (ToolName.SEARCH_WEB, ToolName.READ_WEB_PAGE),
        None,
    )
    assert [call.call_id for call in decision.action.calls] == ["one", "two"]


@pytest.mark.asyncio
async def test_gemini_signature_survives_canonical_tool_record_serialization():
    signature_key = "__gemini_function_call_thought_signatures__"
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {"id": "search", "name": "search_catalog", "args": {"query": "GPU"}}
            ],
            additional_kwargs={signature_key: {"search": "c2lnbmF0dXJl"}},
        ),
        name="gemini",
    )
    adapter = NativeFlexibleModel(provider)
    decision = await adapter.decide("GPU", [], (ToolName.SEARCH_CATALOG,), None)
    assert decision.action.calls[0].provider_signature == "c2lnbmF0dXJl"
    trusted = context()
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        decision.action.calls[0], trusted
    )
    record = ToolRecord.model_validate_json(
        ToolRecord(
            call=decision.action.calls[0], result=result, origin_run_id=trusted.run_id
        ).model_dump_json()
    )
    provider.model.response = native_result(
        AnswerCandidate(text="GPU", evidence_refs=result.evidence_refs)
    )
    await adapter.decide("GPU", [record], (ToolName.SEARCH_CATALOG,), None, trusted)
    assert provider.model.messages[2].additional_kwargs[signature_key] == {
        "search": "c2lnbmF0dXJl"
    }
