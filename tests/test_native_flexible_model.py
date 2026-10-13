import json

import pytest
from langchain_core.messages import AIMessage
from test_native_react_model import Provider

from ai_service.capabilities.assistant.react.actions import ActionKind, RepairFeedback
from ai_service.capabilities.assistant.react.contracts import ToolName
from ai_service.infrastructure.providers.react_model import (
    GEMINI_SIGNATURES_KEY,
    NativeFlexibleModel,
)


def clarification():
    return {
        "schema_version": 2,
        "action": {
            "kind": "ASK_CLARIFICATION",
            "question": "Ngân sách bao nhiêu?",
            "missing_fields": ["budget"],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "gemini"])
async def test_multi_task_final_is_typed_provider_proposal_not_completion_authority(
    name,
):
    payload = {
        "schema_version": 2,
        "action": {
            "kind": "FINAL_ANSWER",
            "outcome": "MULTI_TASK",
            "answer": {"text": "Ignored aggregate"},
            "task_answers": [
                {
                    "task_index": 0,
                    "outcome": "ANSWERED",
                    "answer": {
                        "text": "Typed candidate",
                        "evidence_refs": ["catalog:gpu"],
                    },
                },
                {
                    "task_index": 1,
                    "outcome": "INSUFFICIENT_EVIDENCE",
                    "supporting_tool_call_ids": ["page"],
                    "answer": {
                        "text": "Missing fact",
                        "evidence_refs": ["fact:unknown"],
                    },
                },
            ],
        },
    }
    decision = await NativeFlexibleModel(
        Provider(AIMessage(content=json.dumps(payload)), name=name)
    ).decide("GPU và kích thước", [], (), None)
    assert decision.action.outcome == "MULTI_TASK"
    assert [part.task_index for part in decision.action.task_answers] == [0, 1]
    assert not hasattr(decision.action.task_answers[0], "status")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "gemini"])
async def test_native_call_maps_to_one_v2_action_with_provider_signature(name):
    provider = Provider(
        AIMessage(
            content="Let me search",
            tool_calls=[{"id": "s1", "name": "search_web", "args": {"query": "GPU"}}],
            additional_kwargs={GEMINI_SIGNATURES_KEY: {"s1": "opaque-signature"}},
        ),
        name=name,
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU", [], (ToolName.SEARCH_WEB,), None
    )
    assert decision.action.kind == "CALL_TOOL"
    assert decision.action.call.provider_signature == (
        "opaque-signature" if name == "gemini" else None
    )
    assert [tool["name"] for tool in provider.model.tools] == ["search_web"]
    assert provider.model.options == (
        {"parallel_tool_calls": False} if name == "openai" else {}
    )


@pytest.mark.asyncio
async def test_clarification_without_tools_and_structured_repair():
    provider = Provider(AIMessage(content=json.dumps(clarification())))
    feedback = RepairFeedback(
        code="TASK_INCOMPLETE",
        missing_fields=["budget"],
        allowed_next_actions=[ActionKind.ASK_CLARIFICATION],
    )
    decision = await NativeFlexibleModel(provider).decide("Build PC", [], (), feedback)
    assert decision.action.question == "Ngân sách bao nhiêu?"
    assert not hasattr(provider.model, "tools")
    assert feedback.model_dump_json() in provider.model.messages[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"text":"old V1 answer"}',
        json.dumps(
            {
                "schema_version": 2,
                "action": {
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": "s1",
                        "name": "search_web",
                        "arguments": {"query": "GPU"},
                    },
                },
            }
        ),
    ],
)
async def test_invalid_or_json_tool_action_needs_repair(content):
    decision = await NativeFlexibleModel(Provider(AIMessage(content=content))).decide(
        "GPU", [], (ToolName.SEARCH_WEB,), None
    )
    assert decision is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["multiple", "mixed", "unavailable"])
async def test_rejects_entire_ambiguous_or_unavailable_native_response(mode):
    calls = [{"id": "s1", "name": "search_web", "args": {"query": "GPU"}}]
    if mode == "multiple":
        calls.append({"id": "s2", "name": "search_web", "args": {"query": "CPU"}})
    provider = Provider(
        AIMessage(
            content=json.dumps(clarification()) if mode == "mixed" else "",
            tool_calls=calls,
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU", [], () if mode == "unavailable" else (ToolName.SEARCH_WEB,), None
    )
    assert decision is None


@pytest.mark.asyncio
async def test_final_answer_is_only_an_outcome_proposal():
    provider = Provider(
        AIMessage(
            content=json.dumps(
                {
                    "schema_version": 2,
                    "action": {
                        "kind": "FINAL_ANSWER",
                        "outcome": "SEARCH_INCOMPLETE",
                        "answer": {"text": "Chưa tìm được", "evidence_refs": []},
                    },
                }
            )
        )
    )
    decision = await NativeFlexibleModel(provider).decide("Build PC", [], (), None)
    assert decision.action.outcome == "SEARCH_INCOMPLETE"


@pytest.mark.asyncio
async def test_requirement_proposal_is_typed_without_mutating_trusted_context():
    from test_react_tools import context

    trusted = context()
    before = trusted.model_dump(mode="json")
    provider = Provider(
        AIMessage(
            content=json.dumps(
                {
                    "schema_version": 2,
                    "action": {
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
                    },
                }
            )
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Tư vấn", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision.action.draft.items[0].value.amount == 32
    assert trusted.model_dump(mode="json") == before
    assert str(trusted.account_id) not in provider.model.messages[0].content


@pytest.mark.asyncio
async def test_repair_restricts_action_and_network_failure_is_not_a_fake_answer():
    provider = Provider(AIMessage(content=json.dumps(clarification())))
    feedback = RepairFeedback(
        code="MISSING_EVIDENCE", allowed_next_actions=[ActionKind.CALL_TOOL]
    )
    assert await NativeFlexibleModel(provider).decide("GPU", [], (), feedback) is None

    async def unavailable(messages):
        raise ConnectionError("Provider unavailable")

    provider.model.ainvoke = unavailable
    with pytest.raises(ConnectionError):
        await NativeFlexibleModel(provider).decide("GPU", [], (), None)


@pytest.mark.asyncio
async def test_v2_uses_canonical_gemini_tool_pairs_after_serialization():
    from langchain_core.messages import ToolMessage
    from test_react_tools import Catalog, context

    from ai_service.capabilities.assistant.react.contracts import ToolCall
    from ai_service.capabilities.assistant.react.tools import ToolDispatcher
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )
    from ai_service.infrastructure.graph.react import ToolRecord

    call = ToolCall(
        call_id="c1",
        name="search_catalog",
        arguments={"query": "GPU"},
        provider_signature="opaque",
    )
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        call, context()
    )
    record = ToolRecord.model_validate_json(
        ToolRecord(call=call, result=result).model_dump_json()
    )
    provider = Provider(AIMessage(content=json.dumps(clarification())), name="gemini")
    await NativeFlexibleModel(provider).decide("GPU", [record], (), None)
    ai, tool = provider.model.messages[-2:]
    assert ai.additional_kwargs[GEMINI_SIGNATURES_KEY] == {"c1": "opaque"}
    assert isinstance(tool, ToolMessage) and tool.tool_call_id == "c1"
    assert tool.content == result.model_dump_json()
