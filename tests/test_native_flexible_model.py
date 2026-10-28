import json

import pytest
from langchain_core.messages import AIMessage
from test_native_react_model import Provider
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.actions import ActionKind, RepairFeedback
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolCall,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.providers.react_model import (
    GEMINI_SIGNATURES_KEY,
    NativeFlexibleModel,
)


def clarification():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "id": "clarify",
                "name": "ask_clarification",
                "args": {
                    "question": "Ngân sách bao nhiêu?",
                    "missing_fields": ["budget"],
                },
            }
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["openai", "gemini"])
async def test_multi_task_final_is_typed_provider_proposal_not_completion_authority(
    name,
):
    from test_flexible_outcomes import fact_result

    trusted = context()
    trusted.completion_requirements = [
        CompletionRequirement(kind="ANSWER"),
        CompletionRequirement(kind="ANSWER"),
    ]
    call = ToolCall(
        call_id="catalog", name="search_catalog", arguments={"query": "GPU"}
    )
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        call, trusted
    )
    page = fact_result(trusted, "UNKNOWN")
    records = [
        ToolRecord(call=call, result=result, origin_run_id=trusted.run_id),
        ToolRecord(
            call=ToolCall(
                call_id="page",
                name="read_web_page",
                arguments={"url": page.result.url},
            ),
            result=page,
            origin_run_id=trusted.run_id,
        ),
    ]
    payload = {
        "results": [
            {
                "task_index": 0,
                "text": "Typed candidate",
                "evidence_refs": result.evidence_refs,
            },
            {
                "task_index": 1,
                "text": "Missing fact",
                "supporting_tool_call_ids": ["page"],
                "evidence_refs": page.evidence_refs,
            },
        ]
    }
    decision = await NativeFlexibleModel(
        Provider(
            AIMessage(
                content="",
                tool_calls=[{"id": "submit", "name": "submit_result", "args": payload}],
            ),
            name=name,
        )
    ).decide("GPU và kích thước", records, (), None, trusted)
    assert decision.action.outcome == "MULTI_TASK"
    assert [part.task_index for part in decision.action.task_answers] == [0, 1]
    assert [part.outcome for part in decision.action.task_answers] == [
        "ANSWERED",
        "INSUFFICIENT_EVIDENCE",
    ]
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
    assert decision.action.kind == "CALL_TOOLS"
    assert decision.action.calls[0].provider_signature == (
        "opaque-signature" if name == "gemini" else None
    )
    tool_names = {tool["name"] for tool in provider.model.tools}
    assert "search_web" in tool_names and "search_catalog" not in tool_names
    assert provider.model.options == (
        {"parallel_tool_calls": True} if name == "openai" else {}
    )


@pytest.mark.asyncio
async def test_clarification_without_tools_and_structured_repair():
    provider = Provider(clarification())
    feedback = RepairFeedback(
        code="TASK_INCOMPLETE",
        missing_fields=["budget"],
        allowed_next_actions=[ActionKind.ASK_CLARIFICATION],
    )
    decision = await NativeFlexibleModel(provider).decide("Build PC", [], (), feedback)
    assert decision.action.question == "Ngân sách bao nhiêu?"
    assert [tool["name"] for tool in provider.model.tools] == ["ask_clarification"]
    assert feedback.model_dump_json() in provider.model.messages[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "{invalid json",
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
async def test_batch_calls_reach_application_policy_but_invalid_protocol_needs_repair(
    mode,
):
    calls = [{"id": "s1", "name": "search_web", "args": {"query": "GPU"}}]
    if mode == "multiple":
        calls.append({"id": "s2", "name": "search_web", "args": {"query": "CPU"}})
    provider = Provider(
        AIMessage(
            content=json.dumps(
                {
                    "schema_version": 2,
                    "action": {
                        "kind": "ASK_CLARIFICATION",
                        "question": "Budget?",
                        "missing_fields": ["budget"],
                    },
                }
            )
            if mode == "mixed"
            else "",
            tool_calls=calls,
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU", [], () if mode == "unavailable" else (ToolName.SEARCH_WEB,), None
    )
    if mode == "mixed":
        assert decision is None
    else:
        assert [call.call_id for call in decision.action.calls] == [
            call["id"] for call in calls
        ]


@pytest.mark.asyncio
async def test_legacy_model_selected_outcome_is_rejected_by_live_protocol():
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
    assert decision is None


@pytest.mark.asyncio
async def test_requirement_proposal_is_typed_without_mutating_trusted_context():
    trusted = context()
    before = trusted.model_dump(mode="json")
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "propose",
                    "name": "propose_requirements",
                    "args": {
                        "source_text": "RAM 32GB",
                        "draft": {
                            "items": [
                                {
                                    "subject": "RAM",
                                    "predicate": "capacity",
                                    "quote": "RAM 32GB",
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
            ],
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
    provider = Provider(clarification())
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
    provider = Provider(clarification(), name="gemini")
    await NativeFlexibleModel(provider).decide("GPU", [record], (), None)
    ai, tool = provider.model.messages[-2:]
    assert ai.additional_kwargs[GEMINI_SIGNATURES_KEY] == {"c1": "opaque"}
    assert isinstance(tool, ToolMessage) and tool.tool_call_id == "c1"
    assert tool.content == result.model_dump_json()
