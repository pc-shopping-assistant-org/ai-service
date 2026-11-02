"""Model interaction is text/native calls; application owns business outcomes."""

import json

import pytest
from langchain_core.messages import AIMessage
from test_native_react_model import Provider
from test_react_tools import Catalog, context

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
    NativeFlexibleModel,
    build_react_request,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
async def test_plain_text_becomes_response_without_fabricating_business_outcome(
    provider_name,
):
    text = "Chào bạn! Bạn cần tư vấn cấu hình hay chọn linh kiện?"
    provider = Provider(AIMessage(content=text), name=provider_name)
    decision = await NativeFlexibleModel(provider).decide(
        "Chào bạn", [], (ToolName.SEARCH_CATALOG,), None, context()
    )
    assert decision is not None
    assert decision.action.kind == "RESPOND"
    assert decision.action.text == text
    assert not hasattr(decision.action, "outcome")
    assert not hasattr(decision.action, "task_outcomes")


def test_model_request_exposes_small_native_controls_without_decision_envelope():
    from test_saved_build_explanation import saved_context

    trusted = saved_context()
    trusted.completion_requirements = [CompletionRequirement(kind="ANSWER")]
    messages, schemas = build_react_request(
        "GPU?", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    system = messages[0].content
    assert "FlexibleDecision" not in system
    assert '"discriminator"' not in system
    by_name = {schema["name"]: schema for schema in schemas}
    assert {
        "search_catalog",
        "propose_requirements",
        "ask_clarification",
        "submit_result",
    } <= by_name.keys()
    for name in ("propose_requirements", "ask_clarification", "submit_result"):
        schema = json.dumps(by_name[name]["parameters"])
        assert "FlexibleDecision" not in schema
        assert '"outcome"' not in schema
        assert '"status"' not in schema
        assert "schema_version" not in by_name[name]["parameters"].get("properties", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
@pytest.mark.parametrize("control", ["ask_clarification", "propose_requirements"])
async def test_native_control_call_normalizes_to_internal_action(
    provider_name, control
):
    trusted = context()
    if control == "ask_clarification":
        trusted.completion_requirements = [
            CompletionRequirement(kind="BUILD", constraints_revision=0)
        ]
        arguments = {
            "question": "Bạn có ngân sách bao nhiêu?",
            "missing_fields": ["BUDGET_REQUIRED"],
        }
    else:
        arguments = {
            "source_text": "RAM 32GB",
            "items": [
                {
                    "subject": "RAM",
                    "predicate": "capacity",
                    "quote": "RAM 32GB",
                    "value": 32,
                    "unit": "GB",
                }
            ],
        }
    before = trusted.model_dump(mode="json")
    provider = Provider(
        AIMessage(
            content="Accompanying text is not a separate response",
            tool_calls=[{"id": "control", "name": control, "args": arguments}],
        ),
        name=provider_name,
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Tư vấn PC", [], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision is not None
    assert decision.action.kind == control.upper()
    if control == "ask_clarification":
        assert decision.action.question == arguments["question"]
        assert decision.action.missing_fields == arguments["missing_fields"]
    else:
        assert decision.action.draft.items[0].value.amount == 32
        assert decision.action.source_text == "RAM 32GB"
    assert trusted.model_dump(mode="json") == before
    assert "search_catalog" in {tool["name"] for tool in provider.model.tools}


@pytest.mark.asyncio
async def test_submit_result_selects_canonical_evidence_without_model_outcome():
    trusted, catalog = context(), Catalog()
    trusted.completion_requirements = [CompletionRequirement(kind="ANSWER")]
    call = ToolCall(
        call_id="catalog-read", name="search_catalog", arguments={"query": "GPU"}
    )
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        call, trusted
    )
    record = ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "submit",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "task_index": 0,
                                "text": "Sản phẩm hiện có trong catalog.",
                                "refs": result.evidence_refs,
                            }
                        ]
                    },
                }
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "GPU?", [record], (ToolName.SEARCH_CATALOG,), None, trusted
    )
    assert decision is not None
    assert decision.action.kind == "FINAL_ANSWER"
    assert decision.action.outcome == "ANSWERED"
    assert decision.action.answer.evidence_refs == result.evidence_refs
    assert catalog.calls == 1


@pytest.mark.asyncio
async def test_selecting_build_ref_resolves_its_canonical_refs_before_explanation(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    from ai_service.capabilities.assistant.react.actions import FlexibleDecision
    from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
    from ai_service.capabilities.assistant.react.submissions import (
        ResultSelection,
        SubmitResults,
        derive_final_action,
        evidence_requirements,
    )

    trusted = context(state(), rows(realistic_catalog))
    service = PCBuildApplicationService()
    call = ToolCall(call_id="build", name="build_pc", arguments={})
    result = await ToolDispatcher(Catalog(), service).execute(call, trusted)
    record = ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)
    selected = SubmitResults(
        results=[
            ResultSelection(
                text="Untrusted price 1 VND",
                build_ref=result.result.build_ref,
                explanation="Lựa chọn bám theo ngân sách.",
            )
        ]
    )
    action = derive_final_action(selected, [record])
    assert action.answer.evidence_refs == result.evidence_refs
    validated = validate_final_answer(
        action,
        [result],
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        service,
    )
    assert validated.outcome == "BUILT"
    assert "Untrusted price" not in validated.text

    fallback = FlexibleDecision(
        action={"kind": "RESPOND", "text": "Đây là cấu hình 1 VND"}
    )
    obligations = evidence_requirements(fallback, [record], trusted)
    assert [task.kind for task in obligations] == ["BUILD"]


@pytest.mark.asyncio
async def test_multiple_control_calls_reject_entire_provider_response():
    call = {
        "name": "ask_clarification",
        "args": {"question": "Ngân sách?", "missing_fields": ["BUDGET_REQUIRED"]},
    }
    provider = Provider(
        AIMessage(
            content="Do not publish this accompanying response",
            tool_calls=[call | {"id": "one"}, call | {"id": "two"}],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Build PC", [], (ToolName.SEARCH_CATALOG,), None, context()
    )
    assert decision is None


@pytest.mark.asyncio
async def test_grounded_side_answer_does_not_complete_the_pending_build():
    from ai_service.capabilities.assistant.react.dialogue import validate_response

    trusted, catalog = context(), Catalog()
    task = CompletionRequirement(kind="BUILD", constraints_revision=0)
    trusted.completion_requirements = [task]
    call = ToolCall(call_id="price", name="search_catalog", arguments={"query": "GPU"})
    result = await ToolDispatcher(catalog, PCBuildApplicationService()).execute(
        call, trusted
    )
    records = [ToolRecord(call=call, result=result, origin_run_id=trusted.run_id)]
    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "side-result",
                    "name": "submit_result",
                    "args": {
                        "results": [
                            {
                                "task_index": None,
                                "text": "Giá tra từ catalog.",
                                "refs": result.evidence_refs,
                            }
                        ]
                    },
                }
            ],
        )
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Giá con đó hiện tại?",
        records,
        (ToolName.SEARCH_CATALOG,),
        None,
        trusted,
    )
    assert decision.action.kind == "RESPOND"
    response = validate_response(
        decision.action, trusted, [task], records, continuing_pending=True
    )
    assert "8,000,000 VND" in response.text
    assert not hasattr(response, "outcome")
    with pytest.raises(ValueError, match="business result"):
        validate_response(decision.action, trusted, [task], records)
