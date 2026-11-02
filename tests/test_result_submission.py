import json

import pytest
from pydantic import ValidationError
from test_flexible_outcomes import fact_result
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.contracts import ToolCall, ToolRecord
from ai_service.capabilities.assistant.react.submissions import (
    ResultSelection,
    SubmitResults,
    derive_final_action,
    resolve_result_selection,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)


def test_model_result_submission_does_not_accept_outcome_or_status():
    schema = json.dumps(SubmitResults.model_json_schema())
    assert '"outcome"' not in schema and '"status"' not in schema
    with pytest.raises(ValidationError):
        SubmitResults.model_validate(
            {"results": [{"task_index": 0, "text": "done", "outcome": "BUILT"}]}
        )


@pytest.mark.asyncio
async def test_application_derives_catalog_outcome_from_canonical_selection():
    trusted = context()
    call = ToolCall(
        call_id="catalog", name="search_catalog", arguments={"query": "GPU"}
    )
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        call, trusted
    )
    submission = SubmitResults.model_validate(
        {
            "results": [
                {
                    "task_index": 0,
                    "text": "Lựa chọn để tham khảo",
                    "evidence_refs": result.evidence_refs,
                }
            ]
        }
    )
    action = derive_final_action(submission, [ToolRecord(call=call, result=result)])
    assert action.outcome == "ANSWERED"
    assert action.answer.evidence_refs == result.evidence_refs


@pytest.mark.asyncio
async def test_omitted_selection_never_chooses_between_multiple_catalog_results():
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    records = []
    for identifier in ("first", "second"):
        call = ToolCall(
            call_id=identifier, name="search_catalog", arguments={"query": "GPU"}
        )
        result = await dispatcher.execute(call, context())
        records.append(ToolRecord(call=call, result=result))
    selection, selected = resolve_result_selection(
        ResultSelection(text="Answer without an explicit choice"), records
    )
    assert selected == []
    assert selection.evidence_refs == []


@pytest.mark.parametrize("status", ["UNKNOWN", "CONFLICT"])
def test_missing_fact_outcome_is_derived_without_a_model_claim(status):
    trusted = context()
    result = fact_result(trusted, status)
    call = ToolCall(
        call_id="page", name="read_web_page", arguments={"url": result.result.url}
    )
    submission = SubmitResults.model_validate(
        {
            "results": [
                {
                    "task_index": 0,
                    "text": "Thiếu số đo",
                    "evidence_refs": result.evidence_refs,
                }
            ]
        }
    )
    action = derive_final_action(submission, [ToolRecord(call=call, result=result)])
    assert action.outcome == "INSUFFICIENT_EVIDENCE"
    assert action.supporting_tool_call_ids == ["page"]
