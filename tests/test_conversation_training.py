"""Training targets must use the same text/native-control protocol as serving."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import convert_to_messages
from test_react_tools import context
from test_training_dataset import dataset, sample
from test_training_evaluation import run_config

from ai_service.capabilities.assistant.react.actions import CallToolAction, action_calls
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.tools import allowed_tools
from ai_service.infrastructure.providers.react_model import (
    build_react_request,
    parse_flexible_response,
)
from ai_service.training.dataset import (
    ReactDecisionSample,
    load_dataset,
    validate_label,
)
from ai_service.training.evaluation import evaluate_responses
from ai_service.training.export import build_example

PILOT = Path(__file__).parents[1] / "training" / "pilot-v1"


def parse_export(example, snapshot):
    message = convert_to_messages([example["target"]])[0]
    source = next(
        (
            part["content"]
            for part in example["request"]["messages"]
            if part["role"] == "user"
        ),
        "",
    )
    if source.startswith("{"):
        source = json.loads(source).get("user_message", source)
    return parse_flexible_response(
        message,
        allowed_tools(snapshot.context, snapshot.records),
        message=source,
        context=snapshot.context,
        transcript=snapshot.records,
    )


def test_response_exports_plain_text_and_evaluates_with_live_protocol(tmp_path):
    text = "Chào bạn! Bạn đang tìm linh kiện hay muốn tư vấn cấu hình?"
    authored = sample(
        split="validation",
        review_status="APPROVED",
        reviewed_by="test-reviewer",
        message="Chào bạn",
        target={"action": {"kind": "RESPOND", "text": text}},
    )
    loaded = load_dataset(dataset(tmp_path, [authored]), require_approved=True)
    record = loaded.samples[0]
    snapshot = loaded.snapshots[record.snapshot_id]
    example = build_example(record, snapshot)
    assert example["target"] == {"role": "assistant", "content": text}
    parsed = parse_export(example, snapshot)
    assert parsed is not None and parsed.action == record.target.action
    report = evaluate_responses(
        loaded,
        [
            {
                "sample_id": record.sample_id,
                "request_hash": example["request_hash"],
                "response": example["target"],
            }
        ],
        run_config(),
    )
    assert report["validated"] == 1
    assert report["cases"][0]["matches_authored_target"] is True


@pytest.mark.parametrize("kind", ["ASK_CLARIFICATION", "PROPOSE_REQUIREMENTS"])
def test_control_targets_export_native_calls_matching_serving_schema(tmp_path, kind):
    trusted = context()
    trusted.completion_requirements = [
        CompletionRequirement(kind="BUILD", constraints_revision=0)
    ]
    action = (
        {
            "kind": kind,
            "question": "Bạn có ngân sách bao nhiêu?",
            "missing_fields": ["BUDGET_REQUIRED"],
        }
        if kind == "ASK_CLARIFICATION"
        else {
            "kind": kind,
            "source_text": "RAM 32GB",
            "draft": {
                "items": [
                    {
                        "subject": "RAM",
                        "predicate": "capacity",
                        "start": 0,
                        "end": 8,
                        "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
                    }
                ]
            },
        }
    )
    loaded = load_dataset(
        dataset(
            tmp_path,
            [sample(message="Build PC", target={"action": action})],
            snapshot={"context": trusted.model_dump(mode="json"), "records": []},
        )
    )
    record = loaded.samples[0]
    snapshot = loaded.snapshots[record.snapshot_id]
    example = build_example(record, snapshot)
    target = example["target"]
    assert not target.get("content")
    assert len(target["tool_calls"]) == 1
    function = target["tool_calls"][0]["function"]
    assert function["name"] == kind.lower()
    arguments = json.loads(function["arguments"])
    if kind == "PROPOSE_REQUIREMENTS":
        assert arguments["draft"]["items"][0]["quote"] == "RAM 32GB"
        assert "start" not in arguments["draft"]["items"][0]
        assert "end" not in arguments["draft"]["items"][0]
    else:
        assert arguments == {
            key: value
            for key, value in record.target.action.model_dump(mode="json").items()
            if key != "kind"
        }
    _, serving_schemas = build_react_request(
        record.message,
        snapshot.records,
        allowed_tools(snapshot.context, snapshot.records),
        None,
        snapshot.context,
    )
    assert [
        entry["function"] for entry in example["request"]["tools"]
    ] == serving_schemas
    parsed = parse_export(example, snapshot)
    assert parsed is not None and parsed.action == record.target.action


def test_existing_final_label_exports_submit_result_and_roundtrips_canonical_action():
    loaded = load_dataset(PILOT)
    record = next(item for item in loaded.samples if item.sample_id == "react-23")
    snapshot = loaded.snapshots[record.snapshot_id]
    original_label = record.target.model_dump(mode="json")
    original_snapshot = snapshot.model_dump(mode="json")
    example = build_example(record, snapshot)
    target = example["target"]
    assert not target.get("content")
    assert len(target["tool_calls"]) == 1
    function = target["tool_calls"][0]["function"]
    assert function["name"] == "submit_result"
    payload = json.loads(function["arguments"])
    assert len(payload["results"]) == 1
    assert "outcome" not in payload["results"][0]
    assert "status" not in payload["results"][0]
    parsed = parse_export(example, snapshot)
    assert parsed is not None
    assert parsed.action == record.target.action
    assert parsed.model_dump(exclude={"control_call"}) == record.target.model_dump(
        exclude={"control_call"}
    )
    assert record.target.model_dump(mode="json") == original_label
    assert snapshot.model_dump(mode="json") == original_snapshot


def test_all_pilot_react_labels_roundtrip_without_rewriting_reviewed_data():
    loaded = load_dataset(PILOT)
    checked = 0
    needs_review = set()
    for record in loaded.samples:
        if not isinstance(record, ReactDecisionSample):
            continue
        snapshot = loaded.snapshots[record.snapshot_id]
        original_label = record.model_dump(mode="json")
        original_snapshot = snapshot.model_dump(mode="json")
        try:
            example = build_example(record, snapshot)
        except ValueError as error:
            assert str(error) == f"TARGET_ACTION_NOT_AVAILABLE: {record.sample_id}"
            assert record.review_status == "DRAFT"
            needs_review.add(record.sample_id)
            continue
        parsed = parse_export(example, snapshot)
        assert parsed is not None, record.sample_id
        validate_label(record.model_copy(update={"target": parsed}), snapshot)
        if isinstance(record.target.action, CallToolAction):
            assert action_calls(parsed.action) == [record.target.action.call], (
                record.sample_id
            )
        else:
            assert parsed.action == record.target.action, record.sample_id
        assert record.model_dump(mode="json") == original_label
        assert snapshot.model_dump(mode="json") == original_snapshot
        checked += 1
    assert checked > 0
    # Clarifications are native controls; application validation still checks fields.
    assert needs_review == set()
