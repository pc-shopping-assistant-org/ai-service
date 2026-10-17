import json

import pytest
from test_training_dataset import dataset, sample

from ai_service.training.dataset import DatasetSplit, load_dataset
from ai_service.training.evaluation import EvaluationRun, evaluate_responses
from ai_service.training.export import build_example


def run_config():
    # Test-only metadata, not an actual model baseline.
    return EvaluationRun(
        model_id="scripted-test",
        model_revision="fixture-v1",
        code_revision="test-only",
        decoding={"temperature": 0},
    )


def response_for(loaded, index=0):
    record = loaded.samples[index]
    example = build_example(record, loaded.snapshots[record.snapshot_id])
    return {
        "sample_id": record.sample_id,
        "request_hash": example["request_hash"],
        "response": example["target"],
        "latency_ms": 12,
    }


def reviewed(**changes):
    return sample(
        split="validation", review_status="APPROVED", reviewed_by="reviewer", **changes
    )


def test_evaluation_keeps_missing_and_invalid_in_denominator(tmp_path):
    loaded = load_dataset(
        dataset(
            tmp_path,
            [reviewed(sample_id=f"s-{i}", scenario_id=f"family-{i}") for i in range(3)],
        ),
        require_approved=True,
    )
    valid = response_for(loaded)
    invalid = response_for(loaded, 1)
    # JSON CALL_TOOL is a schema-valid decision but invalid provider protocol.
    invalid["response"] = {
        "role": "assistant",
        "content": loaded.samples[1].target.model_dump_json(),
    }
    report = evaluate_responses(loaded, [valid, invalid], run_config())
    assert report["total"] == 3
    assert report["raw_contract_valid"] == 1 and report["validated"] == 1
    assert [row["status"] for row in report["cases"]] == ["PASS", "FAIL", "MISSING"]
    assert report["semantic_review"] == "REQUIRED"


def test_alternate_allowed_action_is_not_marked_wrong_only_for_gold_mismatch(tmp_path):
    loaded = load_dataset(dataset(tmp_path, [reviewed()]), require_approved=True)
    output = response_for(loaded)
    output["response"] = {
        "role": "assistant",
        "content": json.dumps(
            {
                "schema_version": 2,
                "action": {
                    "kind": "ASK_CLARIFICATION",
                    "question": "Bạn đang dùng hệ điều hành nào?",
                    "missing_fields": ["operating_system"],
                },
            }
        ),
    }
    report = evaluate_responses(loaded, [output], run_config())
    assert report["validated"] == 1
    assert report["cases"][0]["matches_authored_target"] is False
    # Whether clarification is necessary is a reviewer task, not PASS semantics.
    assert report["semantic_review"] == "REQUIRED"


@pytest.mark.parametrize(
    "change,code",
    [
        ({"request_hash": "0" * 64}, "REQUEST_HASH_MISMATCH"),
        ({"sample_id": "not-in-dataset"}, "UNKNOWN_PREDICTION"),
    ],
)
def test_captured_response_must_bind_exact_case_and_prompt(tmp_path, change, code):
    loaded = load_dataset(dataset(tmp_path, [reviewed()]), require_approved=True)
    with pytest.raises(ValueError, match=code):
        evaluate_responses(loaded, [{**response_for(loaded), **change}], run_config())


def test_duplicate_predictions_are_not_silently_selected(tmp_path):
    loaded = load_dataset(dataset(tmp_path, [reviewed()]), require_approved=True)
    output = response_for(loaded)
    with pytest.raises(ValueError, match="DUPLICATE_PREDICTION"):
        evaluate_responses(loaded, [output, output], run_config())


@pytest.mark.parametrize("mode", ["multiple", "malformed_arguments", "mixed"])
def test_invalid_native_tool_protocol_is_not_scored_as_valid(tmp_path, mode):
    loaded = load_dataset(dataset(tmp_path, [reviewed()]), require_approved=True)
    output = response_for(loaded)
    if mode == "multiple":
        output["response"]["tool_calls"].append(output["response"]["tool_calls"][0])
    elif mode == "malformed_arguments":
        output["response"]["tool_calls"][0]["function"]["arguments"] = "{broken"
    else:
        output["response"]["content"] = json.dumps(
            {
                "schema_version": 2,
                "action": {
                    "kind": "ASK_CLARIFICATION",
                    "question": "Hệ điều hành nào?",
                    "missing_fields": ["operating_system"],
                },
            }
        )
    report = evaluate_responses(loaded, [output], run_config())
    assert report["raw_contract_valid"] == 0 and report["validated"] == 0


def test_schema_valid_but_disallowed_build_fails_policy_validation(tmp_path):
    loaded = load_dataset(dataset(tmp_path, [reviewed()]), require_approved=True)
    output = response_for(loaded)
    output["response"]["tool_calls"][0]["function"].update(
        name="build_pc", arguments="{}"
    )
    report = evaluate_responses(loaded, [output], run_config())
    assert report["raw_contract_valid"] == 1 and report["validated"] == 0
    assert report["cases"][0]["error"] == "TARGET_TOOL_NOT_ALLOWED"


def test_test_holdout_is_not_evaluated_by_default(tmp_path):
    loaded = load_dataset(
        dataset(
            tmp_path,
            [
                reviewed(),
                sample(
                    sample_id="test-1",
                    scenario_id="test-family",
                    split="test",
                    review_status="APPROVED",
                    reviewed_by="reviewer",
                ),
            ],
        ),
        require_approved=True,
    )
    assert evaluate_responses(loaded, [], run_config())["total"] == 1
    held_out = evaluate_responses(loaded, [], run_config(), split=DatasetSplit.TEST)
    assert held_out["total"] == 1 and held_out["split"] == "test"


def test_requirement_wrong_amount_is_parsed_but_fails_source_validation(tmp_path):
    record = reviewed(
        task="REQUIREMENT_EXTRACTION",
        message="RAM 32GB",
        target={
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
    )
    loaded = load_dataset(dataset(tmp_path, [record]), require_approved=True)
    output = response_for(loaded)
    output["response"]["content"] = output["response"]["content"].replace(
        '"amount":32', '"amount":64'
    )
    report = evaluate_responses(loaded, [output], run_config())
    assert report["raw_contract_valid"] == 1 and report["validated"] == 0
    assert "QUANTITY" in report["cases"][0]["error"]


def test_cli_export_and_captured_response_report(tmp_path, monkeypatch, capsys):
    from ai_service.training.evaluation import main as evaluate_main
    from ai_service.training.export import main as export_main

    source = dataset(tmp_path / "source", [reviewed()])
    export = tmp_path / "export"
    monkeypatch.setattr("sys.argv", ["export", str(source), "--output", str(export)])
    export_main()
    assert json.loads(capsys.readouterr().out)["tokenizer_ready"] is False
    loaded = load_dataset(source, require_approved=True)
    responses = tmp_path / "responses.jsonl"
    responses.write_text(json.dumps(response_for(loaded)) + "\n")
    config = tmp_path / "run.json"
    config.write_text(run_config().model_dump_json())
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            str(source),
            "--responses",
            str(responses),
            "--run-config",
            str(config),
            "--output",
            str(output),
        ],
    )
    evaluate_main()
    report = json.loads(output.read_text())
    assert report["validated"] == report["total"] == 1
    assert report["dependency_lock_hash"] and report["responses_hash"]
    assert report["run"]["model_id"] == "scripted-test"
