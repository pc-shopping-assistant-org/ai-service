import hashlib
import json
from pathlib import Path

import pytest
from test_react_tools import context

from ai_service.training.dataset import validate_dataset


def sample(**updates):
    return {
        "sample_id": "driver-1",
        "scenario_id": "driver-msi",
        "split": "train",
        "task": "REACT_DECISION",
        "snapshot_id": "empty",
        "message": "Tìm BIOS cho MSI MAG B650 TOMAHAWK WIFI",
        "expected_behavior": "Đọc trang support chính hãng, không tải/cài BIOS.",
        "review_status": "DRAFT",
        "target": {
            "schema_version": 2,
            "action": {
                "kind": "CALL_TOOL",
                "call": {
                    "call_id": "driver",
                    "name": "research_driver",
                    "arguments": {
                        "component_name": "MSI MAG B650 TOMAHAWK WIFI",
                        "software_type": "BIOS_UPDATE",
                    },
                },
            },
        },
        **updates,
    }


def dataset(path, samples, *, snapshot=None):
    path.mkdir(exist_ok=True)
    files = {
        "samples.jsonl": "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in samples
        ),
        "snapshots.json": json.dumps(
            {
                "empty": snapshot
                or {"context": context().model_dump(mode="json"), "records": []}
            }
        ),
    }
    for name, content in files.items():
        (path / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_id": "test-pilot-v1",
        "runtime_contract": "react-v2",
        "source": "AUTHORED_SYNTHETIC",
        "files": {
            name: hashlib.sha256(content.encode()).hexdigest()
            for name, content in files.items()
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_valid_draft_is_not_claimed_train_ready_and_release_requires_approval(tmp_path):
    directory = dataset(tmp_path, [sample()])
    report = validate_dataset(directory)
    assert report.samples == 1 and report.scenarios == 1
    assert report.train_ready is False and report.review_status == {"DRAFT": 1}
    with pytest.raises(ValueError, match="DATASET_REVIEW_REQUIRED"):
        validate_dataset(directory, require_approved=True)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"sample_id": "driver-1", "scenario_id": "another"}, "DUPLICATE_SAMPLE"),
        ({"sample_id": "driver-2", "split": "test"}, "SCENARIO_SPLIT_LEAK"),
    ],
)
def test_duplicate_and_cross_split_scenario_are_rejected(tmp_path, change, code):
    directory = dataset(tmp_path, [sample(), sample(**change)])
    with pytest.raises(ValueError, match=code):
        validate_dataset(directory)


def test_file_changes_need_a_new_manifest_hash(tmp_path):
    directory = dataset(tmp_path, [sample()])
    with (directory / "samples.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
        validate_dataset(directory)


@pytest.mark.parametrize(
    "name,args",
    [
        ("build_pc", {}),
        ("shell", {"command": "hello"}),
        ("research_driver", {"component_name": "MSI", "download": True}),
    ],
)
def test_react_labels_respect_runtime_tool_schema_and_state_policy(
    tmp_path, name, args
):
    record = sample()
    record["target"]["action"]["call"].update(name=name, arguments=args)
    with pytest.raises(ValueError):
        validate_dataset(dataset(tmp_path, [record]))


def test_requirement_span_must_bind_exact_current_message(tmp_path):
    record = sample(
        task="REQUIREMENT_EXTRACTION",
        message="RAM 32GB",
        target={
            "items": [
                {
                    "subject": "RAM",
                    "predicate": "capacity",
                    "start": 0,
                    "end": 99,
                    "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="SOURCE_SPAN_INVALID"):
        validate_dataset(dataset(tmp_path, [record]))


def test_requirement_amount_cannot_disagree_with_its_source(tmp_path):
    record = sample(
        task="REQUIREMENT_EXTRACTION",
        message="RAM 32GB",
        target={
            "items": [
                {
                    "subject": "RAM",
                    "predicate": "capacity",
                    "start": 0,
                    "end": 8,
                    "value": {"kind": "QUANTITY", "amount": 64, "unit": "GB"},
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="QUANTITY"):
        validate_dataset(dataset(tmp_path, [record]))


def test_snapshot_constraints_cannot_carry_conflicting_quantity_evidence(tmp_path):
    from test_requirement_pipeline import state

    trusted = context(requirements=state())
    snapshot = {"context": trusted.model_dump(mode="json"), "records": []}
    entry = next(iter(snapshot["context"]["requirements"]["requirements"].values()))
    entry["value"]["amount"] = 99_000_000
    with pytest.raises(ValueError, match="QUANTITY"):
        validate_dataset(dataset(tmp_path, [sample()], snapshot=snapshot))


def test_final_label_cannot_claim_build_without_optimizer_evidence(tmp_path):
    record = sample(
        message="Build PC 25 triệu",
        target={
            "schema_version": 2,
            "action": {
                "kind": "FINAL_ANSWER",
                "outcome": "BUILT",
                "answer": {"text": "Đã build xong"},
            },
        },
    )
    with pytest.raises(ValueError):
        validate_dataset(dataset(tmp_path, [record]))


def test_pilot_is_versioned_authored_data_with_grouped_splits():
    directory = Path(__file__).parents[1] / "training" / "pilot-v1"
    report = validate_dataset(directory)
    assert report.samples >= 50 and report.scenarios >= 50
    assert set(report.tasks) == {
        "TASK_EXTRACTION",
        "REQUIREMENT_EXTRACTION",
        "REACT_DECISION",
    }
    assert set(report.splits) == {"train", "validation", "test"}
    assert report.train_ready is False
