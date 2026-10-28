import json
from pathlib import Path

import pytest
from test_native_react_model import Provider
from test_training_dataset import dataset, sample

from ai_service.capabilities.assistant.react.tools import allowed_tools
from ai_service.infrastructure.providers.react_model import NativeFlexibleModel
from ai_service.training.dataset import load_dataset
from ai_service.training.export import build_example, export_dataset

PILOT = Path(__file__).parents[1] / "training" / "pilot-v1"


def test_export_refuses_unreviewed_data_without_writing_artifacts(tmp_path):
    with pytest.raises(ValueError, match="DATASET_REVIEW_REQUIRED"):
        export_dataset(PILOT, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_keeps_splits_separate_and_metadata_out_of_model_input(tmp_path):
    approved = [
        sample(
            sample_id=f"driver-{split}",
            scenario_id=f"family-{split}",
            split=split,
            review_status="APPROVED",
            reviewed_by="test-reviewer",
        )
        for split in ("train", "validation", "test")
    ]
    source = dataset(tmp_path / "source", approved)
    output = tmp_path / "export"
    report = export_dataset(source, output)
    assert report["splits"] == {"train": 1, "validation": 1, "test": 1}
    assert not (output / "test.jsonl").exists()
    training = json.loads((output / "train.jsonl").read_text())
    target = training["messages"][-1]
    assert target["role"] == "assistant" and not target.get("content")
    assert target["tool_calls"][0]["function"]["name"] == "research_driver"
    assert "CALL_TOOL" not in target["tool_calls"][0]["function"]["arguments"]
    assert "driver-test" not in (output / "train.jsonl").read_text()
    requests = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 3
    assert all(
        "target" not in row and "expected_behavior" not in row for row in requests
    )
    assert all("reviewer" not in json.dumps(row["request"]) for row in requests)
    assert all(row["request_hash"] for row in requests)


@pytest.mark.asyncio
async def test_export_react_prompt_is_runtime_prompt_with_canonical_tool_pairs():
    from langchain_core.messages import AIMessage, convert_to_openai_messages

    loaded = load_dataset(PILOT)
    record = next(item for item in loaded.samples if item.sample_id == "react-23")
    snapshot = loaded.snapshots[record.snapshot_id]
    example = build_example(record, snapshot)
    provider = Provider(AIMessage(content=record.target.model_dump_json()))
    await NativeFlexibleModel(provider).decide(
        record.message,
        snapshot.records,
        allowed_tools(snapshot.context, snapshot.records),
        None,
        snapshot.context,
    )
    assert example["request"]["messages"] == convert_to_openai_messages(
        provider.model.messages
    )
    assert [tool["function"]["name"] for tool in example["request"]["tools"]] == [
        tool["name"] for tool in provider.model.tools
    ]
    messages = example["request"]["messages"]
    assert messages[-2]["tool_calls"][0]["id"] == messages[-1]["tool_call_id"]
    assert messages[-1]["tool_call_id"] == "pilot-build_pc"
    assert str(snapshot.context.account_id) not in json.dumps(messages)


@pytest.mark.asyncio
async def test_export_structured_prompt_schema_matches_runtime():
    from test_native_requirement_extractor import Budget
    from test_native_requirement_extractor import Provider as DraftProvider

    from ai_service.infrastructure.providers.requirement_extractor import (
        NativeRequirementExtractor,
    )
    from ai_service.infrastructure.providers.task_extractor import NativeTaskExtractor

    loaded = load_dataset(PILOT)
    for task in ("TASK_EXTRACTION", "REQUIREMENT_EXTRACTION"):
        record = next(item for item in loaded.samples if item.task == task)
        snapshot = loaded.snapshots[record.snapshot_id]
        example = build_example(record, snapshot)
        provider = DraftProvider(json.loads(example["target"]["content"]))
        if task == "TASK_EXTRACTION":
            extractor = NativeTaskExtractor(
                provider, Budget(), context=snapshot.context
            )
            await extractor.extract_tasks(record.message)
        else:
            extractor = NativeRequirementExtractor(provider, Budget())
            await extractor.extract(record.message, snapshot.context.requirements)
        assert example["request"]["messages"] == [
            {"role": "system" if role == "system" else "user", "content": content}
            for role, content in provider.model.messages
        ]
        assert (
            example["request"]["response_format"]["json_schema"]["schema"]
            == provider.model.schema.model_json_schema()
        )
        assert "start" not in example["target"]["content"]
        assert "end" not in example["target"]["content"]


@pytest.mark.asyncio
async def test_tool_batch_export_matches_native_agent_protocol(tmp_path):
    from langchain_core.messages import convert_to_messages
    from test_react_tools import context

    from ai_service.training.dataset import validate_label

    row = sample()
    call = row["target"]["action"]["call"]
    row["target"]["action"] = {
        "kind": "CALL_TOOLS",
        "calls": [
            call,
            {
                **call,
                "call_id": "driver-2",
                "arguments": {**call["arguments"], "component_name": "ASUS TUF B650"},
            },
        ],
    }
    trusted = context()
    source = dataset(
        tmp_path,
        [row],
        snapshot={"context": trusted.model_dump(mode="json"), "records": []},
    )
    loaded = load_dataset(source)
    record, snapshot = loaded.samples[0], loaded.snapshots["empty"]
    example = build_example(record, snapshot)
    assert example["request"]["parallel_tool_calls"] is True
    assert [call["function"]["name"] for call in example["target"]["tool_calls"]] == [
        "research_driver",
        "research_driver",
    ]
    response = convert_to_messages([example["target"]])[0]
    decision = await NativeFlexibleModel(Provider(response)).decide(
        record.message, [], allowed_tools(snapshot.context), None, snapshot.context
    )
    assert decision.action == record.target.action
    validate_label(record.model_copy(update={"target": decision}), snapshot)
