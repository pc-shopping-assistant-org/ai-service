"""Approved canonical messages, not tokenizer-specific tensors or a training job."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import convert_to_openai_messages
from pydantic import BaseModel

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.tools import allowed_tools
from ai_service.capabilities.assistant.source_quotes import (
    QuotedRequirementDraft,
    QuotedTaskBatch,
    project_requirement_draft,
    project_task_batch,
)
from ai_service.infrastructure.providers.react_model import (
    REACT_PROTOCOL_VERSION,
    build_react_request,
    project_model_response,
)
from ai_service.infrastructure.providers.requirement_extractor import (
    build_requirement_extraction_request,
)
from ai_service.infrastructure.providers.task_extractor import (
    build_task_extraction_request,
)
from ai_service.training.dataset import (
    DatasetSnapshot,
    DatasetSplit,
    ReactDecisionSample,
    TaskExtractionSample,
    TrainingSample,
    load_dataset,
)


def build_example(sample: TrainingSample, snapshot: DatasetSnapshot) -> dict[str, Any]:
    """No label, reviewer, split or future result may enter request."""
    request: dict[str, Any]
    if isinstance(sample, ReactDecisionSample):
        messages, schemas = build_react_request(
            sample.message,
            snapshot.records,
            allowed_tools(snapshot.context, snapshot.records),
            None,
            snapshot.context,
        )
        request = {
            "messages": convert_to_openai_messages(messages),
            "tools": [{"type": "function", "function": tool} for tool in schemas],
        }
        target = convert_to_openai_messages(
            project_model_response(sample.target, sample.message)
        )
        available = {schema["name"] for schema in schemas}
        if any(
            call["function"]["name"] not in available
            for call in target.get("tool_calls", [])
        ):
            raise ValueError(f"TARGET_ACTION_NOT_AVAILABLE: {sample.sample_id}")
    else:
        schema: type[BaseModel]
        label: BaseModel
        if isinstance(sample, TaskExtractionSample):
            prompt, payload = build_task_extraction_request(
                sample.message, snapshot.context
            )
            schema = QuotedTaskBatch
            label = project_task_batch(sample.target, sample.message)
        else:
            prompt, payload = build_requirement_extraction_request(
                sample.message,
                snapshot.context.requirements,
            )
            schema = QuotedRequirementDraft
            label = project_requirement_draft(sample.target, sample.message)
        request = {
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            },
        }
        target = {"role": "assistant", "content": label.model_dump_json()}
    return {
        "react_protocol": REACT_PROTOCOL_VERSION,
        "sample_id": sample.sample_id,
        "scenario_id": sample.scenario_id,
        "split": sample.split.value,
        "task": sample.task.value,
        "request_hash": canonical_hash(request),
        "request": request,
        "target": target,
    }


def export_dataset(directory: Path, output: Path) -> dict[str, Any]:
    loaded = load_dataset(directory, require_approved=True)
    examples = [
        build_example(sample, loaded.snapshots[sample.snapshot_id])
        for sample in loaded.samples
    ]
    # Never overwrite an earlier artifact/review by accident.
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split in DatasetSplit:
        # Test targets stay in canonical reviewed data, never in an SFT file.
        if split == DatasetSplit.TEST:
            continue
        rows = []
        for example in examples:
            if example["split"] != split:
                continue
            request = example["request"]
            rows.append(
                {**request, "messages": [*request["messages"], example["target"]]}
            )
        files[f"{split.value}.jsonl"] = rows
    files["requests.jsonl"] = [
        {key: value for key, value in example.items() if key != "target"}
        for example in examples
    ]
    hashes = {}
    for name, rows in files.items():
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (output / name).write_text(content, encoding="utf-8")
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
    report = {
        "schema_version": 1,
        "react_protocol": REACT_PROTOCOL_VERSION,
        "dataset": loaded.manifest.model_dump(mode="json"),
        "samples": len(examples),
        "splits": loaded.report.splits,
        "files": hashes,
        "tokenizer_ready": False,
        "format": "canonical-messages-v1",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = export_dataset(args.directory, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"EXPORT_FAILED: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
