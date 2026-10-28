"""Score captured raw model calls offline; never fabricate an inference baseline."""

import argparse
import hashlib
import json
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, convert_to_messages
from pydantic import Field, JsonValue

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.react.actions import CallToolAction
from ai_service.capabilities.assistant.react.contracts import ToolName
from ai_service.capabilities.assistant.source_quotes import (
    QuotedRequirementDraft,
    QuotedTaskBatch,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text
from ai_service.infrastructure.providers.react_model import parse_flexible_response
from ai_service.training.dataset import (
    DatasetSplit,
    LoadedDataset,
    ReactDecisionSample,
    ReviewStatus,
    TaskExtractionSample,
    load_dataset,
    validate_label,
)
from ai_service.training.export import build_example


class CaseStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    MISSING = "MISSING"


class EvaluationRun(Contract):
    model_id: Text
    model_revision: Text
    code_revision: Text
    # Record actual serving settings, not inferred defaults or secrets.
    decoding: dict[str, JsonValue]
    adapter_revision: Text | None = None
    seed: int | None = None


class CapturedResponse(Contract):
    sample_id: Text
    request_hash: Digest
    model_input_hash: Digest | None = None
    raw_response: JsonValue = None
    response: JsonValue = None
    error: Text | None = None
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    usage: dict[str, JsonValue] = Field(default_factory=dict)


class FunctionCall(Contract):
    name: Text
    arguments: str


class NativeToolCall(Contract):
    id: Text
    type: Literal["function"]
    function: FunctionCall


class AssistantResponse(Contract):
    role: Literal["assistant"]
    content: str = ""
    tool_calls: list[NativeToolCall] = Field(default_factory=list)


def evaluate_responses(
    loaded: LoadedDataset,
    responses: list[dict[str, Any]],
    run: EvaluationRun,
    *,
    split: DatasetSplit = DatasetSplit.VALIDATION,
) -> dict[str, Any]:
    if any(sample.review_status != ReviewStatus.APPROVED for sample in loaded.samples):
        raise ValueError("DATASET_REVIEW_REQUIRED")
    samples = [sample for sample in loaded.samples if sample.split == split]
    if not samples:
        raise ValueError("EVALUATION_SPLIT_EMPTY")
    by_id = {sample.sample_id: sample for sample in samples}
    captured = {}
    for raw in responses:
        item = CapturedResponse.model_validate(raw)
        if item.sample_id not in by_id:
            raise ValueError(f"UNKNOWN_PREDICTION: {item.sample_id}")
        if item.sample_id in captured:
            raise ValueError(f"DUPLICATE_PREDICTION: {item.sample_id}")
        captured[item.sample_id] = item
    cases = []
    for sample in samples:
        snapshot = loaded.snapshots[sample.snapshot_id]
        example = build_example(sample, snapshot)
        capture = captured.get(sample.sample_id)
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "scenario_id": sample.scenario_id,
            "task": sample.task.value,
            "request_hash": example["request_hash"],
            "status": CaseStatus.MISSING.value,
            "raw_contract_valid": False,
            "error": "MISSING_RESPONSE",
            "matches_authored_target": None,
        }
        if capture is not None:
            if capture.request_hash != example["request_hash"]:
                raise ValueError(f"REQUEST_HASH_MISMATCH: {sample.sample_id}")
            row.update(capture.model_dump(mode="json"), status=CaseStatus.FAIL.value)
            try:
                if capture.error is not None:
                    raise ValueError("INFERENCE_FAILED: " + capture.error)
                wire = AssistantResponse.model_validate(capture.response)
                if isinstance(sample, ReactDecisionSample):
                    message = convert_to_messages([wire.model_dump()])[0]
                    if not isinstance(message, AIMessage):
                        raise TypeError("ASSISTANT_RESPONSE_REQUIRED")
                    parsed = parse_flexible_response(
                        message,
                        tuple(ToolName),
                        message=sample.message,
                        context=snapshot.context,
                        transcript=snapshot.records,
                    )
                    if parsed is None:
                        raise ValueError("NATIVE_PROTOCOL_OR_CONTRACT_INVALID")
                    target: Any = parsed
                else:
                    if wire.tool_calls:
                        raise ValueError("STRUCTURED_RESPONSE_CANNOT_CALL_TOOLS")
                    schema = (
                        QuotedTaskBatch
                        if isinstance(sample, TaskExtractionSample)
                        else QuotedRequirementDraft
                    )
                    target = schema.model_validate_json(wire.content).bind(
                        sample.message
                    )
                row["raw_contract_valid"] = True
                # Reuse canonical source/policy/grounding/completion checks, not a
                # second evaluator that only compares against the authored answer.
                validate_label(sample.model_copy(update={"target": target}), snapshot)
                matches = target == sample.target
                if isinstance(sample, ReactDecisionSample):
                    matches = target.action == sample.target.action
                if isinstance(sample, ReactDecisionSample) and isinstance(
                    target.action, CallToolAction
                ):
                    gold = sample.target.action
                    matches = isinstance(gold, CallToolAction) and (
                        target.action.call.name == gold.call.name
                        and target.action.call.arguments == gold.call.arguments
                    )
                row.update(
                    status=CaseStatus.PASS.value,
                    error=None,
                    matches_authored_target=matches,
                )
            except (ValueError, TypeError, KeyError) as error:
                row["error"] = str(error)
        cases.append(row)
    groups = {}
    for task in sorted({sample.task.value for sample in samples}):
        subset = [row for row in cases if row["task"] == task]
        groups[task] = {
            "total": len(subset),
            "raw_contract_valid": sum(row["raw_contract_valid"] for row in subset),
            "validated": sum(row["status"] == CaseStatus.PASS for row in subset),
        }
    return {
        "schema_version": 1,
        "run": run.model_dump(mode="json"),
        "dataset": loaded.manifest.model_dump(mode="json"),
        "split": split.value,
        "total": len(cases),
        "raw_contract_valid": sum(row["raw_contract_valid"] for row in cases),
        "validated": sum(row["status"] == CaseStatus.PASS for row in cases),
        "statuses": dict(Counter(row["status"] for row in cases)),
        "groups": groups,
        "cases": cases,
        "semantic_review": "REQUIRED",
        "repair_attempts": 0,
        "scope": "captured-model-calls; not conversation success or factuality",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument(
        "--split",
        type=DatasetSplit,
        choices=list(DatasetSplit),
        default=DatasetSplit.VALIDATION,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        loaded = load_dataset(args.directory, require_approved=True)
        run = EvaluationRun.model_validate_json(args.run_config.read_bytes())
        raw = args.responses.read_bytes()
        responses = [json.loads(line) for line in raw.splitlines() if line.strip()]
        report = evaluate_responses(loaded, responses, run, split=args.split)
        report["responses_hash"] = hashlib.sha256(raw).hexdigest()
        report["dependency_lock_hash"] = hashlib.sha256(
            (Path(__file__).parents[3] / "uv.lock").read_bytes()
        ).hexdigest()
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"EVALUATION_FAILED: {error}\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
