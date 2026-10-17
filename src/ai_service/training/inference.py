"""Capture baseline/adapter calls against explicitly started Qwen serving, then score."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.config.qwen import QWEN_ADAPTER_NAME
from ai_service.infrastructure.providers.qwen import QwenChatClient
from ai_service.infrastructure.providers.qwen_protocol import prepare_qwen_request
from ai_service.training.artifacts import (
    adapter_manifest,
    code_provenance,
    file_hash,
    write_json,
)
from ai_service.training.dataset import DatasetSplit, load_dataset
from ai_service.training.evaluation import EvaluationRun, evaluate_responses
from ai_service.training.export import build_example
from ai_service.training.qwen_config import QwenConfig


async def capture_responses(
    directory: Path,
    output: Path,
    config: QwenConfig,
    *,
    client: QwenChatClient,
    split: DatasetSplit = DatasetSplit.VALIDATION,
    adapter: Path | None = None,
) -> dict[str, Any]:
    loaded = load_dataset(directory, require_approved=True)
    manifest = adapter_manifest(adapter, config) if adapter else None
    model = QWEN_ADAPTER_NAME if manifest else config.model_id
    examples = [
        build_example(sample, loaded.snapshots[sample.snapshot_id])
        for sample in loaded.samples
        if sample.split == split
    ]
    if not examples:
        raise ValueError("EVALUATION_SPLIT_EMPTY")
    provenance = code_provenance()
    run = EvaluationRun(
        model_id=config.model_id,
        model_revision=config.model_revision,
        code_revision=provenance["source_hash"],
        decoding=config.decoding(),
        adapter_revision=canonical_hash(manifest) if manifest else None,
        seed=config.seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run.json", run.model_dump(mode="json"))
    captures = []
    with (output / "responses.jsonl").open("x", encoding="utf-8") as stream:
        for example in examples:
            capture: dict[str, Any] = {
                "sample_id": example["sample_id"],
                "request_hash": example["request_hash"],
                "model_input_hash": canonical_hash(
                    prepare_qwen_request(example["request"])
                ),
            }
            started = perf_counter()
            try:
                response = await client.complete(
                    example["request"], model=model, decoding=config.decoding()
                )
                message = response["choices"][0]["message"]
                capture["raw_response"] = message
                capture["response"] = {
                    "role": message["role"],
                    "content": message.get("content") or "",
                }
                if message.get("tool_calls"):
                    capture["response"]["tool_calls"] = message["tool_calls"]
                capture["usage"] = response.get("usage") or {}
            except (ValueError, KeyError, IndexError, TypeError) as error:
                capture["error"] = f"MODEL_RESPONSE_INVALID: {type(error).__name__}"
            except httpx.HTTPError as error:
                # Do not leak endpoint credentials/response bodies in errors.
                # Cancellation is a BaseException and is intentionally propagated.
                capture["error"] = f"INFERENCE_FAILED: {type(error).__name__}"
            capture["latency_ms"] = (perf_counter() - started) * 1000
            stream.write(json.dumps(capture, ensure_ascii=False) + "\n")
            stream.flush()
            captures.append(capture)
    report = evaluate_responses(loaded, captures, run, split=split)
    report.update(
        code=provenance,
        responses_hash=file_hash(output / "responses.jsonl"),
        dependency_lock_hash=provenance["dependency_lock_hash"],
        qwen_config=config.model_dump(mode="json"),
    )
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--split",
        type=DatasetSplit,
        choices=list(DatasetSplit),
        default=DatasetSplit.VALIDATION,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    async def execute() -> dict[str, Any]:
        client = QwenChatClient(
            args.base_url, api_key=os.getenv("AI_QWEN_API_KEY")
        )
        try:
            return await capture_responses(
                args.directory,
                args.output,
                QwenConfig.model_validate_json(args.config.read_bytes()),
                client=client,
                split=args.split,
                adapter=args.adapter,
            )
        finally:
            await client.aclose()

    try:
        report = asyncio.run(execute())
    except (ValueError, OSError) as error:
        parser.exit(1, f"INFERENCE_FAILED: {error}\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "total",
                    "raw_contract_valid",
                    "validated",
                    "semantic_review",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
