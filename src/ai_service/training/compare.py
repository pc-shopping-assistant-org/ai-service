"""Paired machine-checkable comparison, never automatic model promotion."""

import argparse
import json
from pathlib import Path
from typing import Any

from ai_service.training.artifacts import write_json


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    conditions = ("model_id", "model_revision", "code_revision", "decoding", "seed")
    if (
        any(before["run"][key] != after["run"][key] for key in conditions)
        or before["dataset"] != after["dataset"]
        or before["split"] != after["split"]
        or before.get("dependency_lock_hash") != after.get("dependency_lock_hash")
        or before.get("qwen_config") != after.get("qwen_config")
    ):
        raise ValueError("EXPERIMENT_CONDITIONS_DIFFER")
    if before["run"].get("adapter_revision") or not after["run"].get(
        "adapter_revision"
    ):
        raise ValueError("BASELINE_AND_ADAPTER_REPORTS_REQUIRED")
    left = {row["sample_id"]: row for row in before["cases"]}
    right = {row["sample_id"]: row for row in after["cases"]}
    if (
        len(left) != len(before["cases"])
        or len(right) != len(after["cases"])
        or left.keys() != right.keys()
    ):
        raise ValueError("PAIRED_INPUTS_DIFFER")
    changes = []
    for sample_id, baseline in left.items():
        adapted = right[sample_id]
        if any(
            baseline.get(key) != adapted.get(key)
            for key in ("request_hash", "model_input_hash")
        ):
            raise ValueError("PAIRED_INPUTS_DIFFER")
        changes.append(
            {
                "sample_id": sample_id,
                "before": baseline["status"],
                "after": adapted["status"],
            }
        )
    return {
        "schema_version": 1,
        "total": len(left),
        "baseline": before["run"],
        "adapter": after["run"],
        "raw_contract_valid_delta": after["raw_contract_valid"]
        - before["raw_contract_valid"],
        "validated_delta": after["validated"] - before["validated"],
        "cases": changes,
        "promotion": "MANUAL_REVIEW_REQUIRED",
        "scope": "machine-checkable model calls; not semantic/conversation accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_reports(
            json.loads(args.before.read_bytes()), json.loads(args.after.read_bytes())
        )
        write_json(args.output, result)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"COMPARISON_FAILED: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
