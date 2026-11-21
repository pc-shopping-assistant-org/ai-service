"""Shared Qwen input projection for training, baseline and runtime serving."""

import json
from copy import deepcopy
from typing import Any


def prepare_qwen_request(request: dict[str, Any]) -> dict[str, Any]:
    projected = deepcopy(request)
    response_format = projected.pop("response_format", None)
    if response_format is not None:
        if response_format.get("type") != "json_schema":
            raise ValueError("QWEN_JSON_SCHEMA_REQUIRED")
        raw_schema = response_format["json_schema"]
        schema_spec = raw_schema.get("schema", raw_schema)
        root_keys = schema_spec.get("required") or list(
            schema_spec.get("properties", {}).keys()
        )
        example_root = "{" + ", ".join(f'"{k}": [...]' for k in root_keys) + "}"
        instruction = (
            f"\nReturn only a JSON object starting directly with the root key {example_root} matching this schema: "
            + json.dumps(
                schema_spec,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        messages = projected["messages"]
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    return projected
