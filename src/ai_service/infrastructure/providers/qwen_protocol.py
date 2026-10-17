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
        instruction = "\nReturn only JSON matching this response schema: " + json.dumps(
            response_format["json_schema"],
            ensure_ascii=False,
            sort_keys=True,
        )
        messages = projected["messages"]
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    return projected
