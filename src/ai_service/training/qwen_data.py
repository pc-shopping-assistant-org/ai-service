"""Use the pinned native template and supervise only the final assistant completion."""

from typing import Any, Protocol

from ai_service.infrastructure.providers.qwen_protocol import prepare_qwen_request

IGNORE_LABEL = -100


class ChatTokenizer(Protocol):
    chat_template: str

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


def tokenize_example(
    example: dict[str, Any], tokenizer: ChatTokenizer, *, max_length: int
) -> dict[str, list[int]]:
    request = prepare_qwen_request(example["request"])
    messages = request["messages"]
    options = {
        "tools": request.get("tools"),
        "tokenize": False,
        "enable_thinking": False,
    }
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, **options
    )
    full = tokenizer.apply_chat_template(
        [*messages, example["target"]], add_generation_prompt=False, **options
    )
    tokens = tokenizer(full, add_special_tokens=False)
    if len(tokens["input_ids"]) > max_length:
        raise ValueError(
            f"QWEN_EXAMPLE_TOO_LONG: {example.get('sample_id', 'example')}: {len(tokens['input_ids'])} > {max_length}"
        )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if (
        not full.startswith(prompt)
        or tokens["input_ids"][: len(prompt_ids)] != prompt_ids
    ):
        raise ValueError("QWEN_COMPLETION_PREFIX_MISMATCH")
    if len(tokens["input_ids"]) <= len(prompt_ids):
        raise ValueError("QWEN_COMPLETION_EMPTY")
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": [IGNORE_LABEL] * len(prompt_ids)
        + tokens["input_ids"][len(prompt_ids) :],
    }
