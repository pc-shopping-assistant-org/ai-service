"""One checked-in configuration for the selected single-GPU SFT/QLoRA recipe."""

from typing import Literal

from pydantic import Field, JsonValue

from ai_service.capabilities.assistant.stateful_contracts import Contract
from ai_service.config.qwen import QWEN_MAX_CONTEXT, QWEN_MODEL_ID, QWEN_MODEL_REVISION


class QwenConfig(Contract):
    model_id: Literal["Qwen/Qwen3-8B"] = QWEN_MODEL_ID
    model_revision: str = Field(default=QWEN_MODEL_REVISION, pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(default=16384, ge=128, le=QWEN_MAX_CONTEXT)
    max_new_tokens: int = Field(default=2048, ge=1, le=QWEN_MAX_CONTEXT)
    seed: int = Field(default=42, ge=0)
    epochs: float = Field(default=3, gt=0, allow_inf_nan=False)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)

    def decoding(self) -> dict[str, JsonValue]:
        return {"temperature": 0, "max_tokens": self.max_new_tokens, "seed": self.seed}
