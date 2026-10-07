"""Vendor-neutral shopping planning contracts and deterministic normalization."""

from pydantic import BaseModel

from ai_service.capabilities.assistant.schemas import AssistantIntent

ShoppingIntent = AssistantIntent


class ShoppingInput(BaseModel):
    query: str
    intent: ShoppingIntent = AssistantIntent.SEARCH


class ShoppingOutput(BaseModel):
    query: str
    intent: ShoppingIntent


def normalize_shopping(inputs: ShoppingInput) -> ShoppingOutput:
    return ShoppingOutput(query=" ".join(inputs.query.split()), intent=inputs.intent)


__all__ = ["ShoppingInput", "ShoppingIntent", "ShoppingOutput", "normalize_shopping"]
