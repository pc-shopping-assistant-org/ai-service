"""Compatibility exports for the assistant capability graph."""

from ai_service.capabilities.assistant.graphs.shopping import (
    ShoppingInput,
    ShoppingIntent,
    ShoppingOutput,
    normalize_shopping,
)

__all__ = [
    "ShoppingInput",
    "ShoppingIntent",
    "ShoppingOutput",
    "normalize_shopping",
]
