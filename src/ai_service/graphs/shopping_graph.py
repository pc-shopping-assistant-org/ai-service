"""Compatibility exports for the assistant capability graph."""

from ai_service.capabilities.assistant.graphs.shopping import (
    ShoppingInput,
    ShoppingIntent,
    ShoppingOutput,
    ShoppingState,
    shopping_graph,
)

__all__ = [
    "ShoppingInput",
    "ShoppingIntent",
    "ShoppingOutput",
    "ShoppingState",
    "shopping_graph",
]
