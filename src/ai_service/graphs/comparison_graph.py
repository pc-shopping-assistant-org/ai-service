"""Compatibility exports for the assistant capability graph."""

from ai_service.capabilities.assistant.graphs.comparison import (
    ComparisonInput,
    ComparisonOutput,
    normalize_comparison,
)

__all__ = [
    "ComparisonInput",
    "ComparisonOutput",
    "normalize_comparison",
]
