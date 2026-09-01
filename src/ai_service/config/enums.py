"""Configuration enums shared by application wiring and adapters.

Keeping runtime choices in one place prevents provider/retrieval names from
becoming magic strings spread across the composition root and infrastructure.
The values intentionally remain the same strings used by environment files.
"""

from enum import StrEnum


class AIProvider(StrEnum):
    """LLM provider selected for PydanticAI answer generation."""

    FALLBACK = "fallback"
    OPENAI = "openai"
    GEMINI = "gemini"


class RetrievalBackend(StrEnum):
    """Catalog retrieval strategy selected at runtime."""

    BACKEND = "backend"
    QDRANT = "qdrant"
    HYBRID = "hybrid"


class EmbeddingProviderKind(StrEnum):
    """Embedding adapter used by semantic retrieval/indexing."""

    HTTP = "http"
    HASH = "hash"


__all__ = ["AIProvider", "EmbeddingProviderKind", "RetrievalBackend"]
