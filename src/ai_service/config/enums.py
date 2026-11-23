"""Configuration enums shared by application wiring and adapters.

Keeping runtime choices in one place prevents provider/retrieval names from
becoming magic strings spread across the composition root and infrastructure.
The values intentionally remain the same strings used by environment files.
"""

from enum import StrEnum


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIProvider(StrEnum):
    """LLM provider selected for model answer generation."""

    FALLBACK = "fallback"
    OPENAI = "openai"
    GEMINI = "gemini"
    QWEN = "qwen"


class RetrievalBackend(StrEnum):
    """Catalog retrieval strategy selected at runtime."""

    BACKEND = "backend"
    QDRANT = "qdrant"
    HYBRID = "hybrid"


class EmbeddingProviderKind(StrEnum):
    """Embedding adapter used by semantic retrieval/indexing."""

    HTTP = "http"
    HASH = "hash"


class ThinkingMode(StrEnum):
    """Unified reasoning/thinking control selectable per turn from the UI."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ON = "on"


class ThinkingStrategy(StrEnum):
    """Provider-specific transport used to apply a ThinkingMode."""

    NONE = "none"
    GEMINI_REASONING = "gemini_reasoning"
    OLLAMA_THINK = "ollama_think"
    OPENAI_REASONING = "openai_reasoning"
    QWEN_THINKING = "qwen_thinking"


__all__ = [
    "AIProvider",
    "EmbeddingProviderKind",
    "LogLevel",
    "RetrievalBackend",
    "ThinkingMode",
    "ThinkingStrategy",
]
