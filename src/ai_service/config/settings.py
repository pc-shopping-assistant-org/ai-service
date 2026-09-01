from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_service.config.enums import AIProvider, EmbeddingProviderKind, RetrievalBackend


class Settings(BaseSettings):
    """Runtime configuration for the AI boundary.

    The AI service never assumes a model/API key is available.  Without one it
    still exposes deterministic, backend-grounded responses so the frontend
    contract can be integrated and tested first.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_", extra="ignore")

    backend_api_url: str = "http://localhost:8080/api/v1"
    request_timeout_seconds: float = 5.0
    conversation_max_messages: int = 20
    retrieval_backend: RetrievalBackend = RetrievalBackend.BACKEND
    qdrant_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_QDRANT_URL", "QDRANT_URL"),
    )
    qdrant_collection: str = "products"
    embedding_api_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_EMBEDDING_API_URL", "EMBEDDING_API_URL"),
    )
    embedding_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_EMBEDDING_API_KEY", "EMBEDDING_API_KEY"),
    )
    embedding_provider: EmbeddingProviderKind = EmbeddingProviderKind.HTTP
    embedding_dimension: int = Field(default=384, ge=2, le=4096)
    provider: AIProvider = AIProvider.FALLBACK
    model_name: str | None = None
    openai_model_name: str = "gpt-4o-mini"
    gemini_model_name: str = "gemini-2.5-flash"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AI_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AI_GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ),
    )
    model_system_prompt: str = (
        "You are a PC shopping assistant. Answer in Vietnamese when the user "
        "uses Vietnamese. Use only the catalog context supplied in the prompt; "
        "do not invent prices, stock, specifications or product availability. "
        "If the context is insufficient, say so clearly."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
