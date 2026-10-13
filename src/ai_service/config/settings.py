from functools import lru_cache

from pydantic import AliasChoices, Field, HttpUrl, JsonValue, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_service.config.enums import (
    AIProvider,
    EmbeddingProviderKind,
    RetrievalBackend,
    StatefulReactVersion,
)


class Settings(BaseSettings):
    """Runtime configuration for the AI boundary.

    The AI service never assumes a model/API key is available.  Without one it
    still exposes deterministic, backend-grounded responses so the frontend
    contract can be integrated and tested first.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AI_",
        extra="ignore",
    )

    backend_api_url: str = "http://localhost:8080/api/v1"
    request_timeout_seconds: float = 5.0
    conversation_max_messages: int = 20
    database_url: SecretStr | None = None
    stateful_enabled: bool = False
    # Select one execution mode; both are off until dependencies are configured.
    stateful_build_enabled: bool = False
    stateful_react_enabled: bool = False
    stateful_react_version: StatefulReactVersion = StatefulReactVersion.V1
    hardware_catalog_mapping: dict[str, JsonValue] | None = None
    jwt_jwks_url: HttpUrl | None = None
    jwt_issuer: str = Field(default="http://identity-service", min_length=1)
    jwt_audience: str = Field(default="pc-shopping-api", min_length=1)

    @model_validator(mode="after")
    def require_stateful_configuration(self) -> "Settings":
        if (
            self.stateful_react_version == StatefulReactVersion.V2
            and not self.stateful_react_enabled
        ):
            raise ValueError("V2 requires the explicitly enabled ReAct runtime")
        if self.stateful_build_enabled and self.stateful_react_enabled:
            raise ValueError("Select guided build or ReAct runtime, not both")
        if self.stateful_build_enabled or self.stateful_react_enabled:
            if not self.stateful_enabled or not self.hardware_catalog_mapping:
                raise ValueError(
                    "Stateful execution requires stateful mode and hardware catalog mapping"
                )
            key = (
                self.openai_api_key
                if self.provider == AIProvider.OPENAI
                else self.gemini_api_key
                if self.provider == AIProvider.GEMINI
                else None
            )
            if not key or not key.strip():
                raise ValueError(
                    "Stateful execution requires an explicitly configured model provider/key"
                )
        if self.stateful_enabled:
            if self.database_url is None or self.jwt_jwks_url is None:
                raise ValueError(
                    "Stateful mode requires AI_DATABASE_URL and AI_JWT_JWKS_URL"
                )
            if not self.database_url.get_secret_value().startswith(
                "postgresql+psycopg://"
            ):
                raise ValueError("AI_DATABASE_URL must use postgresql+psycopg")
        return self

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
