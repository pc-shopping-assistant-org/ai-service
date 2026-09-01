from __future__ import annotations

from pydantic_ai.models import Model

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class OpenAIModelProvider(ModelProvider):
    """Lazy PydanticAI adapter for the OpenAI Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.openai_model_name
        self._api_key = settings.openai_api_key

    @property
    def name(self) -> str:
        return "openai"

    def create_model(self) -> Model:
        # Keep optional SDK imports inside the adapter.  Importing the AI
        # service with the deterministic fallback must not require credentials
        # or initialize a network client.
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(api_key=self._api_key)
        return OpenAIChatModel(self._model_name, provider=provider)
