from __future__ import annotations

from typing import cast

from pydantic_ai.models import Model

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class GeminiModelProvider(ModelProvider):
    """Lazy PydanticAI adapter for Google Gemini (AI Studio) models."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.gemini_model_name
        self._api_key = settings.gemini_api_key

    @property
    def name(self) -> str:
        return "gemini"

    def create_model(self) -> Model:
        # See the OpenAI adapter for why imports and client construction are
        # intentionally deferred until a request needs a model.
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        provider = GoogleProvider(api_key=cast(str, self._api_key))
        return GoogleModel(self._model_name, provider=provider)
