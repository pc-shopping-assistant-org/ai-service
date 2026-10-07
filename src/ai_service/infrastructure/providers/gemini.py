from __future__ import annotations

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class GeminiModelProvider(ModelProvider):
    """Lazy LangChain adapter for Google Gemini (AI Studio) models."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.gemini_model_name
        self._api_key = settings.gemini_api_key
        self._timeout = settings.request_timeout_seconds

    @property
    def name(self) -> str:
        return "gemini"

    def create_model(self) -> object:
        # See the OpenAI adapter for why imports and client construction are
        # intentionally deferred until a request needs a model.
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not self._api_key:
            raise ValueError("AI_GEMINI_API_KEY is required")
        return ChatGoogleGenerativeAI(
            model=self._model_name,
            api_key=self._api_key,
            vertexai=False,
            timeout=self._timeout,
            max_retries=0,
        )
