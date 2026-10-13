from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class GeminiModelProvider(ModelProvider):
    """Lazy LangChain adapter for Google Gemini (AI Studio) models."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.gemini_model_name
        self._api_key = settings.gemini_api_key
        self._timeout = settings.request_timeout_seconds
        self._model: ChatGoogleGenerativeAI | None = None
        self._closed = False

    @property
    def name(self) -> str:
        return "gemini"

    def create_model(self) -> object:
        if self._closed:
            raise RuntimeError("Model provider is closed")
        if self._model is not None:
            return self._model
        # See the OpenAI adapter for why imports and client construction are
        # intentionally deferred until a request needs a model.
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not self._api_key:
            raise ValueError("AI_GEMINI_API_KEY is required")
        self._model = ChatGoogleGenerativeAI(
            model=self._model_name,
            api_key=self._api_key,
            vertexai=False,
            timeout=self._timeout,
            max_retries=0,
        )
        return self._model

    async def aclose(self) -> None:
        self._closed = True
        if self._model is not None:
            await self._model.aclose()
