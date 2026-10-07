from __future__ import annotations

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class OpenAIModelProvider(ModelProvider):
    """Lazy LangChain adapter for the OpenAI Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.openai_model_name
        self._api_key = settings.openai_api_key
        self._timeout = settings.request_timeout_seconds

    @property
    def name(self) -> str:
        return "openai"

    def create_model(self) -> object:
        # Keep optional SDK imports inside the adapter.  Importing the AI
        # service with the deterministic fallback must not require credentials
        # or initialize a network client.
        from langchain_openai import ChatOpenAI

        if not self._api_key:
            raise ValueError("AI_OPENAI_API_KEY is required")
        return ChatOpenAI(  # type: ignore[call-arg]
            model=self._model_name,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=0,
        )
