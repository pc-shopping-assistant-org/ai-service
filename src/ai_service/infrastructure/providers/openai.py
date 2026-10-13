from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings


class OpenAIModelProvider(ModelProvider):
    """Lazy LangChain adapter for the OpenAI Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.model_name or settings.openai_model_name
        self._api_key = settings.openai_api_key
        self._timeout = settings.request_timeout_seconds
        self._model: ChatOpenAI | None = None
        self._closed = False

    @property
    def name(self) -> str:
        return "openai"

    def create_model(self) -> object:
        if self._closed:
            raise RuntimeError("Model provider is closed")
        if self._model is not None:
            return self._model
        # Keep optional SDK imports inside the adapter.  Importing the AI
        # service with the deterministic fallback must not require credentials
        # or initialize a network client.
        from langchain_openai import ChatOpenAI

        if not self._api_key:
            raise ValueError("AI_OPENAI_API_KEY is required")
        self._model = ChatOpenAI(  # type: ignore[call-arg]
            model=self._model_name,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=0,
        )
        return self._model

    async def aclose(self) -> None:
        self._closed = True
        if self._model is not None:
            if self._model.root_async_client is not None:
                await self._model.root_async_client.close()
            if self._model.root_client is not None:
                self._model.root_client.close()
