from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.enums import ThinkingMode, ThinkingStrategy
from ai_service.config.models import ResolvedModelEntry
from ai_service.config.settings import Settings


class OpenAIModelProvider(ModelProvider):
    """Lazy LangChain adapter for the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_optional: bool = False,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        thinking_strategy: ThinkingStrategy = ThinkingStrategy.NONE,
        thinking_mode: ThinkingMode | None = None,
    ) -> None:
        if settings is not None:
            self._model_name = model_name or settings.model_name or settings.openai_model_name
            self._base_url = (
                base_url
                if base_url is not None
                else (str(settings.openai_base_url) if settings.openai_base_url else None)
            )
            self._api_key = api_key if api_key is not None else settings.openai_api_key
            self._timeout = timeout if timeout is not None else settings.request_timeout_seconds
            self._max_output_tokens = max_output_tokens
        else:
            if not model_name:
                raise ValueError("model_name is required")
            self._model_name = model_name
            self._base_url = base_url
            self._api_key = api_key
            self._timeout = timeout if timeout is not None else 120.0
            self._max_output_tokens = max_output_tokens
        self._api_key_optional = api_key_optional
        self._thinking_strategy = thinking_strategy
        self._thinking_mode = thinking_mode
        self._model: ChatOpenAI | None = None
        self._closed = False

    @classmethod
    def from_entry(
        cls,
        entry: ResolvedModelEntry,
        thinking_mode: ThinkingMode | None = None,
    ) -> OpenAIModelProvider:
        return cls(
            model_name=entry.model_name,
            base_url=entry.base_url,
            api_key=entry.api_key.get_secret_value() if entry.api_key else None,
            api_key_optional=entry.api_key_optional,
            timeout=entry.timeout_seconds,
            max_output_tokens=entry.max_output_tokens,
            thinking_strategy=entry.thinking.strategy,
            thinking_mode=thinking_mode,
        )

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

        effective_key = self._api_key or ("ollama" if self._api_key_optional else None)
        if not effective_key:
            raise ValueError("AI_OPENAI_API_KEY is required")
        kwargs: dict[str, object] = {
            "model": self._model_name,
            "api_key": effective_key,
            "timeout": self._timeout,
            "max_retries": 0,
        }
        if self._base_url:
            kwargs["base_url"] = self._base_url
        if self._max_output_tokens is not None:
            kwargs["max_tokens"] = self._max_output_tokens
        if (
            self._thinking_strategy == ThinkingStrategy.OLLAMA_THINK
            and self._thinking_mode is not None
        ):
            kwargs["extra_body"] = {
                "think": self._thinking_mode != ThinkingMode.OFF
            }
        elif (
            self._thinking_strategy == ThinkingStrategy.OPENAI_REASONING
            and self._thinking_mode is not None
            and self._thinking_mode != ThinkingMode.OFF
        ):
            kwargs["reasoning_effort"] = (
                "medium"
                if self._thinking_mode == ThinkingMode.ON
                else self._thinking_mode.value
            )
        self._model = ChatOpenAI(**kwargs)  # type: ignore[arg-type]
        return self._model

    async def aclose(self) -> None:
        self._closed = True
        if self._model is not None:
            if self._model.root_async_client is not None:
                await self._model.root_async_client.close()
            if self._model.root_client is not None:
                self._model.root_client.close()
