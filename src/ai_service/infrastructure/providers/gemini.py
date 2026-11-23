from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain_google_genai import ChatGoogleGenerativeAI

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.enums import ThinkingMode, ThinkingStrategy
from ai_service.config.models import ResolvedModelEntry
from ai_service.config.settings import Settings

GeminiReasoningEffort = Literal["minimal", "low", "medium", "high"]

_GEMINI_EFFORT_MAP: dict[ThinkingMode, GeminiReasoningEffort] = {
    ThinkingMode.MINIMAL: "minimal",
    ThinkingMode.LOW: "low",
    ThinkingMode.MEDIUM: "medium",
    ThinkingMode.HIGH: "high",
    ThinkingMode.ON: "medium",
}

_GEMINI_THINKING_BUDGETS: dict[ThinkingMode, int] = {
    ThinkingMode.OFF: 0,
    ThinkingMode.MINIMAL: 512,
    ThinkingMode.LOW: 1024,
    ThinkingMode.MEDIUM: 4096,
    ThinkingMode.HIGH: 8192,
    ThinkingMode.ON: 4096,
}


class GeminiModelProvider(ModelProvider):
    """Lazy LangChain adapter for Google Gemini (AI Studio) models."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        thinking_strategy: ThinkingStrategy = ThinkingStrategy.GEMINI_REASONING,
        thinking_mode: ThinkingMode | None = None,
    ) -> None:
        self._thinking_mode: ThinkingMode | None
        if settings is not None:
            self._model_name = model_name or settings.model_name or settings.gemini_model_name
            self._api_key = api_key if api_key is not None else settings.gemini_api_key
            self._timeout = timeout if timeout is not None else settings.request_timeout_seconds
            self._max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else settings.gemini_max_output_tokens
            )
            self._thinking_mode = (
                thinking_mode
                if thinking_mode is not None
                else ThinkingMode(settings.gemini_reasoning_effort)
            )
        else:
            if not model_name:
                raise ValueError("model_name is required")
            self._model_name = model_name
            self._api_key = api_key
            self._timeout = timeout if timeout is not None else 120.0
            self._max_output_tokens = max_output_tokens if max_output_tokens is not None else 4096
            self._thinking_mode = thinking_mode
        self._thinking_strategy = thinking_strategy
        self._model: ChatGoogleGenerativeAI | None = None
        self._closed = False

    @classmethod
    def from_entry(
        cls,
        entry: ResolvedModelEntry,
        thinking_mode: ThinkingMode | None = None,
    ) -> GeminiModelProvider:
        return cls(
            model_name=entry.model_name,
            api_key=entry.api_key.get_secret_value() if entry.api_key else None,
            timeout=entry.timeout_seconds,
            max_output_tokens=entry.max_output_tokens,
            thinking_strategy=entry.thinking.strategy,
            thinking_mode=thinking_mode,
        )

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
            max_output_tokens=self._max_output_tokens,
        )
        self._apply_thinking(self._model)
        return self._model

    def _apply_thinking(self, model: ChatGoogleGenerativeAI) -> None:
        if (
            self._thinking_strategy != ThinkingStrategy.GEMINI_REASONING
            or self._thinking_mode is None
        ):
            return
        levels = list((model.profile or {}).get("reasoning_effort_levels", []))
        if self._thinking_mode == ThinkingMode.OFF:
            if "minimal" in levels and "2.5" not in self._model_name:
                model.reasoning_effort = "minimal"
            else:
                model.thinking_budget = 0
            return
        effort = _GEMINI_EFFORT_MAP.get(self._thinking_mode)
        if effort is not None and effort in levels:
            model.reasoning_effort = effort
        else:
            model.thinking_budget = _GEMINI_THINKING_BUDGETS[self._thinking_mode]

    async def aclose(self) -> None:
        self._closed = True
        if self._model is not None:
            await self._model.aclose()
