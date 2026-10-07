"""Direct model calls with Pydantic output validation, not an agent runtime."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Protocol, cast

from pydantic import BaseModel

from ai_service.application.ports.answer_generator import (
    AnswerGenerator,
    StreamingAnswerGenerator,
)
from ai_service.config.settings import Settings, get_settings
from ai_service.infrastructure.providers.factory import build_model_provider
from ai_service.schemas.agent import ShoppingAnswer

LOGGER = logging.getLogger(__name__)
Messages = list[tuple[str, str]]


class StructuredModel(Protocol):
    async def ainvoke(self, input: Messages) -> object: ...


class TextChunk(Protocol):
    content: str | list[str | dict[str, object]]


class ChatModel(Protocol):
    def with_structured_output(
        self, schema: type[BaseModel], **kwargs: object
    ) -> StructuredModel: ...

    def astream(self, input: Messages) -> AsyncIterator[TextChunk]: ...


class ModelAnswerGenerator(AnswerGenerator, StreamingAnswerGenerator):
    """Lazy LangChain model adapter; application ports remain vendor-neutral.

    No tool loop, hidden conversation history or agent retries. Streaming uses
    plain text, while non-streaming output is validated with ShoppingAnswer.
    Partial stream failures propagate so the application emits an ERROR event.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: ChatModel | None = None
        self._initialization_attempted = False

    def _get_model(self) -> ChatModel | None:
        if not self._initialization_attempted:
            self._initialization_attempted = True
            try:
                provider = build_model_provider(self.settings)
                if provider is not None:
                    self._model = cast(ChatModel, provider.create_model())
            except Exception as exc:  # noqa: BLE001 - optional provider initialization
                LOGGER.warning("Model initialization failed (%s)", type(exc).__name__)
        return self._model

    def _messages(self, prompt: str) -> Messages:
        return [("system", self.settings.model_system_prompt), ("human", prompt)]

    async def generate(self, prompt: str, fallback: str) -> str:
        model = self._get_model()
        if model is None:
            return fallback
        try:
            result = await model.with_structured_output(
                ShoppingAnswer, method="json_schema"
            ).ainvoke(self._messages(prompt))
            return ShoppingAnswer.model_validate(result).answer.strip() or fallback
        except Exception as exc:  # noqa: BLE001 - provider SDK/network/validation errors
            LOGGER.warning(
                "Model generation failed (%s); using fallback", type(exc).__name__
            )
            return fallback

    async def stream(self, prompt: str, fallback: str) -> AsyncIterator[str]:
        model = self._get_model()
        if model is None:
            yield fallback
            return
        yielded = False
        try:
            async for chunk in model.astream(self._messages(prompt)):
                content = chunk.content
                parts = [content] if isinstance(content, str) else content
                for part in parts:
                    text = (
                        part
                        if isinstance(part, str)
                        else (part.get("text") if part.get("type") == "text" else None)
                    )
                    if isinstance(text, str) and text:
                        yielded = True
                        yield text
        except Exception as exc:
            LOGGER.warning("Model streaming failed (%s)", type(exc).__name__)
            if yielded:
                raise
        if not yielded:
            yield fallback


__all__ = ["ModelAnswerGenerator"]
