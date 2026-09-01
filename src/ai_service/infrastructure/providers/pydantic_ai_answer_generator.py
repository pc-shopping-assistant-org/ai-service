"""PydanticAI answer-generation adapter.

Kept in infrastructure so the application service depends only on the
``AnswerGenerator`` port and can be tested without provider SDKs or secrets.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import cast

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ai_service.agents.shopping_agent import (
    create_shopping_agent,
    create_streaming_shopping_agent,
)
from ai_service.application.ports.answer_generator import (
    AnswerGenerator,
    StreamingAnswerGenerator,
)
from ai_service.config.enums import AIProvider
from ai_service.config.settings import Settings, get_settings
from ai_service.infrastructure.providers.factory import build_model_provider
from ai_service.schemas.agent import ShoppingAnswer

LOGGER = logging.getLogger(__name__)


class PydanticAIAnswerGenerator(AnswerGenerator, StreamingAnswerGenerator):
    """Optional PydanticAI adapter with a safe local fallback.

    The model is intentionally lazy. Local development and tests can run with
    no provider credentials, while a configured provider/model enables the same
    routes to use a real model. A provider failure never turns a catalog
    response into an unhandled 500; the deterministic answer remains available.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._agent: Agent[None, ShoppingAnswer] | None = None
        self._stream_agent: Agent[None, str] | None = None
        self._model: str | Model | None = None
        self._initialization_attempted = False
        self._stream_initialization_attempted = False

    def _get_model(self) -> str | Model | None:
        """Resolve the configured model through the infrastructure port."""
        if self._initialization_attempted:
            return self._model

        configured_fallback_model = (
            self.settings.provider == AIProvider.FALLBACK and not self.settings.model_name
        )
        if configured_fallback_model:
            self._initialization_attempted = True
            return None

        self._initialization_attempted = True
        try:
            provider = build_model_provider(self.settings)
            self._model = (
                cast(str | Model, provider.create_model())
                if provider is not None
                else self.settings.model_name
            )
        except Exception as exc:  # noqa: BLE001 - provider implementations vary
            LOGGER.warning(
                "Unable to initialize AI model provider %r: %s",
                self.settings.provider,
                exc,
            )
            self._model = None
        return self._model

    def _get_agent(self) -> Agent[None, ShoppingAnswer] | None:
        if self._agent is not None:
            return self._agent

        model = self._get_model()
        if model is not None:
            try:
                self._agent = create_shopping_agent(
                    model,
                    system_prompt=self.settings.model_system_prompt,
                )
            except Exception as exc:  # noqa: BLE001 - provider implementations vary
                LOGGER.warning("Unable to initialize AI answer agent: %s", exc)
        return self._agent

    def _get_stream_agent(self) -> Agent[None, str] | None:
        if self._stream_agent is not None:
            return self._stream_agent
        if self._stream_initialization_attempted:
            return None

        self._stream_initialization_attempted = True
        model = self._get_model()
        if model is not None:
            try:
                self._stream_agent = create_streaming_shopping_agent(
                    model,
                    system_prompt=self.settings.model_system_prompt,
                )
            except Exception as exc:  # noqa: BLE001 - provider implementations vary
                LOGGER.warning("Unable to initialize AI streaming agent: %s", exc)
        return self._stream_agent

    async def generate(self, prompt: str, fallback: str) -> str:
        agent = self._get_agent()
        if agent is None:
            return fallback

        try:
            result = await agent.run(prompt)
            answer = result.output.answer.strip()
            return answer or fallback
        except Exception as exc:  # noqa: BLE001 - provider/network errors vary
            LOGGER.warning("AI model call failed; using deterministic fallback: %s", exc)
            return fallback

    async def stream(self, prompt: str, fallback: str) -> AsyncIterator[str]:
        """Yield text deltas and fall back to one deterministic delta on failure."""
        agent = self._get_stream_agent()
        if agent is None:
            yield fallback
            return

        yielded = False
        try:
            async with agent.run_stream(prompt) as result:
                async for delta in result.stream_text(delta=True, debounce_by=None):
                    if delta:
                        yielded = True
                        yield delta
        except Exception as exc:  # noqa: BLE001 - provider/network errors vary
            LOGGER.warning(
                "AI streaming call failed; using deterministic fallback: %s",
                exc,
            )
            if not yielded:
                yield fallback


__all__ = ["PydanticAIAnswerGenerator"]
