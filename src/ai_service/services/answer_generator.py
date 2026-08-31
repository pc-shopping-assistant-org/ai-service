from __future__ import annotations

import logging
from typing import Protocol

from pydantic_ai import Agent

from ai_service.agents.shopping_agent import create_shopping_agent
from ai_service.config.settings import Settings, get_settings
from ai_service.schemas.agent import ShoppingAnswer

LOGGER = logging.getLogger(__name__)


class AnswerGenerator(Protocol):
    """Generate a grounded answer without changing the HTTP response contract."""

    async def generate(self, prompt: str, fallback: str) -> str:
        """Return a model answer or the supplied deterministic fallback."""


class PydanticAIAnswerGenerator:
    """Optional PydanticAI adapter with a safe local fallback.

    The model is intentionally lazy.  Local development and tests can run with
    no provider credentials, while a configured ``AI_MODEL_NAME`` (for example
    ``openai:gpt-4o-mini``) enables the same routes to use a real model.  A
    provider failure never turns a catalog response into an unhandled 500; the
    deterministic answer remains available and the event is logged by the
    caller's process.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._agent: Agent[None, ShoppingAnswer] | None = None
        self._initialization_attempted = False

    def _get_agent(self) -> Agent[None, ShoppingAnswer] | None:
        if not self.settings.model_name or self._initialization_attempted:
            return self._agent

        self._initialization_attempted = True
        try:
            self._agent = create_shopping_agent(
                self.settings.model_name,
                system_prompt=self.settings.model_system_prompt,
            )
        except Exception as exc:  # noqa: BLE001 - provider implementations vary
            # Missing provider credentials or an invalid model must not make
            # the local deterministic API unavailable.
            LOGGER.warning("Unable to initialize AI model %r: %s", self.settings.model_name, exc)
            self._agent = None
        return self._agent

    async def generate(self, prompt: str, fallback: str) -> str:
        agent = self._get_agent()
        if agent is None:
            return fallback

        try:
            result = await agent.run(prompt)
            answer = result.output.answer.strip()
            return answer or fallback
        except Exception as exc:  # noqa: BLE001 - provider/network errors vary
            # Keep catalog-grounded behavior available during provider/network
            # outages.  The route still exposes its normal static message key.
            LOGGER.warning("AI model call failed; using deterministic fallback: %s", exc)
            return fallback
