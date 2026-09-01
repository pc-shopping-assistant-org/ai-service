"""FastAPI dependency adapters for the application composition root."""

from functools import lru_cache

from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.infrastructure.composition import ApplicationContainer, build_container


@lru_cache
def get_container() -> ApplicationContainer:
    """Return one process-scoped container; no provider call happens here."""

    return build_container()


def get_assistant_service() -> AssistantUseCase:
    """Expose the assistant inbound port to FastAPI routes."""

    return get_container().assistant


__all__ = ["get_assistant_service", "get_container"]
