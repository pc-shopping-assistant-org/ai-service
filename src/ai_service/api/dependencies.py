"""FastAPI dependency adapters for the application composition root."""

from functools import lru_cache

from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.capabilities.pc_builder.application.build_check import (
    BuildCompatibilityService,
)
from ai_service.infrastructure.commerce.variant_components import BackendComponentSource
from ai_service.infrastructure.composition import ApplicationContainer, build_container
from ai_service.infrastructure.hardware import LocalHardwareRuleEngine


@lru_cache
def get_container() -> ApplicationContainer:
    """Return one process-scoped container; no provider call happens here."""

    return build_container()


def get_assistant_service() -> AssistantUseCase:
    """Expose the assistant inbound port to FastAPI routes."""

    return get_container().assistant


@lru_cache
def get_build_compatibility() -> BuildCompatibilityService:
    """The rule engine is stateless; the specs it checks are read from the catalog on each request."""

    return BuildCompatibilityService(LocalHardwareRuleEngine(), BackendComponentSource(get_container().backend_client))


__all__ = ["get_assistant_service", "get_build_compatibility", "get_container"]
