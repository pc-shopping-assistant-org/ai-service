"""Lazy, cached registry of per-(model_id, thinking_mode) providers."""

from __future__ import annotations

from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.enums import ThinkingMode
from ai_service.config.models import ModelCatalog, ResolvedModelEntry
from ai_service.infrastructure.providers.gemini import GeminiModelProvider
from ai_service.infrastructure.providers.openai import OpenAIModelProvider
from ai_service.infrastructure.providers.qwen import QwenModelProvider


def _instantiate_provider(
    entry: ResolvedModelEntry,
    thinking_mode: ThinkingMode | None,
) -> ModelProvider:
    if entry.provider == "gemini":
        return GeminiModelProvider.from_entry(entry, thinking_mode)
    if entry.provider == "openai":
        return OpenAIModelProvider.from_entry(entry, thinking_mode)
    if entry.provider == "qwen":
        return QwenModelProvider.from_entry(entry, thinking_mode)  # type: ignore[return-value]
    raise ValueError(f"Unsupported model provider '{entry.provider}'")


class ModelProviderRegistry:
    """Caches provider adapters per (model_id, thinking_mode) for pool reuse."""

    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog
        self._providers: dict[tuple[str, ThinkingMode | None], ModelProvider] = {}
        self._closed = False

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    def resolve_provider(
        self,
        model_id: str | None = None,
        thinking_mode: ThinkingMode | None = None,
    ) -> tuple[ModelProvider, ResolvedModelEntry, ThinkingMode | None]:
        if self._closed:
            raise RuntimeError("ModelProviderRegistry is closed")
        entry, resolved_mode = self._catalog.resolve(model_id, thinking_mode)
        key = (entry.id, resolved_mode)
        provider = self._providers.get(key)
        if provider is None:
            provider = _instantiate_provider(entry, resolved_mode)
            self._providers[key] = provider
        return provider, entry, resolved_mode

    async def aclose(self) -> None:
        self._closed = True
        for provider in list(self._providers.values()):
            await provider.aclose()
        self._providers.clear()
