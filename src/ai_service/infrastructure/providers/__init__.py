"""LLM provider adapters used by the application ports."""

from ai_service.infrastructure.providers.factory import build_model_provider

__all__ = ["build_model_provider"]
