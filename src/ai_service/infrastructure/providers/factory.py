from ai_service.application.ports.model_provider import ModelProvider
from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.gemini import GeminiModelProvider
from ai_service.infrastructure.providers.openai import OpenAIModelProvider


def build_model_provider(settings: Settings) -> ModelProvider | None:
    """Build the configured provider, or ``None`` for deterministic mode.

    The factory only creates a small adapter object.  SDK imports, API-key
    checks and HTTP client construction remain lazy inside ``create_model``.
    This keeps tests and local fallback mode independent of provider secrets.
    """

    if settings.provider == "openai":
        return OpenAIModelProvider(settings)
    if settings.provider == "gemini":
        return GeminiModelProvider(settings)
    return None
