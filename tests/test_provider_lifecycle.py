import pytest

from ai_service.config.settings import Settings
from ai_service.infrastructure.providers.gemini import GeminiModelProvider
from ai_service.infrastructure.providers.openai import OpenAIModelProvider


@pytest.mark.asyncio
async def test_openai_provider_reuses_and_closes_owned_clients_without_network():
    provider = OpenAIModelProvider(
        Settings(_env_file=None, openai_api_key="local-test-key")
    )
    first = provider.create_model()
    assert provider.create_model() is first
    await provider.aclose()
    assert first.root_async_client.is_closed()
    assert first.root_client.is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        provider.create_model()


@pytest.mark.asyncio
async def test_gemini_provider_reuses_model_and_closes_on_current_loop(monkeypatch):
    provider = GeminiModelProvider(
        Settings(_env_file=None, gemini_api_key="local-test-key")
    )
    first = provider.create_model()
    assert provider.create_model() is first
    closed = []
    original_close = type(first).aclose

    async def close(self):
        closed.append(True)
        await original_close(self)

    monkeypatch.setattr(type(first), "aclose", close)
    await provider.aclose()
    assert closed == [True]
    with pytest.raises(RuntimeError, match="closed"):
        provider.create_model()
