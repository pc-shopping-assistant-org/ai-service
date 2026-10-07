import json
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import pytest
import pytest_asyncio
from pydantic import ValidationError

from ai_service.api.dependencies import get_assistant_service
from ai_service.application.ports.assistant import AssistantUseCase
from ai_service.main import app
from ai_service.schemas.response import ApiResponse
from ai_service.services.assistant_service import AssistantService


@pytest_asyncio.fixture
async def api_client() -> AsyncIterator[httpx.AsyncClient]:
    # ASGI transport avoids TestClient's cross-thread blocking portal, and the
    # backend stub keeps HTTP/SSE tests independent of local services and secrets.
    class CatalogFixture:
        async def search_products(self, query: str, limit: int = 10) -> list[dict]:
            return []

        async def get_product(self, product_id: UUID) -> dict | None:
            return None

    service = AssistantService(CatalogFixture())

    async def assistant_override() -> AssistantUseCase:
        return service

    previous = app.dependency_overrides.get(get_assistant_service)
    app.dependency_overrides[get_assistant_service] = assistant_override
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_assistant_service, None)
        else:
            app.dependency_overrides[get_assistant_service] = previous


@pytest.mark.asyncio
async def test_health_uses_canonical_envelope(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["data", "message", "errors"]
    assert body["message"] == "HEALTH_OK"
    assert body["errors"] == []


@pytest.mark.asyncio
async def test_validation_errors_use_static_message_and_array(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "VALIDATION_ERROR"
    assert isinstance(body["errors"], list)
    assert body["errors"][0]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_http_errors_use_canonical_envelope(api_client: httpx.AsyncClient) -> None:
    response = await api_client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["data"] is None
    assert body["message"] == "ENDPOINT_NOT_FOUND"
    assert body["errors"][0]["code"] == "ENDPOINT_NOT_FOUND"


def test_response_rejects_dynamic_top_level_message() -> None:
    with pytest.raises(ValidationError):
        ApiResponse[None](data=None, message="A translated error message")


def test_response_defaults_to_static_success_message() -> None:
    response = ApiResponse[None](data=None)

    assert response.message == "SUCCESS"


@pytest.mark.asyncio
async def test_chat_stream_returns_sse_frames_with_canonical_envelopes(api_client: httpx.AsyncClient) -> None:
    response = await api_client.post("/api/v1/chat/stream", json={"message": "tìm laptop"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [frame["data"]["event"] for frame in frames] == [
        "START",
        "DELTA",
        "COMPLETED",
    ]
    assert all(list(frame) == ["data", "message", "errors"] for frame in frames)
    assert frames[0]["message"] == "AI_CHAT_STREAM_STARTED"
    assert frames[-1]["message"] == "AI_CHAT_STREAM_COMPLETED"
