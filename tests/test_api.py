import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from ai_service.main import app
from ai_service.schemas.response import ApiResponse


def test_health_uses_canonical_envelope() -> None:
    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["data", "message", "errors"]
    assert body["message"] == "HEALTH_OK"
    assert body["errors"] == []


def test_validation_errors_use_static_message_and_array() -> None:
    response = TestClient(app).post("/api/v1/chat", json={"message": ""})

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "VALIDATION_ERROR"
    assert isinstance(body["errors"], list)
    assert body["errors"][0]["code"] == "VALIDATION_ERROR"


def test_http_errors_use_canonical_envelope() -> None:
    response = TestClient(app).get("/api/v1/does-not-exist")

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


def test_chat_stream_returns_sse_frames_with_canonical_envelopes() -> None:
    response = TestClient(app).post("/api/v1/chat/stream", json={"message": "tìm laptop"})

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
