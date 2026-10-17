from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_stateful_api import OwnedStore, Tokens

from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
    StatefulRunService,
)
from ai_service.capabilities.assistant.public_contracts import RunViewV1
from ai_service.config.settings import Settings
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.main import create_app


@pytest.mark.asyncio
async def test_v2_http_schema_requires_version_and_keeps_envelope_owner_boundaries(
    monkeypatch,
):
    owner, saver, received = uuid4(), InMemorySaver(), []

    class Runs:
        async def submit(self, actor, request):
            assert actor == owner
            received.append(request)
            return RunViewV1(
                run_id=uuid4(),
                conversation_id=request.conversation_id,
                request_id=request.request_id,
                status="PENDING",
                base_revision=request.expected_revision,
            )

    resources = StatefulResources(
        StatefulConversationService(
            OwnedStore(),
            PostgresAcceptedCheckpointReader(saver),
            NativeConversationInitializer(saver),
        ),
        Tokens(owner),
        StatefulRunService(Runs(), lambda: None),
    )

    @asynccontextmanager
    async def opened(settings):
        yield resources

    monkeypatch.setattr(
        "ai_service.infrastructure.stateful.open_stateful_resources", opened
    )
    settings = Settings(
        _env_file=None,
        provider="openai",
        openai_api_key="fixture",
        database_url="postgresql+psycopg://fixture:fixture@localhost/ai_db",
        jwt_jwks_url="http://identity.test/jwks",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):  # noqa: SIM117 -- lifespan must start before client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer owner"},
        ) as client:
            created = await client.post("/api/v1/conversations", json={})
            assert created.status_code == 201
            data = created.json()["data"]
            assert data["schema_version"] == 2 and data["flexible"] is None
            assert set(created.json()) == {"data", "message", "errors"}
            identity = UUID(data["conversation_id"])
            path = f"/api/v1/conversations/{identity}/runs"
            body = {
                "conversation_id": str(identity),
                "request_id": str(uuid4()),
                "expected_revision": 0,
                "message": "GPU?",
            }
            assert (await client.post(path, json=body)).status_code == 422
            accepted = await client.post(path, json=body | {"schema_version": 2})
            assert accepted.status_code == 202
            assert accepted.json()["message"] == "AI_RUN_ACCEPTED"
            assert len(received) == 1 and received[0].schema_version == 2
            assert (
                await client.post(
                    path,
                    json=body | {"schema_version": 2},
                    headers={"Authorization": "Bearer invalid"},
                )
            ).status_code == 401
            schema = (await client.get("/openapi.json")).json()
            body_schema = schema["paths"][
                "/api/v1/conversations/{conversation_id}/runs"
            ]["post"]["requestBody"]["content"]["application/json"]["schema"]
            selected = schema["components"]["schemas"][
                body_schema["$ref"].split("/")[-1]
            ]
            assert "schema_version" in selected["required"]
            assert "confirmation" in selected["properties"]
            assert "clarification_reply" not in selected["properties"]


def test_server_serves_only_v2_without_enable_or_version_flags():
    settings = Settings(_env_file=None)
    assert not {
        "stateful_enabled", "stateful_build_enabled", "stateful_react_enabled",
        "stateful_react_version",
    } & Settings.model_fields.keys()
    schema = create_app(settings).openapi()
    assert "TurnRequestV1" not in schema["components"]["schemas"]
    body = schema["paths"]["/api/v1/conversations/{conversation_id}/runs"]["post"]["requestBody"]
    assert body["content"]["application/json"]["schema"]["$ref"].endswith("TurnRequestV2Body")


@pytest.mark.asyncio
async def test_v2_startup_without_dependencies_fails_instead_of_serving_fallback():
    app = create_app(Settings(_env_file=None))
    with pytest.raises(ValueError, match="AI_DATABASE_URL"):
        async with app.router.lifespan_context(app):
            pytest.fail("Missing dependencies must not yield a running app")
