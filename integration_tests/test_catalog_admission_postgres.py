"""Managed V2 clarification/consent with an unavailable hardware backend."""

import asyncio
import json
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.config.settings import Settings
from ai_service.infrastructure.persistence.schema import conversations
from ai_service.main import create_app


@pytest.mark.asyncio
async def test_build_clarification_and_unit_consent_publish_without_hardware_io(
    postgres_dsn,
    monkeypatch,
):
    owner, calls, outbound = uuid4(), [], []

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            return self

        async def ainvoke(self, messages):
            if isinstance(messages[0], tuple):
                assert json.loads(messages[-1][1])["user_message"] == "Build PC"
                return {
                    "tasks": [
                        {"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}
                    ]
                }
            message = messages[1].content
            calls.append(message)
            action = (
                {
                    "kind": "ASK_CLARIFICATION",
                    "question": "Ngân sách và nhu cầu sử dụng của bạn là gì?",
                    "missing_fields": ["BUDGET_REQUIRED", "USE_CASE_REQUIRED"],
                }
                if message == "Build PC"
                else {
                    "kind": "PROPOSE_REQUIREMENTS",
                    "source_text": "RAM 32",
                    "draft": {
                        "items": [
                            {
                                "subject": "RAM",
                                "predicate": "capacity",
                                "start": 0,
                                "end": 6,
                                "value": {"kind": "UNRESOLVED_QUANTITY", "amount": 32},
                            }
                        ]
                    },
                }
            )
            return AIMessage(content=json.dumps({"action": action}))

    class Provider:
        name = "openai"

        def create_model(self):
            return Model()

        async def aclose(self):
            pass

    class Auth:
        async def authenticate(self, token):
            return owner

    def respond(request):
        outbound.append(request.url.path)
        return httpx.Response(503)

    original_client = httpx.AsyncClient

    def client_factory(**kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(respond)
        return original_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        "ai_service.infrastructure.execution.build_runtime.build_model_provider",
        lambda settings: Provider(),
    )
    monkeypatch.setattr(
        "ai_service.infrastructure.stateful.JwtAuthenticator", lambda *args: Auth()
    )
    settings = Settings(
        _env_file=None,
        provider="openai",
        openai_api_key="fixture",
        backend_api_url="https://catalog.test",
        database_url=postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        jwt_jwks_url="http://identity.test/jwks",
        hardware_catalog_mapping={"categories": {str(uuid4()): "CPU"}},
    )
    app, conversation_id = create_app(settings), None
    try:
        async with (
            app.router.lifespan_context(app),
            original_client(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer fixture"},
            ) as client,
        ):
            created = await client.post("/api/v1/conversations", json={})
            assert created.status_code == 201
            conversation_id = UUID(created.json()["data"]["conversation_id"])
            path = f"/api/v1/conversations/{conversation_id}"

            async def turn(message, revision, **extra):
                submitted = await client.post(
                    path + "/runs",
                    json={
                        "schema_version": 2,
                        "conversation_id": str(conversation_id),
                        "request_id": str(uuid4()),
                        "expected_revision": revision,
                        "message": message,
                        **extra,
                    },
                )
                assert submitted.status_code == 202
                run_id = submitted.json()["data"]["run_id"]
                async with asyncio.timeout(5):
                    while True:
                        view = (await client.get(path + f"/runs/{run_id}")).json()[
                            "data"
                        ]
                        if view["status"] not in {
                            "PENDING",
                            "RUNNING",
                            "FAILED_RETRYABLE",
                            "RECOVERING",
                        }:
                            assert view["status"] == "WAITING_INPUT", view
                            break
                        await asyncio.sleep(0.01)
                public = (await client.get(path)).json()["data"]
                assert public["revision"] == revision + 1
                assert public["flexible"]["status"] == "WAITING_INPUT"
                return public

            first = await turn("Build PC", 0)
            assert first["flexible"]["missing_fields"] == [
                "BUDGET_REQUIRED",
                "USE_CASE_REQUIRED",
            ]
            second = await turn("RAM 32", 1)
            pending = second["flexible"]["clarification"]
            assert pending is not None
            third = await turn(
                "GB",
                2,
                unit_reply={
                    "pending_message_id": pending["pending_message_id"],
                    "constraints_revision": 0,
                    "selections": [
                        {
                            "item_index": 0,
                            "unit": "GB",
                            "start": 0,
                            "end": 2,
                            "quote": "GB",
                        }
                    ],
                },
            )
            assert third["flexible"]["proposal"] is not None
            assert third["flexible"]["clarification"] is None
            assert calls == ["Build PC", "RAM 32"] and outbound == []
            history = (await client.get(path + "/messages")).json()["data"]["items"]
            assert [item["role"] for item in history] == ["user", "assistant"] * 3
    finally:
        if conversation_id is not None:
            engine = create_async_engine(
                settings.database_url.get_secret_value(), hide_parameters=True
            )
            try:
                async with engine.begin() as connection:
                    await connection.execute(
                        delete(conversations).where(
                            conversations.c.id == conversation_id
                        )
                    )
            finally:
                await engine.dispose()
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation_id))
