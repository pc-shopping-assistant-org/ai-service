"""Actual lifespan, SQL/saver/executor; model/catalog/auth are explicit fixtures."""

import asyncio
import json
from uuid import UUID, uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.config.settings import Settings
from ai_service.infrastructure.persistence.schema import conversations
from ai_service.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "react_enabled,version,retry,selected",
    [
        (True, "v2", False, False),
        (True, "v2", True, False),
        (True, "v2", False, True),
    ],
)
async def test_native_runtime_claims_http_run_and_closes_provider_after_drain(
    postgres_dsn, monkeypatch, react_enabled, version, retry, selected
):
    owner = uuid4()
    product_id = uuid4()
    closed = []
    calls = []
    web_settings = []
    failures = []
    backend_calls = []
    conversation_contexts = []
    from ai_service.infrastructure.execution.conversation import ConversationExecution

    execute = ConversationExecution.__call__

    async def observed_execution(self, claim, children):
        try:
            await execute(self, claim, children)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(ConversationExecution, "__call__", observed_execution)
    from ai_service.infrastructure.search.duckduckgo_adapter import (
        DuckDuckGoSearchAdapter,
    )
    from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

    def search_adapter(settings):
        web_settings.append(settings)
        return DuckDuckGoSearchAdapter(settings)

    def reader_adapter(settings):
        web_settings.append(settings)
        return TrafilaturaReader(settings)

    monkeypatch.setattr(
        "ai_service.infrastructure.execution.build_runtime.DuckDuckGoSearchAdapter",
        search_adapter,
    )
    monkeypatch.setattr(
        "ai_service.infrastructure.execution.build_runtime.TrafilaturaReader",
        reader_adapter,
    )

    class Model:
        def bind_tools(self, tools, **kwargs):
            return self

        def with_structured_output(self, schema, **kwargs):
            return self

        async def ainvoke(self, messages):
            calls.append(True)
            if react_enabled and not isinstance(messages[0], tuple):
                if version == "v2":
                    instruction = messages[0].content
                    context_json = json.loads(instruction.rsplit("\n", 1)[-1])
                    conversation_contexts.append(context_json.get("history", []))
                read_only = messages[1].content == "Find GPU"
                results = [
                    json.loads(m.content)
                    for m in messages
                    if isinstance(m, ToolMessage)
                ]
                if not results:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "search" if read_only else "build",
                                "name": "get_product_details"
                                if selected
                                else "search_catalog"
                                if read_only
                                else "build_pc",
                                "args": {"product_id": str(product_id)}
                                if selected
                                else {"query": "GPU"}
                                if read_only
                                else {},
                            }
                        ],
                    )
                result = results[-1]
                if retry and not failures:
                    from openai import APITimeoutError

                    raise APITimeoutError(
                        request=httpx.Request("POST", "https://model.test/chat")
                    )
                if version == "v2":
                    return AIMessage(
                        content=json.dumps(
                            {
                                "action": {
                                    "kind": "FINAL_ANSWER",
                                    "answer": {
                                        "text": "Catalog result",
                                        "evidence_refs": result["evidence_refs"],
                                        "build_ref": None,
                                        "explanation": {
                                            "text": "Đối chiếu thông số và nhu cầu trước khi quyết định chọn linh kiện.",
                                            "evidence_refs": result["evidence_refs"],
                                        }
                                        if selected
                                        else None,
                                    },
                                }
                            }
                        )
                    )
                return AIMessage(
                    content=json.dumps(
                        {
                            "text": "Build done",
                            "evidence_refs": result["evidence_refs"],
                            "build_ref": result["result"].get("build_ref"),
                        }
                    )
                )
            if (
                react_enabled
                and json.loads(messages[-1][1])["user_message"] == "Find GPU"
            ):
                return {
                    **({"items": []} if version == "v1" else {}),
                    "tasks": [
                        {"kind": "ANSWER", "start": 0, "end": 8, "quote": "Find GPU"}
                    ],
                }
            return {
                "tasks": [
                    {
                        "kind": "BUILD",
                        "start": 0,
                        "end": 21,
                        "quote": "Build 30 triệu gaming",
                    }
                ]
                if react_enabled
                else [],
                "items": [
                    {
                        "subject": "SETUP",
                        "predicate": "budget",
                        "operator": "LTE",
                        "value": {
                            "kind": "QUANTITY",
                            "amount": 30_000_000,
                            "unit": "VND",
                        },
                        "start": 0,
                        "end": 21,
                    },
                    {
                        "target": "GOAL",
                        "subject": "WORKLOAD",
                        "predicate": "use_case",
                        "value": {"kind": "TEXT", "text": "GAMING_1080P"},
                        "start": 0,
                        "end": 21,
                    },
                ],
            }

    class Provider:
        name = "openai"

        def create_model(self):
            return Model()

        async def aclose(self):
            closed.append(True)

    class Auth:
        async def authenticate(self, token):
            assert token == "test-owner"
            return owner

    def respond(request):
        backend_calls.append(request.url.path)
        if request.url.path == f"/products/{product_id}":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": str(product_id),
                        "name": "Canonical GPU",
                        "status": "ACTIVE",
                        "specifications": {"vram_gb": 8},
                        "variants": [
                            {
                                "id": str(uuid4()),
                                "productId": str(product_id),
                                "price": 8_000_000,
                                "quantity": 1,
                                "status": "ACTIVE",
                            }
                        ],
                    },
                    "message": "SUCCESS",
                    "errors": [],
                },
            )
        if request.url.path == "/products":
            return httpx.Response(
                200,
                json={
                    "data": {
                        "items": [
                            {
                                "id": str(product_id),
                                "name": "Canonical GPU",
                                "status": "ACTIVE",
                                "minPrice": 8_000_000,
                                "maxPrice": 8_000_000,
                            }
                        ],
                        "hasNext": False,
                    },
                    "message": "SUCCESS",
                    "errors": [],
                },
            )
        assert request.url.path == "/products/snapshot"
        assert version != "v2", (
            "Read-only V2 turns must not preload optimizer candidates"
        )
        return httpx.Response(
            200,
            json={
                "data": {"items": [], "hasNext": False, "nextCursor": None},
                "message": "SUCCESS",
                "errors": [],
            },
        )

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
        openai_api_key="local-test-key",
        backend_api_url="https://catalog.test",
        database_url=postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        jwt_jwks_url="http://identity.test/jwks",
        hardware_catalog_mapping={"categories": {str(uuid4()): "CPU"}},
    )
    app = create_app(settings)
    conversation_id = None
    try:
        async with app.router.lifespan_context(app):
            assert web_settings == ([settings, settings] if react_enabled else [])
            async with original_client(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Authorization": "Bearer test-owner"},
            ) as client:
                created = await client.post("/api/v1/conversations", json={})
                assert created.status_code == 201
                if version == "v2":
                    assert created.json()["data"]["schema_version"] == 2
                conversation_id = UUID(created.json()["data"]["conversation_id"])
                path = f"/api/v1/conversations/{conversation_id}"
                accepted = await client.post(
                    path + "/runs",
                    json={
                        "conversation_id": str(conversation_id),
                        "request_id": str(uuid4()),
                        "expected_revision": 0,
                        "message": "Find GPU"
                        if version == "v2"
                        else "Build 30 triệu gaming",
                        **({"schema_version": 2} if version == "v2" else {}),
                        **(
                            {"selected_product_ids": [str(product_id)]}
                            if selected
                            else {}
                        ),
                    },
                )
                assert accepted.status_code == 202
                run_id = accepted.json()["data"]["run_id"]
                async with asyncio.timeout(5):
                    while True:
                        response = await client.get(path + f"/runs/{run_id}")
                        status = response.json()["data"]["status"]
                        if status not in {
                            "PENDING",
                            "RUNNING",
                            "FAILED_RETRYABLE",
                            "RECOVERING",
                        }:
                            assert status == "COMPLETED", failures
                            break
                        await asyncio.sleep(0.01)
                history = await client.get(path + "/messages")
                messages = history.json()["data"]["items"]
                assert [m["role"] for m in messages] == ["user", "assistant"]
                # Empty catalog must be truthful infeasibility, never sample data.
                if version == "v2":
                    assert "8,000,000 VND" in messages[-1]["content"]
                    view = (await client.get(path)).json()["data"]
                    assert view["revision"] == 1
                    assert view["flexible"]["status"] == "COMPLETED"
                    assert view["flexible"]["task_outcomes"][0]["kind"] == "ANSWER"
                else:
                    assert messages[-1]["content"].startswith("INFEASIBLE")
                assert (
                    len(calls)
                    == (2 if selected else 4 if retry else 3 if react_enabled else 1)
                    and closed == []
                )
                if react_enabled:
                    if version == "v2":
                        stream = await client.get(path + f"/runs/{run_id}/events")
                        assert stream.status_code == 200
                        assert (
                            "event: completed" in stream.text
                            and "id: " not in stream.text
                        )
                        if selected:
                            assert "Nhận định tư vấn" in messages[-1]["content"]
                            completed = json.loads(
                                stream.text.strip()
                                .split("\n\n")[-1]
                                .split("data: ", 1)[1]
                            )
                            assert completed["data"]["result"]["products"][0][
                                "id"
                            ] == str(product_id)
                    accepted = await client.post(
                        path + "/runs",
                        json={
                            "conversation_id": str(conversation_id),
                            "request_id": str(uuid4()),
                            "expected_revision": 1,
                            "message": "Find GPU",
                            **({"schema_version": 2} if version == "v2" else {}),
                        },
                    )
                    assert accepted.status_code == 202
                    run_id = accepted.json()["data"]["run_id"]
                    async with asyncio.timeout(5):
                        while True:
                            response = await client.get(path + f"/runs/{run_id}")
                            status = response.json()["data"]["status"]
                            if status not in {"PENDING", "RUNNING"}:
                                assert status == "COMPLETED"
                                break
                            await asyncio.sleep(0.01)
                    history = await client.get(path + "/messages")
                    messages = history.json()["data"]["items"]
                    assert [m["role"] for m in messages] == [
                        "user",
                        "assistant",
                        "user",
                        "assistant",
                    ]
                    assert "8,000,000 VND" in messages[-1]["content"]
                    view = await client.get(path)
                    assert view.json()["data"]["revision"] == 2
                    # Ordinary follow-up retains the prior attachment turn in history.
                    assert len(calls) == (5 if selected else 7 if retry else 6)
                    if version == "v2":
                        last_context = conversation_contexts[-1]
                        assert len(last_context) == 1
                        assert last_context[0]["user_text"] == "Find GPU"
                        assert "8,000,000 VND" in last_context[0]["assistant_text"]
                    if retry:
                        assert len(failures) == 1
                        assert backend_calls.count("/products/snapshot") == 0
                        assert backend_calls.count("/products") == 2
        assert closed == [True] and app.state.stateful is None
    finally:
        if conversation_id is not None:
            engine = create_async_engine(
                settings.database_url.get_secret_value(), hide_parameters=True
            )
            try:
                async with engine.begin() as conn:
                    await conn.execute(
                        delete(conversations).where(
                            conversations.c.id == conversation_id
                        )
                    )
                async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                    await saver.adelete_thread(str(conversation_id))
            finally:
                await engine.dispose()
