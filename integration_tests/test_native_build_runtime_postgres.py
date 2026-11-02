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

from ai_service.capabilities.assistant.source_quotes import (
    QuotedRequirementDraft,
    QuotedTaskBatch,
)
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
        (True, "v2", False, 2),
    ],
)
async def test_native_runtime_claims_http_run_and_closes_provider_after_drain(
    postgres_dsn, monkeypatch, react_enabled, version, retry, selected
):
    owner = uuid4()
    product_id = uuid4()
    selected_ids = [product_id, uuid4()] if selected == 2 else [product_id]
    closed = []
    calls = []
    web_settings = []
    failures = []
    from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator

    publication = PostgresRunCoordinator._publication

    async def observed_publication(self, claim, run):
        try:
            return await publication(self, claim, run)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(PostgresRunCoordinator, "_publication", observed_publication)
    backend_calls = []
    conversation_contexts = []
    fail_model = [False]
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
        async def astream(self, messages):
            from langchain_core.messages import AIMessageChunk

            response = await self.ainvoke(messages)
            yield AIMessageChunk(
                content=response.content,
                tool_calls=response.tool_calls,
                additional_kwargs=response.additional_kwargs,
            )

        def bind_tools(self, tools, **kwargs):
            self.tools = {tool["name"] for tool in tools}
            return self

        def with_structured_output(self, schema, **kwargs):
            self.schema = schema
            return self

        async def ainvoke(self, messages):
            if fail_model[0]:
                raise RuntimeError("Explicit failed-build provider fixture")
            calls.append(True)
            if react_enabled and not isinstance(messages[0], tuple):
                if version == "v2":
                    instruction = messages[0].content
                    context_json = json.loads(
                        instruction.split(" Application context (data): ", 1)[1]
                    )
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
                        tool_calls=(
                            [
                                {
                                    "id": "register",
                                    "name": "register_tasks",
                                    "args": {
                                        "tasks": [
                                            {
                                                "kind": "ANSWER"
                                                if read_only
                                                else "BUILD",
                                                "quote": messages[1].content,
                                            }
                                        ]
                                    },
                                }
                            ]
                            if "register_tasks" in self.tools
                            else []
                        )
                        + [
                            {
                                "id": "search" if read_only else "build",
                                "name": "compare_products"
                                if selected == 2
                                else "get_product_details"
                                if selected
                                else "search_catalog"
                                if read_only
                                else "build_pc",
                                "args": {"product_ids": [str(p) for p in selected_ids]}
                                if selected == 2
                                else {"product_id": str(product_id)}
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
                        content="",
                        tool_calls=[
                            {
                                "id": "submit",
                                "name": "submit_result",
                                "args": {
                                    "results": [
                                        {
                                            "text": "Catalog result",
                                            "refs": result["evidence_refs"],
                                            "build_ref": None,
                                            "explanation": "Đối chiếu thông số và nhu cầu trước khi quyết định chọn linh kiện."
                                            if selected
                                            else None,
                                        }
                                    ],
                                },
                            }
                        ],
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
            source = json.loads(messages[-1][1])["user_message"]
            if react_enabled and self.schema is QuotedTaskBatch:
                return {
                    "tasks": [
                        {
                            "kind": "ANSWER" if source == "Find GPU" else "BUILD",
                            "quote": source,
                        }
                    ],
                }
            if self.schema is QuotedRequirementDraft:
                if source == "Find GPU":
                    return {"items": []}
                return {
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
                            "quote": source,
                        },
                        {
                            "target": "GOAL",
                            "subject": "WORKLOAD",
                            "predicate": "use_case",
                            "value": {"kind": "TEXT", "text": "GAMING_1080P"},
                            "quote": source,
                        },
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
        if request.url.path in {f"/products/{p}" for p in selected_ids}:
            requested_id = UUID(request.url.path.rsplit("/", 1)[1])
            return httpx.Response(
                200,
                json={
                    "data": {
                        "id": str(requested_id),
                        "name": "Canonical GPU",
                        "status": "ACTIVE",
                        "specifications": {"vram_gb": 8},
                        "variants": [
                            {
                                "id": str(uuid4()),
                                "productId": str(requested_id),
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
                payload = {
                    "conversation_id": str(conversation_id),
                    "request_id": str(uuid4()),
                    "expected_revision": 0,
                    "message": "Find GPU"
                    if version == "v2"
                    else "Build 30 triệu gaming",
                    **({"schema_version": 2} if version == "v2" else {}),
                    **(
                        {"selected_product_ids": [str(p) for p in selected_ids]}
                        if selected
                        else {}
                    ),
                }
                accepted = await client.post(
                    path + "/runs",
                    json=payload,
                )
                assert accepted.status_code == 202
                run_id = accepted.json()["data"]["run_id"]
                # Lost admission response: the identical request must reuse the run.
                retried = await client.post(path + "/runs", json=payload)
                assert retried.status_code == 202
                assert retried.json()["data"]["run_id"] == run_id
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
                # This also holds after publication advanced the conversation revision.
                replayed = await client.post(path + "/runs", json=payload)
                assert replayed.status_code == 202
                assert replayed.json()["data"]["run_id"] == run_id
                messages = history.json()["data"]["items"]
                assert [m["role"] for m in messages] == ["user", "assistant"]
                # Empty catalog must be truthful infeasibility, never sample data.
                if version == "v2":
                    assert "8,000,000 VND" in messages[-1]["content"]
                    view = (await client.get(path)).json()["data"]
                    assert view["revision"] == 1
                    assert view["flexible"]["status"] == "COMPLETED"
                    assert view["flexible"]["task_outcomes"][0]["kind"] == (
                        "COMPARE" if selected == 2 else "ANSWER"
                    )
                else:
                    assert messages[-1]["content"].startswith("INFEASIBLE")
                assert (
                    len(calls)
                    == (2 if selected else 3 if retry else 2 if react_enabled else 1)
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
                            assert (
                                "Đối chiếu thông số và nhu cầu"
                                in messages[-1]["content"]
                            )
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
                    first_page = await client.get(
                        path + "/messages", params={"limit": 2}
                    )
                    assert [
                        m["sequence"] for m in first_page.json()["data"]["items"]
                    ] == [1, 2]
                    cursor = first_page.json()["data"]["next_cursor"]
                    assert cursor is not None
                    next_page = await client.get(
                        path + "/messages", params={"limit": 2, "cursor": cursor}
                    )
                    assert [
                        m["sequence"] for m in next_page.json()["data"]["items"]
                    ] == [3, 4]
                    assert next_page.json()["data"]["next_cursor"] is None
                    assert (
                        await client.get(
                            path + "/messages", params={"cursor": str(uuid4())}
                        )
                    ).status_code == 400
                    assert (
                        await client.get(path + "/messages", params={"limit": 0})
                    ).status_code == 422
                    stale = await client.post(
                        path + "/runs",
                        json={
                            "schema_version": 2,
                            "conversation_id": str(conversation_id),
                            "request_id": str(uuid4()),
                            "expected_revision": 1,
                            "message": "Stale revision",
                        },
                    )
                    assert stale.status_code == 409
                    assert stale.json()["message"] == "AI_REVISION_CONFLICT"
                    view = await client.get(path)
                    assert view.json()["data"]["revision"] == 2
                    # Ordinary follow-up retains the prior attachment turn in history.
                    assert len(calls) == (4 if selected else 5 if retry else 4)
                    if version == "v2":
                        last_context = conversation_contexts[-1]
                        assert len(last_context) == 1
                        assert last_context[0]["user_text"] == "Find GPU"
                    if selected:
                        # A terminal failed build must not poison subsequent reads.
                        fail_model[0] = True
                        failed = await client.post(
                            path + "/runs",
                            json={
                                "schema_version": 2,
                                "conversation_id": str(conversation_id),
                                "request_id": str(uuid4()),
                                "expected_revision": 2,
                                "message": "Build PC 25 triệu gaming 1080p",
                            },
                        )
                        assert failed.status_code == 202
                        failed_id = failed.json()["data"]["run_id"]
                        async with asyncio.timeout(5):
                            while (
                                await client.get(path + f"/runs/{failed_id}")
                            ).json()["data"]["status"] in {"PENDING", "RUNNING"}:
                                await asyncio.sleep(0.01)
                        assert (await client.get(path + f"/runs/{failed_id}")).json()[
                            "data"
                        ]["status"] == "FAILED_TERMINAL"
                        assert (await client.get(path)).json()["data"]["revision"] == 2
                        fail_model[0] = False
                        recovered = await client.post(
                            path + "/runs",
                            json={
                                "schema_version": 2,
                                "conversation_id": str(conversation_id),
                                "request_id": str(uuid4()),
                                "expected_revision": 2,
                                "message": "Find GPU",
                                "selected_product_ids": [str(p) for p in selected_ids],
                            },
                        )
                        assert recovered.status_code == 202  # never CONVERSATION_BUSY
                        recovered_id = recovered.json()["data"]["run_id"]
                        async with asyncio.timeout(5):
                            while (
                                await client.get(path + f"/runs/{recovered_id}")
                            ).json()["data"]["status"] in {"PENDING", "RUNNING"}:
                                await asyncio.sleep(0.01)
                        assert (
                            await client.get(path + f"/runs/{recovered_id}")
                        ).json()["data"]["status"] == "COMPLETED"
                        assert (await client.get(path)).json()["data"]["revision"] == 3
                        result = await client.get(path + f"/runs/{recovered_id}/result")
                        assert result.status_code == 200
                        assert (
                            "8,000,000 VND" in result.json()["data"]["result"]["answer"]
                        )
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
