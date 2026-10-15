"""HTTP replay uses actual persisted events, never invokes graph execution."""

import json
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.use_cases.run_events import StatefulRunEventService
from ai_service.application.use_cases.stateful_conversations import StatefulRunService
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.config.settings import Settings
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.events import PostgresRunEventStore
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import agent_run_events, conversations
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [2])
async def test_basic_sse_owner_and_committed_events(postgres_dsn, version):
    owner, other = uuid4(), uuid4()
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    conversation = None

    class Auth:
        async def authenticate(self, token):
            return owner if token == "owner" else other

    def no_execution():
        raise AssertionError("Subscription cannot wake execution")

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            store = PostgresConversationStore(engine)
            conversation = await store.create(owner, "SSE fixture")
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            coordinator = PostgresRunCoordinator(engine, saver, request_versions=(2,))
            submitted = await coordinator.submit(
                owner,
                TurnRequestV2(
                    conversation_id=conversation.id,
                    request_id=uuid4(),
                    expected_revision=0,
                    message="SSE fixture",
                ),
            )
            await coordinator.cancel(owner, conversation.id, submitted.run_id)
            reader = PostgresRunEventStore(engine)
            events = await reader.read(owner, conversation.id, submitted.run_id)
            app = create_app(Settings(_env_file=None))
            app.state.stateful = StatefulResources(
                None,
                Auth(),
                StatefulRunService(coordinator, no_execution),
                StatefulRunEventService(reader, coordinator),
            )
            path = f"/api/v1/conversations/{conversation.id}/runs/{submitted.run_id}/events"
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://fixture",
            ) as client:
                headers = {"Authorization": "Bearer owner"}
                assert (await client.get(path)).status_code == 401
                for target, actor in (
                    (path, "other"),
                    (path.replace(str(submitted.run_id), str(uuid4())), "owner"),
                ):
                    missing = await client.get(
                        target, headers={"Authorization": f"Bearer {actor}"}
                    )
                    assert missing.status_code == 404
                    assert missing.json()["message"] == "AI_CONVERSATION_NOT_FOUND"
                response = await client.get(path, headers=headers)
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                assert response.headers["x-accel-buffering"] == "no"
                frames = response.text.strip().split("\n\n")
                assert len(frames) == 2
                for frame, expected in zip(frames, events, strict=True):
                    lines = frame.splitlines()
                    assert lines[0] == f"event: {expected.envelope.data.event.lower()}"
                    assert json.loads(lines[1][6:]) == expected.envelope.model_dump(
                        mode="json"
                    )
                assert "id: " not in response.text
                schema = app.openapi()["paths"][
                    path.replace(str(conversation.id), "{conversation_id}").replace(
                        str(submitted.run_id), "{run_id}"
                    )
                ]["get"]
                assert "text/event-stream" in schema["responses"]["200"]["content"]
                assert all(p["name"] != "Last-Event-ID" for p in schema["parameters"])
                async with engine.begin() as conn:
                    await conn.execute(
                        delete(agent_run_events).where(
                            agent_run_events.c.id == events[0].event_id
                        )
                    )
                gap = await client.get(path, headers=headers)
                assert gap.status_code == 410
                assert gap.json()["message"] == "AI_STREAM_INTERRUPTED"
                after = await store.get(owner, conversation.id)
                assert after.revision == 0 and after.active_run_id is None
    finally:
        if conversation:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
