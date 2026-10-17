"""Accepted snapshot fixture + real SQL/native managed explanation; no live LLM."""

import asyncio
import json
from typing import TypedDict
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import PublicBuildV1
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequestV2
from ai_service.capabilities.assistant.stateful_contracts import TurnRequestV1
from ai_service.config.settings import Settings
from ai_service.infrastructure.execution.build_runtime import open_build_runtime
from ai_service.infrastructure.execution.conversation import ConversationExecution
from ai_service.infrastructure.execution.supervisor import ManagedExecutor
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import conversations


class SeedState(TypedDict, total=False):
    message: str
    message_id: str
    consultation: dict
    requirement_state: dict
    public_build: dict
    answer: str
    phase: str
    outcome: str


@pytest.mark.asyncio
async def test_v2_rejects_legacy_saved_snapshot_after_orphan_and_reopen(
    postgres_dsn, monkeypatch
):
    owner = uuid4()
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store = PostgresConversationStore(engine)
    conversation = await store.create(owner, "Saved build fixture")
    snapshot = PublicBuildV1(
        objective="BALANCED",
        parts=[
            {
                "slot": "GPU",
                "variant_id": uuid4(),
                "name": "Historical GPU",
                "spending_vnd": 8_000_000,
            }
        ],
        total_spending_vnd=8_000_000,
        stale=True,
    )
    message = "Giải thích cấu hình cũ"
    closed = []

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def bind_tools(self, tools, **kwargs):
            return self

        async def ainvoke(self, messages):
            if isinstance(messages[0], tuple):
                return {
                    "items": [],
                    "tasks": [
                        {
                            "kind": "EXPLAIN_SAVED_BUILD",
                            "start": 0,
                            "end": len(message),
                            "quote": message,
                        }
                    ],
                }
            context = json.loads(messages[0].content.rsplit("\n", 1)[-1])
            return AIMessage(
                content=json.dumps(
                    {
                        "text": "GPU costs 7 VND and is perfect for every game",
                        "evidence_refs": [context["saved_build"]["evidence_ref"]],
                    }
                )
            )

    class Provider:
        name = "openai"

        def create_model(self):
            return Model()

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(
        "ai_service.infrastructure.execution.build_runtime.build_model_provider",
        lambda settings: Provider(),
    )
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            builder = StateGraph(SeedState)

            def seed(state):
                return {
                    "consultation": ConsultationStateV1(
                        build_stale=True, published_build_hash=canonical_hash(snapshot)
                    ).model_dump(mode="json"),
                    "public_build": snapshot.model_dump(mode="json"),
                    "answer": "Fixture accepted historical build",
                    "outcome": "BUILT",
                    "phase": "READY_TO_PUBLISH",
                }

            builder.add_node("seed", seed)
            builder.add_edge(START, "seed")
            builder.add_edge("seed", END)
            graph = DurableConversationGraph(builder.compile(checkpointer=saver))
            coordinator = PostgresRunCoordinator(engine, saver)
            executor = ManagedExecutor(
                coordinator,
                ConversationExecution(
                    coordinator, lambda claim: graph, "seed-fixture-v1"
                ),
                worker_id=uuid4(),
                poll_seconds=0.01,
            )
            await executor.start()
            try:
                run = await coordinator.submit(
                    owner,
                    TurnRequestV1(
                        conversation_id=conversation.id,
                        request_id=uuid4(),
                        expected_revision=0,
                        message="Seed saved snapshot",
                    ),
                )
                async with asyncio.timeout(5):
                    while (
                        await coordinator.get(owner, conversation.id, run.run_id)
                    ).status.value in {"PENDING", "RUNNING"}:
                        await asyncio.sleep(0.01)
                assert (
                    await coordinator.get(owner, conversation.id, run.run_id)
                ).status.value == "COMPLETED"
            finally:
                assert await executor.stop(grace_seconds=5)
            accepted = await store.get(owner, conversation.id)
            base_id = accepted.accepted_checkpoint_id
            orphan = snapshot.model_copy(
                update={
                    "total_spending_vnd": 123,
                    "parts": [
                        snapshot.parts[0].model_copy(update={"spending_vnd": 123})
                    ],
                }
            )
            await graph.graph.aupdate_state(
                {
                    "configurable": {
                        "thread_id": str(conversation.id),
                        "checkpoint_ns": "",
                        "checkpoint_id": base_id,
                    }
                },
                {"public_build": orphan.model_dump(mode="json")},
                as_node="seed",
            )

        # Reopen actual PostgreSQL saver; the newer orphan must not become the base.
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            settings = Settings(
                _env_file=None,
                database_url=postgres_dsn.replace(
                    "postgresql://", "postgresql+psycopg://", 1
                ),
                jwt_jwks_url="http://identity.test/jwks",
                provider="openai",
                openai_api_key="fixture-key",
            )

            def reject_network(request):
                pytest.fail(
                    "Saved snapshot explanation must not fetch current catalog/web"
                )

            async with (
                httpx.AsyncClient(
                    transport=httpx.MockTransport(reject_network)
                ) as client,
                open_build_runtime(settings, engine, saver, client) as service,
            ):
                run = await service.submit(
                    owner,
                    TurnRequestV2(
                        conversation_id=conversation.id,
                        request_id=uuid4(),
                        expected_revision=1,
                        message=message,
                    ),
                )
                async with asyncio.timeout(5):
                    while (
                        await service.get(owner, conversation.id, run.run_id)
                    ).status.value in {"PENDING", "RUNNING"}:
                        await asyncio.sleep(0.01)
                assert (
                    await service.get(owner, conversation.id, run.run_id)
                ).status.value == "FAILED_TERMINAL"
                current = await store.get(owner, conversation.id)
                record = await saver.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": str(conversation.id),
                            "checkpoint_ns": "",
                            "checkpoint_id": current.accepted_checkpoint_id,
                        }
                    }
                )
                values = record.checkpoint["channel_values"]
                assert current.accepted_checkpoint_id == base_id
                assert values["public_build"] == snapshot.model_dump(mode="json")
                assert values["answer"] == "Fixture accepted historical build"
                assert current.revision == 1
        assert closed == [True]
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                delete(conversations).where(conversations.c.id == conversation.id)
            )
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
