"""V2 admission/payload gate, without selecting a V2 executor or publishing."""

from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.capabilities.assistant.react.turn_contracts import (
    TurnRequestV2,
    parse_turn_request,
)
from ai_service.capabilities.assistant.stateful_contracts import TurnRequestV1
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator
from ai_service.infrastructure.persistence.schema import agent_runs, conversations


@pytest.mark.asyncio
async def test_v2_request_admission_preserves_commands_and_idempotency(postgres_dsn):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    conversation = None
    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            owner = uuid4()
            store = PostgresConversationStore(engine)
            conversation = await store.create(owner, None)
            conversation = await store.initialize(
                owner, conversation.id, NativeConversationInitializer(saver)
            )
            request = TurnRequestV2(
                conversation_id=conversation.id,
                request_id=uuid4(),
                expected_revision=0,
                message="OK",
                confirmation={
                    "proposal_id": uuid4(),
                    "proposal_hash": "a" * 64,
                    "target_revision": 1,
                    "base_requirements_hash": "b" * 64,
                },
            )
            # Default V1 coordinator rejects V2 before a run/message is inserted.
            with pytest.raises(
                RunAdmissionError, match="AI_REQUEST_VERSION_UNSUPPORTED"
            ):
                await PostgresRunCoordinator(engine, saver).submit(owner, request)
            coordinator = PostgresRunCoordinator(engine, saver, request_versions=(2,))
            with pytest.raises(
                RunAdmissionError, match="AI_REQUEST_VERSION_UNSUPPORTED"
            ):
                await coordinator.submit(
                    owner,
                    TurnRequestV1(
                        conversation_id=conversation.id,
                        request_id=uuid4(),
                        expected_revision=0,
                        message="Legacy",
                    ),
                )
            submitted = await coordinator.submit(owner, request)
            assert (await coordinator.submit(owner, request)).run_id == submitted.run_id
            changed = request.model_copy(deep=True)
            changed.confirmation.proposal_hash = "c" * 64
            with pytest.raises(RunAdmissionError, match="AI_REQUEST_CONFLICT"):
                await coordinator.submit(owner, changed)
            async with engine.connect() as conn:
                rows = (
                    (
                        await conn.execute(
                            select(agent_runs).where(
                                agent_runs.c.conversation_id == conversation.id
                            )
                        )
                    )
                    .mappings()
                    .all()
                )
            assert len(rows) == 1
            assert parse_turn_request(rows[0]["request_payload"]) == request
            assert rows[0]["status"] == "PENDING"
    finally:
        if conversation is not None:
            async with engine.begin() as conn:
                await conn.execute(
                    delete(conversations).where(conversations.c.id == conversation.id)
                )
            async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
                await saver.adelete_thread(str(conversation.id))
        await engine.dispose()
