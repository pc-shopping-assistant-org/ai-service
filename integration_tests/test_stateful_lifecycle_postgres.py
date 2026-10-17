"""Native lifecycle/readiness/reopen gate. Run after Alembic and saver bootstrap."""

from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import ConversationNotFound
from ai_service.config.settings import Settings
from ai_service.infrastructure.persistence.schema import conversations
from ai_service.infrastructure.stateful import open_stateful_resources


@pytest.mark.asyncio
async def test_migrated_native_resources_reopen_without_losing_owned_conversation(
    postgres_dsn,
    monkeypatch,
):
    dsn = postgres_dsn
    url = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    settings = Settings(
        _env_file=None,
        database_url=url,
        jwt_jwks_url="http://identity.test/.well-known/jwks.json",
        provider="openai", openai_api_key="fixture",
    )
    class Provider:
        def create_model(self):
            pytest.fail("Read-only lifecycle must not invoke the model")

        async def aclose(self):
            pass

    monkeypatch.setattr(
        "ai_service.infrastructure.execution.build_runtime.build_model_provider",
        lambda settings: Provider(),
    )
    owner = uuid4()
    created_id = None
    try:
        async with open_stateful_resources(settings) as first:
            created = await first.conversations.create(owner, "Persistent setup")
            created_id = created.conversation_id
        # All first-process pools/clients have closed; reopen actual adapters.
        async with open_stateful_resources(settings) as second:
            restored = await second.conversations.get(owner, created_id)
            assert restored.conversation_id == created_id
            assert restored.title == "Persistent setup" and restored.revision == 0
            assert restored.state.target_budget_vnd is None
            with pytest.raises(ConversationNotFound):
                await second.conversations.get(uuid4(), created_id)
    finally:
        if created_id is not None:
            engine = create_async_engine(url, hide_parameters=True)
            try:
                async with engine.begin() as connection:
                    await connection.execute(
                        delete(conversations).where(conversations.c.id == created_id)
                    )
            finally:
                await engine.dispose()
