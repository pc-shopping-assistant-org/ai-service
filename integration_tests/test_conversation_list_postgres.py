"""Owner-scoped public conversation pagination through the actual SQL adapter."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.ports.stateful_conversation import InvalidConversationCursor
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.capabilities.assistant.public_contracts import ConversationViewV1
from ai_service.capabilities.assistant.react.public_views import ConversationViewV2
from ai_service.infrastructure.persistence.conversations import (
    PostgresConversationStore,
)
from ai_service.infrastructure.persistence.schema import conversations


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_owned_conversation_pages_ties_cursor_and_public_projection(
    postgres_dsn, version
):
    engine = create_async_engine(
        postgres_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
    )
    store, owner, foreign, created = (
        PostgresConversationStore(engine),
        uuid4(),
        uuid4(),
        [],
    )

    class NoNativeLookup:
        async def read(self, ref):
            raise AssertionError("Uninitialized rows must not read a native head")

    service = StatefulConversationService(
        store, NoNativeLookup(), None, view_version=version
    )
    try:
        assert (await service.list(owner, 2, None)).items == []
        for actor in (owner, owner, owner, foreign):
            created.append(await store.create(actor, "List fixture"))
        tied = datetime(2026, 10, 13, tzinfo=UTC)
        async with engine.begin() as conn:
            await conn.execute(
                update(conversations)
                .where(conversations.c.id.in_([row.id for row in created]))
                .values(updated_at=tied)
            )
        ordered = sorted([row.id for row in created[:3]], reverse=True)
        page = await service.list(owner, 2, None)
        assert [row.conversation_id for row in page.items] == ordered[:2]
        assert page.next_cursor == str(ordered[1])
        last = await service.list(owner, 2, page.next_cursor)
        assert [row.conversation_id for row in last.items] == ordered[2:]
        assert last.next_cursor is None
        assert (await service.list(foreign, 100, None)).items[
            0
        ].conversation_id == created[-1].id
        for view in [*page.items, *last.items]:
            assert type(view) is (
                ConversationViewV2 if version == 2 else ConversationViewV1
            )
            assert view.readiness == "INITIALIZING"
            assert "accepted_checkpoint_id" not in view.model_dump()
        for bad in ("not-a-uuid", str(uuid4()), str(created[-1].id)):
            with pytest.raises(InvalidConversationCursor):
                await service.list(owner, 2, bad)
        for limit in (0, 101):
            with pytest.raises(ValueError):
                await service.list(owner, limit, None)
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                delete(conversations).where(
                    conversations.c.id.in_([row.id for row in created])
                )
            )
        await engine.dispose()
