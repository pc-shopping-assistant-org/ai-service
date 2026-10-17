"""Memory is owner-controlled context, never an implicit requirement merge."""

from datetime import UTC, datetime, timedelta

import pytest
from test_react_tools import context


def test_memory_projects_as_untrusted_context_without_changing_requirements():
    from ai_service.application.ports.user_memory import MemoryEntry, MemoryKind
    from ai_service.infrastructure.providers.react_model import build_react_request

    original = context()
    entry = MemoryEntry(
        key="gpu-brand",
        kind=MemoryKind.PREFERENCE,
        content="Tôi thích NVIDIA",
        updated_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(days=90),
    )
    ctx = original.model_copy(update={"memories": [entry]})
    messages, _ = build_react_request("Build mới", [], (), None, ctx)
    assert "Tôi thích NVIDIA" in messages[0].content
    assert "not authorization" in messages[0].content
    assert ctx.requirements == original.requirements
    assert str(ctx.account_id) not in messages[0].content


def test_memory_schema_is_bounded_and_does_not_accept_fact_kinds():
    from pydantic import ValidationError

    from ai_service.application.ports.user_memory import MemoryInput

    with pytest.raises(ValidationError):
        MemoryInput(kind="PRICE", content="RTX costs 1 dong")
    with pytest.raises(ValidationError):
        MemoryInput(kind="PREFERENCE", content="x" * 401)


@pytest.mark.asyncio
async def test_memory_http_requires_auth_and_uses_authenticated_owner():
    from uuid import uuid4

    import httpx
    from test_stateful_api import Tokens

    from ai_service.api.stateful_conversations import resources
    from ai_service.application.ports.user_memory import MemoryEntry
    from ai_service.infrastructure.stateful import StatefulResources
    from ai_service.main import create_app

    class Store:
        def __init__(self):
            self.entries = {}

        async def put(self, owner, key, value):
            entry = MemoryEntry(
                **value.model_dump(),
                key=key,
                updated_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(days=90),
            )
            self.entries[owner, key] = entry
            return entry

        async def list(self, owner):
            return [v for (actor, _), v in self.entries.items() if actor == owner]

        async def delete(self, owner, key):
            self.entries.pop((owner, key), None)

    owner, store = uuid4(), Store()
    app = create_app()
    app.dependency_overrides[resources] = lambda: StatefulResources(
        None, Tokens(owner), memories=store
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        path = "/api/v1/memories/gpu-brand"
        body = {"kind": "PREFERENCE", "content": "Prefer NVIDIA"}
        assert (await client.put(path, json=body)).status_code == 401
        own = {"Authorization": "Bearer owner"}
        other = {"Authorization": "Bearer other"}
        response = await client.put(path, json=body, headers=own)
        assert response.status_code == 200
        assert set(response.json()) == {"data", "message", "errors"}
        assert response.json()["message"] == "AI_MEMORY_SAVED"
        assert (await client.get("/api/v1/memories", headers=other)).json()[
            "data"
        ] == []
        await client.delete(path, headers=other)
        assert (
            len((await client.get("/api/v1/memories", headers=own)).json()["data"]) == 1
        )
        assert (
            await client.put(path, json=body | {"account_id": str(owner)}, headers=own)
        ).status_code == 422
        await client.delete(path, headers=own)
        assert (await client.get("/api/v1/memories", headers=own)).json()["data"] == []
