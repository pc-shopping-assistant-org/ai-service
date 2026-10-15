import json
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_stateful_api import OwnedStore, Tokens

from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
)
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.outcomes import FlexibleAnswer, TaskOutcome
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.config.settings import Settings
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [2])
async def test_conversation_list_http_auth_paging_envelope_and_version_schema(version):
    class Store(OwnedStore):
        async def list_owned(self, account_id, limit=20, before=None):
            rows = sorted(
                [row for actor, row in self.rows.values() if actor == account_id],
                key=lambda row: (row.updated_at, row.id),
                reverse=True,
            )
            return [
                row
                for row in rows
                if before is None or (row.updated_at, row.id) < before
            ][:limit]

    owner, saver, store = uuid4(), InMemorySaver(), Store()
    service = StatefulConversationService(
        store,
        PostgresAcceptedCheckpointReader(saver),
        NativeConversationInitializer(saver),
    )
    created = [await service.create(owner, title) for title in ("First", "Second")]
    foreign = await service.create(uuid4(), "Foreign")
    if version == 2:

        class Reader(PostgresAcceptedCheckpointReader):
            async def read(self, ref):
                snapshot = await super().read(ref)
                if ref.thread_id == created[1].conversation_id:
                    snapshot.flexible = FlexiblePublishedState(
                        conversation_id=ref.thread_id,
                        task_message_id=uuid4(),
                        task_message="GPU?",
                        status="COMPLETED",
                        answer=FlexibleAnswer(
                            text="Saved GPU answer",
                            status="COMPLETED",
                            outcome="ANSWERED",
                            task_outcomes=[
                                TaskOutcome(
                                    task=CompletionRequirement(kind="ANSWER"),
                                    status="COMPLETED",
                                )
                            ],
                        ),
                    )
                return snapshot

        service.reader = Reader(saver)
    app = create_app(Settings(_env_file=None))
    app.state.stateful = StatefulResources(service, Tokens(owner))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        assert (await client.get("/api/v1/conversations")).status_code == 401
        headers = {"Authorization": "Bearer owner"}
        first = await client.get("/api/v1/conversations?limit=1", headers=headers)
        assert first.status_code == 200
        payload = first.json()
        assert set(payload) == {"data", "message", "errors"}
        assert (
            payload["message"] == "AI_CONVERSATIONS_LISTED" and payload["errors"] == []
        )
        assert payload["data"]["items"][0]["conversation_id"] == str(
            created[1].conversation_id
        )
        assert "accepted_checkpoint_id" not in payload["data"]["items"][0]
        if version == 2:
            assert (
                payload["data"]["items"][0]["flexible"]["answer"] == "Saved GPU answer"
            )
        else:
            assert "flexible" not in payload["data"]["items"][0]
        last = await client.get(
            "/api/v1/conversations",
            params={"limit": 1, "cursor": payload["data"]["next_cursor"]},
            headers=headers,
        )
        assert last.status_code == 200 and last.json()["data"]["next_cursor"] is None
        assert last.json()["data"]["items"][0]["conversation_id"] == str(
            created[0].conversation_id
        )
        for cursor in ("bad", str(uuid4()), str(foreign.conversation_id)):
            bad = await client.get(
                "/api/v1/conversations", params={"cursor": cursor}, headers=headers
            )
            assert bad.status_code == 400
            assert bad.json()["message"] == "VALIDATION_ERROR"
            assert bad.json()["errors"][0]["field"] == "cursor"
        for limit in (0, 101):
            assert (
                await client.get(
                    "/api/v1/conversations", params={"limit": limit}, headers=headers
                )
            ).status_code == 422
    operation = app.openapi()["paths"]["/api/v1/conversations"]["get"]
    schema_ref = operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ]
    assert f"ConversationViewV{version}" in schema_ref
    assert "owner" not in json.dumps(operation["parameters"])
