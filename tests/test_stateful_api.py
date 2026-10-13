"""HTTP boundary checks with explicit fake storage, not native DB evidence."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from ai_service.application.ports.stateful_conversation import (
    ConversationNotFound,
    ConversationRecord,
    InvalidConversationCursor,
    RunAdmissionError,
)
from ai_service.application.use_cases.stateful_conversations import (
    StatefulConversationService,
    StatefulRunService,
)
from ai_service.capabilities.assistant.public_contracts import PageV1, RunViewV1
from ai_service.config.settings import Settings
from ai_service.infrastructure.auth import AuthenticationError
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)
from ai_service.infrastructure.stateful import StatefulResources
from ai_service.main import create_app


class OwnedStore:
    def __init__(self):
        self.rows = {}

    async def create(self, account_id, title):
        row = ConversationRecord(
            id=uuid4(),
            title=title,
            revision=0,
            accepted_checkpoint_id=None,
            active_run_id=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self.rows[row.id] = account_id, row
        return row

    async def get(self, account_id, conversation_id):
        found = self.rows.get(conversation_id)
        if found is None or found[0] != account_id:
            raise ConversationNotFound()
        return found[1]

    async def list_owned(self, account_id, limit=20, before=None):
        return [row for owner, row in self.rows.values() if owner == account_id][:limit]

    async def messages(self, account_id, conversation_id, limit, after):
        await self.get(account_id, conversation_id)
        if after is not None:
            raise InvalidConversationCursor()
        return PageV1(items=[])

    async def initialize(self, account_id, conversation_id, initializer):
        record = await self.get(account_id, conversation_id)
        ref = await initializer.initialize(conversation_id)
        updated = record.model_copy(
            update={"accepted_checkpoint_id": ref.checkpoint_id}
        )
        self.rows[record.id] = account_id, updated
        return updated


class NoAcceptedCheckpoint:
    async def read(self, ref):
        raise AssertionError("New conversation has no accepted checkpoint to read")


class Tokens:
    def __init__(self, owner):
        self.owner = owner

    async def authenticate(self, token):
        if token == "owner":
            return self.owner
        if token == "other":
            return UUID(int=1)
        raise AuthenticationError()


@pytest.mark.asyncio
async def test_enabled_lifecycle_auth_owner_and_public_envelope(monkeypatch):
    owner = uuid4()
    saver = InMemorySaver()
    resources = StatefulResources(
        conversations=StatefulConversationService(
            OwnedStore(),
            PostgresAcceptedCheckpointReader(saver),
            NativeConversationInitializer(saver),
        ),
        authenticator=Tokens(owner),
    )
    closed = []

    @asynccontextmanager
    async def open_resources(settings):
        try:
            yield resources
        finally:
            closed.append(True)

    monkeypatch.setattr(
        "ai_service.infrastructure.stateful.open_stateful_resources", open_resources
    )
    settings = Settings(
        _env_file=None,
        stateful_enabled=True,
        database_url="postgresql+psycopg://postgres:postgres@localhost:5433/ai_db",
        jwt_jwks_url="http://identity.test/.well-known/jwks.json",
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):  # noqa: SIM117 - make lifecycle ordering explicit
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for headers in [
                {},
                {"Authorization": "Bearer bad"},
                {"X-Account-Id": str(owner)},
            ]:
                denied = await client.post(
                    "/api/v1/conversations", json={}, headers=headers
                )
                assert denied.status_code == 401
                assert denied.json()["message"] == "AI_UNAUTHENTICATED"
                assert denied.headers["www-authenticate"] == "Bearer"
            created = await client.post(
                "/api/v1/conversations",
                json={"title": "My setup"},
                headers={"Authorization": "Bearer owner"},
            )
            assert created.status_code == 201
            body = created.json()
            assert set(body) == {"data", "message", "errors"}
            assert body["message"] == "AI_CONVERSATION_CREATED" and body["errors"] == []
            assert body["data"]["state"]["target_budget_vnd"] is None
            assert body["data"]["readiness"] == "READY"
            assert body["data"]["revision"] == 0
            assert "accepted_checkpoint_id" not in body["data"]
            path = "/api/v1/conversations/" + body["data"]["conversation_id"]
            assert (
                await client.get(path, headers={"Authorization": "Bearer owner"})
            ).status_code == 200
            nonowner = await client.get(path, headers={"Authorization": "Bearer other"})
            missing = await client.get(
                "/api/v1/conversations/" + str(uuid4()),
                headers={"Authorization": "Bearer other"},
            )
            assert nonowner.status_code == missing.status_code == 404
            assert nonowner.json() == missing.json()
            messages = await client.get(
                path + "/messages", headers={"Authorization": "Bearer owner"}
            )
            assert (
                messages.status_code == 200 and messages.json()["data"]["items"] == []
            )
            foreign_history = await client.get(
                path + "/messages", headers={"Authorization": "Bearer other"}
            )
            missing_history = await client.get(
                f"/api/v1/conversations/{uuid4()}/messages",
                headers={"Authorization": "Bearer other"},
            )
            assert foreign_history.status_code == missing_history.status_code == 404
            assert foreign_history.json() == missing_history.json()
            invalid = await client.post(
                "/api/v1/conversations",
                json={"account_id": str(owner)},
                headers={"Authorization": "Bearer owner"},
            )
            assert invalid.status_code == 422
            unavailable = await client.post(
                path + "/runs",
                json={
                    "conversation_id": body["data"]["conversation_id"],
                    "request_id": str(uuid4()),
                    "expected_revision": 0,
                    "message": "Build PC",
                },
                headers={"Authorization": "Bearer owner"},
            )
            assert unavailable.status_code == 503
            assert unavailable.json()["data"] is None
    assert closed == [True] and app.state.stateful is None


def test_persistence_is_opt_in_and_enabled_mode_requires_configuration():
    app = create_app(Settings(_env_file=None, stateful_enabled=False))
    assert "/api/v1/conversations" not in app.openapi()["paths"]
    assert not any("/conversations" in path for path in app.openapi()["paths"])
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None, stateful_enabled=True, database_url=None, jwt_jwks_url=None
        )


@pytest.mark.parametrize(
    "runtime_flag", ["stateful_build_enabled", "stateful_react_enabled"]
)
def test_native_build_runtime_requires_explicit_model_and_mapping(runtime_flag):
    base = {
        "_env_file": None,
        "stateful_enabled": True,
        "database_url": "postgresql+psycopg://postgres:postgres@localhost:5433/ai_db",
        "jwt_jwks_url": "http://identity.test/jwks",
        runtime_flag: True,
    }
    with pytest.raises(ValidationError):
        Settings(**base)
    with pytest.raises(ValidationError):
        Settings(**base, hardware_catalog_mapping={"categories": {str(uuid4()): "CPU"}})


def test_guided_and_react_execution_modes_cannot_be_enabled_together():
    with pytest.raises(ValidationError, match="not both"):
        Settings(
            _env_file=None, stateful_build_enabled=True, stateful_react_enabled=True
        )


@pytest.mark.asyncio
async def test_run_http_auth_scope_validation_static_errors_and_wake(monkeypatch):
    owner, conversation_id, request_id, run_id = (uuid4() for _ in range(4))
    wakes = []
    received = []

    class Runs:
        async def submit(self, account, request):
            received.append(request)
            if account != owner:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            if request.message.startswith("AI_"):
                raise RunAdmissionError(request.message)
            return RunViewV1(
                run_id=run_id,
                conversation_id=conversation_id,
                request_id=request_id,
                status="PENDING",
                base_revision=0,
            )

        async def get(self, account, conversation, run):
            if account != owner or conversation != conversation_id or run != run_id:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            return RunViewV1(
                run_id=run_id,
                conversation_id=conversation_id,
                request_id=request_id,
                status="RUNNING",
                base_revision=0,
            )

    saver = InMemorySaver()
    runtime = StatefulResources(
        StatefulConversationService(
            OwnedStore(),
            PostgresAcceptedCheckpointReader(saver),
            NativeConversationInitializer(saver),
        ),
        Tokens(owner),
        runs=StatefulRunService(Runs(), lambda: wakes.append(True)),
    )

    @asynccontextmanager
    async def opened(settings):
        yield runtime

    monkeypatch.setattr(
        "ai_service.infrastructure.stateful.open_stateful_resources", opened
    )
    app = create_app(
        Settings(
            _env_file=None,
            stateful_enabled=True,
            database_url="postgresql+psycopg://postgres:postgres@localhost:5433/ai_db",
            jwt_jwks_url="http://identity.test/jwks",
        )
    )
    async with app.router.lifespan_context(app):  # noqa: SIM117 - explicit lifespan ordering
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            path = f"/api/v1/conversations/{conversation_id}/runs"
            body = {
                "conversation_id": str(conversation_id),
                "request_id": str(request_id),
                "expected_revision": 0,
                "message": "Build PC",
            }
            headers = {"Authorization": "Bearer owner"}
            assert (await client.post(path, json=body)).status_code == 401
            mismatch = await client.post(
                path, json=body | {"conversation_id": str(uuid4())}, headers=headers
            )
            assert mismatch.status_code == 422 and not received
            response = await client.post(path, json=body, headers=headers)
            assert response.status_code == 202
            assert response.json()["message"] == "AI_RUN_ACCEPTED"
            assert response.json()["errors"] == [] and wakes == [True]
            assert set(response.json()) == {"data", "message", "errors"}
            status = await client.get(path + f"/{run_id}", headers=headers)
            assert (
                status.status_code == 200
                and status.json()["message"] == "AI_RUN_RETRIEVED"
            )
            for code in [
                "AI_REVISION_CONFLICT",
                "AI_REQUEST_CONFLICT",
                "AI_CONVERSATION_BUSY",
                "AI_CONVERSATION_NOT_READY",
            ]:
                rejected = await client.post(
                    path, json=body | {"message": code}, headers=headers
                )
                assert (
                    rejected.status_code == 409 and rejected.json()["message"] == code
                )
                assert rejected.json()["data"] is None
            foreign = await client.get(
                path + f"/{run_id}", headers={"Authorization": "Bearer other"}
            )
            absent = await client.get(path + f"/{uuid4()}", headers=headers)
            assert foreign.status_code == absent.status_code == 404
            assert foreign.json() == absent.json()
            assert wakes == [True]
