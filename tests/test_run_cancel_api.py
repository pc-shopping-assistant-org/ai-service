from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from test_stateful_api import Tokens

from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.application.use_cases.stateful_conversations import StatefulRunService
from ai_service.capabilities.assistant.public_contracts import RunViewV1
from ai_service.config.settings import Settings
from ai_service.main import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [2])
async def test_cancel_http_owner_envelope_and_commit_before_signal(version):
    owner, conversation, run, request_id = uuid4(), uuid4(), uuid4(), uuid4()
    calls = []

    class Store:
        async def cancel(self, actor, conversation_id, run_id):
            if actor != owner or conversation_id != conversation or run_id != run:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            calls.append("commit")
            return RunViewV1(
                run_id=run,
                conversation_id=conversation,
                request_id=request_id,
                status="CANCELLED",
                base_revision=0,
            )

    def signal(run_id):
        assert run_id == run and calls[-1] == "commit"
        calls.append("signal")
        return True

    service = StatefulRunService(Store(), lambda: calls.append("wake"), signal)
    app = create_app(Settings(_env_file=None))
    app.state.stateful = SimpleNamespace(authenticator=Tokens(owner), runs=service)
    path = f"/api/v1/conversations/{conversation}/runs/{run}/cancel"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        assert (await client.post(path)).status_code == 401
        forbidden = await client.post(path, headers={"Authorization": "Bearer other"})
        missing = await client.post(
            path.replace(str(run), str(uuid4())),
            headers={"Authorization": "Bearer owner"},
        )
        assert forbidden.status_code == missing.status_code == 404
        assert forbidden.json() == missing.json() and calls == []
        for _ in range(2):
            response = await client.post(
                path, headers={"Authorization": "Bearer owner"}
            )
            assert response.status_code == 202
            body = response.json()
            assert set(body) == {"data", "message", "errors"}
            assert body["message"] == "AI_RUN_CANCEL_REQUESTED" and body["errors"] == []
            assert body["data"]["status"] == "CANCELLED"
            assert "execution_id" not in body["data"]
        assert calls == ["commit", "signal", "wake"] * 2
