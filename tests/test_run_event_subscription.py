from uuid import uuid4

import pytest

from ai_service.application.ports.run_events import (
    EventHistoryUnavailable,
    StoredRunEvent,
)
from ai_service.application.use_cases.run_events import StatefulRunEventService
from ai_service.capabilities.assistant.public_contracts import RunViewV1
from ai_service.capabilities.assistant.stateful_contracts import RunStatus


def event(conversation, run, seq, kind="START"):
    data = {
        "conversation_id": conversation,
        "run_id": run.run_id,
        "sequence": seq,
        "event": kind,
    }
    if kind == "START":
        data["run"] = run.model_dump()
    else:
        data["code"] = "AI_RUN_CANCELLED"
    return StoredRunEvent(
        event_id=uuid4(),
        sequence=seq,
        envelope={
            "data": data,
            "message": "AI_CHAT_STREAM_STARTED"
            if kind == "START"
            else "AI_CHAT_STREAM_FAILED",
            "errors": [],
        },
    )


@pytest.mark.asyncio
async def test_subscription_replays_without_submit_and_disconnect_does_not_cancel():
    owner, conversation, run_id = uuid4(), uuid4(), uuid4()
    run = RunViewV1(
        conversation_id=conversation,
        run_id=run_id,
        request_id=uuid4(),
        base_revision=0,
        status="RUNNING",
    )
    start, end = event(conversation, run, 1), event(conversation, run, 2, "ERROR")

    class Runs:
        async def get(self, actor, cid, rid):
            assert (actor, cid, rid) == (owner, conversation, run_id)
            return run

        async def submit(self, *args):
            raise AssertionError("SSE must not submit execution")

        async def cancel(self, *args):
            raise AssertionError("Disconnect must not cancel execution")

    class Reader:
        def __init__(self):
            self.rows = [start]

        async def read(self, actor, cid, rid, after=None, limit=100):
            assert (actor, cid, rid) == (owner, conversation, run_id)
            return [
                row
                for row in self.rows
                if after is None
                or row.sequence
                > next(item.sequence for item in self.rows if item.event_id == after)
            ]

    reader, service = Reader(), None
    service = StatefulRunEventService(
        reader, Runs(), poll_seconds=0.001, heartbeat_seconds=0.002
    )
    ticket = await service.open(owner, conversation, run_id)
    stream = ticket.events()
    assert (await anext(stream)).event_id == start.event_id
    await stream.aclose()
    assert run.status == "RUNNING"
    reader.rows.append(end)
    run.status = "CANCELLED"
    completed = await service.open(owner, conversation, run_id)
    assert [row.event_id async for row in completed.events() if row] == [
        start.event_id,
        end.event_id,
    ]
    reader.rows = []
    with pytest.raises(EventHistoryUnavailable):
        await service.open(owner, conversation, run_id)


@pytest.mark.asyncio
async def test_final_outcome_waits_for_terminal_run_and_reads_its_own_journal():
    owner, conversation, run_id = uuid4(), uuid4(), uuid4()
    run = RunViewV1(
        conversation_id=conversation, run_id=run_id, request_id=uuid4(),
        base_revision=2, status=RunStatus.RUNNING,
    )
    end = event(conversation, run, 2, "ERROR")

    class Runs:
        async def get(self, actor, cid, rid):
            if (actor, cid, rid) != (owner, conversation, run_id):
                raise PermissionError()
            return run

    class Reader:
        async def read(self, actor, cid, rid, after=None, limit=100):
            assert (actor, cid, rid) == (owner, conversation, run_id)
            return [end]

    service = StatefulRunEventService(Reader(), Runs())
    assert await service.result(owner, conversation, run_id) is None
    run.status = RunStatus.FAILED_RETRYABLE
    assert await service.result(owner, conversation, run_id) is None
    run.status = RunStatus.FAILED_TERMINAL
    assert await service.result(owner, conversation, run_id) == end
    with pytest.raises(PermissionError):
        await service.result(uuid4(), conversation, run_id)


@pytest.mark.asyncio
async def test_result_http_pending_and_saved_final_event_are_owner_scoped():
    import httpx
    from langgraph.checkpoint.memory import InMemorySaver
    from test_stateful_api import OwnedStore, Tokens

    from ai_service.application.ports.stateful_conversation import RunAdmissionError
    from ai_service.application.use_cases.stateful_conversations import (
        StatefulConversationService,
    )
    from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
    from ai_service.capabilities.assistant.public_contracts import (
        to_public_consultation,
    )
    from ai_service.capabilities.assistant.react.public_views import (
        RunCompletedV2,
        TurnResultV2,
    )
    from ai_service.config.settings import Settings
    from ai_service.infrastructure.graph.initial_root import (
        NativeConversationInitializer,
    )
    from ai_service.infrastructure.persistence.accepted_state import (
        PostgresAcceptedCheckpointReader,
    )
    from ai_service.infrastructure.stateful import StatefulResources
    from ai_service.main import create_app

    owner, cid, rid, saver = uuid4(), uuid4(), uuid4(), InMemorySaver()
    run = RunViewV1(conversation_id=cid, run_id=rid, request_id=uuid4(),
                    base_revision=2, status=RunStatus.RUNNING)

    class Runs:
        async def get(self, actor, conversation, requested):
            if (actor, conversation, requested) != (owner, cid, rid):
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            return run

    class Reader:
        def __init__(self):
            self.rows = []

        async def read(self, *args):
            return self.rows

    reader = Reader()
    app = create_app(Settings(_env_file=None))
    app.state.stateful = StatefulResources(
        StatefulConversationService(OwnedStore(), PostgresAcceptedCheckpointReader(saver), NativeConversationInitializer(saver)),
        Tokens(owner), events=StatefulRunEventService(reader, Runs()),
    )
    path = f"/api/v1/conversations/{cid}/runs/{rid}/result"
    headers = {"Authorization": "Bearer owner"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        assert (await client.get(path)).status_code == 401
        pending = await client.get(path, headers=headers)
        assert pending.status_code == 202
        assert pending.json() == {"data": None, "message": "AI_RUN_RETRIEVED", "errors": []}
        run.status = RunStatus.COMPLETED
        reader.rows = [StoredRunEvent(
            event_id=uuid4(), sequence=3, envelope={
                "data": RunCompletedV2(
                    conversation_id=cid, run_id=rid, sequence=3, event="COMPLETED",
                    result=TurnResultV2(run=run, revision=3, state=to_public_consultation(ConsultationStateV1()), answer="Di động hay hiệu năng?"),
                ), "message": "AI_CHAT_STREAM_COMPLETED", "errors": [],
            },
        )]
        saved = await client.get(path, headers=headers)
        assert saved.status_code == 200
        assert saved.json()["data"]["result"]["answer"] == "Di động hay hiệu năng?"
        assert saved.json()["data"]["result"]["run"]["run_id"] == str(rid)
        assert (await client.get(path, headers={"Authorization": "Bearer other"})).status_code == 404
        reader.rows = []
        missing = await client.get(path, headers=headers)
        assert missing.status_code == 410
        assert missing.json()["message"] == "AI_STREAM_INTERRUPTED"
