from uuid import uuid4

import pytest

from ai_service.application.ports.run_events import (
    EventHistoryUnavailable,
    StoredRunEvent,
)
from ai_service.application.use_cases.run_events import StatefulRunEventService
from ai_service.capabilities.assistant.public_contracts import RunViewV1


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
