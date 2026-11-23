"""Read-only subscriptions: execution remains owned by the managed executor."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass
from time import monotonic
from uuid import UUID

from ai_service.application.activity import RunActivityBroker
from ai_service.application.ports.run_events import (
    EventHistoryUnavailable,
    RunEventReader,
    StoredRunEvent,
)
from ai_service.application.ports.stateful_conversation import StatefulRunStore
from ai_service.capabilities.assistant.public_contracts import RunStatusV1
from ai_service.capabilities.assistant.schemas import ChatStreamEventType
from ai_service.capabilities.assistant.stateful_contracts import RunStatus

DEFAULT_POLL_SECONDS = 0.2
DEFAULT_HEARTBEAT_SECONDS = 15.0

type StreamEventItem = StoredRunEvent | RunStatusV1 | None

_terminal = {
    RunStatus.COMPLETED,
    RunStatus.WAITING_INPUT,
    RunStatus.FAILED_TERMINAL,
    RunStatus.CANCELLED,
}


class StatefulRunEventService:
    def __init__(
        self,
        reader: RunEventReader,
        runs: StatefulRunStore,
        *,
        activity_broker: RunActivityBroker | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    ) -> None:
        if poll_seconds <= 0 or heartbeat_seconds <= 0:
            raise ValueError("Subscription intervals must be positive")
        self.reader, self.runs = reader, runs
        self.activity_broker = activity_broker
        self.poll_seconds, self.heartbeat_seconds = poll_seconds, heartbeat_seconds

    async def result(
        self, owner: UUID, conversation_id: UUID, run_id: UUID
    ) -> StoredRunEvent | None:
        """Read the immutable delivery outcome, never the latest conversation head."""
        run = await self.runs.get(owner, conversation_id, run_id)
        if run.status not in _terminal:
            return None
        rows = await self.reader.read(owner, conversation_id, run_id)
        for row in reversed(rows):
            assert row.envelope.data is not None
            if row.envelope.data.event in (
                ChatStreamEventType.COMPLETED,
                ChatStreamEventType.ERROR,
            ):
                return row
        raise EventHistoryUnavailable()

    async def open(
        self,
        owner: UUID,
        conversation_id: UUID,
        run_id: UUID,
    ) -> "RunEventSubscription":
        # Authorize before sending headers. No public reconnect cursor contract.
        run = await self.runs.get(owner, conversation_id, run_id)
        rows = await self.reader.read(owner, conversation_id, run_id)
        if not rows and run.status in _terminal:
            raise EventHistoryUnavailable()
        return RunEventSubscription(self, owner, conversation_id, run_id, None, rows)


@dataclass
class RunEventSubscription:
    service: StatefulRunEventService
    owner: UUID
    conversation_id: UUID
    run_id: UUID
    after: UUID | None
    initial: list[StoredRunEvent]

    async def events(self) -> AsyncIterator[StreamEventItem]:
        rows, after = self.initial, self.after
        heartbeat = monotonic()
        broker = self.service.activity_broker
        subscription = broker.subscribe(self.run_id) if broker is not None else nullcontext(None)

        with subscription as status_queue:
            while True:
                for row in rows:
                    yield row
                    after = row.event_id
                    assert row.envelope.data is not None
                    if row.envelope.data.event in (
                        ChatStreamEventType.COMPLETED,
                        ChatStreamEventType.ERROR,
                    ):
                        return
                    heartbeat = monotonic()
                if status_queue is not None:
                    while not status_queue.empty():
                        yield status_queue.get_nowait()
                        heartbeat = monotonic()
                rows = await self.service.reader.read(
                    self.owner, self.conversation_id, self.run_id, after
                )
                if rows:
                    continue
                run = await self.service.runs.get(
                    self.owner, self.conversation_id, self.run_id
                )
                if run.status in _terminal:
                    # Status may commit between read and get; re-read its atomic events.
                    rows = await self.service.reader.read(
                        self.owner, self.conversation_id, self.run_id, after
                    )
                    if rows:
                        continue
                    raise EventHistoryUnavailable()
                if monotonic() - heartbeat >= self.service.heartbeat_seconds:
                    yield None
                    heartbeat = monotonic()
                if status_queue is None:
                    await asyncio.sleep(self.service.poll_seconds)
                    continue
                deadline = monotonic() + self.service.poll_seconds
                while (remaining := deadline - monotonic()) > 0:
                    try:
                        status = await asyncio.wait_for(
                            status_queue.get(), timeout=remaining
                        )
                    except TimeoutError:
                        break
                    yield status
                    heartbeat = monotonic()
