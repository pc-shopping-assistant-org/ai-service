"""In-memory pub/sub broker for ephemeral live agent activity status events.

Status updates are strictly UI feedback during an active run; durable journal
events (START, DELTA, COMPLETED, ERROR) remain Postgres-backed and sequenced.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager
from uuid import UUID

from ai_service.capabilities.assistant.public_contracts import (
    AgentActivityStage,
    RunStatusV1,
)

ActivitySink = Callable[[AgentActivityStage, str | None, str | None], None]

MAX_BUFFERED_STATUSES_PER_RUN = 16
MAX_TRACKED_RUNS = 256
MAX_SUBSCRIBER_QUEUE = 32


class RunActivityBroker:
    """Process-local ring buffer + subscriber fan-out keyed by run_id."""

    def __init__(
        self,
        *,
        max_per_run: int = MAX_BUFFERED_STATUSES_PER_RUN,
        max_runs: int = MAX_TRACKED_RUNS,
    ) -> None:
        self._max_per_run = max_per_run
        self._max_runs = max_runs
        self._history: OrderedDict[UUID, deque[RunStatusV1]] = OrderedDict()
        self._subscribers: dict[UUID, set[asyncio.Queue[RunStatusV1]]] = {}

    def publish(
        self,
        conversation_id: UUID,
        run_id: UUID,
        stage: AgentActivityStage,
        tool: str | None = None,
        detail: str | None = None,
    ) -> RunStatusV1:
        clean_detail = detail.strip()[:160] if detail and detail.strip() else None
        event = RunStatusV1(
            conversation_id=conversation_id,
            run_id=run_id,
            stage=stage,
            tool=tool,
            detail=clean_detail,
        )
        history = self._history.get(run_id)
        if history is None:
            while len(self._history) >= self._max_runs:
                oldest_run_id, _ = self._history.popitem(last=False)
                self._subscribers.pop(oldest_run_id, None)
            history = deque(maxlen=self._max_per_run)
            self._history[run_id] = history
        else:
            self._history.move_to_end(run_id)
        if (
            history
            and history[-1].stage == event.stage
            and history[-1].tool == event.tool
            and history[-1].detail == event.detail
        ):
            return event
        history.append(event)
        for queue in list(self._subscribers.get(run_id, ())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass
        return event

    def sink(self, conversation_id: UUID, run_id: UUID) -> ActivitySink:
        def emit(
            stage: AgentActivityStage,
            tool: str | None = None,
            detail: str | None = None,
        ) -> None:
            self.publish(conversation_id, run_id, stage, tool=tool, detail=detail)

        return emit

    @contextmanager
    def subscribe(self, run_id: UUID):
        queue: asyncio.Queue[RunStatusV1] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE)
        snapshot = list(self._history.get(run_id, ()))
        for item in snapshot:
            if not queue.full():
                queue.put_nowait(item)
        subs = self._subscribers.setdefault(run_id, set())
        subs.add(queue)
        try:
            yield queue
        finally:
            subs.discard(queue)
            if not subs:
                self._subscribers.pop(run_id, None)

    def finish(self, run_id: UUID) -> None:
        self._history.pop(run_id, None)
        self._subscribers.pop(run_id, None)

