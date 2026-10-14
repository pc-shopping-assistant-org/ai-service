"""Transactional journal; append only from trusted application lifecycle callers."""

from typing import Any
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ai_service.application.ports.run_events import (
    MAX_EVENT_BATCH_SIZE,
    EventHistoryUnavailable,
    InvalidEventCursor,
    PersistedRunEvent,
    StoredRunEvent,
)
from ai_service.application.ports.stateful_conversation import RunAdmissionError
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import RunErrorV1
from ai_service.capabilities.assistant.schemas import ChatStreamEventType
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage

from .schema import agent_run_events, agent_runs, conversations

_events: TypeAdapter[PersistedRunEvent] = TypeAdapter(PersistedRunEvent)
MAX_EVENT_DEDUP_KEY_CHARS = 200
_messages = {
    ChatStreamEventType.START: ResponseMessage.AI_CHAT_STREAM_STARTED,
    ChatStreamEventType.DELTA: ResponseMessage.AI_CHAT_STREAM_DELTA,
    ChatStreamEventType.COMPLETED: ResponseMessage.AI_CHAT_STREAM_COMPLETED,
    ChatStreamEventType.ERROR: ResponseMessage.AI_CHAT_STREAM_FAILED,
}


def _record(row) -> StoredRunEvent:
    result = StoredRunEvent(
        event_id=row["id"], sequence=row["sequence"], envelope=row["payload"]
    )
    data = result.envelope.data
    assert data is not None
    if (data.run_id, data.conversation_id, data.event) != (
        row["run_id"],
        row["conversation_id"],
        row["event_type"],
    ):
        raise ValueError("Stored event identity differs from journal row")
    if result.envelope.message != _messages[ChatStreamEventType(data.event)]:
        raise ValueError("Stored event has an invalid message key")
    return result


async def append_run_event(
    conn: AsyncConnection, run_id: UUID, dedup_key: str, payload: dict[str, Any]
) -> StoredRunEvent:
    if not dedup_key.strip() or len(dedup_key) > MAX_EVENT_DEDUP_KEY_CHARS:
        raise ValueError("Invalid event dedup key")
    if {"run_id", "conversation_id", "sequence"}.intersection(payload):
        raise ValueError("Event identity is application-owned")
    run = (
        (
            await conn.execute(
                select(agent_runs).where(agent_runs.c.id == run_id).with_for_update()
            )
        )
        .mappings()
        .one()
    )
    existing = (
        (
            await conn.execute(
                select(agent_run_events).where(
                    agent_run_events.c.run_id == run_id,
                    agent_run_events.c.dedup_key == dedup_key,
                )
            )
        )
        .mappings()
        .one_or_none()
    )
    if existing is not None:
        stored = _record(existing)
        assert stored.envelope.data is not None
        original = stored.envelope.data.model_dump(
            mode="json", exclude={"run_id", "conversation_id", "sequence"}
        )
        if canonical_hash(original) != canonical_hash(payload):
            raise ValueError("Event dedup key reused with different payload")
        return stored
    sequence = (
        await conn.execute(
            select(func.coalesce(func.max(agent_run_events.c.sequence), 0)).where(
                agent_run_events.c.run_id == run_id
            )
        )
    ).scalar_one() + 1
    event = _events.validate_python(
        payload
        | {
            "run_id": run_id,
            "conversation_id": run["conversation_id"],
            "sequence": sequence,
        }
    )
    errors = (
        [ErrorDetail(code=event.code.value, message="Run did not complete")]
        if isinstance(event, RunErrorV1)
        else []
    )
    envelope = ApiResponse[PersistedRunEvent](
        data=event, message=_messages[ChatStreamEventType(event.event)], errors=errors
    )
    row = (
        (
            await conn.execute(
                insert(agent_run_events)
                .values(
                    id=uuid4(),
                    conversation_id=run["conversation_id"],
                    run_id=run_id,
                    sequence=sequence,
                    dedup_key=dedup_key,
                    event_type=event.event,
                    payload=envelope.model_dump(mode="json"),
                )
                .returning(*agent_run_events.c)
            )
        )
        .mappings()
        .one()
    )
    return _record(row)


class PostgresRunEventStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def read(
        self,
        owner: UUID,
        conversation_id: UUID,
        run_id: UUID,
        after: UUID | None = None,
        limit: int = MAX_EVENT_BATCH_SIZE,
    ) -> list[StoredRunEvent]:
        if not 1 <= limit <= MAX_EVENT_BATCH_SIZE:
            raise ValueError(
                f"Event limit must be between 1 and {MAX_EVENT_BATCH_SIZE}"
            )
        async with self.engine.connect() as conn:
            owned = await conn.scalar(
                select(agent_runs.c.id)
                .join(conversations, agent_runs.c.conversation_id == conversations.c.id)
                .where(
                    conversations.c.account_id == owner,
                    conversations.c.id == conversation_id,
                    agent_runs.c.id == run_id,
                )
            )
            if owned is None:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            sequence = 0
            if after is not None:
                value = await conn.scalar(
                    select(agent_run_events.c.sequence).where(
                        agent_run_events.c.run_id == run_id,
                        agent_run_events.c.id == after,
                    )
                )
                if value is None:
                    raise InvalidEventCursor()
                sequence = value
            rows = (
                (
                    await conn.execute(
                        select(agent_run_events)
                        .where(
                            agent_run_events.c.run_id == run_id,
                            agent_run_events.c.sequence > sequence,
                        )
                        .order_by(agent_run_events.c.sequence)
                        .limit(limit)
                    )
                )
                .mappings()
                .all()
            )
            result = [_record(row) for row in rows]
            if any(
                row.sequence != sequence + index + 1 for index, row in enumerate(result)
            ):
                raise EventHistoryUnavailable()
            return result
