from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from ai_service.application.ports.run_events import StoredRunEvent
from ai_service.application.use_cases.run_events import RunEventSubscription
from ai_service.capabilities.assistant.schemas import (
    ChatStreamEvent,
    ChatStreamEventType,
)
from ai_service.schemas.response import ApiResponse, ErrorDetail, ResponseMessage

logger = logging.getLogger(__name__)


def encode_chat_event[EventDataT](response: ApiResponse[EventDataT]) -> str:
    """Encode one canonical response envelope as an SSE frame."""
    data = response.data
    event_type = "message"
    if isinstance(data, ChatStreamEvent):
        event = data.event
        event_type = (
            event.value.lower()
            if isinstance(event, ChatStreamEventType)
            else str(event).lower()
        )
    payload = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_type}\ndata: {payload}\n\n"


def encode_run_event(record: StoredRunEvent) -> str:
    data = record.envelope.data
    assert data is not None
    return (
        f"event: {data.event.lower()}\n"
        f"data: {record.envelope.model_dump_json()}\n\n"
    )


async def run_event_frames(subscription: RunEventSubscription) -> AsyncIterator[str]:
    try:
        async for record in subscription.events():
            yield encode_run_event(record) if record else ": keep-alive\n\n"
    except Exception:
        logger.exception("Run event subscription interrupted")
        # Transport-only failure: no persisted offset, no pretend completion.
        envelope = ApiResponse[None](
            message=ResponseMessage.AI_STREAM_INTERRUPTED,
            errors=[
                ErrorDetail(
                    code=ResponseMessage.AI_STREAM_INTERRUPTED.value,
                    message="Reload the conversation to read the run result",
                )
            ],
        )
        yield f"event: {ChatStreamEventType.ERROR.lower()}\ndata: {envelope.model_dump_json()}\n\n"
