from __future__ import annotations

import json

from ai_service.capabilities.assistant.schemas import (
    ChatStreamEvent,
    ChatStreamEventType,
)
from ai_service.schemas.response import ApiResponse


def encode_chat_event[EventDataT](response: ApiResponse[EventDataT]) -> str:
    """Encode one canonical response envelope as an SSE frame."""
    data = response.data
    event_type = "message"
    if isinstance(data, ChatStreamEvent):
        event = data.event
        event_type = event.value.lower() if isinstance(event, ChatStreamEventType) else str(event).lower()
    payload = json.dumps(
        response.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_type}\ndata: {payload}\n\n"
