from __future__ import annotations

from collections import defaultdict, deque
from uuid import UUID, uuid4

from ai_service.config.settings import Settings, get_settings
from ai_service.schemas.conversation import ConversationMessage


class ConversationManager:
    """Small in-process context store for the MVP chat API.

    It intentionally has no persistence claim; replacing this class with a
    Redis-backed implementation later does not change the HTTP contract.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        self._histories: dict[UUID, deque[ConversationMessage]] = defaultdict(
            lambda: deque(maxlen=settings.conversation_max_messages)
        )

    def get_or_create(self, conversation_id: UUID | None) -> UUID:
        conversation_id = conversation_id or uuid4()
        self._histories[conversation_id]
        return conversation_id

    def append(self, conversation_id: UUID, message: ConversationMessage) -> None:
        self._histories[conversation_id].append(message)

    def history(self, conversation_id: UUID) -> list[ConversationMessage]:
        return list(self._histories.get(conversation_id, ()))
