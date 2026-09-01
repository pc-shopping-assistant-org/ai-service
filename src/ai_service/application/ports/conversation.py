"""Conversation-context port used by assistant application use cases."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from ai_service.capabilities.assistant.schemas import ConversationMessage


class ConversationStore(Protocol):
    """Read and append request-scoped conversation messages."""

    def get_or_create(self, conversation_id: UUID | None) -> UUID:
        """Return an existing conversation ID or create one."""

    def append(self, conversation_id: UUID, message: ConversationMessage) -> None:
        """Append one validated message."""

    def history(self, conversation_id: UUID) -> list[ConversationMessage]:
        """Return the conversation history in insertion order."""


__all__ = ["ConversationStore"]
