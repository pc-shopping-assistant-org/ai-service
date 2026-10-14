from typing import Protocol
from uuid import UUID

from pydantic import model_validator

from ai_service.capabilities.assistant.public_contracts import StatefulEventV1
from ai_service.capabilities.assistant.react.public_views import RunCompletedV2
from ai_service.capabilities.assistant.stateful_contracts import Contract, PositiveInt
from ai_service.schemas.response import ApiResponse

type PersistedRunEvent = StatefulEventV1 | RunCompletedV2

MAX_EVENT_BATCH_SIZE = 100


class InvalidEventCursor(ValueError):
    """Missing/expired/event-of-other-run cursor; never a checkpoint selector."""


class EventHistoryUnavailable(RuntimeError):
    """The persisted stream cannot be replayed without losing events."""


class StoredRunEvent(Contract):
    event_id: UUID
    sequence: PositiveInt
    envelope: ApiResponse[PersistedRunEvent]

    @model_validator(mode="after")
    def identity(self) -> "StoredRunEvent":
        if self.envelope.data is None or self.envelope.data.sequence != self.sequence:
            raise ValueError("Event envelope sequence differs from journal")
        return self


class RunEventReader(Protocol):
    async def read(
        self,
        owner: UUID,
        conversation_id: UUID,
        run_id: UUID,
        after: UUID | None = None,
        limit: int = MAX_EVENT_BATCH_SIZE,
    ) -> list[StoredRunEvent]: ...
