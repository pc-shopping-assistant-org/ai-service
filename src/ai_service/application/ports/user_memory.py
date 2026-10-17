"""Explicit account memory, advisory context rather than catalog/consent evidence."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Protocol
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

MAX_MEMORIES = 20
MAX_MEMORY_CHARS = 400
MEMORY_RETENTION_DAYS = 90
MemoryKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]


class MemoryKind(StrEnum):
    PREFERENCE = "PREFERENCE"
    OWNED_CONTEXT = "OWNED_CONTEXT"


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=MAX_MEMORY_CHARS)


class MemoryEntry(MemoryInput):
    key: MemoryKey
    updated_at: AwareDatetime
    expires_at: AwareDatetime


class MemoryLimitReached(Exception):
    pass


class UserMemoryStore(Protocol):
    async def list(
        self, owner: UUID, *, at: datetime | None = None
    ) -> list[MemoryEntry]: ...
    async def put(self, owner: UUID, key: str, value: MemoryInput) -> MemoryEntry: ...
    async def delete(self, owner: UUID, key: str) -> None: ...
