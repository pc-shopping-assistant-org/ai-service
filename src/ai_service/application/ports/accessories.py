"""Canonical peripheral snapshots, separate from the eight required PC slots."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class AccessoryType(StrEnum):
    MONITOR = "MONITOR"
    MOUSE = "MOUSE"
    KEYBOARD = "KEYBOARD"
    HEADSET = "HEADSET"


class AccessorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    category: AccessoryType
    name: str = Field(min_length=1)
    price: int = Field(ge=0)
    brand: str | None = None
    extra_specs: dict[str, JsonValue] = Field(default_factory=dict)
