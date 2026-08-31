from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    source: str = "backend-api"
