"""Outbound port for web search and live internet technical research."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class WebSearchCategory(StrEnum):
    COMPATIBILITY = "COMPATIBILITY"
    REVIEWS = "REVIEWS"
    BUILD_GUIDES = "BUILD_GUIDES"
    GAME_REQUIREMENTS = "GAME_REQUIREMENTS"
    HARDWARE_ISSUES = "HARDWARE_ISSUES"
    PSU_TIER = "PSU_TIER"
    GENERAL = "GENERAL"


class WebSearchItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    url: str
    snippet: str


class WebSearchQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1)
    category: WebSearchCategory = WebSearchCategory.GENERAL
    target_domains: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=10)


class WebSearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    category: WebSearchCategory
    items: list[WebSearchItem] = Field(default_factory=list)
    total_results: int = 0
    note: str | None = None


class WebSearchClient(Protocol):
    """Protocol for fetching live web search results."""

    async def search(self, query: WebSearchQuery) -> WebSearchResult:
        """Execute a web search query and return parsed items."""
