"""Outbound port for web search and live internet technical research."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class WebSearchCategory(StrEnum):
    COMPATIBILITY = "COMPATIBILITY"
    REVIEWS = "REVIEWS"
    BUILD_GUIDES = "BUILD_GUIDES"
    GAME_REQUIREMENTS = "GAME_REQUIREMENTS"
    HARDWARE_ISSUES = "HARDWARE_ISSUES"
    PSU_TIER = "PSU_TIER"
    DRIVER_SOFTWARE = "DRIVER_SOFTWARE"
    VENDOR_SPECS = "VENDOR_SPECS"
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


class WebPageReadStatus(StrEnum):
    READABLE = "READABLE"
    UNREADABLE = "UNREADABLE"
    EMPTY = "EMPTY"


class WebExtractionStatus(StrEnum):
    STRUCTURED = "STRUCTURED"
    CONTENT_ONLY = "CONTENT_ONLY"
    FAILED = "FAILED"


class WebFactStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


type Dimension = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ExtractedWebFact(BaseModel):
    """V1 static dimension evidence, not a catalog update or axis inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model: str = Field(min_length=1, max_length=500)
    attribute: Literal["dimensions"] = "dimensions"
    value: tuple[Dimension, Dimension, Dimension] | None
    unit: Literal["mm"] = "mm"
    fact_status: WebFactStatus
    source_url: str = Field(min_length=1, max_length=2000)
    document_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str = Field(min_length=1, max_length=100)
    locator: str = Field(min_length=1, max_length=500)
    quote: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def supported_value(self) -> ExtractedWebFact:
        if (self.fact_status == WebFactStatus.SUPPORTED) != (self.value is not None):
            raise ValueError("Only supported facts have a publishable value")
        return self


class WebPageReadResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    final_url: str
    title: str | None = None
    status: WebPageReadStatus
    content_hash: str | None = None
    content: str | None = None
    evidence_ref: str | None = None
    extraction_status: WebExtractionStatus | None = None
    document_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    facts: list[ExtractedWebFact] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def extraction_boundary(self) -> WebPageReadResult:
        if self.extraction_status is None:
            self.extraction_status = (
                WebExtractionStatus.CONTENT_ONLY
                if self.status == WebPageReadStatus.READABLE
                else WebExtractionStatus.FAILED
            )
        if self.extraction_status == WebExtractionStatus.STRUCTURED:
            if self.status != WebPageReadStatus.READABLE or not self.facts:
                raise ValueError("Structured extraction requires readable facts")
        elif self.facts:
            raise ValueError("Non-structured extraction cannot expose facts")
        if any(
            f.source_url != self.final_url or f.document_hash != self.document_hash
            for f in self.facts
        ):
            raise ValueError("Fact source does not match fetched document")
        return self


class WebPageReader(Protocol):
    """Protocol for fetching and extracting web page text/tables."""

    async def read(self, url: str) -> WebPageReadResult:
        """Fetch page content, clean boilerplate, extract tables and text."""
