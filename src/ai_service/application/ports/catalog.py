"""Outbound ports for canonical catalog access and retrieval.

The application layer knows only these small contracts. HTTP, Qdrant and any
future search index are infrastructure details that implement them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID


@dataclass(frozen=True, slots=True)
class CatalogPage:
    """Cursor page returned by the canonical backend catalog endpoint."""

    items: list[dict[str, Any]]
    next_cursor: str | None = None
    has_next: bool = False


class CatalogClient(Protocol):
    """Minimum catalog gateway needed by assistant use cases."""

    async def search_products(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the canonical backend catalog."""

    async def get_product(self, product_id: UUID) -> dict[str, Any] | None:
        """Read one canonical product, or ``None`` when it is not found."""


class CatalogPageClient(Protocol):
    """Cursor-page surface used by the catalog indexing command."""

    async def list_products(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> CatalogPage:
        """Read one page from the canonical catalog."""


class BackendCatalogClient(Protocol):
    """Small client surface required by the backend keyword retriever."""

    async def search_products(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search products through the canonical backend API."""


class CatalogRetriever(Protocol):
    """Return backend-shaped catalog rows ordered by relevance."""

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Retrieve visible catalog records for an assistant request."""


__all__ = [
    "BackendCatalogClient",
    "CatalogClient",
    "CatalogPage",
    "CatalogPageClient",
    "CatalogRetriever",
]
