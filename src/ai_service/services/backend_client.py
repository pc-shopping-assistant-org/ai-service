from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.catalog import CatalogPage
from ai_service.config.settings import Settings, get_settings

__all__ = ["BackendClient", "BackendUnavailableError", "CatalogPage"]


class BackendClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def search_products(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        page = await self._request_products({"keyword": query, "limit": limit})
        return page.items

    async def list_products(
        self,
        *,
        cursor: str | None = None,
        limit: int = 100,
    ) -> CatalogPage:
        """Read one active-catalog cursor page for indexing/synchronization."""
        params: dict[str, str | int] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request_products(params)

    async def _request_products(self, params: dict[str, str | int]) -> CatalogPage:
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(
                    f"{self.settings.backend_api_url.rstrip('/')}/products",
                    params=params,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Catalog service is unavailable") from exc
        data = payload.get("data", payload)
        if isinstance(data, dict):
            items = data.get("items", data.get("products", []))
            next_cursor = data.get("nextCursor", data.get("next_cursor"))
            has_next = bool(data.get("hasNext", data.get("has_next", next_cursor is not None)))
        else:
            items = data
            next_cursor = None
            has_next = False
        normalized_items = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        return CatalogPage(
            items=normalized_items,
            next_cursor=str(next_cursor) if next_cursor is not None else None,
            has_next=has_next,
        )

    async def get_product(self, product_id: UUID) -> dict[str, Any] | None:
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(
                    f"{self.settings.backend_api_url.rstrip('/')}/products/{product_id}"
                )
                if response.status_code == httpx.codes.NOT_FOUND:
                    return None
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Catalog service is unavailable") from exc
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else None
