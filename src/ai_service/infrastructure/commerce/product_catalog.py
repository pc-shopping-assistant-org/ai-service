"""Budgeted reads of the actual catalog-service product DTOs, not variant IDs."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, JsonValue, StrictBool, StrictInt, ValidationError

from ai_service.application.call_attempts import (
    AttemptBoundary,
    AttemptKind,
    AttemptRejected,
)
from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import ProductEvidence

Status = Literal["ACTIVE", "INACTIVE", "DELETED"]
Money = StrictInt
RESOLUTION_PAGE_SIZE = 100
MAX_RESOLUTION_PRODUCTS = 100
MAX_RESOLUTION_PAGES = 20


class Summary(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    status: Status
    min_price: Money | None = Field(default=None, alias="minPrice", ge=0)
    max_price: Money | None = Field(default=None, alias="maxPrice", ge=0)


class Variant(BaseModel):
    id: UUID
    product_id: UUID = Field(alias="productId")
    price: Money = Field(ge=0)
    quantity: StrictInt = Field(ge=0)
    status: Status


class ResolutionPage(BaseModel):
    items: list[Summary] = Field(max_length=RESOLUTION_PAGE_SIZE)
    has_next: StrictBool = Field(alias="hasNext")
    next_cursor: str | None = Field(default=None, alias="nextCursor", max_length=2000)


class Detail(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    status: Status
    specifications: dict[str, JsonValue] = Field(default_factory=dict)
    variants: list[Variant] = Field(default_factory=list)


class BackendProductCatalog:
    def __init__(
        self, client: httpx.AsyncClient, base_url: str, attempts: AttemptBoundary
    ):
        self.client, self.base_url, self.attempts = (
            client,
            base_url.rstrip("/"),
            attempts,
        )

    async def _get(self, path: str, params: dict[str, JsonValue] | None = None) -> Any:
        async def dispatch():
            response = await self.client.get(self.base_url + path, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            body = response.json()
            if (
                not isinstance(body, dict)
                or set(body) != {"data", "message", "errors"}
                or body["message"] != "SUCCESS"
                or body["errors"] != []
            ):
                raise ValueError("Invalid catalog response envelope")
            return body["data"]

        try:
            return await self.attempts.invoke(
                AttemptKind.TOOL_IO,
                "catalog:" + path,
                canonical_hash({"path": path, "params": params}),
                dispatch,
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Admission failures must propagate, not masquerade as catalog failures.
            if isinstance(exc, AttemptRejected):
                raise
            raise BackendUnavailableError(
                "Catalog returned unavailable or invalid data"
            ) from exc

    async def search(self, query: str, limit: int) -> list[ProductEvidence]:
        if not query.strip() or not 1 <= limit <= 20:
            raise ValueError("Invalid catalog search")
        body = await self._get("/products", {"keyword": query, "limit": limit})
        try:
            if not isinstance(body, dict) or not isinstance(body.get("items"), list):
                raise TypeError("Invalid product page")
            rows = [Summary.model_validate(row) for row in body["items"]]
            if len(rows) > limit or len({row.id for row in rows}) != len(rows):
                raise ValueError("Invalid product page identities/limit")
            return [
                self._evidence(
                    row.id,
                    row.name,
                    row.min_price if row.min_price == row.max_price else None,
                    {},
                    row.model_dump(mode="json"),
                )
                for row in rows
                if row.status == "ACTIVE"
            ]
        except (ValidationError, ValueError, TypeError) as exc:
            raise BackendUnavailableError("Catalog product page is invalid") from exc

    async def resolve(self, mention: str) -> list[ProductEvidence]:
        """Exhaust bounded keyword pages; never treat top-k as uniqueness proof."""
        normalized = " ".join(mention.casefold().split())
        if not normalized or len(mention) > 1000:
            raise ValueError("Invalid product mention")
        matches: list[ProductEvidence] = []
        seen_ids: set[UUID] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        for _ in range(MAX_RESOLUTION_PAGES):
            params: dict[str, JsonValue] = {
                "keyword": mention,
                "limit": RESOLUTION_PAGE_SIZE,
            }
            if cursor is not None:
                params["cursor"] = cursor
            body = await self._get("/products", params)
            try:
                page = ResolutionPage.model_validate(body)
                for row in page.items:
                    if row.id in seen_ids:
                        raise ValueError("Duplicate product across resolution pages")
                    seen_ids.add(row.id)
                    if len(seen_ids) > MAX_RESOLUTION_PRODUCTS:
                        raise ValueError("Product resolution exceeds candidate bound")
                    if row.status == "ACTIVE" and normalized in " ".join(
                        row.name.casefold().split()
                    ):
                        matches.append(
                            self._evidence(
                                row.id,
                                row.name,
                                row.min_price
                                if row.min_price == row.max_price
                                else None,
                                {},
                                row.model_dump(mode="json"),
                            )
                        )
                if not page.has_next:
                    return matches
                cursor = page.next_cursor
                if not cursor or cursor in seen_cursors:
                    raise ValueError("Product resolution cursor does not advance")
                seen_cursors.add(cursor)
            except (ValidationError, ValueError, TypeError) as exc:
                raise BackendUnavailableError(
                    "Canonical product resolution is incomplete or invalid"
                ) from exc
        raise BackendUnavailableError("Product resolution exceeds page bound")

    async def get(self, product_id: UUID) -> ProductEvidence | None:
        body = await self._get(f"/products/{product_id}")
        if body is None:
            return None
        try:
            row = Detail.model_validate(body)
            if row.id != product_id or any(
                v.product_id != row.id for v in row.variants
            ):
                raise ValueError("Product/variant identity mismatch")
            if row.status != "ACTIVE":
                return None
            prices = {v.price for v in row.variants if v.status == "ACTIVE"}
            price = next(iter(prices)) if len(prices) == 1 else None
            return self._evidence(
                row.id, row.name, price, row.specifications, row.model_dump(mode="json")
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise BackendUnavailableError("Catalog product detail is invalid") from exc

    @staticmethod
    def _evidence(
        identifier: UUID,
        name: str,
        price: int | None,
        specifications: dict[str, JsonValue],
        snapshot: dict[str, Any],
    ) -> ProductEvidence:
        return ProductEvidence(
            product_id=identifier,
            name=name,
            price_vnd=price,
            specifications=specifications,
            evidence_ref=f"catalog:product:{identifier}:{canonical_hash(snapshot)}",
            retrieved_at=datetime.now(UTC),
        )
