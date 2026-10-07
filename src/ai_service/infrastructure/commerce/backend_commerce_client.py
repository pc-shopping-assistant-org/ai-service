"""Canonical catalog reads. Unintegrated commerce operations fail closed."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import httpx
from pydantic import ValidationError

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.commerce import (
    CartSummary,
    CatalogFilter,
    CommerceClient,
    OrderStatusView,
    ProductComparisonItem,
    ProductComparisonReport,
    ProductDetailSpec,
    PromotionReport,
    StorePolicyReport,
)
from ai_service.config.settings import Settings, get_settings


class BackendCommerceClient(CommerceClient):
    """Never substitute synthetic commerce facts for missing backend data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                response = await client.get(
                    f"{self.settings.backend_api_url.rstrip('/')}{path}", params=params,
                )
                if response.status_code == httpx.codes.NOT_FOUND:
                    return None
                response.raise_for_status()
                payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("Invalid catalog envelope")
            return payload.get("data", payload)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise BackendUnavailableError("Catalog service is unavailable or returned invalid data") from exc

    async def search_with_filters(self, filter_params: CatalogFilter) -> list[dict[str, Any]]:
        # Advanced filters need a confirmed backend contract; do not silently ignore them.
        if filter_params.category_slug or filter_params.brand_id or filter_params.spec_filters:
            raise BackendUnavailableError("Advanced catalog filters are not integrated")
        data = await self._get("/products", {"keyword": filter_params.keyword or "", "limit": filter_params.limit})
        if data is None:
            raise BackendUnavailableError("Catalog search endpoint is unavailable")
        rows = data.get("items", data.get("content", data.get("products"))) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise BackendUnavailableError("Catalog search returned invalid data")
        results = []
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "ACTIVE":
                continue
            price = row.get("listPrice")
            if filter_params.min_price is not None and (not isinstance(price, int) or price < filter_params.min_price):
                continue
            if filter_params.max_price is not None and (not isinstance(price, int) or price > filter_params.max_price):
                continue
            if filter_params.in_stock_only and not (row.get("inStock") is True or isinstance(row.get("quantity"), int) and row["quantity"] > 0):
                continue
            results.append(row)
        return results[:filter_params.limit]

    async def get_product_detail(self, product_id: UUID | str) -> ProductDetailSpec | None:
        try:
            canonical_id = UUID(str(product_id))
        except ValueError:
            return None
        data = await self._get(f"/products/{canonical_id}")
        if data is None:
            return None
        if not isinstance(data, dict):
            raise BackendUnavailableError("Catalog detail returned invalid data")
        if data.get("status") != "ACTIVE":
            return None
        try:
            detail = ProductDetailSpec.model_validate({
                "id": data.get("id"), "name": data.get("name"),
                "category": data.get("category"), "list_price": data.get("listPrice"),
                "in_stock": data.get("inStock"), "brand": data.get("brand"),
                "warranty_months": data.get("warrantyMonths"),
                "specifications": data.get("specifications") or {},
                "highlights": data.get("highlights") or [],
            })
            if detail.id != canonical_id:
                raise ValueError("Catalog returned a different product")
            return detail
        except (ValidationError, ValueError) as exc:
            raise BackendUnavailableError("Catalog detail is incomplete or invalid") from exc

    async def compare_products(self, product_ids: Sequence[UUID | str]) -> ProductComparisonReport:
        items = []
        for product_id in product_ids:
            detail = await self.get_product_detail(product_id)
            if detail is None:
                raise BackendUnavailableError("A comparison product is unavailable")
            items.append(ProductComparisonItem(
                product_id=detail.id, name=detail.name, price=detail.list_price,
                key_specs={key: str(value) for key, value in detail.specifications.items()},
            ))
        if not 2 <= len(items) <= 5:
            raise ValueError("Compare between two and five products")
        shared = set(items[0].key_specs)
        for item in items[1:]:
            shared &= set(item.key_specs)
        return ProductComparisonReport(
            compared_products=items,
            shared_specs=sorted(key for key in shared if len({item.key_specs[key] for item in items}) == 1),
            differences_summary=[f"Chênh lệch giá: {max(i.price for i in items) - min(i.price for i in items):,}đ"],
            verdict="Chỉ đối chiếu giá và thông số được backend cung cấp; chưa có dữ liệu để kết luận hiệu năng.",
        )

    async def get_applicable_promotions(self, order_value: int, product_ids: list[UUID] | None = None) -> PromotionReport:
        raise BackendUnavailableError("Promotion lookup is not integrated")

    async def get_cart(self, session_token: str | None = None) -> CartSummary | None:
        raise BackendUnavailableError("Authenticated cart lookup is not integrated")

    async def add_item_to_cart(self, variant_id: UUID, quantity: int = 1, session_token: str | None = None) -> CartSummary:
        raise BackendUnavailableError("Authenticated cart mutation is not integrated")

    async def get_order_by_code(self, order_code: str) -> OrderStatusView | None:
        raise BackendUnavailableError("Owned order lookup is not integrated")

    async def get_store_policies(self, topic: str | None = None) -> StorePolicyReport:
        raise BackendUnavailableError("Canonical store policies are not integrated")


__all__ = ["BackendCommerceClient"]
