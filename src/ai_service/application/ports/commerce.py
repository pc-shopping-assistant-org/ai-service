"""Outbound port for commercial operations: filtered search, vouchers, cart & orders."""

from __future__ import annotations

from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CatalogFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    keyword: str | None = None
    category_slug: str | None = None
    brand_id: UUID | None = None
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    in_stock_only: bool = True
    spec_filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=10, ge=1, le=50)


class VoucherOption(BaseModel):
    code: str
    discount_amount: int
    description: str
    min_order_value: int
    applicable: bool


class PromotionReport(BaseModel):
    order_value: int
    best_voucher: VoucherOption | None = None
    available_vouchers: list[VoucherOption] = Field(default_factory=list)
    discounted_total: int
    savings_amount: int
    message: str


class CartItemView(BaseModel):
    product_id: UUID | None = None
    variant_id: UUID
    product_name: str
    unit_price: int
    quantity: int
    subtotal: int


class CartSummary(BaseModel):
    cart_id: UUID
    item_count: int
    subtotal: int
    shipping_estimated: int = 0
    total: int
    items: list[CartItemView] = Field(default_factory=list)


class OrderStatusView(BaseModel):
    order_id: UUID | None = None
    order_code: str
    status: str  # PENDING_PAYMENT, CONFIRMED, SHIPPING, COMPLETED, CANCELLED
    shipping_address: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    total_amount: int
    items_count: int
    note: str


class CommerceClient(Protocol):
    """Client protocol for store commerce operations."""

    async def search_with_filters(self, filter_params: CatalogFilter) -> list[dict[str, Any]]:
        """Search products with multi-attribute filtering."""

    async def get_applicable_promotions(
        self,
        order_value: int,
        product_ids: list[UUID] | None = None,
    ) -> PromotionReport:
        """Find best discounts and vouchers for the current selection/value."""

    async def get_cart(self, session_token: str | None = None) -> CartSummary | None:
        """Fetch current active cart details."""

    async def add_item_to_cart(
        self,
        variant_id: UUID,
        quantity: int = 1,
        session_token: str | None = None,
    ) -> CartSummary:
        """Add an item/variant to the active cart."""

    async def get_order_by_code(self, order_code: str) -> OrderStatusView | None:
        """Lookup order shipping/delivery status."""
