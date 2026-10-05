"""Pydantic schemas for Alibaba-style Shopping Assistant capability tools."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FilterCatalogArgs(BaseModel):
    """Parameters for smart faceted catalog search with multiple constraints."""

    model_config = ConfigDict(extra="ignore")

    keyword: str | None = Field(default=None, description="Free-text product search term (e.g. 'RTX 4060', 'bàn phím cơ').")
    category_slug: str | None = Field(default=None, description="Category slug if specified (e.g. 'vga', 'cpu', 'ram', 'ban-phim').")
    min_price: int | None = Field(default=None, ge=0, description="Minimum price in VND.")
    max_price: int | None = Field(default=None, ge=0, description="Maximum price in VND.")
    in_stock_only: bool = Field(default=True, description="Filter only items currently available in stock.")
    limit: int = Field(default=5, ge=1, le=20)


class CheckPromotionsArgs(BaseModel):
    """Parameters for finding the best discount codes and vouchers for an order."""

    model_config = ConfigDict(extra="ignore")

    order_value: int = Field(
        description="Total current value of the cart or planned build in VND.",
        ge=0,
    )
    product_ids: list[UUID] | None = Field(
        default=None,
        description="Optional list of product IDs to check for specific item-level vouchers.",
    )


class ManageCartArgs(BaseModel):
    """Parameters for cart actions: view current cart, add an item, or update quantity."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["VIEW", "ADD_ITEM"] = Field(
        default="VIEW",
        description="'VIEW' to inspect cart total and items; 'ADD_ITEM' to put a product variant into the cart.",
    )
    variant_id: UUID | None = Field(
        default=None,
        description="Product variant UUID required when action is 'ADD_ITEM'.",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=99,
        description="Quantity of items to add.",
    )


class TrackOrderArgs(BaseModel):
    """Parameters for tracking customer order delivery and shipment status."""

    model_config = ConfigDict(extra="ignore")

    order_code: str = Field(
        description="Order tracking code (e.g. 'ORD-2026-991', 'DH-12345').",
        min_length=3,
    )


class ExportBuildToCartArgs(BaseModel):
    """Parameters for batch adding an entire curated PC build into the customer cart."""

    model_config = ConfigDict(extra="ignore")

    component_variant_ids: list[UUID] = Field(
        description="List of product variant UUIDs corresponding to the 6-8 chosen components of the build.",
        min_length=1,
    )


__all__ = [
    "CheckPromotionsArgs",
    "ExportBuildToCartArgs",
    "FilterCatalogArgs",
    "ManageCartArgs",
    "TrackOrderArgs",
]
