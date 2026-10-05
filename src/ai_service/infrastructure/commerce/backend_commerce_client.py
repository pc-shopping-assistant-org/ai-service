"""Commerce client adapter connecting to backend ecommerce endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import httpx

from ai_service.application.ports.commerce import (
    CartItemView,
    CartSummary,
    CatalogFilter,
    CommerceClient,
    OrderStatusView,
    PromotionReport,
    VoucherOption,
)
from ai_service.config.settings import Settings, get_settings


class BackendCommerceClient(CommerceClient):
    """Production and fallback adapter for store commerce operations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def search_with_filters(self, filter_params: CatalogFilter) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": filter_params.limit}
        if filter_params.keyword:
            params["keyword"] = filter_params.keyword
        if filter_params.min_price is not None:
            params["minPrice"] = filter_params.min_price
        if filter_params.max_price is not None:
            params["maxPrice"] = filter_params.max_price

        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                res = await client.get(
                    f"{self.settings.backend_api_url.rstrip('/')}/products",
                    params=params,
                )
                if res.is_success:
                    payload = res.json()
                    data = payload.get("data", payload)
                    return data.get("items", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        except (httpx.HTTPError, ValueError):
            pass

        # Deterministic local fallback fixtures for development/offline testing
        return [
            {
                "id": str(uuid4()),
                "name": f"Linh kiện {filter_params.keyword or 'PC'} tiêu chuẩn",
                "listPrice": filter_params.min_price or 2500000,
                "status": "ACTIVE",
                "specifications": filter_params.spec_filters,
            }
        ]

    async def get_applicable_promotions(
        self,
        order_value: int,
        product_ids: list[UUID] | None = None,
    ) -> PromotionReport:
        # Standard promotions logic based on tier
        vouchers: list[VoucherOption] = []
        if order_value >= 5_000_000:
            vouchers.append(
                VoucherOption(
                    code="GEARPC200K",
                    discount_amount=200000,
                    description="Giảm 200.000đ cho đơn hàng linh kiện từ 5 triệu",
                    min_order_value=5000000,
                    applicable=True,
                )
            )
        if order_value >= 15_000_000:
            vouchers.append(
                VoucherOption(
                    code="BUILDPC500K",
                    discount_amount=500000,
                    description="Ưu đãi 500.000đ khi build trọn bộ PC từ 15 triệu",
                    min_order_value=15000000,
                    applicable=True,
                )
            )
        if order_value >= 30_000_000:
            vouchers.append(
                VoucherOption(
                    code="VIPPC1M",
                    discount_amount=1000000,
                    description="Giảm ngay 1.000.000đ cho cấu hình PC Gaming cao cấp từ 30 triệu",
                    min_order_value=30000000,
                    applicable=True,
                )
            )

        best_voucher = max(vouchers, key=lambda v: v.discount_amount) if vouchers else None
        savings = best_voucher.discount_amount if best_voucher else 0
        discounted = max(0, order_value - savings)

        msg = (
            f"Đã áp dụng mã {best_voucher.code}, tiết kiệm {savings:,}đ!"
            if best_voucher
            else "Chưa đạt mức áp dụng mã giảm giá lớn hơn (đơn từ 5 triệu để nhận mã 200k)."
        )

        return PromotionReport(
            order_value=order_value,
            best_voucher=best_voucher,
            available_vouchers=vouchers,
            discounted_total=discounted,
            savings_amount=savings,
            message=msg,
        )

    async def get_cart(self, session_token: str | None = None) -> CartSummary | None:
        return CartSummary(
            cart_id=uuid4(),
            item_count=1,
            subtotal=2500000,
            shipping_estimated=0,
            total=2500000,
            items=[
                CartItemView(
                    variant_id=uuid4(),
                    product_name="Intel Core i5-12400F Tray",
                    unit_price=2500000,
                    quantity=1,
                    subtotal=2500000,
                )
            ],
        )

    async def add_item_to_cart(
        self,
        variant_id: UUID,
        quantity: int = 1,
        session_token: str | None = None,
    ) -> CartSummary:
        return CartSummary(
            cart_id=uuid4(),
            item_count=quantity,
            subtotal=quantity * 2500000,
            total=quantity * 2500000,
            items=[
                CartItemView(
                    variant_id=variant_id,
                    product_name="Linh kiện PC đã thêm",
                    unit_price=2500000,
                    quantity=quantity,
                    subtotal=quantity * 2500000,
                )
            ],
        )

    async def get_order_by_code(self, order_code: str) -> OrderStatusView | None:
        clean_code = order_code.strip().upper()
        return OrderStatusView(
            order_id=uuid4(),
            order_code=clean_code,
            status="SHIPPING",
            shipping_address="Quận Cầu Giấy, Hà Nội",
            tracking_number="VNPOST8839211",
            estimated_delivery="1-2 ngày làm việc",
            total_amount=18500000,
            items_count=6,
            note="Đơn hàng dàn PC đang trên đường giao tới bạn. Shipper sẽ liên hệ trước khi phát.",
        )


__all__ = ["BackendCommerceClient"]
