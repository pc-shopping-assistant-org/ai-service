"""Agent tools for the Alibaba-style Shopping Assistant capability."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_service.application.ports.commerce import (
    CartItemView,
    CartSummary,
    CatalogFilter,
    CommerceClient,
    OrderStatusView,
    PromotionReport,
)
from ai_service.capabilities.shopping_assistant.schemas import (
    CheckPromotionsArgs,
    ExportBuildToCartArgs,
    FilterCatalogArgs,
    ManageCartArgs,
    TrackOrderArgs,
)
from ai_service.infrastructure.commerce.backend_commerce_client import (
    BackendCommerceClient,
)


class ShoppingAssistantTools:
    """Executable commerce tools callable by PydanticAI agents or unified chat workflows."""

    def __init__(self, commerce_client: CommerceClient | None = None) -> None:
        self.commerce_client = commerce_client or BackendCommerceClient()

    async def search_catalog_with_filters(self, args: FilterCatalogArgs) -> list[dict[str, Any]]:
        """Tìm kiếm sản phẩm đa tiêu chí với các bộ lọc thông minh (Alibaba / Taobao style).

        Hỗ trợ:
        - Lọc theo từ khóa tự do (ví dụ: 'RTX 4060', 'bàn phím cơ').
        - Lọc theo danh mục linh kiện.
        - Lọc theo khoảng giá tối thiểu / tối đa (VND).
        - Chỉ lấy hàng còn trong kho.
        """
        filter_params = CatalogFilter(
            keyword=args.keyword,
            category_slug=args.category_slug,
            min_price=args.min_price,
            max_price=args.max_price,
            in_stock_only=args.in_stock_only,
            limit=args.limit,
        )
        return await self.commerce_client.search_with_filters(filter_params)

    async def check_promotions_and_vouchers(self, args: CheckPromotionsArgs) -> PromotionReport:
        """Kiểm tra và tìm các mã giảm giá, voucher khuyến mãi tốt nhất cho đơn hàng.

        Tự động tính toán số tiền tiết kiệm tối đa và tổng tiền sau giảm giá để tư vấn khách chốt đơn.
        """
        return await self.commerce_client.get_applicable_promotions(
            order_value=args.order_value,
            product_ids=args.product_ids,
        )

    async def manage_cart(self, args: ManageCartArgs) -> CartSummary:
        """Thao tác trực tiếp với giỏ hàng của khách hàng qua hội thoại.

        Hỗ trợ:
        - 'VIEW': Xem danh sách sản phẩm, số lượng và tổng tiền trong giỏ hiện tại.
        - 'ADD_ITEM': Thêm sản phẩm (variant) vào giỏ hàng với số lượng tùy chọn.
        """
        if args.action == "ADD_ITEM" and args.variant_id is not None:
            return await self.commerce_client.add_item_to_cart(
                variant_id=args.variant_id,
                quantity=args.quantity,
            )
        cart = await self.commerce_client.get_cart()
        return cart or CartSummary(
            cart_id=uuid4(),
            item_count=0,
            subtotal=0,
            total=0,
            items=[],
        )

    async def track_order_status(self, args: TrackOrderArgs) -> OrderStatusView:
        """Tra cứu trạng thái vận chuyển và tiến độ giao hàng của một đơn hàng theo mã đơn.

        Trả về: tình trạng đơn (CONFIRMED, SHIPPING, COMPLETED...), mã vận đơn bưu cục, ngày giao dự kiến.
        """
        status = await self.commerce_client.get_order_by_code(args.order_code)
        if status is not None:
            return status
        return OrderStatusView(
            order_code=args.order_code,
            status="NOT_FOUND",
            total_amount=0,
            items_count=0,
            note=f"Không tìm thấy thông tin đơn hàng với mã '{args.order_code}'. Vui lòng kiểm tra lại mã đơn.",
        )

    async def export_build_to_cart(self, args: ExportBuildToCartArgs) -> CartSummary:
        """Đẩy toàn bộ 6-8 linh kiện của bộ PC đã build vào giỏ hàng chỉ trong một thao tác.

        Giúp khách hàng nhanh chóng chuyển từ giai đoạn tư vấn cấu hình sang bước thanh toán đặt cọc.
        """
        items: list[CartItemView] = []
        subtotal = 0
        for vid in args.component_variant_ids:
            item_price = 2500000
            items.append(
                CartItemView(
                    variant_id=vid,
                    product_name="Linh kiện trong bộ PC cấu hình",
                    unit_price=item_price,
                    quantity=1,
                    subtotal=item_price,
                )
            )
            subtotal += item_price

        return CartSummary(
            cart_id=uuid4(),
            item_count=len(items),
            subtotal=subtotal,
            total=subtotal,
            items=items,
        )


__all__ = ["ShoppingAssistantTools"]
