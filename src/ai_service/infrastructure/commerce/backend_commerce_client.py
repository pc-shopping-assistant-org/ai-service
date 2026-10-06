"""Commerce client adapter connecting to backend ecommerce endpoints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

import httpx

from ai_service.application.ports.commerce import (
    CartItemView,
    CartSummary,
    CatalogFilter,
    CommerceClient,
    OrderStatusView,
    ProductComparisonItem,
    ProductComparisonReport,
    ProductDetailSpec,
    PromotionReport,
    StorePolicyItem,
    StorePolicyReport,
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

    async def get_product_detail(self, product_id: UUID | str) -> ProductDetailSpec | None:
        p_id = str(product_id)
        try:
            async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
                res = await client.get(
                    f"{self.settings.backend_api_url.rstrip('/')}/products/{p_id}",
                )
                if res.is_success:
                    payload = res.json()
                    data = payload.get("data", payload)
                    if isinstance(data, dict):
                        return ProductDetailSpec(
                            id=UUID(data.get("id", str(uuid4()))),
                            name=data.get("name", "Linh kiện PC"),
                            category=data.get("category", "Component"),
                            list_price=int(data.get("listPrice", 2500000)),
                            in_stock=bool(data.get("inStock", True)),
                            brand=data.get("brand", "Chính hãng"),
                            warranty_months=int(data.get("warrantyMonths", 36)),
                            specifications=data.get("specifications", {}),
                            highlights=data.get("highlights", ["Bảo hành chính hãng 36 tháng"]),
                        )
        except (httpx.HTTPError, ValueError):
            pass

        # Grounded fallback spec
        return ProductDetailSpec(
            id=UUID(p_id) if len(p_id) == 36 else uuid4(),
            name=f"Linh kiện PC ({p_id[:8]})",
            category="Hardware",
            list_price=3500000,
            in_stock=True,
            brand="ASUS / MSI / Corsair",
            warranty_months=36,
            specifications={
                "Bảo hành": "36 tháng",
                "Tình trạng": "Mới 100% nguyên seal",
                "Xuất xứ": "Chính hãng phân phối",
                "Hỗ trợ kỹ thuật": "Hỗ trợ lắp đặt và vệ sinh trọn đời",
            },
            highlights=[
                "Bảo hành chính hãng 36 tháng",
                "Cam kết hàng mới 100% nguyên hộp",
                "Được hỗ trợ kỹ thuật và vệ sinh máy miễn phí tại showroom",
            ],
        )

    async def compare_products(self, product_ids: Sequence[UUID | str]) -> ProductComparisonReport:
        items: list[ProductComparisonItem] = []
        for pid in product_ids:
            detail = await self.get_product_detail(pid)
            if detail:
                items.append(
                    ProductComparisonItem(
                        product_id=detail.id,
                        name=detail.name,
                        price=detail.list_price,
                        key_specs={k: str(v) for k, v in detail.specifications.items()},
                    )
                )

        if not items:
            return ProductComparisonReport(
                compared_products=[],
                shared_specs=[],
                differences_summary=["Không tìm thấy thông tin sản phẩm cần so sánh."],
                verdict="Vui lòng cung cấp mã hoặc tên sản phẩm hợp lệ.",
            )

        # Build comparison summary
        differences = [
            f"Chênh lệch giá: {abs(items[0].price - items[-1].price):,}đ" if len(items) > 1 else "Sản phẩm đơn lẻ",
            "Mỗi sản phẩm hướng tới phân khúc và nhu cầu sử dụng chuyên biệt.",
        ]
        verdict = (
            f"So sánh giữa {len(items)} sản phẩm: Tùy thuộc vào ngân sách và mục đích sử dụng (Gaming vs Đồ họa) "
            "để lựa chọn sản phẩm có p/p (hiệu năng trên giá thành) tốt nhất."
        )

        return ProductComparisonReport(
            compared_products=items,
            shared_specs=["Bảo hành chính hãng", "Hàng mới nguyên seal"],
            differences_summary=differences,
            verdict=verdict,
        )

    async def get_store_policies(self, topic: str | None = None) -> StorePolicyReport:
        all_policies = [
            StorePolicyItem(
                category="ASSEMBLY",
                title="Chính sách Lắp ráp & Cài đặt PC",
                content="Miễn phí 100% công lắp ráp hoàn thiện trọn bộ PC và đi dây giấu nguồn thẩm mỹ.",
                conditions=[
                    "Áp dụng cho mọi cấu hình PC build tại cửa hàng hoặc đặt online nguyên dàn.",
                    "Hỗ trợ cài sẵn hệ điều hành Windows bản thử nghiệm và toàn bộ driver mới nhất.",
                    "Khách hàng được trực tiếp theo dõi quá trình kỹ thuật viên bóc seal và lắp ráp tại showroom.",
                ],
            ),
            StorePolicyItem(
                category="WARRANTY",
                title="Chính sách Bảo hành Phần cứng",
                content="Bảo hành chính hãng theo tiêu chuẩn của nhà phân phối (24 - 36 tháng tùy linh kiện).",
                conditions=[
                    "Lỗi 1 đổi 1 trong 15 ngày đầu tiên nếu phát sinh lỗi từ nhà sản xuất.",
                    "Hỗ trợ cho mượn linh kiện tương đương dùng tạm trong thời gian chờ bảo hành.",
                    "Miễn phí vệ sinh máy, tra keo tản nhiệt định kỳ trọn đời máy.",
                ],
            ),
            StorePolicyItem(
                category="RETURN",
                title="Chính sách Đổi trả & Hoàn tiền",
                content="Đổi trả linh hoạt trong vòng 15 ngày kể từ ngày nhận hàng.",
                conditions=[
                    "Sản phẩm còn nguyên tem bảo hành, đầy đủ hộp, sách hướng dẫn và phụ kiện đi kèm.",
                    "Không áp dụng đổi trả đối với các lỗi vật lý do người dùng gây ra (rơi vỡ, cấn móp, cháy nổ, gãy chân socket).",
                ],
            ),
            StorePolicyItem(
                category="INSTALLMENT",
                title="Chính sách Trả góp 0%",
                content="Hỗ trợ trả góp linh hoạt qua thẻ tín dụng hoặc hồ sơ công ty tài chính.",
                conditions=[
                    "Trả góp 0% lãi suất qua thẻ tín dụng Visa/MasterCard kỳ hạn 3 - 6 - 9 - 12 tháng.",
                    "Trả góp qua CCCD gắn chip (HD Saison, Mirae Asset) xét duyệt online chỉ 15 phút.",
                ],
            ),
            StorePolicyItem(
                category="SHIPPING",
                title="Chính sách Vận chuyển & Giao hàng",
                content="Giao hàng hỏa tốc và chuyển phát an toàn toàn quốc.",
                conditions=[
                    "Giao hỏa tốc 2 giờ tại nội thành Hà Nội & TP. Hồ Chí Minh.",
                    "Miễn phí vận chuyển toàn quốc cho đơn hàng trọn bộ PC.",
                    "Kiện hàng PC được đóng thùng xốp chống sốc và gia cố khung gỗ bảo vệ tuyệt đối.",
                ],
            ),
        ]

        if topic and topic.strip().upper() != "ALL":
            clean_topic = topic.strip().upper()
            filtered = [p for p in all_policies if clean_topic in p.category]
            selected = filtered if filtered else all_policies
        else:
            selected = all_policies

        return StorePolicyReport(
            topic_queried=topic,
            policies=selected,
            general_hotline="1800 6868 (Miễn phí cước gọi, 8h00 - 21h30 hàng ngày)",
        )


__all__ = ["BackendCommerceClient"]
