from uuid import uuid4

import pytest

from ai_service.capabilities.shopping_assistant.schemas import (
    CheckPromotionsArgs,
    ExportBuildToCartArgs,
    FilterCatalogArgs,
    ManageCartArgs,
    TrackOrderArgs,
)
from ai_service.capabilities.shopping_assistant.tools import ShoppingAssistantTools


@pytest.fixture
def commerce_tools() -> ShoppingAssistantTools:
    return ShoppingAssistantTools()


@pytest.mark.asyncio
async def test_search_catalog_with_filters(commerce_tools: ShoppingAssistantTools) -> None:
    args = FilterCatalogArgs(keyword="RTX 4060", min_price=7000000, max_price=9000000, limit=3)
    results = await commerce_tools.search_catalog_with_filters(args)
    assert len(results) > 0
    assert "name" in results[0]


@pytest.mark.asyncio
async def test_check_promotions_and_vouchers(commerce_tools: ShoppingAssistantTools) -> None:
    args = CheckPromotionsArgs(order_value=16_000_000)
    report = await commerce_tools.check_promotions_and_vouchers(args)
    assert report.best_voucher is not None
    assert report.savings_amount >= 500_000
    assert report.discounted_total < 16_000_000


@pytest.mark.asyncio
async def test_manage_cart_view_and_add(commerce_tools: ShoppingAssistantTools) -> None:
    # 1. View cart
    view_args = ManageCartArgs(action="VIEW")
    cart = await commerce_tools.manage_cart(view_args)
    assert cart.total >= 0

    # 2. Add item
    add_args = ManageCartArgs(action="ADD_ITEM", variant_id=uuid4(), quantity=2)
    cart_updated = await commerce_tools.manage_cart(add_args)
    assert cart_updated.item_count == 2


@pytest.mark.asyncio
async def test_track_order_status(commerce_tools: ShoppingAssistantTools) -> None:
    args = TrackOrderArgs(order_code="DH-98214")
    order = await commerce_tools.track_order_status(args)
    assert order.status == "SHIPPING"
    assert order.order_code == "DH-98214"


@pytest.mark.asyncio
async def test_export_build_to_cart(commerce_tools: ShoppingAssistantTools) -> None:
    part_ids = [uuid4() for _ in range(6)]
    args = ExportBuildToCartArgs(component_variant_ids=part_ids)
    cart = await commerce_tools.export_build_to_cart(args)
    assert cart.item_count == 6
    assert len(cart.items) == 6


@pytest.mark.asyncio
async def test_compare_products(commerce_tools: ShoppingAssistantTools) -> None:
    from ai_service.capabilities.shopping_assistant.schemas import CompareProductsArgs

    args = CompareProductsArgs(product_ids=[str(uuid4()), str(uuid4())])
    report = await commerce_tools.compare_products(args)
    assert len(report.compared_products) == 2
    assert len(report.differences_summary) > 0


@pytest.mark.asyncio
async def test_get_product_detail_specs(commerce_tools: ShoppingAssistantTools) -> None:
    from ai_service.capabilities.shopping_assistant.schemas import (
        GetProductDetailSpecsArgs,
    )

    args = GetProductDetailSpecsArgs(product_id=str(uuid4()))
    detail = await commerce_tools.get_product_detail_specs(args)
    assert detail.name != ""
    assert detail.warranty_months >= 24
    assert len(detail.specifications) > 0


@pytest.mark.asyncio
async def test_query_store_policies(commerce_tools: ShoppingAssistantTools) -> None:
    from ai_service.capabilities.shopping_assistant.schemas import (
        QueryStorePoliciesArgs,
    )

    # 1. Query warranty
    args_w = QueryStorePoliciesArgs(topic="WARRANTY")
    report_w = await commerce_tools.query_store_policies(args_w)
    assert any("Bảo hành" in p.title for p in report_w.policies)

    # 2. Query all
    args_all = QueryStorePoliciesArgs(topic="ALL")
    report_all = await commerce_tools.query_store_policies(args_all)
    assert len(report_all.policies) >= 5
