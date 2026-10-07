"""Commerce tools never substitute fixture success for an unintegrated backend."""

from uuid import uuid4

import httpx
import pytest

from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.shopping_assistant.schemas import (
    CheckPromotionsArgs,
    CompareProductsArgs,
    ExportBuildToCartArgs,
    FilterCatalogArgs,
    GetProductDetailSpecsArgs,
    ManageCartArgs,
    QueryStorePoliciesArgs,
    TrackOrderArgs,
)
from ai_service.capabilities.shopping_assistant.tools import ShoppingAssistantTools


@pytest.fixture
def commerce_tools(monkeypatch: pytest.MonkeyPatch) -> ShoppingAssistantTools:
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        row = {
            "id": request.url.path.rsplit("/", 1)[-1] if not request.url.path.endswith("/products") else str(uuid4()),
            "name": "Fixture GPU", "status": "ACTIVE", "category": "GPU",
            "listPrice": 8_000_000, "inStock": True, "brand": "Fixture",
            "warrantyMonths": 24, "specifications": {"VRAM": 8},
        }
        data = {"items": [row]} if request.url.path.endswith("/products") else row
        return httpx.Response(200, json={"data": data, "message": "SUCCESS", "errors": []})

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    return ShoppingAssistantTools()


@pytest.mark.asyncio
async def test_search_catalog_with_filters(commerce_tools: ShoppingAssistantTools) -> None:
    results = await commerce_tools.search_catalog_with_filters(
        FilterCatalogArgs(keyword="GPU", min_price=7_000_000, max_price=9_000_000, limit=3),
    )
    assert len(results) == 1
    assert results[0]["listPrice"] == 8_000_000


@pytest.mark.asyncio
async def test_check_promotions_and_vouchers(commerce_tools: ShoppingAssistantTools) -> None:
    with pytest.raises(BackendUnavailableError):
        await commerce_tools.check_promotions_and_vouchers(CheckPromotionsArgs(order_value=16_000_000))


@pytest.mark.asyncio
async def test_manage_cart_view_and_add(commerce_tools: ShoppingAssistantTools) -> None:
    for args in (ManageCartArgs(action="VIEW"), ManageCartArgs(action="ADD_ITEM", variant_id=uuid4(), quantity=2)):
        with pytest.raises(BackendUnavailableError):
            await commerce_tools.manage_cart(args)


@pytest.mark.asyncio
async def test_track_order_status(commerce_tools: ShoppingAssistantTools) -> None:
    with pytest.raises(BackendUnavailableError):
        await commerce_tools.track_order_status(TrackOrderArgs(order_code="DH-98214"))


@pytest.mark.asyncio
async def test_export_build_to_cart(commerce_tools: ShoppingAssistantTools) -> None:
    with pytest.raises(BackendUnavailableError):
        await commerce_tools.export_build_to_cart(ExportBuildToCartArgs(component_variant_ids=[uuid4()]))


@pytest.mark.asyncio
async def test_compare_products(commerce_tools: ShoppingAssistantTools) -> None:
    report = await commerce_tools.compare_products(CompareProductsArgs(product_ids=[str(uuid4()), str(uuid4())]))
    assert len(report.compared_products) == 2
    assert report.shared_specs == ["VRAM"]
    assert report.differences_summary == ["Chênh lệch giá: 0đ"]


@pytest.mark.asyncio
async def test_get_product_detail_specs(commerce_tools: ShoppingAssistantTools) -> None:
    detail = await commerce_tools.get_product_detail_specs(GetProductDetailSpecsArgs(product_id=str(uuid4())))
    assert detail is not None
    assert detail.warranty_months == 24
    assert detail.specifications == {"VRAM": 8}


@pytest.mark.asyncio
async def test_query_store_policies(commerce_tools: ShoppingAssistantTools) -> None:
    with pytest.raises(BackendUnavailableError):
        await commerce_tools.query_store_policies(QueryStorePoliciesArgs(topic="WARRANTY"))
