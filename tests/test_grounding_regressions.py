"""Grounding regressions: unavailable dependencies must never invent commerce facts."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
import pytest

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.commerce import CatalogFilter
from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchQuery,
    WebSearchResult,
)
from ai_service.capabilities.shopping_assistant.schemas import (
    ExportBuildToCartArgs,
    ManageCartArgs,
)
from ai_service.capabilities.shopping_assistant.tools import ShoppingAssistantTools
from ai_service.capabilities.web_search.schemas import ProductReviewSearchArgs
from ai_service.capabilities.web_search.tools import WebSearchTools
from ai_service.infrastructure.commerce.backend_commerce_client import (
    BackendCommerceClient,
)
from ai_service.infrastructure.search.duckduckgo_adapter import DuckDuckGoSearchAdapter

PRODUCT_ID = UUID("12345678-1234-1234-1234-123456789abc")


def stub_http(monkeypatch: pytest.MonkeyPatch, mode: str, payload: Any = None) -> None:
    """Use the real HTTP client with an in-process transport, never the network."""
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        if mode == "timeout":
            raise httpx.ReadTimeout("Dependency unavailable", request=request)
        if mode == "invalid_json":
            return httpx.Response(200, text="not JSON")
        if mode == "empty_html":
            return httpx.Response(200, text="<html>No search results</html>")
        return httpx.Response(int(mode), json=payload)

    def create_client(**kwargs: Any) -> httpx.AsyncClient:
        return original_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "503", "invalid_json"])
async def test_catalog_failure_does_not_fabricate_products(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    stub_http(monkeypatch, mode)
    with pytest.raises(BackendUnavailableError):
        await BackendCommerceClient().search_with_filters(CatalogFilter(keyword="RTX 4060"))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "503", "invalid_json"])
async def test_product_detail_failure_does_not_fabricate_specs(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    stub_http(monkeypatch, mode)
    with pytest.raises(BackendUnavailableError):
        await BackendCommerceClient().get_product_detail(PRODUCT_ID)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["INACTIVE", "DELETED"])
async def test_product_detail_hides_inactive_catalog_rows(
    monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    stub_http(monkeypatch, "200", {"data": {
        "id": str(PRODUCT_ID), "name": "Hidden product", "listPrice": 1_000_000,
        "status": status,
    }})
    assert await BackendCommerceClient().get_product_detail(PRODUCT_ID) is None


@pytest.mark.asyncio
async def test_missing_product_detail_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_http(monkeypatch, "404", {"data": None, "message": "PRODUCT_NOT_FOUND", "errors": []})
    assert await BackendCommerceClient().get_product_detail(PRODUCT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "kwargs"), [
    ("get_cart", {}),
    ("add_item_to_cart", {"variant_id": PRODUCT_ID}),
    ("get_order_by_code", {"order_code": "ORDER-123"}),
    ("get_applicable_promotions", {"order_value": 30_000_000}),
    ("get_store_policies", {"topic": "WARRANTY"}),
])
async def test_unintegrated_commerce_operations_fail_closed(
    method: str, kwargs: dict[str, Any],
) -> None:
    with pytest.raises(BackendUnavailableError):
        await getattr(BackendCommerceClient(), method)(**kwargs)


@pytest.mark.asyncio
async def test_export_build_without_atomic_backend_capability_does_not_fake_success() -> None:
    tools = ShoppingAssistantTools(BackendCommerceClient())
    with pytest.raises(BackendUnavailableError):
        await tools.export_build_to_cart(ExportBuildToCartArgs(component_variant_ids=[PRODUCT_ID]))


@pytest.mark.asyncio
async def test_missing_cart_is_not_replaced_with_a_synthetic_cart() -> None:
    class MissingCart:
        async def get_cart(self) -> None:
            return None

    assert await ShoppingAssistantTools(MissingCart()).manage_cart(ManageCartArgs(action="VIEW")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "503"])
async def test_failed_web_search_returns_unavailable_without_invented_sources(
    monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    stub_http(monkeypatch, mode)
    result = await DuckDuckGoSearchAdapter().search(WebSearchQuery(query="RTX 4060 benchmarks"))
    assert result.items == []
    assert result.total_results == 0
    assert result.note == "WEB_SEARCH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_empty_web_search_does_not_invent_a_result(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_http(monkeypatch, "empty_html")
    result = await DuckDuckGoSearchAdapter().search(WebSearchQuery(query="RTX 4060 benchmarks"))
    assert result.items == []
    assert result.total_results == 0


@pytest.mark.asyncio
async def test_search_tool_preserves_unavailable_note_and_does_not_claim_live_evidence() -> None:
    class UnavailableSearch:
        async def search(self, query: WebSearchQuery) -> WebSearchResult:
            return WebSearchResult(
                query=query.query, category=WebSearchCategory.REVIEWS,
                items=[], total_results=0, note="WEB_SEARCH_UNAVAILABLE",
            )

    response = await WebSearchTools(UnavailableSearch()).search_product_reviews(
        ProductReviewSearchArgs(product_name="RTX 4060"),
    )
    assert response.results_found == 0
    assert response.results == []
    assert response.model_dump().get("note") == "WEB_SEARCH_UNAVAILABLE"
    assert "thông tin tra cứu mới nhất" not in response.analysis_prompt
    assert "Hãy dùng thông tin thực tế trên" not in response.analysis_prompt
