from uuid import uuid4

import httpx
import pytest

from ai_service.application.errors import BackendUnavailableError
from ai_service.infrastructure.commerce.product_catalog import BackendProductCatalog


class Budget:
    def __init__(self):
        self.calls = []

    async def invoke(self, kind, key, digest, dispatch):
        self.calls.append(kind)
        return await dispatch()


@pytest.mark.asyncio
async def test_task_lookup_exhausts_cursor_pages_instead_of_assuming_first_unique():
    ids = [uuid4(), uuid4()]
    budget = Budget()
    calls = []

    def respond(request):
        calls.append(dict(request.url.params))
        index = 1 if request.url.params.get("cursor") else 0
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": str(ids[index]),
                            "name": f"Vendor {index} RTX 4060",
                            "status": "ACTIVE",
                        }
                    ],
                    "hasNext": index == 0,
                    "nextCursor": "second" if index == 0 else None,
                },
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await BackendProductCatalog(
            client, "https://catalog.test", budget
        ).resolve("RTX 4060")
    assert [p.product_id for p in result] == ids
    assert len(budget.calls) == 2
    assert calls[1]["cursor"] == "second"


@pytest.mark.asyncio
async def test_task_lookup_rejects_missing_cursor_instead_of_returning_partial_matches():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": {"items": [], "hasNext": True, "nextCursor": None},
                    "message": "SUCCESS",
                    "errors": [],
                },
            )
        )
    ) as client:
        with pytest.raises(BackendUnavailableError):
            await BackendProductCatalog(
                client, "https://catalog.test", Budget()
            ).resolve("RTX 4060")


@pytest.mark.asyncio
async def test_microservice_summary_range_is_not_reported_as_exact_price():
    identity = uuid4()
    budget = Budget()

    def respond(request):
        assert request.url.params["keyword"] == "GPU"
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": str(identity),
                            "name": "GPU",
                            "status": "ACTIVE",
                            "minPrice": 8_000_000,
                            "maxPrice": 9_000_000,
                        },
                        {"id": str(uuid4()), "name": "Hidden", "status": "INACTIVE"},
                    ],
                    "hasNext": False,
                },
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await BackendProductCatalog(
            client, "https://catalog.test/api/v1", budget
        ).search("GPU", 5)
    assert len(result) == 1 and result[0].product_id == identity
    assert result[0].price_vnd is None
    assert budget.calls == ["TOOL_IO"]


@pytest.mark.asyncio
async def test_detail_uses_active_variant_price_and_rechecks_product_identity():
    product_id = uuid4()
    response = {
        "id": str(product_id),
        "name": "RAM",
        "status": "ACTIVE",
        "specifications": {"capacity_gb": 32},
        "variants": [
            {
                "id": str(uuid4()),
                "productId": str(product_id),
                "price": 2_000_000,
                "quantity": 5,
                "status": "ACTIVE",
            },
            {
                "id": str(uuid4()),
                "productId": str(product_id),
                "price": 100,
                "quantity": 5,
                "status": "DELETED",
            },
        ],
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"data": response, "message": "SUCCESS", "errors": []}
            )
        )
    ) as client:
        catalog = BackendProductCatalog(client, "https://catalog.test/api/v1", Budget())
        detail = await catalog.get(product_id)
        assert detail.price_vnd == 2_000_000
        assert detail.specifications["capacity_gb"] == 32
        with pytest.raises(BackendUnavailableError):
            await catalog.get(uuid4())


@pytest.mark.asyncio
async def test_catalog_error_envelope_never_becomes_successful_empty_search():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": None,
                    "message": "CATALOG_FAILURE",
                    "errors": [{"message": "failed"}],
                },
            )
        )
    ) as client:
        with pytest.raises(BackendUnavailableError):
            await BackendProductCatalog(
                client, "https://catalog.test/api/v1", Budget()
            ).search("GPU", 5)
