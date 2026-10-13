from uuid import uuid4

import httpx
import pytest

from ai_service.application.call_attempts import AttemptRejected
from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.infrastructure.commerce.hardware_catalog import (
    BackendRequirementCatalog,
    HardwareCatalogMapping,
)


@pytest.mark.asyncio
async def test_sixteen_products_are_complete_without_exhausting_call_budget():
    category = uuid4()
    products = [uuid4() for _ in range(16)]
    requests = []

    class Budget:
        calls = 0

        async def invoke(self, kind, key, digest, dispatch):
            if self.calls >= 16:
                raise AttemptRejected("CALL_BUDGET_EXHAUSTED")
            self.calls += 1
            return await dispatch()

    def respond(request):
        requests.append(request.url.path)
        assert request.url.path == "/products/snapshot"
        return httpx.Response(
            200,
            json={
                "message": "SUCCESS",
                "errors": [],
                "data": {
                    "hasNext": False,
                    "nextCursor": None,
                    "items": [
                        {
                            "id": str(p),
                            "name": "GPU",
                            "categoryId": str(category),
                            "status": "ACTIVE",
                            "specifications": {},
                            "variants": [
                                {
                                    "id": str(p),
                                    "productId": str(p),
                                    "sku": str(p),
                                    "price": 1_000_000,
                                    "quantity": 1,
                                    "status": "ACTIVE",
                                }
                            ],
                        }
                        for p in products
                    ],
                },
            },
        )

    budget = Budget()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            budget,
            HardwareCatalogMapping(categories={category: "GPU"}),
        )
        rows = await catalog.load(Draft(items=[]), CanonicalRequirementState())
    assert {r.variant_id for r in rows} == set(products)
    assert budget.calls == 1
    assert requests == ["/products/snapshot"]


@pytest.mark.asyncio
async def test_summary_shaped_response_is_not_an_empty_hardware_snapshot():
    category = uuid4()

    class Budget:
        async def invoke(self, kind, key, digest, dispatch):
            return await dispatch()

    def respond(request):
        return httpx.Response(
            200,
            json={
                "message": "SUCCESS",
                "errors": [],
                "data": {
                    "hasNext": False,
                    "nextCursor": None,
                    "items": [
                        {
                            "id": str(uuid4()),
                            "name": "GPU",
                            "categoryId": str(category),
                            "status": "ACTIVE",
                        }
                    ],
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            Budget(),
            HardwareCatalogMapping(categories={category: "GPU"}),
        )
        with pytest.raises(BackendUnavailableError, match="invalid"):
            await catalog.load(Draft(items=[]), CanonicalRequirementState())
