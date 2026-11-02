from uuid import UUID, uuid4

import httpx
import pytest

from ai_service.application.call_attempts import AttemptRejected
from ai_service.application.errors import BackendUnavailableError
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.infrastructure.commerce.hardware_catalog import (
    BackendRequirementCatalog,
    HardwareCatalogMapping,
)


@pytest.mark.asyncio
async def test_seed_sized_catalog_fits_runtime_context_without_entering_model_prompt():
    from ai_service.capabilities.assistant.react.contracts import ToolContext

    count = 2391
    identities = [uuid4() for _ in range(count)]
    category = uuid4()

    class Budget:
        calls = 0

        async def invoke(self, kind, key, digest, dispatch):
            self.calls += 1
            assert self.calls <= 48
            return await dispatch()

    def respond(request):
        cursor = request.url.params.get("cursor")
        start = identities.index(UUID(cursor)) + 1 if cursor else 0
        selected = identities[start : start + 100]
        more = start + len(selected) < count
        return httpx.Response(
            200,
            json={
                "message": "SUCCESS",
                "errors": [],
                "data": {
                    "items": [
                        {
                            "id": str(p),
                            "name": "GPU",
                            "categoryId": str(category),
                            "hardwareType": "GPU",
                            "status": "ACTIVE",
                            "specifications": {},
                            "variants": [
                                {
                                    "id": str(p),
                                    "productId": str(p),
                                    "sku": str(p),
                                    "price": 1000000,
                                    "quantity": 10,
                                    "status": "ACTIVE",
                                }
                            ],
                        }
                        for p in selected
                    ],
                    "hasNext": more,
                    "nextCursor": str(selected[-1]) if more else None,
                },
            },
        )

    budget = Budget()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client, "https://catalog.test", budget, HardwareCatalogMapping()
        )
        rows = await catalog.load(
            RequirementDraft(items=[]), CanonicalRequirementState()
        )
    context = ToolContext(
        account_id=uuid4(),
        conversation_id=uuid4(),
        run_id=uuid4(),
        requirements=CanonicalRequirementState(),
        catalog=rows,
        read_catalog_allowed=True,
        execution_allowed=True,
    )
    assert len(context.catalog) == count
    assert budget.calls == 24

    import json

    from ai_service.capabilities.assistant.react.contracts import ReactLimits
    from ai_service.infrastructure.providers.react_model import build_react_request

    messages, _ = build_react_request("Xin chào", [], (), None, context)
    assert str(identities[0]) not in messages[0].content
    # The graph keeps canonical rows in working context, provenance and accepted
    # preparation; this is distinct from the bounded model-facing projection.
    persisted = json.dumps(
        {"catalog_context": {"rows": [r.model_dump(mode="json") for r in rows]}}
    ).encode()
    assert len(persisted) * 3 < ReactLimits().max_state_bytes


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
                            "hardwareType": "GPU",
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
            HardwareCatalogMapping(),
        )
        rows = await catalog.load(
            RequirementDraft(items=[]), CanonicalRequirementState()
        )
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
            HardwareCatalogMapping(),
        )
        with pytest.raises(BackendUnavailableError, match="invalid"):
            await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
