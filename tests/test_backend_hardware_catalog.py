from uuid import uuid4

import httpx
import pytest
from test_backend_product_catalog import Budget

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
async def test_hardware_type_is_loaded_from_each_snapshot_without_category_env():
    product, category = uuid4(), uuid4()
    hardware_type: str | None = "RAM"

    def respond(request):
        return httpx.Response(
            200,
            json={
                "message": "SUCCESS",
                "errors": [],
                "data": {
                    "items": [
                        {
                            "id": str(product),
                            "name": "Renamed product",
                            "categoryId": str(category),
                            "hardwareType": hardware_type,
                            "status": "ACTIVE",
                            "specifications": {},
                            "variants": [
                                {
                                    "id": str(product),
                                    "productId": str(product),
                                    "sku": "PART",
                                    "price": 100,
                                    "quantity": 1,
                                    "status": "ACTIVE",
                                }
                            ],
                        }
                    ],
                    "hasNext": False,
                    "nextCursor": None,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client, "https://catalog.test", Budget(), HardwareCatalogMapping()
        )
        rows = await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
        assert rows[0].component.category == "RAM"
        hardware_type = "GPU"
        rows = await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
        assert rows[0].component.category == "GPU"
        hardware_type = None
        assert await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState()) == []
        hardware_type = "UNSUPPORTED"
        with pytest.raises(BackendUnavailableError):
            await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())


@pytest.mark.asyncio
async def test_variant_capacity_price_stock_and_visibility_are_canonical():
    category, product, capacity_option, first, second = (uuid4() for _ in range(5))
    other_capacity_option = uuid4()
    variants = [
        {
            "id": str(first),
            "productId": str(product),
            "sku": "RAM-32",
            "price": 2_000_000,
            "quantity": 0,
            "status": "ACTIVE",
            "options": [
                {"id": str(capacity_option), "name": "Capacity", "value": "32"}
            ],
        },
        {
            "id": str(second),
            "productId": str(product),
            "sku": "RAM-64",
            "price": 3_000_000,
            "quantity": 5,
            "status": "ACTIVE",
            "options": [
                {"id": str(other_capacity_option), "name": "Capacity", "value": "64"}
            ],
        },
        {
            "id": str(uuid4()),
            "productId": str(product),
            "sku": "HIDDEN",
            "price": 1,
            "quantity": 1,
            "status": "DELETED",
        },
    ]

    def respond(request):
        assert request.url.path == "/products/snapshot"
        data = {
            "items": [
                {
                    "id": str(product),
                    "name": "RAM",
                    "categoryId": str(category),
                    "hardwareType": "RAM",
                    "status": "ACTIVE",
                    "specifications": {"ram_type": "DDR5", "capacity_gb": 16},
                    "variants": variants,
                }
            ],
            "hasNext": False,
            "nextCursor": None,
        }
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    budget = Budget()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            budget,
            HardwareCatalogMapping(
                option_fields={
                    capacity_option: "capacity_gb",
                    other_capacity_option: "capacity_gb",
                },
            ),
        )
        rows = await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
        assert [row.variant_id for row in rows] == [first, second]
        assert [row.component.capacity_gb for row in rows] == [32, 64]
        assert [row.component.price for row in rows] == [2_000_000, 3_000_000]
        assert [row.quantity for row in rows] == [0, 5]
        assert rows[0].component.socket is None
        assert rows[0].component.performance_score is None
        assert rows[0].evidence_refs != rows[1].evidence_refs
        variants[0]["options"][0]["id"] = str(uuid4())
        with pytest.raises(BackendUnavailableError, match="mapping"):
            await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
    assert budget.calls == ["TOOL_IO"] * 2


@pytest.mark.asyncio
async def test_accessory_taxonomy_maps_canonical_price_without_fake_core_category():
    from ai_service.application.ports.accessories import AccessorySpec

    category, product, variant = uuid4(), uuid4(), uuid4()

    def respond(request):
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {
                            "id": str(product),
                            "name": "Mouse",
                            "categoryId": str(category),
                            "hardwareType": "MOUSE",
                            "brandName": "Logitech",
                            "status": "ACTIVE",
                            "specifications": {},
                            "variants": [
                                {
                                    "id": str(variant),
                                    "productId": str(product),
                                    "sku": "GEAR-1",
                                    "price": 500_000,
                                    "quantity": 2,
                                    "status": "ACTIVE",
                                }
                            ],
                        }
                    ],
                    "hasNext": False,
                    "nextCursor": None,
                },
                "message": "SUCCESS",
                "errors": [],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            Budget(),
            HardwareCatalogMapping(),
        )
        result = await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())
    assert len(result) == 1
    assert isinstance(result[0].component, AccessorySpec)
    assert result[0].component.id == variant
    assert result[0].component.category == "MOUSE"
    assert result[0].component.price == 500_000


@pytest.mark.asyncio
async def test_candidate_scan_uses_all_pages_and_rejects_truncation():
    category, p1, p2 = uuid4(), uuid4(), uuid4()

    def respond(request):
        assert request.url.path == "/products/snapshot"
        cursor = request.url.params.get("cursor")
        identity = p1 if cursor is None else p2
        data = {
            "items": [
                {
                    "id": str(identity),
                    "name": "GPU",
                    "categoryId": str(category),
                    "hardwareType": "GPU",
                    "status": "ACTIVE",
                    "specifications": {},
                    "variants": [
                        {
                            "id": str(identity),
                            "productId": str(identity),
                            "sku": str(identity),
                            "price": 1_000_000,
                            "quantity": 1,
                            "status": "ACTIVE",
                        }
                    ],
                }
            ],
            "hasNext": cursor is None,
            "nextCursor": str(p1) if cursor is None else None,
        }
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            Budget(),
            HardwareCatalogMapping(),
        )
        assert (
            len(await catalog.load(RequirementDraft(items=[]), CanonicalRequirementState())) == 2
        )
        bounded = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            Budget(),
            HardwareCatalogMapping(),
            max_rows=1,
        )
        with pytest.raises(BackendUnavailableError, match="bound"):
            await bounded.load(RequirementDraft(items=[]), CanonicalRequirementState())


@pytest.mark.asyncio
async def test_backend_snapshot_flows_into_real_optimizer_and_satisfaction(
    realistic_catalog,
):
    from test_requirement_pipeline import state

    from ai_service.capabilities.assistant.requirements.build import (
        optimize_requirements,
        verify_receipt,
    )
    from ai_service.capabilities.assistant.requirements.verifier import (
        SatisfactionReport,
    )
    from ai_service.capabilities.pc_builder.application.service import (
        PCBuildApplicationService,
    )

    categories = {part.category: uuid4() for part in realistic_catalog}
    details = {}
    for index, part in enumerate(realistic_catalog):
        product_id, variant_id = uuid4(), uuid4()
        specs = part.model_dump(
            mode="json",
            exclude={"id", "name", "category", "price", "extra_specs"},
            exclude_none=True,
        )
        specs.update(part.extra_specs)
        details[str(product_id)] = {
            "id": str(product_id),
            "name": part.name,
            "status": "ACTIVE",
            "categoryId": str(categories[part.category]),
            "hardwareType": part.category,
            "specifications": specs,
            "variants": [
                {
                    "id": str(variant_id),
                    "productId": str(product_id),
                    "sku": f"SKU-{index}",
                    "price": part.price,
                    "quantity": 1,
                    "status": "ACTIVE",
                }
            ],
        }

    def respond(request):
        assert request.url.path == "/products/snapshot"
        data = {
            "items": list(details.values()),
            "hasNext": False,
            "nextCursor": None,
        }
        return httpx.Response(
            200, json={"data": data, "message": "SUCCESS", "errors": []}
        )

    budget = Budget()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        catalog = BackendRequirementCatalog(
            client,
            "https://catalog.test",
            budget,
            HardwareCatalogMapping(),
        )
        requirements = state()
        rows = await catalog.load(RequirementDraft(items=[]), requirements)
        receipt = optimize_requirements(
            requirements, rows, PCBuildApplicationService()
        )
    assert receipt.outcome == "BUILT" and receipt.build is not None
    satisfaction = verify_receipt(requirements, receipt, rows)
    assert satisfaction is not None
    assert SatisfactionReport.model_validate(satisfaction.model_dump()).hard_satisfied
    assert receipt.build.total_price <= 30_000_000
    assert len(budget.calls) == 1
