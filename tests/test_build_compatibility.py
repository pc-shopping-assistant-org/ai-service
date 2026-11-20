from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio

from ai_service.api.dependencies import get_build_compatibility
from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.application.build_check import (
    BuildCompatibilityService,
    CheckStatus,
    CompatibilityCheckRequest,
    ComponentNotFoundError,
    check_build,
)
from ai_service.infrastructure.commerce.variant_components import BackendComponentSource
from ai_service.infrastructure.hardware import LocalHardwareRuleEngine
from ai_service.main import app

ENGINE = LocalHardwareRuleEngine()


def part(category: ComponentCategory, **fields: object) -> ComponentSpec:
    return ComponentSpec.model_validate({"id": uuid4(), "name": category.value, "category": category, **fields})


def build() -> dict[str, ComponentSpec]:
    return {
        "CPU": part(ComponentCategory.CPU, socket="AM5", tdp_watts=105),
        "MAINBOARD": part(
            ComponentCategory.MAINBOARD, socket="AM5", ram_type="DDR5", ram_slots=4, form_factor="ATX"
        ),
        "RAM": part(ComponentCategory.RAM, ram_type="DDR5", kit_modules=2),
        "GPU": part(ComponentCategory.GPU, gpu_length_mm=300, tdp_watts=250),
        "CASE": part(
            ComponentCategory.CASE, supported_form_factors=["ATX", "mATX"], max_gpu_length_mm=400, max_cooler_height_mm=165,
        ),
        "COOLER": part(ComponentCategory.COOLER, supported_sockets=["AM5", "LGA1700"], cooler_height_mm=150),
        "SSD": part(ComponentCategory.STORAGE, interface="NVMe", form_factor="M.2"),
        "PSU": part(ComponentCategory.PSU, wattage=850),
    }


def by_rule(parts: dict[str, ComponentSpec]) -> dict[str, CheckStatus]:
    return {result.rule: result.status for result in check_build(ENGINE, parts).results}


def test_a_matching_build_is_ok_on_every_rule() -> None:
    result = check_build(ENGINE, build())

    assert result.overall == CheckStatus.OK
    assert {r.rule for r in result.results} == {
        "SOCKET", "RAM_TYPE", "RAM_SLOTS", "FORM_FACTOR", "GPU_CLEARANCE", "COOLER_CLEARANCE", "COOLER_SOCKET", "PSU_WATTAGE", "SSD_INTERFACE",
    }
    assert all(r.status == CheckStatus.OK and r.missing_attributes == [] for r in result.results)


def test_aio_cooler_matching_build_is_ok() -> None:
    parts = build()
    parts["COOLER"] = part(ComponentCategory.COOLER, supported_sockets=["AM5", "LGA1700"], radiator_mm=360)
    parts["CASE"] = part(
        ComponentCategory.CASE, supported_form_factors=["ATX", "mATX"], max_gpu_length_mm=400, max_radiator_mm=360,
    )
    result = check_build(ENGINE, parts)
    assert result.overall == CheckStatus.OK
    assert next(r for r in result.results if r.rule == "COOLER_CLEARANCE").status == CheckStatus.OK


def test_aio_cooler_exceeding_case_radiator_is_incompatible() -> None:
    parts = build()
    parts["COOLER"] = part(ComponentCategory.COOLER, supported_sockets=["AM5", "LGA1700"], radiator_mm=360)
    parts["CASE"] = part(
        ComponentCategory.CASE, supported_form_factors=["ATX", "mATX"], max_gpu_length_mm=400, max_radiator_mm=240,
    )
    result = check_build(ENGINE, parts)
    assert result.overall == CheckStatus.INCOMPATIBLE
    assert next(r for r in result.results if r.rule == "COOLER_CLEARANCE").status == CheckStatus.INCOMPATIBLE


def test_a_wrong_socket_is_incompatible_and_names_both_slots() -> None:
    parts = build()
    parts["CPU"] = part(ComponentCategory.CPU, socket="LGA1700")

    result = check_build(ENGINE, parts)
    socket = next(r for r in result.results if r.rule == "SOCKET")

    assert result.overall == CheckStatus.INCOMPATIBLE
    assert (socket.status, socket.slot_a, socket.slot_b) == (CheckStatus.INCOMPATIBLE, "CPU", "MAINBOARD")
    assert socket.message
    assert by_rule(parts)["COOLER_SOCKET"] == CheckStatus.OK  # AM5 or LGA1700: this cooler fits the new CPU too


def test_ram_gpu_cooler_and_case_clearances_are_each_reported() -> None:
    parts = build()
    parts["RAM"] = part(ComponentCategory.RAM, ram_type="DDR4")
    parts["GPU"] = part(ComponentCategory.GPU, gpu_length_mm=450)
    parts["COOLER"] = part(ComponentCategory.COOLER, supported_sockets=["AM5"], cooler_height_mm=170)
    parts["CASE"] = part(
        ComponentCategory.CASE, supported_form_factors=["mATX"], max_gpu_length_mm=400, max_cooler_height_mm=165,
    )

    statuses = by_rule(parts)

    assert statuses["RAM_TYPE"] == CheckStatus.INCOMPATIBLE
    assert statuses["GPU_CLEARANCE"] == CheckStatus.INCOMPATIBLE
    assert statuses["COOLER_CLEARANCE"] == CheckStatus.INCOMPATIBLE
    assert statuses["FORM_FACTOR"] == CheckStatus.INCOMPATIBLE


def test_ram_kit_exceeding_mainboard_slots_is_incompatible() -> None:
    parts = build()
    parts["MAINBOARD"] = part(
        ComponentCategory.MAINBOARD, socket="AM5", ram_type="DDR5", ram_slots=2, form_factor="ATX"
    )
    parts["RAM"] = part(ComponentCategory.RAM, ram_type="DDR5", kit_modules=4)

    result = check_build(ENGINE, parts)
    ram_slots = next(r for r in result.results if r.rule == "RAM_SLOTS")

    assert result.overall == CheckStatus.INCOMPATIBLE
    assert ram_slots.status == CheckStatus.INCOMPATIBLE


@pytest.mark.parametrize(
    ("mainboard_fields", "ram_fields", "missing"),
    [
        ({"ram_type": "DDR5"}, {"ram_type": "DDR5", "kit_modules": 2}, ["MAINBOARD.ram_slots"]),
        ({"ram_type": "DDR5", "ram_slots": 4}, {"ram_type": "DDR5"}, ["RAM.kit_modules"]),
    ],
)
def test_ram_slot_rule_is_unknown_when_slot_or_kit_count_is_missing(
    mainboard_fields: dict[str, object],
    ram_fields: dict[str, object],
    missing: list[str],
) -> None:
    parts = build()
    parts["MAINBOARD"] = part(ComponentCategory.MAINBOARD, **mainboard_fields)
    parts["RAM"] = part(ComponentCategory.RAM, **ram_fields)

    result = next(r for r in check_build(ENGINE, parts).results if r.rule == "RAM_SLOTS")

    assert result.status == CheckStatus.UNKNOWN
    assert result.missing_attributes == missing


def test_m2_ssd_without_m2_slots_on_mainboard_is_incompatible() -> None:
    parts = build()
    parts["MAINBOARD"] = part(ComponentCategory.MAINBOARD, socket="AM5", ram_type="DDR5", form_factor="ATX", m2_slots=0)
    parts["SSD"] = part(ComponentCategory.STORAGE, interface="NVMe", form_factor="M.2")
    result = check_build(ENGINE, parts)
    assert result.overall == CheckStatus.INCOMPATIBLE
    assert next(r for r in result.results if r.rule == "SSD_INTERFACE").status == CheckStatus.INCOMPATIBLE


def test_sata_ssd_or_hdd_is_compatible() -> None:
    parts = build()
    parts["MAINBOARD"] = part(ComponentCategory.MAINBOARD, socket="AM5", ram_type="DDR5", form_factor="ATX", m2_slots=0)
    parts["SSD"] = part(ComponentCategory.STORAGE, interface="SATA", form_factor="3.5")
    result = check_build(ENGINE, parts)
    assert next(r for r in result.results if r.rule == "SSD_INTERFACE").status == CheckStatus.OK


def test_a_weak_power_supply_is_only_a_warning() -> None:
    parts = build()
    parts["PSU"] = part(ComponentCategory.PSU, wattage=200)

    result = check_build(ENGINE, parts)

    assert next(r for r in result.results if r.rule == "PSU_WATTAGE").status == CheckStatus.WARNING
    assert result.overall == CheckStatus.WARNING


def test_missing_data_is_unknown_never_ok() -> None:
    parts = build()
    parts["MAINBOARD"] = part(ComponentCategory.MAINBOARD, socket="AM5", ram_type="DDR5")  # no form factor
    parts["GPU"] = part(ComponentCategory.GPU)  # no length

    results = {r.rule: r for r in check_build(ENGINE, parts).results}

    assert results["FORM_FACTOR"].status == CheckStatus.UNKNOWN
    assert results["FORM_FACTOR"].missing_attributes == ["MAINBOARD.form_factor"]
    assert results["GPU_CLEARANCE"].status == CheckStatus.UNKNOWN
    assert results["GPU_CLEARANCE"].missing_attributes == ["GPU.gpu_length_mm"]
    assert check_build(ENGINE, parts).overall == CheckStatus.UNKNOWN


def test_a_rule_is_left_out_until_both_its_slots_are_chosen() -> None:
    assert check_build(ENGINE, {"CPU": build()["CPU"]}).results == []
    only_cpu_and_board = {k: v for k, v in build().items() if k in {"CPU", "MAINBOARD"}}
    assert [r.rule for r in check_build(ENGINE, only_cpu_and_board).results] == ["SOCKET"]


class Parts:
    def __init__(self, parts: dict[UUID, ComponentSpec], down: bool = False) -> None:
        self.parts, self.down = parts, down

    async def component(self, variant_id: UUID, category: ComponentCategory) -> ComponentSpec | None:
        if self.down:
            raise BackendUnavailableError("down")
        found = self.parts.get(variant_id)
        return found.model_copy(update={"category": category}) if found else None


@pytest.mark.asyncio
async def test_the_service_reads_each_chosen_variant_and_skips_slots_that_are_not_hardware() -> None:
    cpu, board = build()["CPU"], build()["MAINBOARD"]
    service = BuildCompatibilityService(ENGINE, Parts({cpu.id: cpu, board.id: board}))  # type: ignore[dict-item]
    request = CompatibilityCheckRequest(slots={"cpu": cpu.id, "MAINBOARD": board.id, "MONITOR": uuid4()})  # type: ignore[dict-item]

    result = await service.check(request)

    assert [r.rule for r in result.results] == ["SOCKET"]


@pytest.mark.asyncio
async def test_an_unknown_variant_names_its_slot() -> None:
    service = BuildCompatibilityService(ENGINE, Parts({}))
    missing = uuid4()

    with pytest.raises(ComponentNotFoundError) as raised:
        await service.check(CompatibilityCheckRequest(slots={"GPU": missing}))

    assert (raised.value.slot, raised.value.variant_id) == ("GPU", missing)


class Catalog:
    def __init__(self, variant: dict | None, product: dict | None) -> None:
        self.variant, self.product = variant, product

    async def get_variant(self, variant_id: UUID) -> dict | None:
        return self.variant

    async def get_product(self, product_id: UUID) -> dict | None:
        return self.product


@pytest.mark.asyncio
async def test_the_catalog_source_reads_specs_and_keeps_a_part_whose_one_value_is_unreadable() -> None:
    variant_id, product_id = uuid4(), uuid4()
    catalog = Catalog(
        {"id": str(variant_id), "productId": str(product_id), "price": 5_000_000},
        {
            "name": "Ryzen 7", "brandName": "AMD",
            "specifications": {"socket": "AM5", "tdp_watts": "105W", "unrelated": 1},  # "105W" is not a number
        },
    )

    spec = await BackendComponentSource(catalog).component(variant_id, ComponentCategory.CPU)  # type: ignore[arg-type]

    assert spec is not None
    assert (spec.id, spec.name, spec.price, spec.socket, spec.brand) == (variant_id, "Ryzen 7", 5_000_000, "AM5", "AMD")
    assert spec.tdp_watts is None


@pytest.mark.asyncio
async def test_the_catalog_source_keeps_extra_hardware_specs() -> None:
    variant_id, product_id = uuid4(), uuid4()
    catalog = Catalog(
        {"id": str(variant_id), "productId": str(product_id), "price": 2_000_000},
        {
            "name": "AIO Cooler",
            "specifications": {
                "radiator_mm": 360,
                "max_radiator_mm": 420,
                "m2_slots": 3,
                "interface": "NVMe",
                "cooler_type": "aio",
                "kit_modules": 2,
            },
        },
    )

    spec = await BackendComponentSource(catalog).component(
        variant_id, ComponentCategory.COOLER  # type: ignore[arg-type]
    )

    assert spec is not None
    assert spec.extra_specs == {
        "radiator_mm": 360,
        "max_radiator_mm": 420,
        "m2_slots": 3,
        "interface": "NVMe",
        "cooler_type": "aio",
    }
    assert (spec.radiator_mm, spec.max_radiator_mm, spec.m2_slots, spec.interface) == (
        360,
        420,
        3,
        "NVMe",
    )
    assert spec.kit_modules == 2


@pytest.mark.asyncio
async def test_the_catalog_source_returns_none_for_an_unknown_variant_or_product() -> None:
    assert await BackendComponentSource(Catalog(None, None)).component(uuid4(), ComponentCategory.CPU) is None  # type: ignore[arg-type]
    gone = Catalog({"productId": str(uuid4()), "price": 1}, None)
    assert await BackendComponentSource(gone).component(uuid4(), ComponentCategory.CPU) is None  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def client_with(request: pytest.FixtureRequest) -> AsyncIterator[httpx.AsyncClient]:
    service: BuildCompatibilityService = request.param
    app.dependency_overrides[get_build_compatibility] = lambda: service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_build_compatibility, None)


_CPU, _BOARD = build()["CPU"], build()["MAINBOARD"]
_SERVICE = BuildCompatibilityService(ENGINE, Parts({_CPU.id: _CPU, _BOARD.id: _BOARD}))  # type: ignore[dict-item]


@pytest.mark.asyncio
@pytest.mark.parametrize("client_with", [_SERVICE], indirect=True)
async def test_the_endpoint_answers_in_the_envelope_without_a_sign_in(client_with: httpx.AsyncClient) -> None:
    response = await client_with.post("/api/v1/compatibility/check", json={"slots": {"CPU": str(_CPU.id), "MAINBOARD": str(_BOARD.id)}})

    body = response.json()
    assert response.status_code == 200
    assert (body["message"], body["errors"]) == ("SUCCESS", [])
    assert body["data"]["overall"] == "OK"
    assert body["data"]["results"][0] == {
        "rule": "SOCKET", "status": "OK", "slot_a": "CPU", "slot_b": "MAINBOARD",
        "message": None, "suggestion": None, "missing_attributes": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("client_with", [BuildCompatibilityService(ENGINE, Parts({}))], indirect=True)
async def test_the_endpoint_reports_a_part_the_catalog_does_not_have(client_with: httpx.AsyncClient) -> None:
    response = await client_with.post("/api/v1/compatibility/check", json={"slots": {"GPU": str(uuid4())}})

    body = response.json()
    assert body["data"] is None
    assert body["message"] == "AI_PRODUCT_NOT_FOUND"
    assert body["errors"][0]["field"] == "slots.GPU"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_with", [BuildCompatibilityService(ENGINE, Parts({}, down=True))], indirect=True)
async def test_the_endpoint_reports_an_unreachable_catalog(client_with: httpx.AsyncClient) -> None:
    body = (await client_with.post("/api/v1/compatibility/check", json={"slots": {"CPU": str(uuid4())}})).json()

    assert (body["data"], body["message"]) == (None, "AI_BACKEND_UNAVAILABLE")


@pytest.mark.asyncio
@pytest.mark.parametrize("client_with", [_SERVICE], indirect=True)
async def test_the_endpoint_rejects_a_slot_that_is_not_an_id(client_with: httpx.AsyncClient) -> None:
    response = await client_with.post("/api/v1/compatibility/check", json={"slots": {"CPU": "not-an-id"}})

    assert response.status_code == 422
