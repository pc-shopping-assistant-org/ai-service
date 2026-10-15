from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.hardware import ComponentCategory as C
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.schemas import FindAlternativesArgs
from ai_service.capabilities.pc_builder.tools import PCBuilderTools
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine


def test_alternatives_require_catalog_instead_of_embedded_products() -> None:
    tools = PCBuilderTools(LocalHardwareRuleEngine())
    with pytest.raises(BackendUnavailableError):
        tools.find_compatible_alternatives(FindAlternativesArgs(slot=C.GPU))


@pytest.mark.parametrize("ceiling", [0, 100, 1_000])
def test_price_ceiling_never_falls_back_to_overpriced_products(ceiling: int) -> None:
    result = PCBuilderTools(LocalHardwareRuleEngine()).find_compatible_alternatives(
        FindAlternativesArgs(slot=C.GPU, max_price=ceiling),
        [ComponentSpec(id=uuid4(), name="Catalog GPU", category=C.GPU, price=2_000)],
    )
    assert result.alternatives == []


def test_empty_catalog_returns_no_alternatives() -> None:
    result = PCBuilderTools(LocalHardwareRuleEngine()).find_compatible_alternatives(
        FindAlternativesArgs(slot=C.MAINBOARD, required_socket="AM5"), [],
    )
    assert result.alternatives == []


def test_alternatives_preserve_catalog_identity_price_and_known_specs() -> None:
    good = ComponentSpec(
        id=uuid4(), name="Catalog board", category=C.MAINBOARD,
        price=1_500, socket="AM5", ram_type="DDR5", form_factor="ATX",
    )
    candidates = [
        good.model_copy(update={"id": uuid4(), "name": "Expensive", "price": 2_001}),
        good.model_copy(update={"id": uuid4(), "name": "Wrong socket", "socket": "AM4"}),
        good.model_copy(update={"id": uuid4(), "name": "Wrong RAM", "ram_type": "DDR4"}),
        good.model_copy(update={"id": uuid4(), "name": "Unknown socket", "socket": None}),
        good.model_copy(update={"id": uuid4(), "name": "Unknown RAM", "ram_type": None}),
        good.model_copy(update={"id": uuid4(), "name": "Wrong category", "category": C.CPU}),
        good.model_copy(update={"id": None, "name": "Unidentified"}),
        good,
    ]
    result = PCBuilderTools(LocalHardwareRuleEngine()).find_compatible_alternatives(
        FindAlternativesArgs(
            slot=C.MAINBOARD, required_socket="am-5", required_ram_type="ddr 5", max_price=2_000,
        ), candidates,
    )
    assert len(result.alternatives) == 1
    part = result.alternatives[0]
    assert (part.id, part.name, part.price) == (good.id, good.name, good.price)
    assert part.specs_summary == "AM5; DDR5; ATX"
    assert part.in_stock is None  # ComponentSpec carries no stock evidence.
    assert result.message == "CATALOG_ALTERNATIVES_FOUND"


def test_cooler_socket_support_and_limit_are_catalog_bound() -> None:
    ids = sorted([uuid4(), uuid4()], key=str)
    candidates = [
        ComponentSpec(id=ids[1], name="Second", category=C.COOLER, price=100, supported_sockets=["AM4", "AM5"]),
        ComponentSpec(id=ids[0], name="First", category=C.COOLER, price=100, supported_sockets=["AM5"]),
        ComponentSpec(id=uuid4(), name="Unknown support", category=C.COOLER, price=50),
    ]
    args = FindAlternativesArgs(slot=C.COOLER, required_socket="AM5", limit=1)
    tools = PCBuilderTools(LocalHardwareRuleEngine())
    first = tools.find_compatible_alternatives(args, candidates)
    second = tools.find_compatible_alternatives(args, list(reversed(candidates)))
    assert first == second
    assert [part.id for part in first.alternatives] == [ids[0]]


def test_negative_price_ceiling_is_invalid() -> None:
    with pytest.raises(ValidationError):
        FindAlternativesArgs(slot=C.GPU, max_price=-1)
