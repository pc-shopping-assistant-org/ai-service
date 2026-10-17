"""Verify canonical requirements against results, not against compiler verdicts."""

from uuid import UUID, uuid5

from pydantic import Field

from ai_service.application.ports.accessories import AccessorySpec, AccessoryType
from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.stateful_contracts import Contract
from ai_service.capabilities.pc_builder.optimizer import (
    SYNTHETIC_NAMESPACE,
    calculate_power_estimate,
    check_compatibility_status,
    stable_component_id,
)
from ai_service.capabilities.pc_builder.schemas import RankedBuild, UseCaseProfile

from .compiler import Compilation, predicate_value
from .contracts import (
    CanonicalRequirementState,
    QuantityValue,
    ReferenceValue,
    TextValue,
)
from .setup import accessory_match


class Satisfaction(Contract):
    requirement_id: UUID
    satisfied: bool
    strength: str
    reason: str


class SatisfactionReport(Contract):
    state_hash: str
    results: list[Satisfaction]
    errors: list[str] = Field(default_factory=list)

    @property
    def hard_satisfied(self) -> bool:
        return not self.errors and all(
            r.satisfied for r in self.results if r.strength == "HARD"
        )


def _synthetic_matches(
    part: ComponentSpec, cpu: ComponentSpec, category: ComponentCategory
) -> bool:
    if category == ComponentCategory.GPU and cpu.has_integrated_graphics:
        expected = ComponentSpec(
            id=uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:igpu"),
            name=f"Đồ họa tích hợp {cpu.name} (iGPU)",
            category=category,
            price=0,
            tdp_watts=cpu.integrated_graphics_power_watts,
            performance_score=cpu.integrated_graphics_score,
            is_integrated=True,
            brand=cpu.brand,
        )
        return (
            cpu.integrated_graphics_score is not None
            and cpu.integrated_graphics_power_watts is not None
            and part == expected
        )
    if category == ComponentCategory.COOLER and cpu.includes_stock_cooler:
        expected = ComponentSpec(
            id=uuid5(SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:stock_cooler"),
            name=f"Tản nhiệt kèm theo CPU {cpu.name} (Stock Cooler)",
            category=category,
            price=0,
            cooler_height_mm=cpu.stock_cooler_height_mm,
            performance_score=cpu.stock_cooler_score,
            tdp_watts=cpu.stock_cooler_tdp_watts,
            supported_sockets=[cpu.socket] if cpu.socket else [],
            brand=cpu.brand,
        )
        return (
            cpu.stock_cooler_height_mm is not None
            and cpu.stock_cooler_score is not None
            and part == expected
        )
    return False


def verify_satisfaction(
    state: CanonicalRequirementState,
    compilation: Compilation,
    build: RankedBuild,
    catalog: list[CatalogRowV1],
    *,
    observed_use_case: UseCaseProfile,
) -> SatisfactionReport:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    build = RankedBuild.model_validate(build.model_dump())
    state_hash = canonical_hash(state.model_dump(mode="json"))
    report = SatisfactionReport(state_hash=state_hash, results=[])
    if set(build.parts) != set(ComponentCategory):
        report.errors.append("MISSING_REQUIRED_CORE_SLOTS")
    if build.compatibility_status != "COMPATIBLE":
        report.errors.append("COMPATIBILITY_NOT_CONFIRMED")
    try:
        power = calculate_power_estimate(build.parts, observed_use_case)
        if (
            power != build.power_estimate
            or check_compatibility_status(build.parts, power.minimum_capacity_watts)
            != "COMPATIBLE"
        ):
            report.errors.append("NATIVE_COMPATIBILITY_OR_POWER_MISMATCH")
    except ValueError:
        report.errors.append("POWER_EVIDENCE_INSUFFICIENT")
    requirements = [*state.goals.values(), *state.requirements.values()]
    if compilation.state_hash != state_hash:
        report.errors.append("COMPILED_STATE_MISMATCH")
    ids = [
        m.requirement_id for m in compilation.mapping if m.requirement_id is not None
    ]
    if set(ids) != {r.id for r in requirements} or len(ids) != len(set(ids)):
        report.errors.append("MAPPING_COVERAGE_MISMATCH")
    if not compilation.ready:
        report.errors.append("COMPILATION_NOT_READY")
    rows = {row.variant_id: row for row in catalog}
    if len(rows) != len(catalog):
        report.errors.append("DUPLICATE_CATALOG_IDENTITY")
    expected_owned: set[ComponentCategory] = set()
    spending = 0
    cpu = build.parts.get(ComponentCategory.CPU)
    for category, part in build.parts.items():
        row = rows.get(part.id) if part.id else None
        if row is None and cpu is not None and _synthetic_matches(part, cpu, category):
            if build.spending_prices.get(category) != 0:
                report.errors.append("INVALID_SYNTHETIC_SPENDING")
            continue
        if (
            row is None
            or row.component.model_dump() != part.model_dump()
            or part.category != category
        ):
            report.errors.append("RESULT_NOT_CANONICAL")
            continue
        selected_owned = any(
            r.subject == category
            and r.predicate == "use_resource"
            and isinstance(r.value, ReferenceValue)
            and r.value.variant_id == part.id
            for r in requirements
        )
        if selected_owned and any(
            resource.variant_id == part.id
            for resource in state.available_resources.values()
        ):
            expected_owned.add(category)
            actual_spending = 0
        else:
            actual_spending = part.price
            if row.quantity == 0:
                report.errors.append("PURCHASE_PART_UNAVAILABLE")
        if build.spending_prices.get(category) != actual_spending:
            report.errors.append("SPENDING_SNAPSHOT_MISMATCH")
        spending += actual_spending
    if spending != build.total_price or set(build.owned_categories) != expected_owned:
        report.errors.append("TOTAL_OR_OWNED_MISMATCH")
    accessory_spending = 0
    accessories: dict[AccessoryType, AccessorySpec] = {}
    if compilation.setup:
        for item in compilation.setup.accessories:
            row = rows.get(item.row.variant_id)
            accessory = item.row.component
            if (
                not isinstance(accessory, AccessorySpec)
                or row != item.row
                or accessory.category in accessories
            ):
                report.errors.append("ACCESSORY_NOT_CANONICAL_OR_DUPLICATE")
                continue
            rules = [r for r in requirements if r.subject == accessory.category]
            is_owned = any(
                r.predicate == "use_resource"
                and isinstance(r.value, ReferenceValue)
                and r.value.variant_id == accessory.id
                for r in rules
            )
            is_pinned = any(
                r.predicate == "identity"
                and isinstance(r.value, ReferenceValue)
                and r.value.variant_id == accessory.id
                for r in rules
            )
            if (
                item.owned != is_owned
                or item.pinned != is_pinned
                or (
                    is_owned
                    and accessory.id
                    not in {
                        resource.variant_id
                        for resource in state.available_resources.values()
                    }
                )
                or (not is_owned and row.quantity == 0)
                or not rules
            ):
                report.errors.append("ACCESSORY_BINDING_OR_STOCK_MISMATCH")
            accessories[accessory.category] = accessory
            accessory_spending += 0 if is_owned else accessory.price
        if set(accessories) != {
            AccessoryType(r.subject)
            for r in requirements
            if r.subject in set(AccessoryType)
        }:
            report.errors.append("ACCESSORY_SLOT_MISMATCH")
    for r in requirements:
        value = r.value
        ok = False
        if (
            r.subject == "SETUP"
            and r.predicate == "budget"
            and r.operator == "LTE"
            and isinstance(value, QuantityValue)
            and value.unit == "VND"
        ):
            ok = spending + accessory_spending <= value.amount
        elif (
            r.subject == "SETUP"
            and r.predicate == "accessory_budget"
            and isinstance(value, QuantityValue)
        ):
            ok = accessory_spending <= value.amount
        elif (
            r.subject == "SETUP"
            and r.predicate == "scope"
            and isinstance(value, TextValue)
        ):
            ok = (
                compilation.setup is not None
                and compilation.setup.scope.value == value.text
            )
        elif r.subject in set(AccessoryType):
            selected_accessory = accessories.get(AccessoryType(r.subject))
            ok = (
                selected_accessory is not None
                and accessory_match(r, selected_accessory) is True
            )
        elif (
            r.subject == "WORKLOAD"
            and r.predicate == "use_case"
            and isinstance(value, TextValue)
        ):
            ok = observed_use_case.value == value.text
        elif r.subject in set(ComponentCategory):
            category = ComponentCategory(r.subject)
            selected = build.parts.get(category)
            if selected is not None:
                if r.predicate == "model" and r.strength == "SOFT":
                    ok = predicate_value(r, selected) is True
                elif r.predicate in ("identity", "use_resource") and isinstance(
                    value, ReferenceValue
                ):
                    ok = selected.id == value.variant_id and (
                        r.predicate != "use_resource" or category in expected_owned
                    )
                elif (
                    r.predicate == "capacity"
                    and isinstance(value, QuantityValue)
                    and value.unit in ("GB", "TB")
                    and selected.capacity_gb is not None
                ):
                    amount = value.amount * (1000 if value.unit == "TB" else 1)
                    ok = (
                        selected.capacity_gb == amount
                        if r.operator == "EQ"
                        else selected.capacity_gb >= amount
                        if r.operator == "GTE"
                        else selected.capacity_gb <= amount
                    )
                elif r.predicate in ("brand", "media") and isinstance(value, TextValue):
                    actual = (
                        selected.brand
                        if r.predicate == "brand"
                        else selected.extra_specs.get("storage_media")
                    )
                    ok = (
                        isinstance(actual, str)
                        and actual.casefold() == value.text.casefold()
                    )
        report.results.append(
            Satisfaction(
                requirement_id=r.id,
                satisfied=ok,
                strength=r.strength,
                reason="SATISFIED" if ok else "UNSATISFIED_OR_UNKNOWN",
            )
        )
    return report
