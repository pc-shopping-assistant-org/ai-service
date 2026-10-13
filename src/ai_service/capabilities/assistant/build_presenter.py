"""Allowlisted projection of a current completed build, not a publish operation."""

from typing import Literal
from uuid import uuid5

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.build_stages import (
    OptimizationStage,
    stage_recommendation,
)
from ai_service.capabilities.assistant.conversation_core import (
    ConsultationStateV1,
    StageStatus,
)
from ai_service.capabilities.assistant.grounded_build import PreparedBuildV1
from ai_service.capabilities.assistant.public_contracts import (
    PublicBuildPartV1,
    PublicBuildV1,
)
from ai_service.capabilities.pc_builder.optimizer import (
    SYNTHETIC_NAMESPACE,
    stable_component_id,
)


def to_public_build(stage: OptimizationStage) -> PublicBuildV1 | None:
    state = ConsultationStateV1.model_validate(stage.state.model_dump())
    prepared = PreparedBuildV1.model_validate(stage.prepared.model_dump())
    normalized = prepared.optimizer_input
    marker = state.optimization
    if (
        normalized is None
        or marker.status != StageStatus.COMPLETED
        or marker.result is None
    ):
        return None
    if marker.result.outcome == "INFEASIBLE":
        return None  # Historical builds need their original saved setup snapshot.
    result = marker.result.result
    if (
        result.target_budget_vnd != normalized.constraints.target_budget_vnd.value
        or result.use_case != normalized.constraints.use_case.value
    ):
        raise ValueError("Current build constraints mismatch")
    build = stage_recommendation(stage).build
    canonical = {part.id: part for part in normalized.candidates}
    if set(build.owned_categories) != set(state.owned_parts):
        raise ValueError("Completed build ownership mismatch")
    parts = []
    for slot, part in build.parts.items():
        if slot not in build.spending_prices:
            raise ValueError("Missing explicit component spending")
        synthetic_kind: Literal["INTEGRATED_GPU", "STOCK_COOLER"] | None = None
        variant_id = part.id
        if variant_id not in canonical:
            cpu = build.parts[ComponentCategory.CPU]
            if slot == ComponentCategory.GPU and cpu.has_integrated_graphics:
                synthetic_kind = "INTEGRATED_GPU"
                suffix = "igpu"
            elif slot == ComponentCategory.COOLER and cpu.includes_stock_cooler:
                synthetic_kind = "STOCK_COOLER"
                suffix = "stock_cooler"
            else:
                raise ValueError("Build component has no canonical source")
            expected_id = uuid5(
                SYNTHETIC_NAMESPACE, f"{stable_component_id(cpu)}:{suffix}"
            )
            if part.id != expected_id or part.price != 0:
                raise ValueError("Synthetic identity/price mismatch")
            variant_id = None
        elif canonical[variant_id] != part or part.category != slot:
            raise ValueError("Build component differs from its canonical snapshot")
        pin = state.pinned_parts.get(slot)
        owned = state.owned_parts.get(slot)
        if pin is not None and pin.variant_id != variant_id:
            raise ValueError("Completed build replaced a paid pin")
        if owned is not None and owned.variant_id != variant_id:
            raise ValueError("Completed build replaced an owned part")
        spending = build.spending_prices[slot]
        if spending != (0 if owned is not None else part.price):
            raise ValueError("Completed build spending mismatch")
        parts.append(
            PublicBuildPartV1(
                slot=slot,
                variant_id=variant_id,
                synthetic_kind=synthetic_kind,
                name=part.name,
                spending_vnd=spending,
                is_owned=owned is not None,
                is_pinned=pin is not None,
            )
        )
    accessory_spending = 0
    for row in prepared.accessories:
        request = state.accessories[row.accessory_type]
        spending = 0 if request.kind == "OWNED" else row.price_vnd
        accessory_spending += spending
        parts.append(
            PublicBuildPartV1(
                slot=row.accessory_type,
                variant_id=row.variant_id,
                name=row.name,
                spending_vnd=spending,
                is_owned=request.kind == "OWNED",
                is_pinned=request.kind == "PINNED",
            )
        )
    if accessory_spending != prepared.accessory_spending_vnd:
        raise ValueError("Resolved accessory spending mismatch")
    return PublicBuildV1(
        objective=build.objective,
        parts=parts,
        total_spending_vnd=build.total_price + accessory_spending,
        stale=state.build_stale,
    )
