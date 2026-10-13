"""P3 canonical preparation; no DB, provider calls or invented catalog facts.

Inputs are resolved backend snapshots, never client-supplied prices/specs.
Unresolved accessories return routing reasons instead of a guessed allocation.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from ai_service.capabilities.assistant.conversation_core import (
    ConsultationStateV1,
    SetupBudgetV1,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1, OptimizerInputV1
from ai_service.capabilities.assistant.stateful_contracts import (
    AccessoryType,
    BudgetScope,
    Contract,
    NonNegativeInt,
    Text,
)
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintValue,
    OwnedComponent,
    PCBuildConstraints,
)


class ClarificationReason(StrEnum):
    BUDGET_REQUIRED = "BUDGET_REQUIRED"
    USE_CASE_REQUIRED = "USE_CASE_REQUIRED"
    BUDGET_SCOPE_REQUIRED = "BUDGET_SCOPE_REQUIRED"
    ACCESSORY_SCOPE_CONFIRMATION = "ACCESSORY_SCOPE_CONFIRMATION"
    ACCESSORY_REQUIREMENTS_REQUIRED = "ACCESSORY_REQUIREMENTS_REQUIRED"
    ACCESSORY_BUDGET_REQUIRED = "ACCESSORY_BUDGET_REQUIRED"
    ACCESSORY_CATALOG_REQUIRED = "ACCESSORY_CATALOG_REQUIRED"
    ACCESSORY_UNAVAILABLE = "ACCESSORY_UNAVAILABLE"
    ACCESSORY_BUDGET_EXCEEDED = "ACCESSORY_BUDGET_EXCEEDED"
    PART_REFERENCE_REQUIRED = "PART_REFERENCE_REQUIRED"
    PART_CATEGORY_MISMATCH = "PART_CATEGORY_MISMATCH"
    PIN_UNAVAILABLE = "PIN_UNAVAILABLE"


class AccessoryCatalogRowV1(Contract):
    """Canonical selected accessory, supplied by the grounded catalog resolver."""

    variant_id: UUID
    accessory_type: AccessoryType
    sku: Text
    name: Text
    price_vnd: NonNegativeInt
    quantity: NonNegativeInt
    product_active: bool
    variant_active: bool
    retrieved_at: AwareDatetime
    evidence_refs: list[Text] = Field(min_length=1)


class PreparedBuildV1(Contract):
    clarifications: list[ClarificationReason] = Field(default_factory=list)
    optimizer_input: OptimizerInputV1 | None = None
    accessories: list[AccessoryCatalogRowV1] = Field(default_factory=list)
    accessory_spending_vnd: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_routing(self) -> PreparedBuildV1:
        if bool(self.clarifications) == (self.optimizer_input is not None):
            raise ValueError("Prepare either clarifies or supplies optimizer input")
        return self


def prepare_build_input(
    state: ConsultationStateV1,
    catalog: list[CatalogRowV1],
    selected_accessories: list[AccessoryCatalogRowV1] | None = None,
) -> PreparedBuildV1:
    state = ConsultationStateV1.model_validate(state.model_dump())
    catalog = [CatalogRowV1.model_validate(row.model_dump()) for row in catalog]
    selected = [
        AccessoryCatalogRowV1.model_validate(row.model_dump())
        for row in selected_accessories or []
    ]
    by_id = {row.variant_id: row for row in catalog}
    by_type = {row.accessory_type: row for row in selected}
    if len(by_id) != len(catalog) or len(by_type) != len(selected):
        raise ValueError(
            "Canonical catalog must not contain duplicate identities/slots"
        )
    if by_type.keys() - state.accessories.keys():
        raise ValueError("Resolver cannot add unrequested accessories")
    reasons: list[ClarificationReason] = []
    requirements = state.constraints
    budget = requirements.target_budget_vnd.value
    scope = requirements.budget_scope.value
    if budget is None:
        reasons.append(ClarificationReason.BUDGET_REQUIRED)
    if requirements.use_case.value is None:
        reasons.append(ClarificationReason.USE_CASE_REQUIRED)
    if scope is None:
        reasons.append(ClarificationReason.BUDGET_SCOPE_REQUIRED)
    if scope == BudgetScope.BUILD_PC and state.accessories:
        reasons.append(ClarificationReason.ACCESSORY_SCOPE_CONFIRMATION)
    if scope == BudgetScope.FULL_SETUP and not state.accessories:
        reasons.append(ClarificationReason.ACCESSORY_REQUIREMENTS_REQUIRED)

    spending = 0
    for kind, request in state.accessories.items():
        row = by_type.get(kind)
        if (
            request.kind == "RECOMMEND"
            and requirements.accessory_budget_vnd.value is None
        ):
            reasons.append(ClarificationReason.ACCESSORY_BUDGET_REQUIRED)
        if row is None:
            reasons.append(ClarificationReason.ACCESSORY_CATALOG_REQUIRED)
            continue
        if request.variant_id is not None and row.variant_id != request.variant_id:
            raise ValueError("Resolved accessory must preserve the bound variant")
        if (
            not row.product_active
            or not row.variant_active
            or (request.kind != "OWNED" and row.quantity == 0)
        ):
            reasons.append(ClarificationReason.ACCESSORY_UNAVAILABLE)
        if request.kind != "OWNED":
            spending += row.price_vnd
    cap = requirements.accessory_budget_vnd.value
    if (budget is not None and spending > budget) or (
        cap is not None and spending > cap
    ):
        reasons.append(ClarificationReason.ACCESSORY_BUDGET_EXCEEDED)

    owned = []
    for bindings, is_owned in ((state.owned_parts, True), (state.pinned_parts, False)):
        for category, ref in bindings.items():
            core_row = by_id.get(ref.variant_id)
            if core_row is None:
                reasons.append(ClarificationReason.PART_REFERENCE_REQUIRED)
            elif core_row.component.category != category:
                reasons.append(ClarificationReason.PART_CATEGORY_MISMATCH)
            elif not is_owned and core_row.quantity == 0:
                reasons.append(ClarificationReason.PIN_UNAVAILABLE)
            elif is_owned:
                owned.append(
                    OwnedComponent(category=category, component=core_row.component)
                )
    if reasons:
        return PreparedBuildV1(clarifications=list(dict.fromkeys(reasons)))

    assert budget is not None and requirements.use_case.value is not None
    core_budget = SetupBudgetV1(
        total_budget_vnd=budget, accessory_spending_vnd=spending
    ).core_budget_vnd
    payload = requirements.model_dump(exclude={"budget_scope", "accessory_budget_vnd"})
    payload["target_budget_vnd"] = ConstraintValue[int](
        **(requirements.target_budget_vnd.model_dump() | {"value": core_budget})
    ).model_dump()
    constraints = PCBuildConstraints.model_validate(
        payload
        | {
            "owned_parts": owned,
            "pinned_parts": {
                category: ref.variant_id for category, ref in state.pinned_parts.items()
            },
        }
    )
    owned_ids = {ref.variant_id for ref in state.owned_parts.values()}
    available = sorted(
        (row for row in catalog if row.quantity > 0 or row.variant_id in owned_ids),
        key=lambda row: (row.component.category, str(row.variant_id)),
    )
    normalized = OptimizerInputV1(
        constraints=constraints,
        candidates=[row.component for row in available],
        pinned_parts=constraints.pinned_parts,
        effective_spending_vnd={
            row.variant_id: 0 if row.variant_id in owned_ids else row.component.price
            for row in available
        },
    )
    return PreparedBuildV1(
        optimizer_input=normalized,
        accessories=selected,
        accessory_spending_vnd=spending,
    )
