"""Finite compiler to the existing optimizer. Unsupported semantics stay visible."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.assistant.provenance import (
    CatalogRowV1,
    OptimizerInputV1,
    canonical_hash,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    ConstraintValue,
    OwnedComponent,
    PCBuildConstraints,
    UseCaseProfile,
)

from .contracts import (
    CanonicalRequirementState,
    QuantityValue,
    ReferenceValue,
    Requirement,
    TextValue,
)

COMPILER_VERSION: Literal["requirements-v1"] = "requirements-v1"


class Mapping(Contract):
    requirement_id: UUID | None
    status: Literal["APPLIED", "NEEDS_CLARIFICATION", "UNSUPPORTED", "CONFLICT"]
    reason: Text
    strength: Literal["HARD", "SOFT"] | None = None
    subject: str | None = None
    predicate: str | None = None
    target_bindings: list[UUID] = Field(default_factory=list)


class Compilation(Contract):
    compiler_version: Literal["requirements-v1"] = COMPILER_VERSION
    state_hash: Text
    mapping: list[Mapping]
    optimizer_input: OptimizerInputV1 | None = None

    @property
    def ready(self) -> bool:
        return self.optimizer_input is not None and all(
            m.status == "APPLIED" for m in self.mapping
        )


def predicate_value(requirement: Requirement, part: ComponentSpec) -> bool | None:
    """Known match/known mismatch/unknown; never infer specs from a product name."""
    value = requirement.value
    if requirement.predicate == "identity" and isinstance(value, ReferenceValue):
        return part.id == value.variant_id
    if requirement.predicate == "capacity" and isinstance(value, QuantityValue):
        if value.unit not in ("GB", "TB") or part.capacity_gb is None:
            return None
        requested = value.amount * (1000 if value.unit == "TB" else 1)
        return {
            "EQ": part.capacity_gb == requested,
            "GTE": part.capacity_gb >= requested,
            "LTE": part.capacity_gb <= requested,
        }[requirement.operator]
    if requirement.predicate in ("brand", "media") and isinstance(value, TextValue):
        actual = (
            part.brand
            if requirement.predicate == "brand"
            else part.extra_specs.get("storage_media")
        )
        return (
            actual.casefold() == value.text.casefold()
            if isinstance(actual, str)
            else None
        )
    return None


def compile_requirements(
    state: CanonicalRequirementState, catalog: list[CatalogRowV1]
) -> Compilation:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    catalog = [CatalogRowV1.model_validate(row.model_dump()) for row in catalog]
    by_id = {row.variant_id: row for row in catalog}
    if len(by_id) != len(catalog):
        raise ValueError("Duplicate canonical catalog identity")
    mapping: list[Mapping] = []
    budget: int | None = None
    use_case: UseCaseProfile | None = None
    pinned: dict[ComponentCategory, UUID] = {}
    owned: dict[ComponentCategory, UUID] = {}
    predicates: dict[ComponentCategory, list[Requirement]] = {}

    def outcome(
        r: Requirement,
        status: Literal["APPLIED", "NEEDS_CLARIFICATION", "UNSUPPORTED", "CONFLICT"],
        reason: str,
    ) -> None:
        mapping.append(
            Mapping(
                requirement_id=r.id,
                status=status,
                reason=reason,
                strength=r.strength,
                subject=r.subject,
                predicate=r.predicate,
                target_bindings=[r.value.variant_id]
                if isinstance(r.value, ReferenceValue)
                else [],
            )
        )

    for r in sorted(
        [*state.goals.values(), *state.requirements.values()], key=lambda r: str(r.id)
    ):
        v = r.value
        if r.strength == "SOFT":
            outcome(r, "UNSUPPORTED", "SOFT_SCORER_NOT_IMPLEMENTED")
        elif (
            r.subject == "SETUP"
            and r.predicate == "budget"
            and r.operator == "LTE"
            and isinstance(v, QuantityValue)
            and v.unit == "VND"
        ):
            budget = min(budget, v.amount) if budget is not None else v.amount
            outcome(r, "APPLIED", "BUDGET_CAP")
        elif (
            r.subject == "WORKLOAD"
            and r.predicate == "use_case"
            and r.operator == "EQ"
            and isinstance(v, TextValue)
        ):
            try:
                requested = UseCaseProfile(v.text)
            except ValueError:
                outcome(r, "UNSUPPORTED", "UNKNOWN_WORKLOAD")
                continue
            if use_case is not None and use_case != requested:
                outcome(r, "CONFLICT", "CONFLICTING_WORKLOAD")
            else:
                use_case = requested
                outcome(r, "APPLIED", "WORKLOAD_PROFILE")
        elif r.subject in set(ComponentCategory):
            category = ComponentCategory(r.subject)
            if (
                r.predicate in ("identity", "use_resource")
                and r.operator == "EQ"
                and isinstance(v, ReferenceValue)
            ):
                row = by_id.get(v.variant_id)
                is_owned = r.predicate == "use_resource"
                if row is None:
                    outcome(r, "NEEDS_CLARIFICATION", "VARIANT_NOT_RESOLVED")
                elif row.component.category != category:
                    outcome(r, "CONFLICT", "CATEGORY_MISMATCH")
                elif is_owned and not any(
                    resource.variant_id == v.variant_id
                    for resource in state.available_resources.values()
                ):
                    outcome(r, "CONFLICT", "RESOURCE_NOT_AVAILABLE")
                elif not is_owned and row.quantity == 0:
                    outcome(r, "NEEDS_CLARIFICATION", "PIN_UNAVAILABLE")
                elif category in pinned or category in owned:
                    prior = owned.get(category) if is_owned else pinned.get(category)
                    if prior == v.variant_id:
                        outcome(r, "APPLIED", "IDENTITY_ALREADY_BOUND")
                    else:
                        outcome(r, "CONFLICT", "SLOT_BINDING_CONFLICT")
                else:
                    (owned if is_owned else pinned)[category] = v.variant_id
                    outcome(
                        r, "APPLIED", "OWNED_BINDING" if is_owned else "PINNED_IDENTITY"
                    )
            elif (
                (
                    r.predicate == "capacity"
                    and category in (ComponentCategory.RAM, ComponentCategory.STORAGE)
                    and isinstance(v, QuantityValue)
                    and v.unit in ("GB", "TB")
                )
                or (
                    r.predicate == "brand"
                    and r.operator == "EQ"
                    and isinstance(v, TextValue)
                )
                or (
                    r.predicate == "media"
                    and category == ComponentCategory.STORAGE
                    and r.operator == "EQ"
                    and isinstance(v, TextValue)
                )
            ):
                predicates.setdefault(category, []).append(r)
                outcome(r, "APPLIED", "CANDIDATE_PREDICATE")
            else:
                outcome(r, "UNSUPPORTED", "PREDICATE_NOT_SUPPORTED")
        else:
            outcome(r, "UNSUPPORTED", "SUBJECT_PREDICATE_NOT_SUPPORTED")

    for identifier, binding in state.resolved_bindings.items():
        requirement = state.requirements[identifier]
        if (
            not isinstance(requirement.value, ReferenceValue)
            or requirement.value.variant_id != binding.variant_id
        ):
            mapping.append(
                Mapping(
                    requirement_id=identifier,
                    status="CONFLICT",
                    reason="BINDING_MISMATCH",
                )
            )
    if budget is None:
        mapping.append(
            Mapping(
                requirement_id=None,
                status="NEEDS_CLARIFICATION",
                reason="BUDGET_REQUIRED",
            )
        )
    if use_case is None:
        mapping.append(
            Mapping(
                requirement_id=None,
                status="NEEDS_CLARIFICATION",
                reason="USE_CASE_REQUIRED",
            )
        )
    if state.unresolved:
        mapping.append(
            Mapping(
                requirement_id=None,
                status="NEEDS_CLARIFICATION",
                reason="UNRESOLVED_REQUIREMENTS",
            )
        )

    usable = [
        row for row in catalog if row.quantity > 0 or row.variant_id in owned.values()
    ]
    for category, rules in predicates.items():
        candidates = [row for row in usable if row.component.category == category]
        for rule in rules:
            if not candidates or all(
                predicate_value(rule, row.component) is None for row in candidates
            ):
                mapping = [
                    m.model_copy(
                        update={
                            "status": "NEEDS_CLARIFICATION",
                            "reason": "METADATA_INSUFFICIENT",
                        }
                    )
                    if m.requirement_id == rule.id
                    else m
                    for m in mapping
                ]
            bound_id = pinned.get(category) or owned.get(category)
            if (
                bound_id is not None
                and predicate_value(rule, by_id[bound_id].component) is not True
            ):
                mapping = [
                    m.model_copy(
                        update={
                            "status": "CONFLICT",
                            "reason": "BOUND_PART_VIOLATES_REQUIREMENT",
                        }
                    )
                    if m.requirement_id == rule.id
                    else m
                    for m in mapping
                ]
        usable = [
            row
            for row in usable
            if row.component.category != category
            or all(predicate_value(rule, row.component) is True for rule in rules)
        ]
    result = Compilation(
        state_hash=canonical_hash(state.model_dump(mode="json")), mapping=mapping
    )
    if any(m.status != "APPLIED" for m in mapping):
        return result
    assert budget is not None and use_case is not None
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](
            value=budget, source=ConstraintSource.USER
        ),
        use_case=ConstraintValue[UseCaseProfile](
            value=use_case, source=ConstraintSource.USER
        ),
        pinned_parts=pinned,
        owned_parts=[
            OwnedComponent(category=category, component=by_id[identifier].component)
            for category, identifier in owned.items()
        ],
    )
    result.optimizer_input = OptimizerInputV1(
        constraints=constraints,
        candidates=[row.component for row in usable],
        pinned_parts=pinned,
        effective_spending_vnd={
            row.variant_id: 0
            if row.variant_id in owned.values()
            else row.component.price
            for row in usable
        },
    )
    return result
