"""Canonical accessory selection under an explicit spending cap, never a budget ratio."""

from itertools import product
from typing import Literal
from uuid import UUID

from pydantic import Field

from ai_service.application.ports.accessories import AccessorySpec, AccessoryType
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.stateful_contracts import BudgetScope, Contract

from .contracts import (
    CanonicalRequirementState,
    QuantityValue,
    ReferenceValue,
    Requirement,
    TextValue,
)

SETUP_SELECTION_POLICY: Literal["explicit-cap-match-price-v1"] = (
    "explicit-cap-match-price-v1"
)
type SetupStatus = Literal["APPLIED", "NEEDS_CLARIFICATION", "UNSUPPORTED", "CONFLICT"]


class SelectedAccessory(Contract):
    row: CatalogRowV1
    owned: bool = False
    pinned: bool = False

    @property
    def spending(self) -> int:
        return 0 if self.owned else self.row.component.price


class SetupPlan(Contract):
    scope: BudgetScope
    policy_version: Literal["explicit-cap-match-price-v1"] = SETUP_SELECTION_POLICY
    accessories: list[SelectedAccessory] = Field(default_factory=list, max_length=4)

    @property
    def spending(self) -> int:
        return sum(item.spending for item in self.accessories)


def accessory_match(rule: Requirement, part: AccessorySpec) -> bool | None:
    if rule.predicate in ("identity", "use_resource") and isinstance(
        rule.value, ReferenceValue
    ):
        return rule.value.variant_id == part.id
    if rule.predicate == "selection" and isinstance(rule.value, TextValue):
        return rule.value.text == "RECOMMEND"
    if rule.predicate == "brand" and isinstance(rule.value, TextValue):
        return (
            part.brand.casefold() == rule.value.text.casefold() if part.brand else None
        )
    return None


def compile_setup(
    state: CanonicalRequirementState, catalog: list[CatalogRowV1]
) -> tuple[
    SetupPlan | None, dict[UUID, tuple[SetupStatus, str]], list[tuple[SetupStatus, str]]
]:
    rules = [*state.goals.values(), *state.requirements.values()]
    special = [
        r
        for r in rules
        if r.subject in set(AccessoryType)
        or (r.subject == "SETUP" and r.predicate in ("scope", "accessory_budget"))
    ]
    if not special:
        return None, {}, []
    decisions: dict[UUID, tuple[SetupStatus, str]] = {}
    problems: list[tuple[SetupStatus, str]] = []
    scope = BudgetScope.BUILD_PC
    scopes = set()
    caps = []
    per_slot: dict[AccessoryType, list[Requirement]] = {}
    for rule in special:
        value = rule.value
        decisions[rule.id] = ("APPLIED", "SETUP_SELECTION")
        if rule.subject in set(AccessoryType):
            per_slot.setdefault(AccessoryType(rule.subject), []).append(rule)
            if not (
                rule.operator == "EQ"
                and (
                    (
                        rule.predicate in ("identity", "use_resource")
                        and isinstance(value, ReferenceValue)
                        and rule.strength == "HARD"
                    )
                    or (
                        rule.predicate == "selection"
                        and isinstance(value, TextValue)
                        and value.text == "RECOMMEND"
                        and rule.strength == "HARD"
                    )
                    or (rule.predicate == "brand" and isinstance(value, TextValue))
                )
            ):
                decisions[rule.id] = (
                    "UNSUPPORTED",
                    "ACCESSORY_PREDICATE_NOT_SUPPORTED",
                )
        elif (
            rule.predicate == "scope"
            and rule.operator == "EQ"
            and rule.strength == "HARD"
            and isinstance(value, TextValue)
            and value.text in set(BudgetScope)
        ):
            scopes.add(BudgetScope(value.text))
        elif (
            rule.predicate == "accessory_budget"
            and rule.operator == "LTE"
            and rule.strength == "HARD"
            and isinstance(value, QuantityValue)
            and value.unit == "VND"
        ):
            caps.append(value.amount)
        else:
            decisions[rule.id] = ("UNSUPPORTED", "SETUP_PREDICATE_NOT_SUPPORTED")
    if len(scopes) > 1:
        problems.append(("CONFLICT", "CONFLICTING_BUDGET_SCOPE"))
    elif scopes:
        scope = next(iter(scopes))
    if per_slot and scope != BudgetScope.FULL_SETUP:
        problems.append(("NEEDS_CLARIFICATION", "ACCESSORY_SCOPE_CONFIRMATION"))
    if scope == BudgetScope.FULL_SETUP and not per_slot:
        problems.append(("NEEDS_CLARIFICATION", "ACCESSORY_REQUIREMENTS_REQUIRED"))
    pools: list[list[SelectedAccessory]] = []
    for slot, slot_rules in sorted(per_slot.items()):
        if any(decisions[r.id][0] != "APPLIED" for r in slot_rules):
            continue
        refs = {
            r.value.variant_id
            for r in slot_rules
            if isinstance(r.value, ReferenceValue)
        }
        owned = any(r.predicate == "use_resource" for r in slot_rules)
        pinned = any(r.predicate == "identity" for r in slot_rules)
        if len(refs) > 1 or (owned and pinned):
            problems.append(("CONFLICT", "ACCESSORY_SLOT_BINDING_CONFLICT"))
            continue
        if owned and not refs <= {
            resource.variant_id for resource in state.available_resources.values()
        }:
            problems.append(("CONFLICT", "RESOURCE_NOT_AVAILABLE"))
            continue
        if not refs and not caps:
            problems.append(("NEEDS_CLARIFICATION", "ACCESSORY_BUDGET_REQUIRED"))
            continue
        candidates = [
            SelectedAccessory(row=row, owned=owned, pinned=pinned)
            for row in catalog
            if isinstance(row.component, AccessorySpec)
            and row.component.category == slot
            and (owned or row.quantity > 0)
            and all(
                accessory_match(r, row.component) is True
                for r in slot_rules
                if r.strength == "HARD"
            )
        ]
        if not candidates:
            problems.append(
                ("NEEDS_CLARIFICATION", "ACCESSORY_UNAVAILABLE_OR_METADATA_MISSING")
            )
            continue
        pools.append(candidates)
    plan = SetupPlan(scope=scope)
    if problems or any(status != "APPLIED" for status, _ in decisions.values()):
        return plan, decisions, problems
    # Bounded canonical catalog (100 rows, four slots). No inferred quality score.
    best_key: tuple[int, int, tuple[str, ...]] | None = None
    for combination in product(*pools):
        spending = sum(item.spending for item in combination)
        if caps and spending > min(caps):
            continue
        matches = sum(
            accessory_match(r, item.row.component) is True
            for item in combination
            if isinstance(item.row.component, AccessorySpec)
            for r in per_slot[item.row.component.category]
            if r.strength == "SOFT"
        )
        key = (
            -matches,
            spending,
            tuple(str(item.row.variant_id) for item in combination),
        )
        if best_key is None or key < best_key:
            best_key = key
            plan.accessories = list(combination)
    if best_key is None:
        problems.append(("NEEDS_CLARIFICATION", "ACCESSORY_BUDGET_EXCEEDED"))
    return plan, decisions, problems
