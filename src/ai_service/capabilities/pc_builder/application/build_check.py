"""Compatibility of a build as the part picker shows it: one verdict for each pair of chosen slots."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from ai_service.application.ports.hardware import (
    CompatibilityIssue,
    ComponentCategory,
    ComponentSource,
    ComponentSpec,
    HardwareRuleEngine,
    WattageReport,
)

MAX_SLOTS = 16
MAX_CANDIDATES = 100
CANDIDATE_LOAD_CONCURRENCY = 16


class CheckStatus(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"
    INCOMPATIBLE = "INCOMPATIBLE"


# Worst first, so the verdict of a build is the first status any rule reached.
SEVERITY = (CheckStatus.INCOMPATIBLE, CheckStatus.WARNING, CheckStatus.UNKNOWN, CheckStatus.OK)

SLOT_CATEGORIES: dict[str, ComponentCategory] = {
    "CPU": ComponentCategory.CPU,
    "MAINBOARD": ComponentCategory.MAINBOARD,
    "RAM": ComponentCategory.RAM,
    "GPU": ComponentCategory.GPU,
    "PSU": ComponentCategory.PSU,
    "CASE": ComponentCategory.CASE,
    "COOLER": ComponentCategory.COOLER,
    "SSD": ComponentCategory.STORAGE,
}


@dataclass(frozen=True)
class PairRule:
    """One check between two slots (or one slot alone) and the attributes it cannot be answered without."""

    code: str
    slot_a: str
    slot_b: str | None
    needs: tuple[tuple[str, str], ...]
    # Categories the rule engine reports for this rule, and which of them mean "could not tell" rather than "no".
    engine_categories: frozenset[str]
    unknown_categories: frozenset[str] = frozenset()


RULES = (
    PairRule("SOCKET", "CPU", "MAINBOARD", (("CPU", "socket"), ("MAINBOARD", "socket")), frozenset({"SOCKET_MISMATCH"})),
    PairRule("RAM_TYPE", "MAINBOARD", "RAM", (("MAINBOARD", "ram_type"), ("RAM", "ram_type")), frozenset({"RAM_TYPE_MISMATCH"})),
    PairRule(
        "RAM_SLOTS", "MAINBOARD", "RAM", (("MAINBOARD", "ram_slots"), ("RAM", "kit_modules")),
        frozenset({"RAM_SLOTS_EXCEEDED"}),
    ),
    PairRule(
        "FORM_FACTOR", "MAINBOARD", "CASE", (("MAINBOARD", "form_factor"), ("CASE", "supported_form_factors")),
        frozenset({"FORM_FACTOR_MISMATCH", "FORM_FACTOR_UNKNOWN"}), frozenset({"FORM_FACTOR_UNKNOWN"}),
    ),
    PairRule("GPU_CLEARANCE", "GPU", "CASE", (("GPU", "gpu_length_mm"), ("CASE", "max_gpu_length_mm")), frozenset({"GPU_CLEARANCE"})),
    PairRule(
        "COOLER_CLEARANCE", "COOLER", "CASE", (("COOLER", "cooler_height_mm"), ("CASE", "max_cooler_height_mm")),
        frozenset({"COOLER_CLEARANCE"}),
    ),
    PairRule(
        "COOLER_SOCKET", "CPU", "COOLER", (("CPU", "socket"), ("COOLER", "supported_sockets")),
        frozenset({"COOLER_SOCKET_MISMATCH", "UNKNOWN_COOLER_SOCKET"}), frozenset({"UNKNOWN_COOLER_SOCKET"}),
    ),
    PairRule("PSU_WATTAGE", "PSU", None, (("PSU", "wattage"),), frozenset({"INSUFFICIENT_PSU"})),
    PairRule(
        "SSD_INTERFACE", "MAINBOARD", "SSD", (),
        frozenset({"STORAGE_INTERFACE_MISMATCH"}),
    ),
)


class CompatibilityCheckRequest(BaseModel):
    """The variant chosen for each slot; slots that are not hardware (monitor, mouse...) have nothing to check."""

    slots: dict[str, UUID] = Field(max_length=MAX_SLOTS)


class RuleResult(BaseModel):
    rule: str
    status: CheckStatus
    slot_a: str
    slot_b: str | None = None
    message: str | None = None
    suggestion: str | None = None
    missing_attributes: list[str] = Field(default_factory=list)


class BuildCheck(BaseModel):
    overall: CheckStatus
    results: list[RuleResult]
    # Estimated draw and the supply it calls for; None until the build has a CPU or a graphics card to size it by.
    power: WattageReport | None = None


class CandidateStatus(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class CandidateCheckRequest(BaseModel):
    """The parts already chosen, the slot being picked for, and the variants on offer for it."""

    slots: dict[str, UUID] = Field(max_length=MAX_SLOTS)
    slot: str
    candidates: list[UUID] = Field(max_length=MAX_CANDIDATES)


class CandidateVerdicts(BaseModel):
    verdicts: dict[UUID, CandidateStatus]


class ComponentNotFoundError(LookupError):
    def __init__(self, slot: str, variant_id: UUID) -> None:
        self.slot = slot
        self.variant_id = variant_id
        super().__init__(f"No {slot} part for variant {variant_id}")


def attribute_of(component: ComponentSpec, name: str) -> Any:
    return getattr(component, name, None) or component.extra_specs.get(name)


def check_build(engine: HardwareRuleEngine, parts: Mapping[str, ComponentSpec]) -> BuildCheck:
    """Runs the rule engine and says, for every pair of chosen slots, whether it is fine, wrong or could not be told."""

    report = engine.check_compatibility(list(parts.values()))
    issues: dict[str, CompatibilityIssue] = {}
    for found in report.issues:
        for rule in RULES:
            if found.category in rule.engine_categories:
                issues.setdefault(rule.code, found)

    results = []
    for rule in RULES:
        if rule.slot_a not in parts or (rule.slot_b is not None and rule.slot_b not in parts):
            continue
        if rule.code == "COOLER_CLEARANCE":
            cooler_part = parts["COOLER"]
            is_aio = attribute_of(cooler_part, "radiator_mm") is not None or attribute_of(cooler_part, "cooler_type") == "aio"
            if is_aio:
                needed_pairs = (("COOLER", "radiator_mm"), ("CASE", "max_radiator_mm"))
            else:
                needed_pairs = (("COOLER", "cooler_height_mm"), ("CASE", "max_cooler_height_mm"))
            missing = [f"{slot}.{name}" for slot, name in needed_pairs if attribute_of(parts[slot], name) is None]
        else:
            missing = [f"{slot}.{name}" for slot, name in rule.needs if not attribute_of(parts[slot], name)]
        issue = issues.get(rule.code)
        if issue is not None:
            status = (
                CheckStatus.UNKNOWN if issue.category in rule.unknown_categories
                else CheckStatus.WARNING if issue.level == "WARNING"
                else CheckStatus.INCOMPATIBLE
            )
            message, suggestion = issue.message, issue.suggestion
        elif missing:
            status, message, suggestion = CheckStatus.UNKNOWN, "Thiếu dữ liệu chưa thể xác nhận tương thích.", None
        else:
            status, message, suggestion = CheckStatus.OK, None, None
        results.append(RuleResult(
            rule=rule.code, status=status, slot_a=rule.slot_a, slot_b=rule.slot_b,
            message=message, suggestion=suggestion, missing_attributes=missing if status == CheckStatus.UNKNOWN else [],
        ))
    present = {result.status for result in results}
    return BuildCheck(
        overall=next((s for s in SEVERITY if s in present), CheckStatus.OK), results=results, power=power_of(engine, parts),
    )


def power_of(engine: HardwareRuleEngine, parts: Mapping[str, ComponentSpec]) -> WattageReport | None:
    if "CPU" not in parts and "GPU" not in parts:
        return None
    rated = attribute_of(parts["PSU"], "wattage") if "PSU" in parts else None
    return engine.calculate_wattage(list(parts.values()), selected_psu_watts=int(rated) if rated else None)


def verdict_of(results: Sequence[RuleResult], slot: str) -> CandidateStatus:
    """What the rules that involve one slot say about a part offered for it; no applicable rule yet is not a yes."""
    involved = {r.status for r in results if slot in (r.slot_a, r.slot_b)}
    if CheckStatus.INCOMPATIBLE in involved:
        return CandidateStatus.INCOMPATIBLE
    if CheckStatus.UNKNOWN in involved or not involved:
        return CandidateStatus.UNKNOWN
    return CandidateStatus.COMPATIBLE


class BuildCompatibilityService:
    def __init__(self, engine: HardwareRuleEngine, components: ComponentSource) -> None:
        self.engine = engine
        self.components = components

    async def check(self, request: CompatibilityCheckRequest) -> BuildCheck:
        return check_build(self.engine, await self._load(_hardware_slots(request.slots)))

    async def candidates(self, request: CandidateCheckRequest) -> CandidateVerdicts:
        """Checks each variant on offer for a slot against the parts chosen for the other slots."""
        slot = request.slot.upper()
        if slot not in SLOT_CATEGORIES:
            return CandidateVerdicts(verdicts=dict.fromkeys(request.candidates, CandidateStatus.UNKNOWN))
        selected = await self._load({s: v for s, v in _hardware_slots(request.slots).items() if s != slot})
        gate = asyncio.Semaphore(CANDIDATE_LOAD_CONCURRENCY)

        async def judge(variant_id: UUID) -> tuple[UUID, CandidateStatus]:
            async with gate:
                part = await self.components.component(variant_id, SLOT_CATEGORIES[slot])
            if part is None:
                return variant_id, CandidateStatus.UNKNOWN
            return variant_id, verdict_of(check_build(self.engine, {**selected, slot: part}).results, slot)

        return CandidateVerdicts(verdicts=dict(await asyncio.gather(*(judge(c) for c in request.candidates))))

    async def _load(self, chosen: Mapping[str, UUID]) -> dict[str, ComponentSpec]:
        loaded = await asyncio.gather(*(
            self.components.component(variant_id, SLOT_CATEGORIES[slot]) for slot, variant_id in chosen.items()
        ))
        parts: dict[str, ComponentSpec] = {}
        for (slot, variant_id), component in zip(chosen.items(), loaded, strict=True):
            if component is None:
                raise ComponentNotFoundError(slot, variant_id)
            parts[slot] = component
        return parts


def _hardware_slots(slots: Mapping[str, UUID]) -> dict[str, UUID]:
    return {slot.upper(): variant_id for slot, variant_id in slots.items() if slot.upper() in SLOT_CATEGORIES}
