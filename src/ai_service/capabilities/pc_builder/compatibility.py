"""Explicit case support shared by checking, enumeration and pruning."""

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.schemas import CompatibilityStatus


def case_supported_form_factors(case: ComponentSpec) -> frozenset[str] | None:
    supported = frozenset(normalize_form_factor(value) for value in case.supported_form_factors)
    return supported if supported and "" not in supported else None


def normalize_form_factor(value: str | None) -> str:
    return (value or "").strip().upper().replace(" ", "").replace("-", "")


def check_case_form_factor(mainboard: ComponentSpec, case: ComponentSpec) -> CompatibilityStatus:
    form_factor = normalize_form_factor(mainboard.form_factor)
    supported = case_supported_form_factors(case)
    if not form_factor or supported is None:
        return CompatibilityStatus.UNKNOWN
    if form_factor not in supported:
        return CompatibilityStatus.INCOMPATIBLE
    return CompatibilityStatus.COMPATIBLE
