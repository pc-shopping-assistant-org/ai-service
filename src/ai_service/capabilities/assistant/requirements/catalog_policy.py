"""Determine when requirement processing actually needs canonical candidates."""

from ai_service.capabilities.assistant.provenance import CatalogRowV1

from .compiler import compile_requirements
from .contracts import CanonicalRequirementState
from .resolver import RequirementDraft

CATALOG_DEPENDENT_REASONS = frozenset(
    {
        "VARIANT_NOT_RESOLVED",
        "METADATA_INSUFFICIENT",
        "ACCESSORY_UNAVAILABLE_OR_METADATA_MISSING",
    }
)
REFERENCE_PREDICATES = frozenset({"identity", "use_resource"})


def build_needs_catalog(state: CanonicalRequirementState) -> bool:
    """Wait for catalog-independent goals/semantics before acquiring candidates."""
    preflight = compile_requirements(state, [])
    return all(
        item.status == "APPLIED"
        or (
            item.status == "NEEDS_CLARIFICATION"
            and item.reason in CATALOG_DEPENDENT_REASONS
        )
        for item in preflight.mapping
    )


def draft_needs_catalog(draft: RequirementDraft) -> bool:
    """Scalar/unit changes and CLEAR do not require SKU resolution."""
    checked = type(draft).model_validate(draft.model_dump())
    return any(
        item.operation == "SET"
        and (item.target == "RESOURCE" or item.predicate in REFERENCE_PREDICATES)
        for item in checked.items
    )


def build_clarification_reasons(
    state: CanonicalRequirementState,
    catalog: list[CatalogRowV1],
    *,
    catalog_ready: bool,
) -> set[str]:
    """Unacquired candidates cannot prove missing SKU/spec metadata."""
    return {
        item.reason
        for item in compile_requirements(state, catalog).mapping
        if item.status == "NEEDS_CLARIFICATION"
        and (catalog_ready or item.reason not in CATALOG_DEPENDENT_REASONS)
    }
