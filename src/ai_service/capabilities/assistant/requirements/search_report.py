"""Derive scoped search outcomes from optimizer observations and canonical inputs."""

from uuid import UUID

from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.react.search_evidence import BuildSearchEvidence
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

from .build import BuildReceipt, verify_receipt
from .compiler import compile_requirements
from .contracts import CanonicalRequirementState, requirements_hash


def search_report(
    state: CanonicalRequirementState,
    catalog: list[CatalogRowV1],
    service: PCBuildApplicationService,
    receipt: BuildReceipt,
    *,
    build_ref: UUID,
) -> BuildSearchEvidence | None:
    receipt = BuildReceipt.model_validate(receipt.model_dump())
    trace = receipt.search_trace
    if trace is None or receipt.search_policy_hash is None:
        return None  # Legacy checkpoints have no observed execution evidence.
    compiled = compile_requirements(state, catalog)
    if (
        not compiled.ready
        or compiled.optimizer_input is None
        or compiled != receipt.compilation
    ):
        raise ValueError("Search result does not match current normalized input")
    if receipt.search_policy_hash != canonical_hash(service.configuration_snapshot()):
        raise ValueError("Search policy changed since execution")
    if receipt.outcome not in {"BUILT", "INFEASIBLE"}:
        raise ValueError("No optimizer execution outcome")
    pool_ids = [
        identifier
        for identifiers in trace.pool_variant_ids.values()
        for identifier in identifiers
    ]
    supplied_ids = {part.id for part in compiled.optimizer_input.candidates}
    if any(
        identifier is not None and identifier not in supplied_ids
        for identifier in pool_ids
    ):
        raise ValueError("Search trace contains candidates outside normalized input")
    coverage = (
        trace.metadata_rejections == 0
        and trace.scoring_rejections == 0
        and all(identifier is not None for identifier in pool_ids)
    )
    # No removals is trivially lossless. Do not assert the Pareto implementation
    # preserves all feasible configurations merely because it is called pruning.
    pruning_sound = trace.pruned_candidates == 0
    found = receipt.outcome == "BUILT"
    if found:
        verify_receipt(state, receipt, catalog)
    # Accessory selection is a staged policy, not exhaustive joint core/setup search.
    # A failed chosen allocation must not prove all other accessory selections infeasible.
    unbound_accessories = compiled.setup is not None and any(
        not item.owned and not item.pinned for item in compiled.setup.accessories
    )
    proven = trace.enumeration_complete and coverage and pruning_sound and not unbound_accessories
    return BuildSearchEvidence(
        outcome="FEASIBLE"
        if found
        else "INFEASIBLE"
        if proven
        else "SEARCH_INCOMPLETE",
        scope="SUPPLIED_NORMALIZED_CANDIDATES",
        requirements_hash=requirements_hash(state),
        optimizer_input_hash=canonical_hash(compiled.optimizer_input),
        catalog_snapshot_hash=canonical_hash(
            [row.model_dump(mode="json") for row in catalog]
        ),
        algorithm_version=trace.algorithm_version,
        pruning_version="compatibility-aware-pareto-v2",
        policy_version=receipt.search_policy_hash,
        search_complete=trace.enumeration_complete,
        coverage_complete=coverage and not unbound_accessories,
        pruning_sound=pruning_sound,
        termination="FOUND_RESULT"
        if found
        else "EXHAUSTED"
        if trace.enumeration_complete
        else "HEURISTIC_STOP",
        build_ref=build_ref if found else None,
    )
