from uuid import uuid4

import pytest
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.requirements.build import optimize_requirements
from ai_service.capabilities.assistant.requirements.search_report import search_report
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer


def test_exact_empty_affordable_scope_can_prove_scoped_infeasibility(realistic_catalog):
    canonical = state()
    next(iter(canonical.requirements.values())).value.amount = 1
    catalog = rows(realistic_catalog)
    service = PCBuildApplicationService(DeterministicPCOptimizer(enable_pruning=False))
    receipt = optimize_requirements(canonical, catalog, service)
    report = search_report(canonical, catalog, service, receipt, build_ref=uuid4())
    assert report.outcome == "INFEASIBLE" and report.build_ref is None
    assert report.search_complete and report.coverage_complete and report.pruning_sound
    assert report.scope == "SUPPLIED_NORMALIZED_CANDIDATES"


def test_missing_scoring_data_is_search_incomplete_not_physical_infeasibility(
    realistic_catalog,
):
    catalog = rows(
        [
            part.model_copy(update={"performance_score": None})
            if part.category == "CPU"
            else part
            for part in realistic_catalog
        ]
    )
    canonical, service = (
        state(),
        PCBuildApplicationService(DeterministicPCOptimizer(enable_pruning=False)),
    )
    receipt = optimize_requirements(canonical, catalog, service)
    report = search_report(canonical, catalog, service, receipt, build_ref=uuid4())
    assert report.outcome == "SEARCH_INCOMPLETE"
    assert receipt.search_trace.scoring_rejections > 0
    assert not report.coverage_complete


def test_found_build_is_feasible_and_report_binds_actual_input_catalog_policy(
    realistic_catalog,
):
    canonical, catalog, service = (
        state(),
        rows(realistic_catalog),
        PCBuildApplicationService(),
    )
    receipt = optimize_requirements(canonical, catalog, service)
    identifier = uuid4()
    report = search_report(canonical, catalog, service, receipt, build_ref=identifier)
    assert report.outcome == "FEASIBLE" and report.build_ref == identifier
    assert report.termination == "FOUND_RESULT"
    assert receipt.search_trace.enumeration_complete
    assert len(report.optimizer_input_hash) == len(report.catalog_snapshot_hash) == 64
    # A legacy receipt without actual execution trace is not auto-certified.
    assert (
        search_report(
            canonical,
            catalog,
            service,
            receipt.model_copy(update={"search_trace": None}),
            build_ref=identifier,
        )
        is None
    )


def test_changed_policy_or_input_cannot_relabel_a_prior_execution(realistic_catalog):
    canonical, catalog, service = (
        state(),
        rows(realistic_catalog),
        PCBuildApplicationService(),
    )
    receipt = optimize_requirements(canonical, catalog, service)
    changed = canonical.model_copy(deep=True)
    next(iter(changed.requirements.values())).value.amount += 1
    with pytest.raises(ValueError, match="normalized input"):
        search_report(changed, catalog, service, receipt, build_ref=uuid4())
    service._optimizer.enable_pruning = False
    with pytest.raises(ValueError, match="policy changed"):
        search_report(canonical, catalog, service, receipt, build_ref=uuid4())


@pytest.mark.asyncio
async def test_dispatcher_captures_actual_typed_search_evidence(realistic_catalog):
    from test_react_tools import Catalog, context

    from ai_service.capabilities.assistant.react.contracts import ToolCall
    from ai_service.capabilities.assistant.react.tools import ToolDispatcher

    trusted = context(state(), rows(realistic_catalog))
    result = await ToolDispatcher(Catalog(), PCBuildApplicationService()).execute(
        ToolCall(call_id="build", name="build_pc", arguments={}), trusted
    )
    assert result.result.search_evidence.outcome == "FEASIBLE"
    assert result.result.search_evidence.build_ref == result.result.build_ref


def test_actual_pruning_prevents_certifying_absence_without_losslessness_proof(
    realistic_catalog,
):
    ram = next(part for part in realistic_catalog if part.category == "RAM")
    modified = [
        part.model_copy(update={"socket": "NO_MATCH"})
        if part.category == "MAINBOARD"
        else part
        for part in realistic_catalog
    ]
    modified.append(
        ram.model_copy(
            update={"name": "More expensive identical RAM", "price": ram.price + 1}
        )
    )
    canonical, catalog, service = state(), rows(modified), PCBuildApplicationService()
    receipt = optimize_requirements(canonical, catalog, service)
    assert receipt.search_trace.pruned_candidates > 0
    report = search_report(canonical, catalog, service, receipt, build_ref=uuid4())
    assert report.outcome == "SEARCH_INCOMPLETE" and not report.pruning_sound
