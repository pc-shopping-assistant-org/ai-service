from copy import deepcopy

import pytest
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.requirements.build import (
    grounded_summary,
    optimize_requirements,
    verify_receipt,
)
from ai_service.capabilities.assistant.requirements.publication import public_projection
from ai_service.capabilities.pc_builder.application.explanation_context import (
    build_explanation_context,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.errors import NoFeasibleBuildError
from ai_service.capabilities.pc_builder.schemas import PCBuildConstraints


def without_scores(catalog):
    parts = deepcopy(catalog)
    for part in parts:
        part.performance_score = None
        part.psu_tier = None
        part.has_integrated_graphics = False
        part.includes_stock_cooler = False
    return parts


def test_missing_scores_keep_verified_build_without_fake_score(realistic_catalog):
    parts = without_scores(realistic_catalog)
    service = PCBuildApplicationService()
    result = service.build_pc(PCBuildConstraints(), parts)
    assert result.optimization.builds == {}
    build = result.recommendation.build
    assert build.ranking_status == "UNRANKED"
    assert build.objective is None and build.objective_score is None
    assert build.total_price <= 20_000_000
    assert build.compatibility_status == "COMPATIBLE"
    assert not any(e.metric == "objective_score" for e in build.evidence)
    explanation = build_explanation_context(result)
    assert explanation["ranking_status"] == "UNRANKED"
    assert explanation["objective_score"] is None
    reverse = service.build_pc(PCBuildConstraints(), list(reversed(parts)))
    assert reverse.recommendation.build == build


def test_unranked_result_passes_verifier_and_publication(realistic_catalog):
    catalog = rows(without_scores(realistic_catalog))
    canonical = state()
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    report = verify_receipt(canonical, receipt, catalog)
    assert report.hard_satisfied
    _, published = public_projection(canonical, receipt, None)
    assert published.objective is None
    assert published.ranking_status == "UNRANKED"
    assert published.model_dump(mode="json")["ranking_status"] == "UNRANKED"
    assert (
        "ranking_status"
        not in published.model_json_schema(mode="serialization")["required"]
    )
    assert "chưa đủ dữ liệu" in grounded_summary(receipt, report).lower()


def test_integrated_parts_do_not_need_benchmarks_for_physical_feasibility(
    realistic_catalog,
):
    parts = deepcopy(realistic_catalog)
    for part in parts:
        part.performance_score = None
        part.psu_tier = None
        part.integrated_graphics_score = None
        part.stock_cooler_score = None
    catalog = rows([part for part in parts if part.category not in {"GPU", "COOLER"}])
    canonical = state()
    receipt = optimize_requirements(canonical, catalog, PCBuildApplicationService())
    assert receipt.outcome == "BUILT"
    assert receipt.build.ranking_status == "UNRANKED"
    assert receipt.build.parts["GPU"].is_integrated
    assert (
        receipt.build.spending_prices["GPU"]
        == receipt.build.spending_prices["COOLER"]
        == 0
    )
    assert verify_receipt(canonical, receipt, catalog).hard_satisfied


def test_unranked_preserves_owned_spending_and_paid_pin(realistic_catalog):
    from ai_service.capabilities.pc_builder.schemas import OwnedComponent

    catalog = rows(without_scores(realistic_catalog))
    gpu = next(row.component for row in catalog if row.component.category == "GPU")
    ram = next(row.component for row in catalog if row.component.category == "RAM")
    constraints = PCBuildConstraints(
        owned_parts=[OwnedComponent(category="RAM", component=ram)],
        pinned_parts={"GPU": gpu.id},
    )
    build = (
        PCBuildApplicationService()
        .build_pc(constraints, [row.component for row in catalog])
        .recommendation.build
    )
    assert build.ranking_status == "UNRANKED"
    assert build.parts["GPU"].id == gpu.id and build.parts["RAM"].id == ram.id
    assert build.spending_prices["RAM"] == 0
    assert build.spending_prices["GPU"] == gpu.price


@pytest.mark.parametrize("failure", ["power", "budget", "socket"])
def test_missing_scores_do_not_bypass_hard_checks(realistic_catalog, failure):
    parts = without_scores(realistic_catalog)
    for part in parts:
        if failure == "power" and part.category == "GPU":
            part.tdp_watts = None
        if failure == "budget":
            part.price = 50_000_000
        if failure == "socket" and part.category == "MAINBOARD":
            part.socket = "IMPOSSIBLE"
    with pytest.raises(NoFeasibleBuildError):
        PCBuildApplicationService().build_pc(PCBuildConstraints(), parts)
