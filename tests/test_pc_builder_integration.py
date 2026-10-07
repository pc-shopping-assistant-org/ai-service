"""End-to-end integration tests for PC Builder Application Layer and Tools Adapter.

Verifies that:
1. `service.build_pc(constraints, catalog)` runs completely without Agent or LLM.
2. `tools.build_pc(...)` and `tools.build_pc_from_args(...)` bridge constraints and catalog deterministically.
3. `tools.explain_build(result)` produces factual explanation context ready for LLM presentation.
4. Recommendation policy versions and reason codes are strictly observed.
5. Invariants hold: no score recalculation, exact part prices, power snapshot matching, and zero LLM coupling.
"""

from __future__ import annotations

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.application import (
    RECOMMENDATION_POLICY_VERSION,
    BuildApplicationResult,
    PCBuildApplicationService,
    RecommendationReason,
)
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    CompatibilityStatus,
    ConstraintSource,
    ConstraintValue,
    PCBuildConstraints,
    RecommendBuildArgs,
    UseCaseProfile,
)
from ai_service.capabilities.pc_builder.tools import PCBuilderTools


def test_service_build_pc_completely_standalone(realistic_catalog: list[ComponentSpec]) -> None:
    """Prove that service.build_pc(constraints, catalog) operates completely without Agent or LLM."""
    service = PCBuildApplicationService()
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue(
            value=25_000_000,
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
        use_case=ConstraintValue(
            value=UseCaseProfile.GAMING_1080P,
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
    )

    result = service.build_pc(constraints=constraints, catalog=realistic_catalog)

    assert isinstance(result, BuildApplicationResult)
    assert len(result.optimization.builds) >= 1
    assert result.recommendation.policy_version == RECOMMENDATION_POLICY_VERSION
    assert result.recommendation.reason_code == RecommendationReason.PROFILE_DEFAULT
    assert result.recommendation.objective == BuildObjective.BALANCED
    assert result.recommendation.build.compatibility_status == CompatibilityStatus.COMPATIBLE
    assert result.recommendation.build.total_price <= int(25_000_000 * 1.05)


def test_tools_build_pc_and_explanation_context_pipeline(realistic_catalog: list[ComponentSpec]) -> None:
    """Verify that PCBuilderTools acts as a clean adapter to the application service and produces factual context."""
    tools = PCBuilderTools()
    args = RecommendBuildArgs(
        budget_vnd=35_000_000,
        use_case=UseCaseProfile.GAMING_4K,
    )

    app_result = tools.build_pc_from_args(args=args, catalog=realistic_catalog)

    # Gaming 4K profile policy maps to PERFORMANCE
    assert app_result.recommendation.objective == BuildObjective.PERFORMANCE
    assert app_result.recommendation.reason_code == RecommendationReason.PROFILE_DEFAULT
    assert app_result.recommendation.policy_version == "v1"

    # Produce factual explanation context
    ctx = tools.explain_build(app_result)

    assert ctx["recommended_objective"] == "PERFORMANCE"
    assert ctx["policy_version"] == "v1"
    assert ctx["reason_code"] == "PROFILE_DEFAULT_OBJECTIVE"
    assert ctx["total_price"] == app_result.recommendation.build.total_price
    assert ctx["objective_score"] == app_result.recommendation.build.objective_score
    assert ctx["compatibility_status"] == "COMPATIBLE"
    assert "PERFORMANCE" in ctx["objective_builds"]
    assert ctx["objective_builds"]["PERFORMANCE"]["total_price"] == app_result.recommendation.build.total_price

    # Check power snapshot consistency
    power_ctx = ctx["power"]
    rec_power = app_result.recommendation.build.power_estimate
    assert power_ctx["recommended_capacity_watts"] == rec_power.recommended_capacity_watts
    assert power_ctx["minimum_capacity_watts"] == rec_power.minimum_capacity_watts
    assert power_ctx["estimated_sustained_peak_watts"] == rec_power.estimated_sustained_peak_watts

    # Verify no fabricated metrics exist in explanation context
    disallowed_fabricated_keys = {"fps", "bottleneck_percentage", "temperature", "noise_dba"}
    assert disallowed_fabricated_keys.isdisjoint(set(ctx.keys()))


def test_user_requested_objective_override(realistic_catalog: list[ComponentSpec]) -> None:
    """When user explicitly specifies UPGRADE_FRIENDLY objective, policy selects it with USER_SELECTED reason."""
    tools = PCBuilderTools()
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue(
            value=30_000_000,
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
        use_case=ConstraintValue(
            value=UseCaseProfile.GAMING_1080P,  # Profile default is BALANCED
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
    )

    app_result = tools.build_pc(
        constraints=constraints,
        catalog=realistic_catalog,
        requested_objective=BuildObjective.UPGRADE_FRIENDLY,
    )

    assert app_result.recommendation.objective == BuildObjective.UPGRADE_FRIENDLY
    assert app_result.recommendation.reason_code == RecommendationReason.USER_SELECTED
    assert app_result.recommendation.policy_version == RECOMMENDATION_POLICY_VERSION
    assert app_result.recommendation.build.objective == BuildObjective.UPGRADE_FRIENDLY


def test_ai_data_science_profile_recommends_performance(realistic_catalog: list[ComponentSpec]) -> None:
    """AI_DATA_SCIENCE use case defaults to PERFORMANCE build according to policy."""
    service = PCBuildApplicationService()
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue(
            value=45_000_000,
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
        use_case=ConstraintValue(
            value=UseCaseProfile.AI_DATA_SCIENCE,
            source=ConstraintSource.USER,
            confidence=1.0,
        ),
    )

    result = service.build_pc(constraints=constraints, catalog=realistic_catalog)

    assert result.recommendation.objective == BuildObjective.PERFORMANCE
    assert result.recommendation.reason_code == RecommendationReason.PROFILE_DEFAULT
    assert result.recommendation.policy_version == "v1"
