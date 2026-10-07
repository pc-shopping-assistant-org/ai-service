"""Unit and architectural tests for the PC Builder Application Layer.

Tests:
- Recommendation policy selection & versioning
- Deterministic fallback when preferred objective is unavailable
- Application service orchestration without LLM/Agent coupling
- Explanation context factual serialization (zero recalculation)
- AST-level verification of strict one-way dependency direction
"""

import ast
from pathlib import Path

import pytest

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.application import (
    DEFAULT_OBJECTIVE_BY_PROFILE,
    FALLBACK_OBJECTIVE_ORDER,
    RECOMMENDATION_POLICY_VERSION,
    BuildApplicationResult,
    PCBuildApplicationService,
    RecommendationDecision,
    RecommendationPolicyError,
    RecommendationReason,
    build_explanation_context,
    choose_recommended_build,
)
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    CompatibilityStatus,
    ConstraintSource,
    ConstraintValue,
    MetricEvidence,
    OptimizationResult,
    PCBuildConstraints,
    PowerEstimate,
    RankedBuild,
    UseCaseProfile,
)


def _make_dummy_build(
    objective: BuildObjective,
    total_price: int = 25_000_000,
    score: float = 85.0,
) -> RankedBuild:
    """Helper to construct a minimal valid RankedBuild for policy unit testing."""
    parts = {
        ComponentCategory.CPU: ComponentSpec(
            name="Test CPU",
            category=ComponentCategory.CPU,
            price=5_000_000,
            socket="AM5",
            tdp_watts=65,
        ),
        ComponentCategory.GPU: ComponentSpec(
            name="Test GPU",
            category=ComponentCategory.GPU,
            price=10_000_000,
            tdp_watts=160,
        ),
    }
    power = PowerEstimate(
        cpu_estimated_peak_watts=65.0,
        gpu_estimated_peak_watts=160.0,
        platform_estimated_watts=65.0,
        estimated_sustained_peak_watts=290.0,
        transient_allowance_watts=32.0,
        minimum_capacity_watts=550,
        recommended_capacity_watts=650,
        sizing_rationale="Adequate headroom",
    )
    evidence = [
        MetricEvidence(
            metric="gpu_perf",
            raw_value="85.0",
            normalized_score=85.0,
            weight=0.35,
            contribution=29.75,
            source="spec",
            label="Hiệu năng GPU",
        ),
    ]
    return RankedBuild(
        objective=objective,
        objective_score=score,
        total_price=total_price,
        parts=parts,
        power_estimate=power,
        evidence=evidence,
        compatibility_status=CompatibilityStatus.COMPATIBLE,
    )


def _make_dummy_optimization_result(
    use_case: UseCaseProfile,
    target_budget_vnd: int,
    builds: dict[BuildObjective, RankedBuild],
) -> OptimizationResult:
    return OptimizationResult(
        use_case=use_case,
        target_budget_vnd=target_budget_vnd,
        builds=builds,
        candidates_evaluated=100,
        pruned_count=20,
    )


def test_every_use_case_has_default_recommendation_policy() -> None:
    """Completeness invariant: every UseCaseProfile enum variant must have a default objective mapped."""
    assert set(DEFAULT_OBJECTIVE_BY_PROFILE) == set(UseCaseProfile)


def test_fallback_covers_every_objective() -> None:
    """Completeness invariant: fallback sequence must cover all BuildObjective variants deterministically."""
    assert set(FALLBACK_OBJECTIVE_ORDER) == set(BuildObjective)
    assert len(FALLBACK_OBJECTIVE_ORDER) == len(BuildObjective)


def test_unmapped_use_case_profile_raises_policy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an unmapped UseCaseProfile is provided, raise a clear RecommendationPolicyError."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=28_000_000,
        builds={BuildObjective.BALANCED: build_bal},
    )
    test_dict = dict(DEFAULT_OBJECTIVE_BY_PROFILE)
    del test_dict[UseCaseProfile.GAMING_1080P]
    monkeypatch.setattr(
        "ai_service.capabilities.pc_builder.application.recommendation_policy.DEFAULT_OBJECTIVE_BY_PROFILE",
        test_dict,
    )
    with pytest.raises(RecommendationPolicyError, match="No recommendation policy configured"):
        choose_recommended_build(result)


def test_user_explicitly_chooses_performance() -> None:
    """User requested objective is strictly honored regardless of use case profile."""
    build_perf = _make_dummy_build(BuildObjective.PERFORMANCE, 26_000_000, 92.0)
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=28_000_000,
        builds={
            BuildObjective.PERFORMANCE: build_perf,
            BuildObjective.BALANCED: build_bal,
        },
    )

    decision = choose_recommended_build(
        result=result,
        requested_objective=BuildObjective.PERFORMANCE,
    )

    assert decision.objective == BuildObjective.PERFORMANCE
    assert decision.build is build_perf
    assert decision.reason_code == RecommendationReason.USER_SELECTED
    assert decision.policy_version == RECOMMENDATION_POLICY_VERSION


def test_user_requests_unavailable_objective_raises_value_error() -> None:
    """If the user requests an objective that has no feasible build, raise ValueError."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=28_000_000,
        builds={BuildObjective.BALANCED: build_bal},
    )

    with pytest.raises(ValueError, match="No feasible build for requested objective"):
        choose_recommended_build(
            result=result,
            requested_objective=BuildObjective.PERFORMANCE,
        )


def test_no_requested_objective_uses_profile_default() -> None:
    """When no objective is requested, use the profile-configured default."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    build_perf = _make_dummy_build(BuildObjective.PERFORMANCE, 26_000_000, 92.0)
    result_1440p = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1440P,
        target_budget_vnd=28_000_000,
        builds={
            BuildObjective.BALANCED: build_bal,
            BuildObjective.PERFORMANCE: build_perf,
        },
    )

    decision = choose_recommended_build(result_1440p)
    assert decision.objective == BuildObjective.BALANCED
    assert decision.build is build_bal
    assert decision.reason_code == RecommendationReason.PROFILE_DEFAULT
    assert decision.policy_version == RECOMMENDATION_POLICY_VERSION


def test_gaming_4k_and_ai_profile_defaults_performance() -> None:
    """Profiles demanding extreme compute default to PERFORMANCE objective."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 50_000_000, 85.0)
    build_perf = _make_dummy_build(BuildObjective.PERFORMANCE, 55_000_000, 95.0)

    for profile in (UseCaseProfile.GAMING_4K, UseCaseProfile.AI_DATA_SCIENCE):
        result = _make_dummy_optimization_result(
            use_case=profile,
            target_budget_vnd=60_000_000,
            builds={
                BuildObjective.BALANCED: build_bal,
                BuildObjective.PERFORMANCE: build_perf,
            },
        )
        decision = choose_recommended_build(result)
        assert decision.objective == BuildObjective.PERFORMANCE
        assert decision.build is build_perf
        assert decision.reason_code == RecommendationReason.PROFILE_DEFAULT


def test_fallback_when_default_unavailable() -> None:
    """If the default objective is unavailable, fall back deterministically: BALANCED -> PERFORMANCE -> UPGRADE_FRIENDLY."""
    # GAMING_4K defaults to PERFORMANCE, but only UPGRADE_FRIENDLY is feasible
    build_upg = _make_dummy_build(BuildObjective.UPGRADE_FRIENDLY, 45_000_000, 80.0)
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_4K,
        target_budget_vnd=50_000_000,
        builds={BuildObjective.UPGRADE_FRIENDLY: build_upg},
    )

    decision = choose_recommended_build(result)
    assert decision.objective == BuildObjective.UPGRADE_FRIENDLY
    assert decision.build is build_upg
    assert decision.reason_code == RecommendationReason.DEFAULT_UNAVAILABLE


def test_no_builds_raises_value_error() -> None:
    """If optimization result is completely empty, raise ValueError."""
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=20_000_000,
        builds={},
    )
    with pytest.raises(ValueError, match="contains no feasible builds"):
        choose_recommended_build(result)


def test_same_optimization_result_gives_same_recommendation() -> None:
    """Recommendation decision is strictly idempotent and reproducible."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    build_perf = _make_dummy_build(BuildObjective.PERFORMANCE, 26_000_000, 92.0)
    result = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1440P,
        target_budget_vnd=28_000_000,
        builds={
            BuildObjective.BALANCED: build_bal,
            BuildObjective.PERFORMANCE: build_perf,
        },
    )

    decisions = [choose_recommended_build(result) for _ in range(10)]
    first = decisions[0]
    for other in decisions[1:]:
        assert other.objective == first.objective
        assert other.reason_code == first.reason_code
        assert other.policy_version == first.policy_version
        assert other.build is first.build


def test_application_service_e2e_without_agent() -> None:
    """PCBuildApplicationService executes end-to-end with pure Python inputs without any agent or LLM."""
    catalog = [
        ComponentSpec(name="AMD Ryzen 5 7600", category=ComponentCategory.CPU, price=5_200_000, socket="AM5", tdp_watts=65, performance_score=85),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Gigabyte RTX 4060 Eagle", category=ComponentCategory.GPU, price=8_200_000, tdp_watts=115, gpu_length_mm=272, vram_gb=8, performance_score=75),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Deepcool AK400", category=ComponentCategory.COOLER, price=600_000, cooler_height_mm=155, performance_score=78, supported_sockets=["AM5"]),
        ComponentSpec(name="Kingston NV2 1TB NVMe", category=ComponentCategory.STORAGE, price=1_450_000, capacity_gb=1000, performance_score=72),
        ComponentSpec(name="MSI MAG A650BN 650W", category=ComponentCategory.PSU, price=1_350_000, wattage=650, psu_tier="C", performance_score=75),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )

    service = PCBuildApplicationService()
    app_result = service.build_pc(constraints=constraints, catalog=catalog)

    assert isinstance(app_result, BuildApplicationResult)
    assert isinstance(app_result.optimization, OptimizationResult)
    assert isinstance(app_result.recommendation, RecommendationDecision)
    assert app_result.recommendation.objective == BuildObjective.BALANCED
    assert app_result.recommendation.build.compatibility_status == CompatibilityStatus.COMPATIBLE
    assert app_result.recommendation.build.total_price <= 25_000_000


def test_application_result_immutability() -> None:
    """BuildApplicationResult is frozen and does not mutate the internal OptimizationResult."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    opt = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=25_000_000,
        builds={BuildObjective.BALANCED: build_bal},
    )
    rec = RecommendationDecision(
        objective=BuildObjective.BALANCED,
        build=build_bal,
        reason_code=RecommendationReason.PROFILE_DEFAULT,
        policy_version=RECOMMENDATION_POLICY_VERSION,
    )
    app_result = BuildApplicationResult(optimization=opt, recommendation=rec)

    with pytest.raises(AttributeError):
        # Frozen dataclass prevents mutation
        app_result.recommendation = rec  # type: ignore[misc]


def test_explanation_context_factual_only() -> None:
    """Explanation context contains only verified facts, exact prices, power snapshot, and optimizer evidence."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    build_perf = _make_dummy_build(BuildObjective.PERFORMANCE, 25_000_000, 92.0)
    opt = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1440P,
        target_budget_vnd=25_000_000,
        builds={
            BuildObjective.BALANCED: build_bal,
            BuildObjective.PERFORMANCE: build_perf,
        },
    )
    rec = RecommendationDecision(
        objective=BuildObjective.BALANCED,
        build=build_bal,
        reason_code=RecommendationReason.PROFILE_DEFAULT,
        policy_version=RECOMMENDATION_POLICY_VERSION,
    )
    app_result = BuildApplicationResult(optimization=opt, recommendation=rec)

    ctx = build_explanation_context(app_result)

    assert ctx["recommended_objective"] == "BALANCED"
    assert ctx["reason_code"] == "PROFILE_DEFAULT_OBJECTIVE"
    assert ctx["policy_version"] == "v1"
    assert ctx["total_price"] == 24_000_000
    assert ctx["objective_score"] == 88.0
    assert ctx["compatibility_status"] == "COMPATIBLE"
    assert "CPU" in ctx["parts"]
    assert ctx["parts"]["CPU"]["name"] == "Test CPU"
    assert ctx["parts"]["CPU"]["price"] == 5_000_000
    assert ctx["power"]["recommended_capacity_watts"] == 650
    assert len(ctx["evidence"]) == 1
    assert ctx["evidence"][0]["metric"] == "gpu_perf"
    assert ctx["objective_builds"]["PERFORMANCE"]["total_price"] == 25_000_000
    assert ctx["objective_builds"]["PERFORMANCE"]["objective_score"] == 92.0

    # Ensure no fabricated metrics exist in the context
    disallowed_keys = {"fps", "bottleneck_percentage", "temperature", "noise_dba"}
    assert disallowed_keys.isdisjoint(set(ctx.keys()))


def test_explanation_context_does_not_recalculate() -> None:
    """Exact invariant test: all fields are copied directly from the build without score or power recalculation."""
    build_bal = _make_dummy_build(BuildObjective.BALANCED, 24_000_000, 88.0)
    opt = _make_dummy_optimization_result(
        use_case=UseCaseProfile.GAMING_1080P,
        target_budget_vnd=25_000_000,
        builds={BuildObjective.BALANCED: build_bal},
    )
    rec = RecommendationDecision(
        objective=BuildObjective.BALANCED,
        build=build_bal,
        reason_code=RecommendationReason.PROFILE_DEFAULT,
        policy_version=RECOMMENDATION_POLICY_VERSION,
    )
    result = BuildApplicationResult(optimization=opt, recommendation=rec)
    context = build_explanation_context(result)

    assert context["objective_score"] == result.recommendation.build.objective_score
    assert context["evidence"] == [e.model_dump() for e in result.recommendation.build.evidence]
    assert context["power"] == result.recommendation.build.power_estimate.model_dump()
    assert context["compatibility_status"] == result.recommendation.build.compatibility_status.value


def test_explanation_context_anti_regression_ast() -> None:
    """Anti-regression test: explanation_context.py must NOT import score/power calculation functions."""
    target = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "ai_service"
        / "capabilities"
        / "pc_builder"
        / "application"
        / "explanation_context.py"
    )
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    forbidden_calculation_functions = {
        "score_build",
        "calculate_power_estimate",
        "resolve_component_metric",
        "pareto_prune_all_objectives",
        "check_compatibility",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_calculation_functions:
                    assert forbidden not in alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name not in forbidden_calculation_functions, (
                    f"explanation_context.py must not import '{alias.name}'. Zero recalculation invariant violated."
                )


def test_core_zero_coupling_ast_direction() -> None:
    """Architectural test: core modules (optimizer, pruning, schemas) must NOT import application or agent modules."""
    core_dir = Path(__file__).resolve().parent.parent / "src" / "ai_service" / "capabilities" / "pc_builder"
    core_files = [
        core_dir / "schemas.py",
        core_dir / "pruning.py",
        core_dir / "optimizer.py",
    ]

    forbidden_prefixes = (
        "ai_service.capabilities.pc_builder.application",
        "ai_service.capabilities.pc_builder.tools",
        "ai_service.agents",
        "pydantic_ai",
        "fastapi",
        ".application",
        ".tools",
    )

    for core_file in core_files:
        assert core_file.exists(), f"Core file {core_file} does not exist"
        tree = ast.parse(core_file.read_text(encoding="utf-8"), filename=str(core_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_prefixes:
                        assert not alias.name.startswith(forbidden), (
                            f"Illegal import in {core_file.name}: 'import {alias.name}' "
                            f"violates zero-coupling direction rule."
                        )
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                # Check relative import level
                if node.level > 0 and module_name:
                    full_target = "." * node.level + module_name
                elif node.level > 0:
                    full_target = "." * node.level
                else:
                    full_target = module_name

                for forbidden in forbidden_prefixes:
                    assert not full_target.startswith(forbidden), (
                        f"Illegal import in {core_file.name}: 'from {full_target} import ...' "
                        f"violates zero-coupling direction rule."
                    )
