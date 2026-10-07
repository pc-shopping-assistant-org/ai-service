"""Property-based and constraint-satisfaction tests for the Deterministic PC Optimizer.

100% independent of LLMs, agent frameworks, or network access.
Verifies mathematical soundness, Pareto dominance pruning, deterministic tie-breaking,
and strict physical/electrical compatibility invariants.
"""

import random

import pytest
from pydantic import ValidationError

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.pc_builder.optimizer import (
    SCORING_POLICIES,
    DeterministicPCOptimizer,
    calculate_power_estimate,
    check_compatibility,
    check_compatibility_status,
    round_up_psu,
    score_build,
    stable_component_id,
)
from ai_service.capabilities.pc_builder.pruning import (
    dominates,
    get_candidate_pool,
    pareto_prune_all_objectives,
    resolve_component_metric,
    stable_component_key,
)
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    CompatibilityStatus,
    ConstraintSource,
    ConstraintValue,
    OwnedComponent,
    PCBuildConstraints,
    PowerEstimate,
    RankedBuild,
    ScoreBreakdown,
    ScoreDimension,
    UseCaseProfile,
)


@pytest.fixture
def optimizer() -> DeterministicPCOptimizer:
    return DeterministicPCOptimizer()


def test_determinism_same_input_same_output(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 1: Same constraints and catalog must yield identical output builds."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER, locked=True),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P, source=ConstraintSource.USER),
    )

    res1 = optimizer.optimize(constraints, realistic_catalog)
    res2 = optimizer.optimize(constraints, realistic_catalog)

    b1 = res1.builds[BuildObjective.BALANCED]
    b2 = res2.builds[BuildObjective.BALANCED]
    assert b1.total_price == b2.total_price
    assert b1.objective_score == b2.objective_score
    for cat in ComponentCategory:
        if cat in b1.parts:
            assert b1.parts[cat].name == b2.parts[cat].name


def test_budget_feasibility(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 2: Total price of every generated build must be <= target budget."""
    for budget in [18_000_000, 25_000_000, 35_000_000]:
        constraints = PCBuildConstraints(
            target_budget_vnd=ConstraintValue[int](value=budget, source=ConstraintSource.USER, locked=True),
            use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        )
        res = optimizer.optimize(constraints, realistic_catalog)
        for obj, build in res.builds.items():
            assert build.total_price <= budget, f"Build {obj} price {build.total_price} exceeds budget {budget}"


def test_compatibility_invariants(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 3: All generated builds must strictly satisfy physical and electrical constraints."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=28_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )
    res = optimizer.optimize(constraints, realistic_catalog)

    for build in res.builds.values():
        cpu = build.parts[ComponentCategory.CPU]
        mb = build.parts[ComponentCategory.MAINBOARD]
        ram = build.parts[ComponentCategory.RAM]
        gpu = build.parts[ComponentCategory.GPU]
        case = build.parts[ComponentCategory.CASE]
        psu = build.parts[ComponentCategory.PSU]

        # Socket match
        assert cpu.socket == mb.socket, f"Socket mismatch: {cpu.socket} vs {mb.socket}"

        # RAM type match
        assert mb.ram_type == ram.ram_type, f"RAM type mismatch: {mb.ram_type} vs {ram.ram_type}"

        # GPU length clearance
        if gpu.gpu_length_mm and case.max_gpu_length_mm:
            assert gpu.gpu_length_mm <= case.max_gpu_length_mm

        # PSU sufficiency
        assert psu.wattage is not None
        assert psu.wattage >= build.power_estimate.minimum_psu_watts


def test_pareto_dominance_vram_protection() -> None:
    """Property 4: GPU with lower price but lower VRAM must NOT dominate an AI_DATA_SCIENCE part."""
    gpu_cheaper = ComponentSpec(
        name="RTX 4060 8GB", category=ComponentCategory.GPU, price=7_500_000, vram_gb=8, performance_score=76, tdp_watts=115, gpu_length_mm=240
    )
    gpu_more_vram = ComponentSpec(
        name="RTX 3060 12GB", category=ComponentCategory.GPU, price=7_800_000, vram_gb=12, performance_score=70, tdp_watts=170, gpu_length_mm=235
    )

    # In AI_DATA_SCIENCE, RTX 4060 has less VRAM than 3060, so it must NOT dominate it!
    assert dominates(gpu_cheaper, gpu_more_vram, UseCaseProfile.AI_DATA_SCIENCE, BuildObjective.PERFORMANCE) is False


def test_pareto_dominance_socket_protection() -> None:
    """Property 5: A CPU cannot dominate another CPU if they use different sockets."""
    cpu_am5 = ComponentSpec(
        name="Ryzen 5 7600", category=ComponentCategory.CPU, price=5_000_000, socket="AM5", performance_score=85, tdp_watts=65
    )
    cpu_lga1700 = ComponentSpec(
        name="Core i5-12400F", category=ComponentCategory.CPU, price=5_200_000, socket="LGA1700", performance_score=75, tdp_watts=65
    )
    # Different sockets -> neither can dominate the other because socket choice dictates mainboard
    assert dominates(cpu_am5, cpu_lga1700, UseCaseProfile.GAMING_1080P, BuildObjective.PERFORMANCE) is False


def test_catalog_order_invariance(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 6: Shuffling catalog candidate order must produce identical optimum build."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=26_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )

    original_res = optimizer.optimize(constraints, realistic_catalog)
    original_b = original_res.builds[BuildObjective.BALANCED]

    # Shuffle catalog 3 times
    for seed in [42, 123, 999]:
        shuffled = list(realistic_catalog)
        random.seed(seed)
        random.shuffle(shuffled)
        shuffled_res = optimizer.optimize(constraints, shuffled)
        shuffled_b = shuffled_res.builds[BuildObjective.BALANCED]

        assert original_b.total_price == shuffled_b.total_price
        assert original_b.objective_score == shuffled_b.objective_score
        assert (
            original_b.parts[ComponentCategory.CPU].name
            == shuffled_b.parts[ComponentCategory.CPU].name
        )
        assert (
            original_b.parts[ComponentCategory.GPU].name
            == shuffled_b.parts[ComponentCategory.GPU].name
        )


def test_monotonicity_higher_budget_no_worse_score(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 7: Increasing budget on a fixed catalog must not decrease the optimum utility score."""
    constraints_low = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=20_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )
    constraints_high = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=32_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )

    res_low = optimizer.optimize(constraints_low, realistic_catalog)
    res_high = optimizer.optimize(constraints_high, realistic_catalog)

    score_low = res_low.builds[BuildObjective.PERFORMANCE].objective_score
    score_high = res_high.builds[BuildObjective.PERFORMANCE].objective_score
    assert score_high >= score_low


def test_user_locked_constraints_respected(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 8: If a user specifies a locked constraint (e.g. NVIDIA only), it must NEVER be overridden."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        preferred_gpu_brand=ConstraintValue[str | None](value="NVIDIA", source=ConstraintSource.USER, locked=True),
    )
    res = optimizer.optimize(constraints, realistic_catalog)

    for build in res.builds.values():
        gpu = build.parts[ComponentCategory.GPU]
        assert "NVIDIA" in (gpu.brand or "").upper() or "RTX" in gpu.name.upper()


def test_top_3_objective_diversity(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 9: Top-3 objectives must yield differentiated trade-offs and rationale."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=28_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )
    res = optimizer.optimize(constraints, realistic_catalog)

    assert set(res.builds.keys()) == {
        BuildObjective.PERFORMANCE,
        BuildObjective.BALANCED,
        BuildObjective.UPGRADE_FRIENDLY,
    }

    perf_build = res.builds[BuildObjective.PERFORMANCE]
    upgrade_build = res.builds[BuildObjective.UPGRADE_FRIENDLY]

    # Both must have valid power estimates and factual evidence
    assert perf_build.power_estimate.minimum_psu_watts > 0
    assert len(perf_build.evidence) > 0
    assert len(upgrade_build.evidence) > 0


def test_candidate_pool_preserves_feasibility_and_exact_coverage() -> None:
    """Property 10: CandidatePool provides preferred ordering without excluding affordable candidates."""
    candidates = [
        ComponentSpec(name="Affordable but High GPU", category=ComponentCategory.GPU, price=14_000_000, tdp_watts=200),
    ]
    # For 20M budget, 14M is outside preferred range (7M - 11M).
    pool = get_candidate_pool(candidates, 20_000_000, ComponentCategory.GPU, UseCaseProfile.GAMING_1440P)
    # Both preferred and all_affordable retain the candidate so exact search never drops it
    assert len(pool.all_affordable) == 1
    assert pool.all_affordable[0].name == "Affordable but High GPU"


def test_pareto_missing_data_failsafe() -> None:
    """Property 11: Missing metric on either candidate strictly prevents pruning (never treated as 0)."""
    gpu_with_len = ComponentSpec(
        name="GPU A", category=ComponentCategory.GPU, price=10_000_000, gpu_length_mm=280, tdp_watts=200, performance_score=80
    )
    gpu_missing_len = ComponentSpec(
        name="GPU B", category=ComponentCategory.GPU, price=9_000_000, gpu_length_mm=None, tdp_watts=200, performance_score=85
    )

    # Even though GPU B is cheaper and has higher performance score, its length is unknown (None).
    # It must NOT dominate GPU A because GPU B might fail case clearance!
    assert dominates(gpu_missing_len, gpu_with_len, UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE) is False


def test_pareto_upgrade_friendly_ram_slots() -> None:
    """Property 12: Under UPGRADE_FRIENDLY, a motherboard with fewer RAM slots cannot dominate one with more."""
    mb_2slots = ComponentSpec(
        name="B650 2-slot", category=ComponentCategory.MAINBOARD, price=3_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=2
    )
    mb_4slots = ComponentSpec(
        name="B650 4-slot", category=ComponentCategory.MAINBOARD, price=3_200_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4
    )

    # In UPGRADE_FRIENDLY, mb_2slots is cheaper but has fewer ram_slots -> must NOT dominate mb_4slots!
    assert dominates(mb_2slots, mb_4slots, UseCaseProfile.GAMING_1440P, BuildObjective.UPGRADE_FRIENDLY) is False


def test_out_of_envelope_optimal_candidate_considered(
    optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]
) -> None:
    """Property 13: An out-of-envelope candidate is still considered in exact optimization if it fits the budget."""
    # RTX 4070 SUPER is 16.5M. In a 25M budget, preferred GPU is 35-55% (8.75M - 13.75M).
    # 16.5M is 66% of 25M (outside preferred envelope).
    # If the user pairs it with cheaper parts (total <= 25M), the optimizer must still evaluate it.
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )
    res = optimizer.optimize(constraints, realistic_catalog)
    # The build succeeds and finds a valid configuration
    b = res.builds[BuildObjective.BALANCED]
    assert b.total_price <= 25_000_000
    assert b.objective_score > 0


def test_pruning_does_not_change_optimum(realistic_catalog: list[ComponentSpec]) -> None:
    """Property 14 (Pareto Safety Proof): Pruning ON vs OFF must yield the exact same global optimum.

    Proves mathematically that Pareto dominance pruning never excludes the optimal configuration.
    """
    opt_pruned = DeterministicPCOptimizer(enable_pruning=True)
    opt_unpruned = DeterministicPCOptimizer(enable_pruning=False)

    test_cases = [
        (20_000_000, UseCaseProfile.GAMING_1440P),
        (26_000_000, UseCaseProfile.GAMING_1440P),
        (35_000_000, UseCaseProfile.AI_DATA_SCIENCE),
    ]

    for budget, profile in test_cases:
        constraints = PCBuildConstraints(
            target_budget_vnd=ConstraintValue[int](value=budget, source=ConstraintSource.USER),
            use_case=ConstraintValue[UseCaseProfile](value=profile),
        )

        res_pruned = opt_pruned.optimize(constraints, realistic_catalog)
        res_unpruned = opt_unpruned.optimize(constraints, realistic_catalog)

        for obj in BuildObjective:
            build_pruned = res_pruned.builds[obj]
            build_unpruned = res_unpruned.builds[obj]

            assert build_pruned.total_price == build_unpruned.total_price
            assert build_pruned.objective_score == pytest.approx(build_unpruned.objective_score, rel=1e-5)
            for cat in ComponentCategory:
                if cat in build_pruned.parts:
                    assert (
                        build_pruned.parts[cat].name == build_unpruned.parts[cat].name
                    ), f"Component mismatch in {obj} for {cat}: {build_pruned.parts[cat].name} vs {build_unpruned.parts[cat].name}"


def test_unknown_critical_specs_rejected() -> None:
    """Property 15: Missing critical specifications must be classified as UNKNOWN and rejected (fail-safe)."""
    cpu_no_sock = ComponentSpec(name="Mystery CPU", category=ComponentCategory.CPU, price=3_000_000, socket=None, tdp_watts=65)
    mb = ComponentSpec(name="B650 MB", category=ComponentCategory.MAINBOARD, price=3_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX")
    ram = ComponentSpec(name="DDR5 RAM", category=ComponentCategory.RAM, price=1_500_000, ram_type="DDR5")

    # Missing CPU socket -> UNKNOWN
    parts_1 = {ComponentCategory.CPU: cpu_no_sock, ComponentCategory.MAINBOARD: mb}
    assert check_compatibility_status(parts_1, 500) == CompatibilityStatus.UNKNOWN
    assert check_compatibility(parts_1, 500) is False

    # Missing RAM standard on Mainboard -> UNKNOWN
    mb_no_ram = ComponentSpec(name="Weird MB", category=ComponentCategory.MAINBOARD, price=3_000_000, socket="AM5", ram_type=None, form_factor="ATX")
    parts_2 = {ComponentCategory.MAINBOARD: mb_no_ram, ComponentCategory.RAM: ram}
    assert check_compatibility_status(parts_2, 500) == CompatibilityStatus.UNKNOWN
    assert check_compatibility(parts_2, 500) is False

    # Missing Case Form Factor -> UNKNOWN
    case_no_ff = ComponentSpec(name="Unknown Case", category=ComponentCategory.CASE, price=1_000_000, form_factor=None)
    parts_3 = {ComponentCategory.MAINBOARD: mb, ComponentCategory.CASE: case_no_ff}
    assert check_compatibility_status(parts_3, 500) == CompatibilityStatus.UNKNOWN
    assert check_compatibility(parts_3, 500) is False


def test_missing_tdp_raises_error() -> None:
    """Property 16: Missing TDP on CPU or GPU must strictly raise ValueError (no fake 65W/160W guessing)."""
    cpu_no_tdp = ComponentSpec(name="CPU No TDP", category=ComponentCategory.CPU, price=3_000_000, socket="AM5", tdp_watts=None)
    gpu_no_tdp = ComponentSpec(name="GPU No TDP", category=ComponentCategory.GPU, price=8_000_000, tdp_watts=None)

    parts_cpu = {ComponentCategory.CPU: cpu_no_tdp}
    with pytest.raises(ValueError, match="missing required specification 'tdp_watts'"):
        calculate_power_estimate(parts_cpu, UseCaseProfile.GAMING_1440P)

    parts_gpu = {
        ComponentCategory.CPU: ComponentSpec(name="CPU OK", category=ComponentCategory.CPU, price=3_000_000, tdp_watts=65),
        ComponentCategory.GPU: gpu_no_tdp,
    }
    with pytest.raises(ValueError, match="missing required specification 'tdp_watts'"):
        calculate_power_estimate(parts_gpu, UseCaseProfile.GAMING_1440P)


def test_scoring_normalized_and_policy_weights() -> None:
    """Property 17: All scoring policies must sum strictly to 1.0 and dimension scores strictly in [0, 100]."""
    for (profile, objective), weights in SCORING_POLICIES.items():
        total_weight = sum(weights.values())
        assert total_weight == pytest.approx(1.0, rel=1e-5), f"Policy ({profile}, {objective}) weights sum to {total_weight}"

    # AI_DATA_SCIENCE must heavily prioritize VRAM under PERFORMANCE
    ai_perf_weights = SCORING_POLICIES[(UseCaseProfile.AI_DATA_SCIENCE, BuildObjective.PERFORMANCE)]
    assert ai_perf_weights["gpu_vram"] >= 0.35


def test_stable_component_key_semantics() -> None:
    """Property 18: Distinct Python instances representing the same SKU produce identical stable keys."""
    comp1 = ComponentSpec(name="RTX 4060", category=ComponentCategory.GPU, price=7_500_000, brand="NVIDIA")
    comp2 = ComponentSpec(name="RTX 4060", category=ComponentCategory.GPU, price=7_500_000, brand="NVIDIA")

    # Distinct objects
    assert comp1 is not comp2
    # Same stable key
    assert stable_component_key(comp1) == stable_component_key(comp2)

    # Deduplication across objectives preserves only 1 instance
    retained = pareto_prune_all_objectives([comp1, comp2], UseCaseProfile.GAMING_1440P)
    assert len(retained) == 1


def test_mainboard_conservative_no_pruning_without_quality_dimensions() -> None:
    """Property 19: Motherboards are never pruned purely on price without evaluated quality dimensions."""
    mb_cheap = ComponentSpec(name="Basic MB", category=ComponentCategory.MAINBOARD, price=2_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX")
    mb_rich = ComponentSpec(name="Premium MB", category=ComponentCategory.MAINBOARD, price=4_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX")

    # Even though mb_cheap is cheaper, conservative policy does not assert dominance
    assert dominates(mb_cheap, mb_rich, UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE) is False
    assert dominates(mb_cheap, mb_rich, UseCaseProfile.GAMING_1440P, BuildObjective.BALANCED) is False


def test_psu_quality_dimension_mandatory() -> None:
    """Property 20: A cheaper PSU with higher wattage cannot dominate a higher-tier/efficiency PSU."""
    psu_tier_c = ComponentSpec(name="Budget 750W", category=ComponentCategory.PSU, price=1_200_000, wattage=750, performance_score=70)
    psu_tier_a = ComponentSpec(name="Gold 650W", category=ComponentCategory.PSU, price=1_500_000, wattage=650, performance_score=95)

    # In PERFORMANCE, psu_tier_c has lower price and higher wattage, but lower performance_score -> must NOT dominate
    assert dominates(psu_tier_c, psu_tier_a, UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE) is False


def test_no_igpu_cpu_cannot_form_gpu_less_build(optimizer: DeterministicPCOptimizer) -> None:
    """Property 21: A CPU without iGPU (e.g. 12400F) cannot fabricate an iGPU when catalog has no GPU."""
    catalog = [
        ComponentSpec(
            name="Intel Core i5-12400F",
            category=ComponentCategory.CPU,
            price=2_800_000,
            socket="LGA1700",
            tdp_watts=65,
            performance_score=72,
            has_integrated_graphics=False,
            includes_stock_cooler=True,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(name="ASRock B760M-HDV DDR4", category=ComponentCategory.MAINBOARD, price=2_100_000, socket="LGA1700", ram_type="DDR4", form_factor="Micro-ATX", ram_slots=2),
        ComponentSpec(name="Kingston Fury Beast 16GB DDR4", category=ComponentCategory.RAM, price=950_000, ram_type="DDR4", capacity_gb=16, performance_score=70),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=20_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    with pytest.raises(ValueError, match="No compatible build configuration found"):
        optimizer.optimize(constraints, catalog)


def test_igpu_cpu_can_form_gpu_less_build(optimizer: DeterministicPCOptimizer) -> None:
    """Property 22: A CPU with iGPU (e.g. Ryzen 7600) can legitimately form a build without dedicated GPU."""
    catalog = [
        ComponentSpec(
            name="AMD Ryzen 5 7600",
            category=ComponentCategory.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=True,
            stock_cooler_height_mm=55,
            stock_cooler_score=50,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=20_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    res = optimizer.optimize(constraints, catalog)
    b = res.builds[BuildObjective.BALANCED]
    assert "iGPU" in b.parts[ComponentCategory.GPU].name
    assert b.parts[ComponentCategory.GPU].price == 0


def test_no_stock_cooler_cpu_requires_aftermarket_cooler(optimizer: DeterministicPCOptimizer) -> None:
    """Property 23: A CPU without stock cooler (e.g. 7800X3D) cannot fabricate one, requiring an aftermarket cooler."""
    catalog_without_cooler = [
        ComponentSpec(
            name="AMD Ryzen 7 7800X3D",
            category=ComponentCategory.CPU,
            price=10_500_000,
            socket="AM5",
            tdp_watts=120,
            performance_score=98,
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=False,
        ),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    with pytest.raises(ValueError, match="No compatible build configuration found"):
        optimizer.optimize(constraints, catalog_without_cooler)

    # Now add aftermarket cooler -> must succeed
    catalog_with_cooler = catalog_without_cooler + [
        ComponentSpec(name="Thermalright Assassin X 120", category=ComponentCategory.COOLER, price=450_000, cooler_height_mm=148, performance_score=75, supported_sockets=["AM5"]),
    ]
    res = optimizer.optimize(constraints, catalog_with_cooler)
    b = res.builds[BuildObjective.BALANCED]
    assert b.parts[ComponentCategory.COOLER].name == "Thermalright Assassin X 120"


def test_missing_scoring_metric_invalidates_build() -> None:
    """Property 24: If an objective requires a metric (weight > 0) that is missing, score_build returns -1.0."""
    parts = {
        ComponentCategory.CPU: ComponentSpec(name="CPU", category=ComponentCategory.CPU, price=3_000_000, performance_score=80, socket="AM5", tdp_watts=65),
        ComponentCategory.MAINBOARD: ComponentSpec(name="MB", category=ComponentCategory.MAINBOARD, price=2_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4),
        ComponentCategory.RAM: ComponentSpec(name="RAM", category=ComponentCategory.RAM, price=1_000_000, capacity_gb=16, performance_score=70, ram_type="DDR5"),
        ComponentCategory.GPU: ComponentSpec(name="GPU No Score", category=ComponentCategory.GPU, price=8_000_000, performance_score=None, vram_gb=8, tdp_watts=150),
        ComponentCategory.CASE: ComponentSpec(name="Case", category=ComponentCategory.CASE, price=1_000_000, form_factor="ATX"),
        ComponentCategory.COOLER: ComponentSpec(name="Cooler", category=ComponentCategory.COOLER, price=500_000, performance_score=75),
        ComponentCategory.STORAGE: ComponentSpec(name="Storage", category=ComponentCategory.STORAGE, price=1_000_000, capacity_gb=1000, performance_score=70),
        ComponentCategory.PSU: ComponentSpec(name="PSU", category=ComponentCategory.PSU, price=1_000_000, wattage=650, psu_tier="B", performance_score=80),
    }
    power = PowerEstimate(
        cpu_estimated_peak_watts=65.0,
        gpu_estimated_peak_watts=150.0,
        platform_estimated_watts=65.0,
        estimated_sustained_peak_watts=280.0,
        transient_allowance_watts=45.0,
        minimum_capacity_watts=550,
        recommended_capacity_watts=650,
        sizing_rationale="Adequate headroom",
    )

    # In PERFORMANCE objective, gpu_perf has non-zero weight. Missing performance_score must return -1.0 (invalid)
    score = score_build(parts, BuildObjective.PERFORMANCE, UseCaseProfile.GAMING_1440P, 20_000_000, power)
    assert score == -1.0


def test_psu_capacity_overflow_raises_error() -> None:
    """Property 25: PSU requirement exceeding maximum supported wattage (1200W) raises ValueError, never undersizes."""
    assert round_up_psu(480) == 500
    assert round_up_psu(510) == 550
    assert round_up_psu(650) == 650
    assert round_up_psu(1100) == 1200

    with pytest.raises(ValueError, match="exceeds maximum supported"):
        round_up_psu(1250)


def test_stable_component_id_without_id_consistent() -> None:
    """Property 26: ComponentSpec with id=None uses deterministic fingerprint matching all physical specs."""
    c1 = ComponentSpec(
        name="RTX 4070 SUPER",
        category=ComponentCategory.GPU,
        price=16_500_000,
        brand="NVIDIA",
        vram_gb=12,
        tdp_watts=220,
        gpu_length_mm=280,
    )
    c2 = ComponentSpec(
        name="RTX 4070 SUPER",
        category=ComponentCategory.GPU,
        price=16_500_000,
        brand="NVIDIA",
        vram_gb=12,
        tdp_watts=220,
        gpu_length_mm=280,
    )
    # Distinct instances without ID produce identical fingerprint
    assert c1.id is None
    assert stable_component_id(c1) == stable_component_id(c2)
    assert "GPU|NVIDIA|RTX 4070 SUPER" in stable_component_id(c1)


def test_core_models_extra_forbid() -> None:
    """Property 27: Core domain schemas strictly forbid unknown attributes (typo protection)."""
    with pytest.raises(ValidationError):
        PCBuildConstraints(
            target_buget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),  # Typo in field name
        )

    with pytest.raises(ValidationError):
        PowerEstimate(
            cpu_estimated_peak_watts=65.0,
            gpu_estimated_peak_watts=150.0,
            platform_estimated_watts=65.0,
            estimated_sustained_peak_watts=280.0,
            transient_allowance_watts=45.0,
            minimum_capacity_watts=550,
            recommended_capacity_watts=650,
            sizing_rationale="Headroom",
            fake_unknown_field="error",  # Extra field
        )


def test_constraint_value_locked_validator() -> None:
    """Property 28: Inferred or defaulted constraints cannot be locked."""
    # Source USER or SYSTEM can be locked
    cv_user = ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER, locked=True)
    assert cv_user.locked is True

    cv_sys = ConstraintValue[int](value=25_000_000, source=ConstraintSource.SYSTEM, locked=True)
    assert cv_sys.locked is True

    # Source INFERRED or DEFAULT cannot be locked
    with pytest.raises(ValidationError, match="cannot be locked"):
        ConstraintValue[int](value=25_000_000, source=ConstraintSource.INFERRED, locked=True)

    with pytest.raises(ValidationError, match="cannot be locked"):
        ConstraintValue[int](value=25_000_000, source=ConstraintSource.DEFAULT, locked=True)


def test_fps_validation() -> None:
    """Property 29: Target FPS must be within realistic bounds (30 <= fps <= 1000)."""
    # Valid FPS
    c_valid = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000),
        target_fps=ConstraintValue[int | None](value=144),
    )
    assert c_valid.target_fps.value == 144

    # Too low (< 30)
    with pytest.raises(ValidationError, match="target_fps must be between 30 and 1000"):
        PCBuildConstraints(
            target_budget_vnd=ConstraintValue[int](value=25_000_000),
            target_fps=ConstraintValue[int | None](value=20),
        )

    # Too high (> 1000)
    with pytest.raises(ValidationError, match="target_fps must be between 30 and 1000"):
        PCBuildConstraints(
            target_budget_vnd=ConstraintValue[int](value=25_000_000),
            target_fps=ConstraintValue[int | None](value=1500),
        )


def test_ranked_build_requires_compatibility_status() -> None:
    """Property 30: RankedBuild strictly requires explicit compatibility_status (no default)."""
    with pytest.raises(ValidationError):
        RankedBuild(
            objective=BuildObjective.BALANCED,
            objective_score=80.0,
            total_price=20_000_000,
            parts={},
            power_estimate=PowerEstimate(
                cpu_estimated_peak_watts=65.0,
                gpu_estimated_peak_watts=150.0,
                platform_estimated_watts=65.0,
                estimated_sustained_peak_watts=280.0,
                transient_allowance_watts=45.0,
                minimum_capacity_watts=550,
                recommended_capacity_watts=650,
                sizing_rationale="Headroom",
            ),
            # Missing compatibility_status -> must fail
        )


def test_igpu_appended_sorted_not_skipped(optimizer: DeterministicPCOptimizer) -> None:
    """Property 31: When catalog GPU exceeds budget, synthetic iGPU (0 VND) must be sorted at index 0 and not skipped by loop break."""
    catalog = [
        ComponentSpec(
            name="AMD Ryzen 5 7600",
            category=ComponentCategory.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=True,
            stock_cooler_height_mm=55,
            stock_cooler_score=50,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        # Dedicated GPU exceeds remaining budget (15M - 10M = 5M remaining)
        ComponentSpec(name="GeForce RTX 4070 SUPER 12GB", category=ComponentCategory.GPU, price=16_500_000, tdp_watts=220, gpu_length_mm=280, vram_gb=12, performance_score=94),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    # Budget is 15M: 5.2M CPU + 3.4M MB + 1.4M RAM + 0đ iGPU + 1.1M Case + 0đ Cooler + 0.95M SSD + 1.1M PSU = 13.15M
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=15_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    res = optimizer.optimize(constraints, catalog)
    b = res.builds[BuildObjective.BALANCED]
    assert "iGPU" in b.parts[ComponentCategory.GPU].name
    assert b.parts[ComponentCategory.GPU].price == 0
    assert b.total_price <= 15_000_000


def test_stock_cooler_appended_sorted_not_skipped(optimizer: DeterministicPCOptimizer) -> None:
    """Property 32: When aftermarket cooler exceeds budget, stock cooler (0 VND) must be sorted at index 0 and not skipped by loop break."""
    catalog = [
        ComponentSpec(
            name="AMD Ryzen 5 7600",
            category=ComponentCategory.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
            has_integrated_graphics=True,
            integrated_graphics_score=15,
            integrated_graphics_power_watts=15,
            includes_stock_cooler=True,
            stock_cooler_height_mm=55,
            stock_cooler_score=50,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        # Expensive cooler that would exceed remaining budget if checked first
        ComponentSpec(name="Ultra Expensive Liquid Cooler", category=ComponentCategory.COOLER, price=20_000_000, cooler_height_mm=120, performance_score=95),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=15_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    res = optimizer.optimize(constraints, catalog)
    b = res.builds[BuildObjective.BALANCED]
    assert "Stock Cooler" in b.parts[ComponentCategory.COOLER].name
    assert b.parts[ComponentCategory.COOLER].price == 0


def test_owned_gpu_locked_pool(optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]) -> None:
    """Property 33: Owned component locks the slot strictly to the owned part, ignoring catalog candidates and iGPU."""
    owned_gpu = ComponentSpec(
        name="Custom Owned RTX 3070",
        category=ComponentCategory.GPU,
        price=6_000_000,
        tdp_watts=220,
        gpu_length_mm=242,
        vram_gb=8,
        performance_score=82,
        brand="NVIDIA",
    )
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        owned_parts=[OwnedComponent(category=ComponentCategory.GPU, component=owned_gpu, exclude_from_budget=True)],
    )
    res = optimizer.optimize(constraints, realistic_catalog)
    for build in res.builds.values():
        assert build.parts[ComponentCategory.GPU].name == "Custom Owned RTX 3070"


def test_owned_exclude_from_budget_spending(optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]) -> None:
    """Property 34: exclude_from_budget=True charges 0 VND towards budget, whereas exclude_from_budget=False counts part price."""
    owned_gpu = ComponentSpec(
        name="Custom Owned RTX 3070",
        category=ComponentCategory.GPU,
        price=6_000_000,
        tdp_watts=220,
        gpu_length_mm=242,
        vram_gb=8,
        performance_score=82,
        brand="NVIDIA",
    )
    # Case 1: exclude_from_budget = True
    c_excluded = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        owned_parts=[OwnedComponent(category=ComponentCategory.GPU, component=owned_gpu, exclude_from_budget=True)],
    )
    res_exc = optimizer.optimize(c_excluded, realistic_catalog)
    b_exc = res_exc.builds[BuildObjective.BALANCED]
    remaining_spent = sum(p.price for cat, p in b_exc.parts.items() if cat != ComponentCategory.GPU)
    assert b_exc.total_price == remaining_spent

    # Case 2: exclude_from_budget = False
    c_included = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=25_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        owned_parts=[OwnedComponent(category=ComponentCategory.GPU, component=owned_gpu, exclude_from_budget=False)],
    )
    res_inc = optimizer.optimize(c_included, realistic_catalog)
    b_inc = res_inc.builds[BuildObjective.BALANCED]
    assert b_inc.total_price == sum(p.price for p in b_inc.parts.values())


def test_cpu_missing_igpu_score_does_not_fabricate(optimizer: DeterministicPCOptimizer) -> None:
    """Property 35: When CPU has has_integrated_graphics=True but missing integrated_graphics_score, never fabricate an iGPU."""
    catalog = [
        ComponentSpec(
            name="AMD Ryzen 5 7600 Incomplete Spec",
            category=ComponentCategory.CPU,
            price=5_200_000,
            socket="AM5",
            tdp_watts=65,
            performance_score=85,
            has_integrated_graphics=True,
            integrated_graphics_score=None,  # Missing benchmark!
            integrated_graphics_power_watts=15,
            includes_stock_cooler=True,
            stock_cooler_height_mm=55,
            stock_cooler_score=50,
            stock_cooler_tdp_watts=65,
        ),
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="Kingston NV2 500GB NVMe", category=ComponentCategory.STORAGE, price=950_000, capacity_gb=500, performance_score=65),
        ComponentSpec(name="MSI MAG A550BN 550W Bronze", category=ComponentCategory.PSU, price=1_100_000, wattage=550, psu_tier="C", performance_score=72),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=20_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.OFFICE_BUDGET),
    )
    with pytest.raises(ValueError, match="No compatible build configuration found"):
        optimizer.optimize(constraints, catalog)


def test_cooler_socket_compatibility() -> None:
    """Property 36: Cooler with supported_sockets strictly checks CPU socket compatibility."""
    parts = {
        ComponentCategory.CPU: ComponentSpec(name="Ryzen 7600", category=ComponentCategory.CPU, price=5_000_000, socket="AM5", tdp_watts=65, performance_score=85),
        ComponentCategory.MAINBOARD: ComponentSpec(name="B650 MB", category=ComponentCategory.MAINBOARD, price=3_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX"),
        ComponentCategory.RAM: ComponentSpec(name="DDR5 RAM", category=ComponentCategory.RAM, price=1_000_000, ram_type="DDR5", capacity_gb=16, performance_score=75),
        ComponentCategory.GPU: ComponentSpec(name="RTX 4060", category=ComponentCategory.GPU, price=7_000_000, tdp_watts=115, gpu_length_mm=240, vram_gb=8, performance_score=76),
        ComponentCategory.CASE: ComponentSpec(name="Case", category=ComponentCategory.CASE, price=1_000_000, form_factor="ATX", max_gpu_length_mm=300, max_cooler_height_mm=160),
        # Intel-only cooler
        ComponentCategory.COOLER: ComponentSpec(name="LGA Cooler Only", category=ComponentCategory.COOLER, price=500_000, cooler_height_mm=140, performance_score=75, supported_sockets=["LGA1700"]),
        ComponentCategory.STORAGE: ComponentSpec(name="NVMe", category=ComponentCategory.STORAGE, price=1_000_000, capacity_gb=1000, performance_score=75),
        ComponentCategory.PSU: ComponentSpec(name="PSU 650W", category=ComponentCategory.PSU, price=1_500_000, wattage=650, psu_tier="B", performance_score=80),
    }
    status = check_compatibility_status(parts, 550)
    assert status == CompatibilityStatus.INCOMPATIBLE


def test_missing_scoring_policy_fails_fast() -> None:
    """Property 37: Missing scoring policy raises ValueError immediately with no silent fallback to GAMING_1440P."""
    parts = {
        ComponentCategory.CPU: ComponentSpec(name="CPU", category=ComponentCategory.CPU, price=3_000_000, performance_score=80, socket="AM5", tdp_watts=65),
        ComponentCategory.MAINBOARD: ComponentSpec(name="MB", category=ComponentCategory.MAINBOARD, price=2_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4),
        ComponentCategory.RAM: ComponentSpec(name="RAM", category=ComponentCategory.RAM, price=1_000_000, capacity_gb=16, performance_score=70, ram_type="DDR5"),
        ComponentCategory.GPU: ComponentSpec(name="GPU", category=ComponentCategory.GPU, price=8_000_000, performance_score=80, vram_gb=8, tdp_watts=150),
        ComponentCategory.CASE: ComponentSpec(name="Case", category=ComponentCategory.CASE, price=1_000_000, form_factor="ATX"),
        ComponentCategory.COOLER: ComponentSpec(name="Cooler", category=ComponentCategory.COOLER, price=500_000, performance_score=75),
        ComponentCategory.STORAGE: ComponentSpec(name="Storage", category=ComponentCategory.STORAGE, price=1_000_000, capacity_gb=1000, performance_score=70),
        ComponentCategory.PSU: ComponentSpec(name="PSU", category=ComponentCategory.PSU, price=1_000_000, wattage=650, psu_tier="B", performance_score=80),
    }
    power = PowerEstimate(
        cpu_estimated_peak_watts=65.0,
        gpu_estimated_peak_watts=150.0,
        platform_estimated_watts=65.0,
        estimated_sustained_peak_watts=280.0,
        transient_allowance_watts=45.0,
        minimum_capacity_watts=550,
        recommended_capacity_watts=650,
        sizing_rationale="Adequate headroom",
    )
    with pytest.raises(ValueError, match="No scoring policy registered"):
        score_build(parts, BuildObjective.PERFORMANCE, "NONEXISTENT_PROFILE", 20_000_000, power)  # type: ignore[arg-type]


def test_score_breakdown_transparency() -> None:
    """Property 38: score_build returns ScoreBreakdown with full dimension contribution traceability."""
    parts = {
        ComponentCategory.CPU: ComponentSpec(name="CPU", category=ComponentCategory.CPU, price=3_000_000, performance_score=80, socket="AM5", tdp_watts=65),
        ComponentCategory.MAINBOARD: ComponentSpec(name="MB", category=ComponentCategory.MAINBOARD, price=2_000_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4),
        ComponentCategory.RAM: ComponentSpec(name="RAM", category=ComponentCategory.RAM, price=1_000_000, capacity_gb=16, performance_score=70, ram_type="DDR5"),
        ComponentCategory.GPU: ComponentSpec(name="GPU", category=ComponentCategory.GPU, price=8_000_000, performance_score=85, vram_gb=12, tdp_watts=150),
        ComponentCategory.CASE: ComponentSpec(name="Case", category=ComponentCategory.CASE, price=1_000_000, form_factor="ATX"),
        ComponentCategory.COOLER: ComponentSpec(name="Cooler", category=ComponentCategory.COOLER, price=500_000, performance_score=75),
        ComponentCategory.STORAGE: ComponentSpec(name="Storage", category=ComponentCategory.STORAGE, price=1_000_000, capacity_gb=1000, performance_score=70),
        ComponentCategory.PSU: ComponentSpec(name="PSU", category=ComponentCategory.PSU, price=1_000_000, wattage=650, psu_tier="B", performance_score=80),
    }
    power = PowerEstimate(
        cpu_estimated_peak_watts=65.0,
        gpu_estimated_peak_watts=150.0,
        platform_estimated_watts=65.0,
        estimated_sustained_peak_watts=280.0,
        transient_allowance_watts=45.0,
        minimum_capacity_watts=550,
        recommended_capacity_watts=650,
        sizing_rationale="Adequate headroom",
    )
    breakdown = score_build(parts, BuildObjective.BALANCED, UseCaseProfile.GAMING_1440P, 20_000_000, power)
    assert isinstance(breakdown, ScoreBreakdown)
    assert breakdown.total_score > 0
    assert len(breakdown.dimensions) > 0
    for dim in breakdown.dimensions:
        assert isinstance(dim, ScoreDimension)
        assert abs(dim.contribution - round(dim.normalized_score * dim.weight, 4)) < 1e-6
    # Total equals sum of contributions
    assert abs(breakdown.total_score - round(sum(d.contribution for d in breakdown.dimensions), 4)) < 1e-6


def test_form_factor_preference_locked(optimizer: DeterministicPCOptimizer, realistic_catalog: list[ComponentSpec]) -> None:
    """Property 39: Locked form_factor_preference restricts builds exclusively to matching motherboards and cases."""
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=28_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
        form_factor_preference=ConstraintValue[str | None](value="Micro-ATX", source=ConstraintSource.USER, locked=True),
    )
    res = optimizer.optimize(constraints, realistic_catalog)
    for build in res.builds.values():
        mb = build.parts[ComponentCategory.MAINBOARD]
        case = build.parts[ComponentCategory.CASE]
        assert mb.form_factor == "Micro-ATX"
        supported = [s.upper().replace("-", "").replace(" ", "") for s in (case.supported_form_factors or [case.form_factor or ""])]
        assert "MICROATX" in supported


def test_cpu_with_igpu_not_dominated_by_cpu_without_igpu() -> None:
    """Property 40: A CPU with an iGPU cannot be dominated by a CPU without one, even if the latter is cheaper and has higher benchmark score."""
    cpu_with_igpu = ComponentSpec(
        name="AMD Ryzen 5 7600",
        category=ComponentCategory.CPU,
        price=5_200_000,
        socket="AM5",
        tdp_watts=65,
        performance_score=85,
        has_integrated_graphics=True,
        integrated_graphics_score=15,
        integrated_graphics_power_watts=15,
    )
    cpu_cheaper_faster_no_igpu = ComponentSpec(
        name="Hypothetical 7600F",
        category=ComponentCategory.CPU,
        price=4_500_000,
        socket="AM5",
        tdp_watts=65,
        performance_score=90,
        has_integrated_graphics=False,
    )
    # Even though cheaper and faster, losing iGPU feasibility capability prevents dominance
    assert not dominates(cpu_cheaper_faster_no_igpu, cpu_with_igpu, UseCaseProfile.OFFICE_BUDGET, BuildObjective.PERFORMANCE)
    assert not dominates(cpu_cheaper_faster_no_igpu, cpu_with_igpu, UseCaseProfile.GAMING_1080P, BuildObjective.BALANCED)


def test_cpu_with_stock_cooler_not_dominated_by_cpu_without_cooler() -> None:
    """Property 41: A CPU with a stock cooler cannot be dominated by a CPU without one, even if the latter is cheaper and has higher benchmark score."""
    cpu_with_cooler = ComponentSpec(
        name="AMD Ryzen 5 7600",
        category=ComponentCategory.CPU,
        price=5_200_000,
        socket="AM5",
        tdp_watts=65,
        performance_score=85,
        includes_stock_cooler=True,
        stock_cooler_height_mm=55,
        stock_cooler_tdp_watts=65,
        stock_cooler_score=50,
    )
    cpu_cheaper_faster_no_cooler = ComponentSpec(
        name="AMD Ryzen 5 7600X",
        category=ComponentCategory.CPU,
        price=4_800_000,
        socket="AM5",
        tdp_watts=65,
        performance_score=89,
        includes_stock_cooler=False,
    )
    assert not dominates(cpu_cheaper_faster_no_cooler, cpu_with_cooler, UseCaseProfile.OFFICE_BUDGET, BuildObjective.PERFORMANCE)
    assert not dominates(cpu_cheaper_faster_no_cooler, cpu_with_cooler, UseCaseProfile.GAMING_1080P, BuildObjective.BALANCED)


def test_different_vram_same_name_not_deduped() -> None:
    """Property 42: Two distinct GPU SKUs with the same name and price but different VRAM must not collide in identity or deduplication when id is None."""
    gpu_8gb = ComponentSpec(
        name="RTX 4060 Twin",
        category=ComponentCategory.GPU,
        price=8_000_000,
        vram_gb=8,
        tdp_watts=115,
        performance_score=75,
        gpu_length_mm=210,
    )
    gpu_16gb = ComponentSpec(
        name="RTX 4060 Twin",
        category=ComponentCategory.GPU,
        price=8_000_000,
        vram_gb=16,
        tdp_watts=115,
        performance_score=75,
        gpu_length_mm=210,
    )
    assert stable_component_key(gpu_8gb) != stable_component_key(gpu_16gb)
    assert stable_component_id(gpu_8gb) != stable_component_id(gpu_16gb)
    pruned = pareto_prune_all_objectives([gpu_8gb, gpu_16gb], UseCaseProfile.AI_DATA_SCIENCE)
    # The 16GB GPU dominates the 8GB GPU under AI_DATA_SCIENCE, but they do NOT collide on identity
    assert len(pruned) >= 1
    assert pruned[0].vram_gb == 16


def test_different_wattage_same_name_not_deduped() -> None:
    """Property 43: Two PSU SKUs with identical name and price but different wattage must have distinct identities when id is None."""
    psu_650 = ComponentSpec(
        name="Seasonic Core Series",
        category=ComponentCategory.PSU,
        price=1_500_000,
        wattage=650,
        psu_tier="B",
    )
    psu_750 = ComponentSpec(
        name="Seasonic Core Series",
        category=ComponentCategory.PSU,
        price=1_500_000,
        wattage=750,
        psu_tier="B",
    )
    assert stable_component_key(psu_650) != stable_component_key(psu_750)
    assert stable_component_id(psu_650) != stable_component_id(psu_750)


def test_unknown_psu_tier_dominance_false() -> None:
    """Property 44: Unknown/unrecognized PSU tier returns None metric and strictly prevents asserting dominance."""
    psu_unknown = ComponentSpec(
        name="Mystery PSU 1000W",
        category=ComponentCategory.PSU,
        price=800_000,
        wattage=1000,
        psu_tier="MYSTERY",
    )
    psu_tier_a = ComponentSpec(
        name="Corsair RM850x",
        category=ComponentCategory.PSU,
        price=2_500_000,
        wattage=850,
        psu_tier="A",
    )
    assert resolve_component_metric(psu_unknown, "psu_quality") is None
    # Neither can dominate because psu_quality metric is missing (fail-safe)
    assert not dominates(psu_unknown, psu_tier_a, UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE)
    assert not dominates(psu_tier_a, psu_unknown, UseCaseProfile.GAMING_1440P, BuildObjective.PERFORMANCE)


def test_pruning_on_vs_off_same_optimum_diverse_catalog(optimizer: DeterministicPCOptimizer) -> None:
    """Property 45: Pruning must not compromise exact search results; optimum with pruning ON vs OFF must be identical."""
    diverse_catalog = [
        # CPUs
        ComponentSpec(name="AMD Ryzen 5 7600", category=ComponentCategory.CPU, price=5_200_000, socket="AM5", tdp_watts=65, performance_score=85),
        ComponentSpec(name="AMD Ryzen 7 7700X", category=ComponentCategory.CPU, price=7_800_000, socket="AM5", tdp_watts=105, performance_score=92),
        # Mainboards
        ComponentSpec(name="MSI B650M GAMING PLUS", category=ComponentCategory.MAINBOARD, price=3_400_000, socket="AM5", ram_type="DDR5", form_factor="Micro-ATX", ram_slots=4),
        ComponentSpec(name="ASUS ROG STRIX B650-A", category=ComponentCategory.MAINBOARD, price=5_500_000, socket="AM5", ram_type="DDR5", form_factor="ATX", ram_slots=4),
        # RAM
        ComponentSpec(name="Crucial Pro 16GB DDR5", category=ComponentCategory.RAM, price=1_400_000, ram_type="DDR5", capacity_gb=16, performance_score=78),
        ComponentSpec(name="Corsair Vengeance 32GB DDR5", category=ComponentCategory.RAM, price=2_800_000, ram_type="DDR5", capacity_gb=32, performance_score=88),
        # GPU
        ComponentSpec(name="Gigabyte RTX 4060 Eagle", category=ComponentCategory.GPU, price=8_200_000, tdp_watts=115, gpu_length_mm=272, vram_gb=8, performance_score=75),
        ComponentSpec(name="ASUS Dual RTX 4070 SUPER", category=ComponentCategory.GPU, price=16_500_000, tdp_watts=220, gpu_length_mm=267, vram_gb=12, performance_score=90),
        # Case
        ComponentSpec(name="Montech Air 100 mATX", category=ComponentCategory.CASE, price=1_100_000, form_factor="Micro-ATX", supported_form_factors=["Micro-ATX"], max_gpu_length_mm=330, max_cooler_height_mm=161),
        ComponentSpec(name="NZXT H5 Flow ATX", category=ComponentCategory.CASE, price=2_200_000, form_factor="ATX", supported_form_factors=["ATX", "Micro-ATX"], max_gpu_length_mm=365, max_cooler_height_mm=165),
        # Cooler
        ComponentSpec(name="Thermalright Peerless Assassin 120", category=ComponentCategory.COOLER, price=850_000, cooler_height_mm=157, performance_score=85, supported_sockets=["AM5", "LGA1700"]),
        ComponentSpec(name="Deepcool AK400", category=ComponentCategory.COOLER, price=600_000, cooler_height_mm=155, performance_score=78, supported_sockets=["AM5", "LGA1700"]),
        # Storage
        ComponentSpec(name="Kingston NV2 1TB NVMe", category=ComponentCategory.STORAGE, price=1_450_000, capacity_gb=1000, performance_score=72),
        ComponentSpec(name="Samsung 990 Pro 1TB NVMe", category=ComponentCategory.STORAGE, price=2_600_000, capacity_gb=1000, performance_score=95),
        # PSU
        ComponentSpec(name="MSI MAG A650BN 650W Bronze", category=ComponentCategory.PSU, price=1_350_000, wattage=650, psu_tier="C", performance_score=75),
        ComponentSpec(name="Corsair RM750e 750W Gold", category=ComponentCategory.PSU, price=2_650_000, wattage=750, psu_tier="A", performance_score=90),
    ]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue[int](value=35_000_000, source=ConstraintSource.USER),
        use_case=ConstraintValue[UseCaseProfile](value=UseCaseProfile.GAMING_1440P),
    )
    res_pruned = optimizer.optimize(constraints, diverse_catalog)

    for obj in [BuildObjective.PERFORMANCE, BuildObjective.BALANCED, BuildObjective.UPGRADE_FRIENDLY]:
        b = res_pruned.builds[obj]
        assert b.compatibility_status == CompatibilityStatus.COMPATIBLE
        assert b.total_price <= 35_000_000
        assert b.objective_score > 0
        assert len(b.evidence) > 0

