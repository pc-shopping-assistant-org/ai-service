from ai_service.application.ports.hardware import (
    BuildPurpose,
    ComponentCategory,
    ComponentSpec,
)
from ai_service.capabilities.pc_builder.schemas import (
    CalculateWattageArgs,
    CheckCompatibilityArgs,
    FindAlternativesArgs,
    RecommendBuildArgs,
)
from ai_service.capabilities.pc_builder.tools import PCBuilderTools


def test_pc_builder_tools_check_compatibility() -> None:
    tools = PCBuilderTools()
    args = CheckCompatibilityArgs(
        components=[
            ComponentSpec(name="i5-13400F", category=ComponentCategory.CPU, socket="LGA1700"),
            ComponentSpec(name="B760M", category=ComponentCategory.MAINBOARD, socket="LGA1700"),
        ]
    )
    result = tools.check_pc_compatibility(args)
    assert result.is_compatible is True


def test_pc_builder_tools_calculate_wattage() -> None:
    tools = PCBuilderTools()
    args = CalculateWattageArgs(
        components=[
            ComponentSpec(name="i7-14700K", category=ComponentCategory.CPU, tdp_watts=253),
            ComponentSpec(name="RTX 4080", category=ComponentCategory.GPU, tdp_watts=320),
        ],
        selected_psu_watts=1000,
    )
    result = tools.calculate_psu_wattage(args)
    assert result.estimated_tdp_watts > 500
    assert result.is_sufficient is True


def test_pc_builder_tools_recommend_build() -> None:
    tools = PCBuilderTools()
    args = RecommendBuildArgs(budget_vnd=15_000_000, purpose=BuildPurpose.GAMING_ESPORTS)
    result = tools.recommend_pc_build(args)
    assert result.total_estimated_price <= 16_000_000
    assert len(result.parts) > 5


def test_pc_builder_tools_find_alternatives() -> None:
    tools = PCBuilderTools()
    args = FindAlternativesArgs(
        slot=ComponentCategory.MAINBOARD,
        required_socket="AM5",
        required_ram_type="DDR5",
        limit=3,
    )
    result = tools.find_compatible_alternatives(args)
    assert len(result.alternatives) > 0
    assert any("B650" in a.name for a in result.alternatives)


def test_pc_builder_tools_bottleneck() -> None:
    from ai_service.application.ports.hardware import ResolutionTier
    from ai_service.capabilities.pc_builder.schemas import AnalyzeBottleneckArgs

    tools = PCBuilderTools()
    res = tools.analyze_bottleneck_balance(
        AnalyzeBottleneckArgs(
            cpu_name="Ryzen 5 7600",
            gpu_name="GeForce RTX 4070 SUPER",
            resolution=ResolutionTier.RES_1440P,
        )
    )
    assert res.status.value == "BALANCED"


def test_pc_builder_tools_upgrade_path() -> None:
    from ai_service.capabilities.pc_builder.schemas import AssessUpgradePathArgs

    tools = PCBuilderTools()
    res = tools.assess_upgrade_path(
        AssessUpgradePathArgs(
            socket="LGA1700",
            psu_wattage=650,
            ram_type="DDR4",
        )
    )
    assert "MATURE" in res.platform_lifecycle
    assert "DDR4" in res.ram_upgradeability


def test_pc_builder_tools_peripherals() -> None:
    from ai_service.capabilities.pc_builder.schemas import RecommendPeripheralsArgs

    tools = PCBuilderTools()
    res = tools.recommend_monitor_and_peripherals(
        RecommendPeripheralsArgs(
            gpu_name="GeForce RTX 4080 SUPER",
            target_use_case="Gaming AAA Đồ Họa Đẹp",
        )
    )
    assert "4K" in res.recommended_display_resolution or "2K" in res.recommended_display_resolution
    assert len(res.display_examples) > 0
