import pytest

from ai_service.application.ports.hardware import (
    BuildPurpose,
    ComponentCategory,
    ComponentSpec,
)
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine


@pytest.fixture
def engine() -> LocalHardwareRuleEngine:
    return LocalHardwareRuleEngine()


def test_socket_matching_success(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="Intel Core i5-13400F", category=ComponentCategory.CPU, socket="LGA1700"),
        ComponentSpec(name="MSI B760M GAMING", category=ComponentCategory.MAINBOARD, socket="LGA-1700"),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is True
    assert report.errors_count == 0


def test_socket_mismatch_fails(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="Intel Core i5-13400F", category=ComponentCategory.CPU, socket="LGA1700"),
        ComponentSpec(name="MSI B650M AM5", category=ComponentCategory.MAINBOARD, socket="AM5"),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is False
    assert report.errors_count == 1
    assert any(i.category == "SOCKET_MISMATCH" for i in report.issues)


def test_ram_type_mismatch_fails(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="MSI B760M DDR4", category=ComponentCategory.MAINBOARD, ram_type="DDR4"),
        ComponentSpec(name="Corsair Vengeance 32GB DDR5", category=ComponentCategory.RAM, ram_type="DDR5"),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is False
    assert any(i.category == "RAM_TYPE_MISMATCH" for i in report.issues)


def test_form_factor_mismatch_fails(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="ASUS ROG MAXIMUS ATX", category=ComponentCategory.MAINBOARD, form_factor="ATX"),
        ComponentSpec(name="Cooler Master NR200 ITX", category=ComponentCategory.CASE, supported_form_factors=["Mini-ITX"]),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is False
    assert any(i.category == "FORM_FACTOR_MISMATCH" for i in report.issues)


def test_gpu_clearance_fails(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="RTX 4090 3 Fans", category=ComponentCategory.GPU, gpu_length_mm=340),
        ComponentSpec(name="Mini Case", category=ComponentCategory.CASE, max_gpu_length_mm=300),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is False
    assert any(i.category == "GPU_CLEARANCE" for i in report.issues)


def test_cooler_clearance_fails(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="Noctua NH-D15", category=ComponentCategory.COOLER, cooler_height_mm=165),
        ComponentSpec(name="Slim Case", category=ComponentCategory.CASE, max_cooler_height_mm=150),
    ]
    report = engine.check_compatibility(components)
    assert report.is_compatible is False
    assert any(i.category == "COOLER_CLEARANCE" for i in report.issues)


def test_wattage_calculation(engine: LocalHardwareRuleEngine) -> None:
    components = [
        ComponentSpec(name="Intel Core i5-13600K", category=ComponentCategory.CPU, tdp_watts=125),
        ComponentSpec(name="RTX 4070 SUPER", category=ComponentCategory.GPU, tdp_watts=220),
    ]
    report = engine.calculate_wattage(components, selected_psu_watts=500)
    # 125 + 220 + 70 = 415W. Recommended = 415 * 1.35 ~ 560 -> 650W.
    assert report.estimated_tdp_watts == 415
    assert report.recommended_psu_watts >= 650
    assert report.is_sufficient is False  # 500W < 415*1.15


def test_recommend_build(engine: LocalHardwareRuleEngine) -> None:
    from ai_service.application.errors import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        engine.recommend_build(budget=20_000_000, purpose=BuildPurpose.GAMING_AAA)


def test_analyze_bottleneck_balanced(engine: LocalHardwareRuleEngine) -> None:
    report = engine.analyze_bottleneck(
        cpu_name="Intel Core i5-12400F",
        gpu_name="GeForce RTX 4060",
    )
    assert report.status.value == "BALANCED"
    assert report.bottleneck_percentage <= 8.0


def test_analyze_bottleneck_cpu_bound(engine: LocalHardwareRuleEngine) -> None:
    from ai_service.application.ports.hardware import ResolutionTier

    report = engine.analyze_bottleneck(
        cpu_name="Core i3-12100F",
        gpu_name="GeForce RTX 4080",
        resolution=ResolutionTier.RES_1080P,
    )
    assert report.status.value == "CPU_BOTTLENECK"
    assert report.bottleneck_percentage > 15.0


def test_assess_upgrade_path_am5(engine: LocalHardwareRuleEngine) -> None:
    report = engine.assess_upgrade_path(
        socket="AM5",
        psu_wattage=750,
        ram_type="DDR5",
    )
    assert "ACTIVE" in report.platform_lifecycle
    assert any("7800X3D" in opt for opt in report.cpu_upgrade_options)
    assert "DDR5" in report.ram_upgradeability


def test_recommend_peripherals(engine: LocalHardwareRuleEngine) -> None:
    report = engine.recommend_peripherals(
        gpu_name="GeForce RTX 4070 SUPER",
        target_use_case="Gaming AAA",
    )
    assert "2K" in report.recommended_display_resolution
    assert len(report.display_examples) > 0
    assert len(report.recommended_peripherals) > 0
