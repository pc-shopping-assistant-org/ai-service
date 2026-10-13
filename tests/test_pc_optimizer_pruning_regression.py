"""Case compatibility and pruning require explicit motherboard support."""

import pytest

from ai_service.application.ports.hardware import ComponentCategory as C
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.errors import (
    CaseSupportMetadataError,
    NoFeasibleBuildError,
)
from ai_service.capabilities.pc_builder.optimizer import (
    DeterministicPCOptimizer,
    check_compatibility_status,
)
from ai_service.capabilities.pc_builder.pruning import dominates
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    CompatibilityStatus,
    PCBuildConstraints,
    UseCaseProfile,
)
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        ([], CompatibilityStatus.UNKNOWN),
        ([""], CompatibilityStatus.UNKNOWN),
        (["ATX"], CompatibilityStatus.INCOMPATIBLE),
        (["ATX", "micro-atx"], CompatibilityStatus.COMPATIBLE),
    ],
)
def test_case_support_is_explicit(
    supported: list[str], expected: CompatibilityStatus,
) -> None:
    parts = {
        C.MAINBOARD: ComponentSpec(name="Micro board", category=C.MAINBOARD, form_factor="Micro-ATX"),
        C.CASE: ComponentSpec(name="ATX case", category=C.CASE, form_factor="ATX", supported_form_factors=supported),
    }
    assert check_compatibility_status(parts, min_psu_watts=0) == expected
    report = LocalHardwareRuleEngine().check_compatibility(list(parts.values()))
    assert report.is_compatible == (expected == CompatibilityStatus.COMPATIBLE)
    if expected == CompatibilityStatus.UNKNOWN:
        assert any(issue.category == "FORM_FACTOR_UNKNOWN" for issue in report.issues)


def test_case_without_support_cannot_dominate_verified_case() -> None:
    unknown = ComponentSpec(
        name="Unknown ATX case", category=C.CASE, price=900_000,
        form_factor="ATX", max_gpu_length_mm=400, max_cooler_height_mm=180,
    )
    verified = ComponentSpec(
        name="Verified ATX case", category=C.CASE, price=1_200_000,
        form_factor="ATX", supported_form_factors=["ATX"],
        max_gpu_length_mm=330, max_cooler_height_mm=161,
    )
    for objective in BuildObjective:
        assert not dominates(unknown, verified, UseCaseProfile.GAMING_1080P, objective)
        assert not dominates(verified, unknown, UseCaseProfile.GAMING_1080P, objective)


def test_pruning_preserves_builds_with_verified_case_support(
    realistic_catalog: list[ComponentSpec],
) -> None:
    # Keep the original A/B case geometry and prices, but require verified
    # Micro-ATX support instead of the removed rank fallback.
    catalog = [
        part
        for part in realistic_catalog
        if part.category != C.CASE
        and not (part.category == C.MAINBOARD and part.form_factor == "ATX")
    ]
    catalog.extend([
        ComponentSpec(
            name="Verified Micro-ATX case",
            category=C.CASE,
            price=1_200_000,
            form_factor="ATX",
            supported_form_factors=["ATX", "Micro-ATX"],
            max_gpu_length_mm=330,
            max_cooler_height_mm=161,
        ),
        ComponentSpec(
            name="Explicit ATX-only case",
            category=C.CASE,
            price=1_100_000,
            form_factor="ATX",
            supported_form_factors=["ATX"],
            max_gpu_length_mm=340,
            max_cooler_height_mm=165,
        ),
    ])
    constraints = PCBuildConstraints()

    exhaustive = DeterministicPCOptimizer(enable_pruning=False).optimize(constraints, catalog)
    assert set(exhaustive.builds) == set(BuildObjective)
    for build in exhaustive.builds.values():
        assert build.compatibility_status == CompatibilityStatus.COMPATIBLE
        assert build.parts[C.MAINBOARD].form_factor == "Micro-ATX"
        assert build.parts[C.CASE].name == "Verified Micro-ATX case"
        assert build.total_price <= constraints.target_budget_vnd.value

    pruned = DeterministicPCOptimizer(enable_pruning=True).optimize(constraints, catalog)

    assert set(pruned.builds) == set(exhaustive.builds)
    for objective, expected in exhaustive.builds.items():
        actual = pruned.builds[objective]
        assert actual.objective_score == expected.objective_score
        assert actual.total_price == expected.total_price
        assert actual.parts == expected.parts


@pytest.mark.parametrize("pruning", [False, True])
def test_missing_case_support_is_metadata_error_not_infeasible(
    realistic_catalog: list[ComponentSpec], pruning: bool,
) -> None:
    catalog = [
        part.model_copy(update={"supported_form_factors": []})
        if part.category == C.CASE else part
        for part in realistic_catalog
    ]
    with pytest.raises(CaseSupportMetadataError, match="CASE_SUPPORT_METADATA_REQUIRED") as failure:
        DeterministicPCOptimizer(enable_pruning=pruning).optimize(PCBuildConstraints(), catalog)
    assert not isinstance(failure.value, NoFeasibleBuildError)


@pytest.mark.parametrize("pruning", [False, True])
def test_unknown_case_does_not_block_verified_build(
    realistic_catalog: list[ComponentSpec], pruning: bool,
) -> None:
    unknown = next(part for part in realistic_catalog if part.category == C.CASE).model_copy(
        update={"id": None, "name": "Unknown support case", "price": 1, "supported_form_factors": []},
    )
    result = DeterministicPCOptimizer(enable_pruning=pruning).optimize(
        PCBuildConstraints(), [*realistic_catalog, unknown],
    )
    assert result.builds
    assert all(build.parts[C.CASE].name != unknown.name for build in result.builds.values())
    assert "Unknown support case: CASE_SUPPORT_METADATA_REQUIRED" in result.rejected_candidates
