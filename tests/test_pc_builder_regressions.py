from uuid import uuid4

import pytest

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.hardware import ComponentCategory as C
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.application import (
    PCBuildApplicationService,
    build_explanation_context,
)
from ai_service.capabilities.pc_builder.optimizer import (
    DeterministicPCOptimizer,
    check_compatibility_status,
)
from ai_service.capabilities.pc_builder.schemas import (
    CompatibilityStatus,
    ConstraintSource,
    ConstraintValue,
    OwnedComponent,
    PCBuildConstraints,
    RecommendBuildArgs,
    UseCaseProfile,
)
from ai_service.capabilities.pc_builder.tools import PCBuilderTools
from ai_service.infrastructure.hardware.rule_engine import LocalHardwareRuleEngine


def test_recommendation_without_canonical_catalog_fails_closed() -> None:
    with pytest.raises(BackendUnavailableError):
        PCBuilderTools().recommend_pc_build(
            RecommendBuildArgs(budget_vnd=5_000_000, use_case=UseCaseProfile.OFFICE_BUDGET)
        )


def test_legacy_rule_engine_cannot_return_hardcoded_build() -> None:
    with pytest.raises(BackendUnavailableError):
        LocalHardwareRuleEngine().recommend_build(budget=5_000_000)


def test_compatibility_tool_does_not_certify_missing_cooler_socket() -> None:
    from ai_service.capabilities.pc_builder.schemas import CheckCompatibilityArgs

    report = PCBuilderTools().check_pc_compatibility(CheckCompatibilityArgs(components=[
        ComponentSpec(name="CPU", category=C.CPU, socket="AM5"),
        ComponentSpec(name="Unknown cooler", category=C.COOLER),
    ]))
    assert report.is_compatible is False


def test_recommendation_entrypoint_uses_application_pipeline(realistic_catalog: list[ComponentSpec]) -> None:
    result = PCBuilderTools().recommend_pc_build(
        RecommendBuildArgs(budget_vnd=25_000_000, use_case=UseCaseProfile.OFFICE_BUDGET),
        catalog=realistic_catalog,
    )
    assert result.optimization.use_case == UseCaseProfile.OFFICE_BUDGET
    assert result.recommendation.build.total_price <= 25_000_000


def test_missing_cooler_socket_support_is_unknown() -> None:
    status = check_compatibility_status({
        C.CPU: ComponentSpec(name="CPU", category=C.CPU, socket="AM5"),
        C.COOLER: ComponentSpec(name="Unverified cooler", category=C.COOLER),
    }, 450)
    assert status == CompatibilityStatus.UNKNOWN


def test_synthetic_igpu_cannot_bypass_locked_brand(realistic_catalog: list[ComponentSpec]) -> None:
    catalog = [c for c in realistic_catalog if c.category != C.CPU or c.name == "AMD Ryzen 5 7600"]
    constraints = PCBuildConstraints(
        target_budget_vnd=ConstraintValue(value=14_000_000),
        use_case=ConstraintValue(value=UseCaseProfile.OFFICE_BUDGET),
        preferred_gpu_brand=ConstraintValue(value="NVIDIA", source=ConstraintSource.USER, locked=True),
    )
    with pytest.raises(ValueError, match="No compatible build"):
        DeterministicPCOptimizer().optimize(constraints, catalog)


@pytest.mark.parametrize("category", [C.CPU, C.GPU])
def test_missing_tdp_candidate_does_not_abort_valid_search(category: C, realistic_catalog: list[ComponentSpec]) -> None:
    invalid = next(c for c in realistic_catalog if c.category == category).model_copy(
        update={"id": uuid4(), "tdp_watts": None}
    )
    result = DeterministicPCOptimizer().optimize(
        PCBuildConstraints(target_budget_vnd=ConstraintValue(value=35_000_000)),
        [invalid, *realistic_catalog],
    )
    assert result.builds
    assert all(b.parts[category].tdp_watts is not None for b in result.builds.values())


def test_owned_part_explanation_preserves_spending(realistic_catalog: list[ComponentSpec]) -> None:
    owned = next(c for c in realistic_catalog if c.category == C.PSU)
    result = PCBuildApplicationService().build_pc(
        PCBuildConstraints(owned_parts=[OwnedComponent(category=C.PSU, component=owned)]),
        realistic_catalog,
    )
    context = build_explanation_context(result)
    assert context["parts"]["PSU"]["is_owned"] is True
    assert context["parts"]["PSU"]["spending_price"] == 0
    assert context["parts"]["PSU"]["price"] == owned.price
    assert sum(part["spending_price"] for part in context["parts"].values()) == context["total_price"]
