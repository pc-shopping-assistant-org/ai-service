"""Versioned power policy: iGPU has no discrete-GPU transient allowance."""

import pytest

from ai_service.application.ports.hardware import ComponentCategory as C
from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.optimizer import calculate_power_estimate
from ai_service.capabilities.pc_builder.schemas import UseCaseProfile


@pytest.mark.parametrize(
    ("integrated", "gpu_watts", "transient"),
    [(True, 15, 0.0), (False, 15, 1.8), (False, 179, 21.5),
     (False, 180, 36.0), (False, 279, 55.8), (False, 280, 84.0)],
)
def test_transient_policy_distinguishes_igpu_and_discrete(
    integrated: bool, gpu_watts: int, transient: float,
) -> None:
    parts = {
        C.CPU: ComponentSpec(name="CPU", category=C.CPU, price=0, tdp_watts=65),
        C.GPU: ComponentSpec(
            name="GPU", category=C.GPU, price=0, tdp_watts=gpu_watts,
            is_integrated=integrated,
        ),
    }
    estimate = calculate_power_estimate(parts, UseCaseProfile.OFFICE_BUDGET)
    assert estimate.transient_allowance_watts == pytest.approx(transient)
    assert estimate.estimated_sustained_peak_watts == pytest.approx(71.5 + gpu_watts + 65)


def test_power_policy_version_is_in_reuse_configuration() -> None:
    snapshot = PCBuildApplicationService().configuration_snapshot()
    assert isinstance(snapshot, dict)
    assert snapshot["power_sizing_version"] == "eng-power-sizing-v1"
