from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.capabilities.assistant.conversation_core import (
    ConsultationStateV1,
    apply_turn_patch,
)
from ai_service.capabilities.assistant.grounded_build import (
    AccessoryCatalogRowV1,
    prepare_build_input,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.stateful_contracts import TurnPatchV1
from ai_service.capabilities.pc_builder.application import PCBuildApplicationService
from ai_service.capabilities.pc_builder.errors import NoFeasibleBuildError
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    PCBuildConstraints,
)


def rows(catalog):
    return [
        CatalogRowV1(
            variant_id=(part_id := uuid4()),
            sku=f"SKU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=part.model_copy(update={"id": part_id}),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["backend/catalog"],
        )
        for index, part in enumerate(catalog)
    ]


def state(**patch):
    return apply_turn_patch(
        ConsultationStateV1(),
        TurnPatchV1(
            schema_version=1,
            constraints={
                "target_budget_vnd": {"op": "SET", "value": 30_000_000},
                "use_case": {"op": "SET", "value": "GAMING_1080P"},
            },
            **patch,
        ),
        source=ConstraintSource.USER,
    )


def test_unknown_requirements_clarify_without_optimizer_defaults():
    result = prepare_build_input(ConsultationStateV1(), [])
    assert set(result.clarifications) == {"BUDGET_REQUIRED", "USE_CASE_REQUIRED"}
    assert result.optimizer_input is None


def test_paid_pin_survives_pruning_and_synthetic_alternatives(realistic_catalog):
    catalog = rows(realistic_catalog)
    gpu = next(row for row in catalog if "RTX 3060" in row.component.name)
    cooler = next(
        row for row in catalog if row.component.category == ComponentCategory.COOLER
    )
    prepared = prepare_build_input(
        state(
            pinned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": gpu.variant_id}},
                "COOLER": {"op": "SET", "value": {"variant_id": cooler.variant_id}},
            }
        ),
        catalog,
    )
    assert not prepared.clarifications
    normalized = prepared.optimizer_input
    assert normalized is not None
    built = PCBuildApplicationService().build_pc(
        normalized.constraints, normalized.candidates
    )
    assert built.optimization.builds
    for build in built.optimization.builds.values():
        assert build.parts[ComponentCategory.GPU].id == gpu.variant_id
        assert build.parts[ComponentCategory.COOLER].id == cooler.variant_id
        assert build.spending_prices[ComponentCategory.GPU] == gpu.component.price
        assert ComponentCategory.GPU not in build.owned_categories


def test_owned_out_of_stock_is_free_but_paid_pin_cannot_be_substituted(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    gpu = next(
        row for row in catalog if row.component.category == ComponentCategory.GPU
    )
    gpu.quantity = 0
    prepared = prepare_build_input(
        state(
            owned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": gpu.variant_id}},
            }
        ),
        catalog,
    )
    assert prepared.optimizer_input is not None
    built = PCBuildApplicationService().build_pc(
        prepared.optimizer_input.constraints, prepared.optimizer_input.candidates
    )
    assert built.optimization.builds
    for build in built.optimization.builds.values():
        assert build.parts[ComponentCategory.GPU].price == gpu.component.price
        assert build.spending_prices[ComponentCategory.GPU] == 0
    unavailable = prepare_build_input(
        state(
            pinned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": gpu.variant_id}},
            }
        ),
        catalog,
    )
    assert "PIN_UNAVAILABLE" in unavailable.clarifications
    assert unavailable.optimizer_input is None


def test_category_mismatch_and_missing_reference_never_create_fake_components(
    realistic_catalog,
):
    catalog = rows(realistic_catalog)
    cpu = catalog[0]
    wrong = prepare_build_input(
        state(
            pinned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": cpu.variant_id}},
            }
        ),
        catalog,
    )
    assert "PART_CATEGORY_MISMATCH" in wrong.clarifications
    missing = prepare_build_input(
        state(
            owned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": uuid4()}},
            }
        ),
        catalog,
    )
    assert "PART_REFERENCE_REQUIRED" in missing.clarifications


def test_full_setup_requires_accessory_budget_and_resolution_not_implicit_allocation():
    setup = apply_turn_patch(
        state(),
        TurnPatchV1(
            schema_version=1,
            constraints={"budget_scope": {"op": "SET", "value": "FULL_SETUP"}},
            accessories={"MONITOR": {"op": "SET", "value": {"kind": "RECOMMEND"}}},
        ),
        source=ConstraintSource.USER,
    )
    prepared = prepare_build_input(setup, [])
    assert "ACCESSORY_BUDGET_REQUIRED" in prepared.clarifications
    assert prepared.optimizer_input is None

    unspecified = apply_turn_patch(
        state(),
        TurnPatchV1(
            schema_version=1,
            constraints={
                "budget_scope": {"op": "SET", "value": "FULL_SETUP"},
            },
        ),
        source=ConstraintSource.USER,
    )
    assert (
        "ACCESSORY_REQUIREMENTS_REQUIRED"
        in prepare_build_input(unspecified, []).clarifications
    )


def test_core_constraint_rejects_owned_and_pinned_on_same_slot(realistic_catalog):
    gpu = next(p for p in realistic_catalog if p.category == ComponentCategory.GPU)
    with pytest.raises(ValidationError):
        PCBuildConstraints(
            pinned_parts={"GPU": uuid4()},
            owned_parts=[
                {
                    "category": "GPU",
                    "component": gpu,
                }
            ],
        )


def test_resolved_full_setup_counts_paid_accessories_but_not_owned(realistic_catalog):
    monitor_id, mouse_id = uuid4(), uuid4()
    setup = apply_turn_patch(
        state(),
        TurnPatchV1(
            schema_version=1,
            constraints={"budget_scope": {"op": "SET", "value": "FULL_SETUP"}},
            accessories={
                "MONITOR": {
                    "op": "SET",
                    "value": {"kind": "PINNED", "variant_id": monitor_id},
                },
                "MOUSE": {
                    "op": "SET",
                    "value": {"kind": "OWNED", "variant_id": mouse_id},
                },
            },
        ),
        source=ConstraintSource.USER,
    )
    accessories = [
        AccessoryCatalogRowV1(
            variant_id=variant_id,
            accessory_type=kind,
            sku=kind,
            name=kind,
            price_vnd=price,
            quantity=stock,
            product_active=True,
            variant_active=True,
            retrieved_at=datetime.now(UTC),
            evidence_refs=["catalog"],
        )
        for variant_id, kind, price, stock in [
            (monitor_id, "MONITOR", 3_000_000, 1),
            (mouse_id, "MOUSE", 500_000, 0),
        ]
    ]
    prepared = prepare_build_input(setup, rows(realistic_catalog), accessories)
    assert prepared.optimizer_input is not None
    assert prepared.accessory_spending_vnd == 3_000_000
    assert prepared.optimizer_input.constraints.target_budget_vnd.value == 27_000_000
    result = PCBuildApplicationService().build_pc(
        prepared.optimizer_input.constraints, prepared.optimizer_input.candidates
    )
    assert result.optimization.builds
    assert all(
        build.total_price + prepared.accessory_spending_vnd <= 30_000_000
        for build in result.optimization.builds.values()
    )
    accessories[0].price_vnd = 31_000_000
    rejected = prepare_build_input(setup, rows(realistic_catalog), accessories)
    assert "ACCESSORY_BUDGET_EXCEEDED" in rejected.clarifications
    assert rejected.optimizer_input is None


def test_missing_hard_pin_is_infeasible_not_synthetic_or_other_sku(realistic_catalog):
    constraints = PCBuildConstraints(pinned_parts={"GPU": uuid4()})
    with pytest.raises(
        NoFeasibleBuildError, match="No compatible build configuration"
    ) as failure:
        PCBuildApplicationService().build_pc(constraints, realistic_catalog)
    assert failure.value.result.builds == {}
    assert failure.value.result.target_budget_vnd == constraints.target_budget_vnd.value
    assert any(
        "pinned" in reason for reason in failure.value.result.rejected_candidates
    )
