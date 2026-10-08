from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.conversation_core import (
    ConsultationStateV1,
    apply_turn_patch,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    AccessoryRequestV1,
    BudgetScope,
    OwnedPartRefV1,
    TurnPatchV1,
)
from ai_service.capabilities.pc_builder.schemas import ConstraintSource


def test_owned_cannot_count_as_a_purchase_and_accessory_quantity_is_one():
    assert OwnedPartRefV1(variant_id=uuid4()).exclude_from_budget is True
    with pytest.raises(ValidationError):
        OwnedPartRefV1(variant_id=uuid4(), exclude_from_budget=False)
    with pytest.raises(ValidationError):
        OwnedPartRefV1(variant_id=uuid4(), exclude_from_budget=1)
    assert AccessoryRequestV1(kind="RECOMMEND").quantity == 1
    for body in [
        {"kind": "RECOMMEND", "quantity": 2},
        {"kind": "PINNED"},
        {"kind": "OWNED", "variant_id": uuid4(), "price": 500_000},
    ]:
        with pytest.raises(ValidationError):
            AccessoryRequestV1.model_validate(body)


def test_full_setup_patch_preserves_unknown_accessory_budget_and_clears_sparsely():
    initial = ConsultationStateV1()
    assert initial.constraints.budget_scope.value == BudgetScope.BUILD_PC
    patch = TurnPatchV1.model_validate(
        {
            "schema_version": 1,
            "constraints": {
                "budget_scope": {"op": "SET", "value": "FULL_SETUP"},
                "target_budget_vnd": {"op": "SET", "value": 25_000_000},
            },
            "accessories": {"MONITOR": {"op": "SET", "value": {"kind": "RECOMMEND"}}},
        }
    )
    full = apply_turn_patch(initial, patch, source=ConstraintSource.USER)
    assert full.constraints.accessory_budget_vnd.value is None  # No 80/20 default.
    assert full.accessories["MONITOR"].quantity == 1
    assert full.constraints_revision == 1
    cleared = apply_turn_patch(
        full,
        TurnPatchV1.model_validate(
            {
                "schema_version": 1,
                "accessories": {"MONITOR": {"op": "CLEAR"}},
            }
        ),
        source=ConstraintSource.USER,
    )
    assert cleared.constraints.target_budget_vnd.value == 25_000_000
    assert not cleared.accessories
    assert ConsultationStateV1.model_validate_json(full.model_dump_json()) == full


def test_accessory_binding_cannot_be_invented_by_inference():
    patch = TurnPatchV1.model_validate(
        {
            "schema_version": 1,
            "accessories": {
                "KEYBOARD": {
                    "op": "SET",
                    "value": {
                        "kind": "OWNED",
                        "variant_id": uuid4(),
                    },
                }
            },
        }
    )
    with pytest.raises(ValueError):
        apply_turn_patch(ConsultationStateV1(), patch, source=ConstraintSource.INFERRED)


def test_accessory_cap_cannot_exceed_total_and_invalid_turn_does_not_mutate_state():
    state = ConsultationStateV1()
    patch = TurnPatchV1.model_validate(
        {
            "schema_version": 1,
            "constraints": {
                "budget_scope": {"op": "SET", "value": "FULL_SETUP"},
                "target_budget_vnd": {"op": "SET", "value": 10_000_000},
                "accessory_budget_vnd": {"op": "SET", "value": 12_000_000},
            },
        }
    )
    with pytest.raises(ValueError):
        apply_turn_patch(state, patch, source=ConstraintSource.USER)
    assert state.constraints.target_budget_vnd.value is None
