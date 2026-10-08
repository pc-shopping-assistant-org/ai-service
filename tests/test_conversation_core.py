from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.capabilities.assistant.conversation_core import (
    CompletedOptimizationV1,
    CompletionMarkerV1,
    ConsultationStateV1,
    MergeConflict,
    SetupBudgetV1,
    StageStatus,
    StageVersionsV1,
    apply_turn_patch,
)
from ai_service.capabilities.assistant.stateful_contracts import TurnPatchV1
from ai_service.capabilities.pc_builder.application import PCBuildApplicationService
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    ConstraintValue,
    OptimizationResult,
    PCBuildConstraints,
    UseCaseProfile,
)


def apply(state, **patch):
    return apply_turn_patch(
        state, TurnPatchV1(schema_version=1, **patch), source=ConstraintSource.USER
    )


def test_multi_turn_preserves_budget_clear_is_unknown_and_restore_is_lossless():
    initial = ConsultationStateV1()
    assert initial.schema_version == 1
    assert initial.constraints.target_budget_vnd.value is None
    assert initial.constraints.use_case.value is None
    first = apply(
        initial, constraints={"target_budget_vnd": {"op": "SET", "value": 25_000_000}}
    )
    second = apply(
        first, constraints={"use_case": {"op": "SET", "value": "GAMING_1440P"}}
    )
    assert second.constraints.target_budget_vnd.value == 25_000_000
    assert second.constraints_revision == 2
    cleared = apply(second, constraints={"use_case": {"op": "CLEAR"}})
    assert cleared.constraints.use_case.value is None
    assert cleared.constraints.target_budget_vnd.value == 25_000_000
    assert ConsultationStateV1.model_validate_json(cleared.model_dump_json()) == cleared
    assert initial.constraints_revision == 0  # Never mutate accepted input.


def test_owned_to_pinned_switch_is_atomic_and_overlap_rejects_without_partial_merge():
    variant = uuid4()
    first = apply(
        ConsultationStateV1(),
        owned_parts={"GPU": {"op": "SET", "value": {"variant_id": variant}}},
    )
    before = first.model_dump_json()
    with pytest.raises(MergeConflict):
        apply(
            first,
            constraints={"target_budget_vnd": {"op": "SET", "value": 30_000_000}},
            pinned_parts={"GPU": {"op": "SET", "value": {"variant_id": variant}}},
        )
    assert first.model_dump_json() == before
    switched = apply(
        first,
        owned_parts={"GPU": {"op": "CLEAR"}},
        pinned_parts={"GPU": {"op": "SET", "value": {"variant_id": variant}}},
    )
    assert not switched.owned_parts
    assert switched.pinned_parts["GPU"].variant_id == variant


def test_user_locked_change_needs_confirmation_and_system_cannot_be_overridden():
    state = ConsultationStateV1()
    state.constraints.preferred_gpu_brand = ConstraintValue[str | None](
        value="NVIDIA", source=ConstraintSource.USER, locked=True
    )
    patch = TurnPatchV1(
        schema_version=1, constraints={"preferred_gpu_brand": {"op": "CLEAR"}}
    )
    with pytest.raises(MergeConflict):
        apply_turn_patch(state, patch, source=ConstraintSource.USER)
    cleared = apply_turn_patch(
        state,
        patch,
        source=ConstraintSource.USER,
        confirmed_fields=frozenset({"preferred_gpu_brand"}),
    )
    assert cleared.constraints.preferred_gpu_brand.value is None
    assert cleared.constraints.preferred_gpu_brand.locked is True
    state.constraints.preferred_gpu_brand.source = ConstraintSource.SYSTEM
    with pytest.raises(MergeConflict):
        apply_turn_patch(
            state,
            patch,
            source=ConstraintSource.USER,
            confirmed_fields=frozenset({"preferred_gpu_brand"}),
        )


def test_inference_cannot_overwrite_explicit_user_intent_or_invent_ownership():
    state = apply(
        ConsultationStateV1(),
        constraints={"target_budget_vnd": {"op": "SET", "value": 25_000_000}},
    )
    patch = TurnPatchV1(
        schema_version=1,
        constraints={"target_budget_vnd": {"op": "SET", "value": 30_000_000}},
    )
    with pytest.raises(MergeConflict):
        apply_turn_patch(state, patch, source=ConstraintSource.INFERRED)
    owned = TurnPatchV1(
        schema_version=1,
        owned_parts={"GPU": {"op": "SET", "value": {"variant_id": uuid4()}}},
    )
    with pytest.raises(MergeConflict):
        apply_turn_patch(state, owned, source=ConstraintSource.INFERRED)


def test_completed_work_requires_hash_result_versions_and_only_matching_input_reuses():
    versions = StageVersionsV1(
        graph="g1", implementation="optimizer1", policy="policy1"
    )
    marker = CompletionMarkerV1[str](
        status=StageStatus.COMPLETED,
        input_hash="a" * 64,
        result="saved",
        versions=versions,
    )
    assert marker.can_reuse("a" * 64, versions)
    assert not marker.can_reuse("b" * 64, versions)
    assert not marker.can_reuse(
        "a" * 64,
        StageVersionsV1(graph="g2", implementation="optimizer1", policy="policy1"),
    )
    with pytest.raises(ValidationError):
        CompletionMarkerV1[str](status=StageStatus.COMPLETED, result="saved")
    assert not CompletionMarkerV1[str]().can_reuse("a" * 64, versions)


def test_noop_keeps_completed_markers_but_changed_requirements_invalidate():
    state = apply(
        ConsultationStateV1(),
        constraints={"target_budget_vnd": {"op": "SET", "value": 25_000_000}},
    )
    state.explanation = CompletionMarkerV1[str](
        status=StageStatus.COMPLETED,
        input_hash="a" * 64,
        result="saved answer",
        versions=StageVersionsV1(graph="g1", implementation="explain1", policy="p1"),
    )
    same = apply(
        state, constraints={"target_budget_vnd": {"op": "SET", "value": 25_000_000}}
    )
    assert same.constraints_revision == state.constraints_revision
    assert same.explanation == state.explanation
    changed = apply(
        state, constraints={"target_budget_vnd": {"op": "SET", "value": 30_000_000}}
    )
    assert changed.explanation.status == StageStatus.NOT_STARTED
    assert state.explanation.result == "saved answer"


def test_total_setup_cap_reserves_resolved_accessory_spending_not_all_budget_for_core():
    budget = SetupBudgetV1(
        total_budget_vnd=25_000_000, accessory_spending_vnd=3_000_000
    )
    assert budget.core_budget_vnd == 22_000_000
    with pytest.raises(ValidationError):
        SetupBudgetV1(total_budget_vnd=2_000_000, accessory_spending_vnd=3_000_000)


def test_changed_requirements_keep_real_historical_build_but_invalidate_work(
    realistic_catalog,
):
    optimization = (
        PCBuildApplicationService()
        .build_pc(PCBuildConstraints(), realistic_catalog)
        .optimization
    )
    assert optimization.builds
    state = ConsultationStateV1(
        last_successful_build=optimization,
        optimization=CompletionMarkerV1[CompletedOptimizationV1](
            status=StageStatus.COMPLETED,
            input_hash="a" * 64,
            result=CompletedOptimizationV1(outcome="BUILT", result=optimization),
            versions=StageVersionsV1(
                graph="g1", implementation="optimizer1", policy="p1"
            ),
        ),
    )
    changed = apply(
        state, constraints={"target_budget_vnd": {"op": "SET", "value": 30_000_000}}
    )
    assert changed.optimization.status == StageStatus.NOT_STARTED
    assert changed.last_successful_build == optimization
    assert changed.build_stale is True
    assert state.build_stale is False


def test_infeasible_is_completed_computation_not_successful_build_or_retry_failure():
    result = OptimizationResult(
        target_budget_vnd=25_000_000,
        use_case=UseCaseProfile.GAMING_1440P,
        builds={},
        candidates_evaluated=0,
        pruned_count=0,
    )
    outcome = CompletedOptimizationV1(outcome="INFEASIBLE", result=result)
    marker = CompletionMarkerV1[CompletedOptimizationV1](
        status=StageStatus.COMPLETED,
        input_hash="a" * 64,
        result=outcome,
        versions=StageVersionsV1(graph="g1", implementation="optimizer1", policy="p1"),
    )
    assert marker.can_reuse("a" * 64, marker.versions)
    with pytest.raises(ValidationError):
        CompletedOptimizationV1(outcome="BUILT", result=result)
