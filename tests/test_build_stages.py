from datetime import UTC, datetime
from uuid import uuid4

import pytest

from ai_service.capabilities.assistant.build_presenter import to_public_build
from ai_service.capabilities.assistant.build_stages import (
    ExplanationConfigV1,
    explain_build_stage,
    optimize_build_stage,
)
from ai_service.capabilities.assistant.conversation_core import (
    ConsultationStateV1,
    StageStatus,
    StageVersionsV1,
    apply_turn_patch,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.stateful_contracts import TurnPatchV1
from ai_service.capabilities.pc_builder.application import PCBuildApplicationService
from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer
from ai_service.capabilities.pc_builder.schemas import ConstraintSource

GENERATION = ExplanationConfigV1(
    provider="test", model="deterministic", model_revision="v1"
)


class CountingBuildService(PCBuildApplicationService):
    calls = 0

    def build_pc(self, *args, **kwargs):
        self.calls += 1
        return super().build_pc(*args, **kwargs)


class RecoveringExplainer:
    calls = 0
    broken = True

    async def generate(self, prompt, fallback):
        self.calls += 1
        assert "spending_price" in prompt
        if self.broken:
            raise TimeoutError("transient explanation failure")
        return fallback


def inputs(catalog, budget=30_000_000):
    state = apply_turn_patch(
        ConsultationStateV1(),
        TurnPatchV1(
            schema_version=1,
            constraints={
                "target_budget_vnd": {"op": "SET", "value": budget},
                "use_case": {"op": "SET", "value": "GAMING_1080P"},
            },
        ),
        source=ConstraintSource.USER,
    )
    rows = [
        CatalogRowV1(
            variant_id=(identity := uuid4()),
            sku=f"SKU-{index}",
            product_status="ACTIVE",
            variant_status="ACTIVE",
            quantity=1,
            component=part.model_copy(update={"id": identity}),
            retrieved_at=datetime.now(UTC),
            evidence_refs=["backend/catalog"],
        )
        for index, part in enumerate(catalog)
    ]
    versions = StageVersionsV1(graph="g1", implementation="core1", policy="p1")
    return state, rows, versions


@pytest.mark.asyncio
async def test_explain_failure_and_restart_reuse_committed_optimization(
    realistic_catalog,
):
    state, catalog, versions = inputs(realistic_catalog)
    before = state.model_dump_json()
    service = CountingBuildService()
    optimized = optimize_build_stage(state, catalog, versions, service=service)
    assert optimized.state.optimization.status == StageStatus.COMPLETED
    explainer = RecoveringExplainer()
    with pytest.raises(TimeoutError):
        await explain_build_stage(
            optimized, versions, explainer, generation_config=GENERATION
        )
    restored = ConsultationStateV1.model_validate_json(
        optimized.state.model_dump_json()
    )
    assert restored.optimization.status == StageStatus.COMPLETED
    assert restored.explanation.status == StageStatus.NOT_STARTED
    resumed = optimize_build_stage(restored, catalog, versions, service=service)
    assert service.calls == 1
    explainer.broken = False
    explained = await explain_build_stage(
        resumed, versions, explainer, generation_config=GENERATION
    )
    assert explained.explanation.status == StageStatus.COMPLETED
    rerun = optimize_build_stage(explained, catalog, versions, service=service)
    assert (
        await explain_build_stage(
            rerun, versions, explainer, generation_config=GENERATION
        )
        == explained
    )
    assert (
        explainer.calls == 2
    )  # Failed call + one successful call, not another LLM call.
    await explain_build_stage(
        rerun,
        versions,
        explainer,
        generation_config=GENERATION.model_copy(update={"model_revision": "v2"}),
    )
    assert explainer.calls == 3
    assert service.calls == 1
    assert state.model_dump_json() == before


def test_catalog_semantics_or_actual_policy_change_invalidates_reuse(realistic_catalog):
    state, catalog, versions = inputs(realistic_catalog)
    optimizer = DeterministicPCOptimizer()
    service = CountingBuildService(optimizer)
    optimized = optimize_build_stage(state, catalog, versions, service=service)
    catalog[0].retrieved_at = datetime.now(UTC)
    assert (
        optimize_build_stage(optimized.state, catalog, versions, service=service).state
        == optimized.state
    )
    assert service.calls == 1
    catalog[0].component.price += 10_000
    changed = optimize_build_stage(optimized.state, catalog, versions, service=service)
    assert service.calls == 2
    assert (
        changed.state.optimization.input_hash != optimized.state.optimization.input_hash
    )
    changed_versions = versions.model_copy(update={"implementation": "core2"})
    optimize_build_stage(changed.state, catalog, changed_versions, service=service)
    assert service.calls == 3
    optimizer.enable_pruning = False
    optimize_build_stage(changed.state, catalog, versions, service=service)
    assert service.calls == 4  # Same version labels, different actual engine config.


@pytest.mark.asyncio
async def test_infeasible_is_completed_but_unexpected_value_error_is_not(
    realistic_catalog,
):
    state, catalog, versions = inputs(realistic_catalog, budget=1)
    stage = optimize_build_stage(state, catalog, versions)
    assert stage.state.optimization.result.outcome == "INFEASIBLE"
    assert stage.state.optimization.status == StageStatus.COMPLETED
    explained = await explain_build_stage(
        stage, versions, RecoveringExplainer(), generation_config=GENERATION
    )
    assert explained.explanation.status == StageStatus.COMPLETED
    assert "ngân sách" in explained.explanation.result

    class BrokenService(PCBuildApplicationService):
        def build_pc(self, *args, **kwargs):
            raise ValueError("unexpected bug")

    original = state.model_dump_json()
    with pytest.raises(ValueError, match="unexpected bug"):
        optimize_build_stage(state, catalog, versions, service=BrokenService())
    assert state.model_dump_json() == original


@pytest.mark.asyncio
async def test_empty_explanation_is_not_a_completed_success(realistic_catalog):
    state, catalog, versions = inputs(realistic_catalog)
    stage = optimize_build_stage(state, catalog, versions)

    class EmptyExplainer:
        async def generate(self, prompt, fallback):
            return "   "

    with pytest.raises(ValueError):
        await explain_build_stage(
            stage, versions, EmptyExplainer(), generation_config=GENERATION
        )
    assert stage.state.explanation.status == StageStatus.NOT_STARTED
    assert stage.state.optimization.status == StageStatus.COMPLETED


def test_public_build_uses_real_skus_exact_spending_and_labeled_synthetic_parts(
    realistic_catalog,
):
    state, catalog, versions = inputs(realistic_catalog)
    gpu = next(row for row in catalog if row.component.category.value == "GPU")
    state = apply_turn_patch(
        state,
        TurnPatchV1(
            schema_version=1,
            owned_parts={
                "GPU": {"op": "SET", "value": {"variant_id": gpu.variant_id}},
            },
        ),
        source=ConstraintSource.USER,
    )
    stage = optimize_build_stage(state, catalog, versions)
    view = to_public_build(stage)
    assert view is not None
    assert view.total_spending_vnd == sum(part.spending_vnd for part in view.parts)
    owned = next(part for part in view.parts if part.slot.value == "GPU")
    assert (
        owned.variant_id == gpu.variant_id
        and owned.is_owned
        and owned.spending_vnd == 0
    )
    assert all(
        part.variant_id in {row.variant_id for row in catalog}
        or part.synthetic_kind is not None
        for part in view.parts
    )
    assert "input_hash" not in view.model_dump_json()


def test_infeasible_and_clarification_do_not_expose_a_fake_current_build(
    realistic_catalog,
):
    state, catalog, versions = inputs(realistic_catalog, budget=1)
    assert to_public_build(optimize_build_stage(state, catalog, versions)) is None
    assert (
        to_public_build(optimize_build_stage(ConsultationStateV1(), catalog, versions))
        is None
    )


def test_generated_igpu_and_stock_cooler_ids_are_not_exposed_as_catalog_skus(
    realistic_catalog,
):
    state, catalog, versions = inputs(realistic_catalog, budget=13_000_000)
    state = apply_turn_patch(
        state,
        TurnPatchV1(
            schema_version=1,
            constraints={"use_case": {"op": "SET", "value": "OFFICE_BUDGET"}},
        ),
        source=ConstraintSource.USER,
    )
    view = to_public_build(optimize_build_stage(state, catalog, versions))
    assert view is not None
    synthetic = [part for part in view.parts if part.synthetic_kind is not None]
    assert {part.synthetic_kind for part in synthetic} == {
        "INTEGRATED_GPU",
        "STOCK_COOLER",
    }
    assert all(part.variant_id is None and part.spending_vnd == 0 for part in synthetic)
