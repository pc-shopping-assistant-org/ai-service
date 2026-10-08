from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.assistant.conversation_core import (
    CompletedOptimizationV1,
    ConsultationStateV1,
    apply_turn_patch,
)
from ai_service.capabilities.assistant.provenance import (
    CatalogRowV1,
    OptimizerInputV1,
    OptimizerPolicySnapshotV1,
    OptimizerVersionsV1,
    RecommendationSnapshotV1,
    ReplaySnapshotV1,
    canonical_hash,
)
from ai_service.capabilities.assistant.stateful_contracts import TurnPatchV1
from ai_service.capabilities.pc_builder.application import PCBuildApplicationService
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    PCBuildConstraints,
)


def test_canonical_hash_is_order_stable_but_candidate_order_and_prices_matter():
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})
    assert (
        canonical_hash({"a": 1})
        == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    )
    assert canonical_hash([1, 2]) != canonical_hash([2, 1])
    assert canonical_hash({"price": 1}) != canonical_hash({"price": 2})
    with pytest.raises(ValueError):
        canonical_hash({"score": float("nan")})


def test_replay_input_requires_explicit_budget_usecase_and_canonical_catalog():
    with pytest.raises(ValidationError):
        OptimizerInputV1(constraints=PCBuildConstraints(), candidates=[])
    variant = uuid4()
    spec = ComponentSpec(id=variant, name="CPU", category="CPU", price=2_000_000)
    row = CatalogRowV1(
        variant_id=variant,
        sku="CPU-1",
        component=spec,
        quantity=1,
        retrieved_at=datetime.now(UTC),
        evidence_refs=["backend:CPU-1"],
        product_status="ACTIVE",
        variant_status="ACTIVE",
    )
    assert row.variant_id == row.component.id
    with pytest.raises(ValidationError):
        CatalogRowV1.model_validate({**row.model_dump(), "variant_status": "INACTIVE"})
    with pytest.raises(ValidationError):
        CatalogRowV1.model_validate({**row.model_dump(), "variant_id": uuid4()})


def test_saved_optimizer_snapshot_survives_catalog_changes_and_rejects_tampering(
    realistic_catalog,
):
    catalog = [part.model_copy(update={"id": uuid4()}) for part in realistic_catalog]
    constraints = PCBuildConstraints(
        target_budget_vnd={"value": 25_000_000, "source": "USER"},
        use_case={"value": "GAMING_1080P", "source": "USER"},
    )
    output = PCBuildApplicationService().build_pc(constraints, catalog)
    state = apply_turn_patch(
        ConsultationStateV1(),
        TurnPatchV1.model_validate(
            {
                "schema_version": 1,
                "constraints": {
                    "target_budget_vnd": {"op": "SET", "value": 25_000_000},
                    "use_case": {"op": "SET", "value": "GAMING_1080P"},
                },
            }
        ),
        source=ConstraintSource.USER,
    )
    body = {
        "conversation_id": uuid4(),
        "run_id": uuid4(),
        "base_revision": 0,
        "constraints_revision": state.constraints_revision,
        "consultation": state,
        "catalog_snapshot": [
            {
                "variant_id": part.id,
                "sku": f"test-{part.id}",
                "component": part,
                "quantity": 1,
                "retrieved_at": datetime.now(UTC),
                "evidence_refs": ["fixture"],
                "product_status": "ACTIVE",
                "variant_status": "ACTIVE",
            }
            for part in catalog
        ],
        "normalized_input": {
            "constraints": constraints,
            "candidates": catalog,
            "effective_spending_vnd": {part.id: part.price for part in catalog},
            "synthetic_sources": [
                {
                    "kind": kind,
                    "category": category,
                    "parent_variant_id": cpu.id,
                    "evidence_refs": ["fixture"],
                }
                for cpu in catalog
                if cpu.category == "CPU"
                for kind, category, available in [
                    ("INTEGRATED_GPU", "GPU", cpu.has_integrated_graphics),
                    ("STOCK_COOLER", "COOLER", cpu.includes_stock_cooler),
                ]
                if available
            ],
        },
        "versions": {
            "graph": "g1",
            "normalization": "n1",
            "compatibility": "c1",
            "optimizer": "o1",
            "pruning": "p1",
            "scoring": "s1",
            "recommendation": "v1",
            "code_artifact_digest": "a" * 64,
            "python": "3.12",
            "dependency_lock_digest": "b" * 64,
        },
        "policy_snapshot": {
            "weights": {"GAMING_1080P": {"performance": 1.0}},
            "thresholds": {},
            "calibration_tables": {},
            "search_limits": {},
            "tie_break_order": ["price", "variant_id"],
            "default_objectives": {"GAMING_1080P": "BALANCED"},
            "fallback_objectives": ["BALANCED"],
        },
        "optimization": {"outcome": "BUILT", "result": output.optimization},
        "recommendation": {
            "objective": output.recommendation.objective,
            "reason": output.recommendation.reason_code,
            "policy_version": "v1",
        },
    }
    # Prepare typed hash inputs; production snapshot capture is P3, not this fixture.
    draft = ReplaySnapshotV1.model_construct(
        **{
            **body,
            "normalized_input": OptimizerInputV1.model_validate(
                body["normalized_input"]
            ),
            "versions": OptimizerVersionsV1.model_validate(body["versions"]),
            "policy_snapshot": OptimizerPolicySnapshotV1.model_validate(
                body["policy_snapshot"]
            ),
            "optimization": CompletedOptimizationV1.model_validate(
                body["optimization"]
            ),
            "recommendation": RecommendationSnapshotV1.model_validate(
                body["recommendation"]
            ),
            "catalog_snapshot": [
                CatalogRowV1.model_validate(row) for row in body["catalog_snapshot"]
            ],
        }
    )
    snapshot = ReplaySnapshotV1.model_validate({**body, **draft.expected_hashes()})
    restored = ReplaySnapshotV1.model_validate_json(snapshot.model_dump_json())
    assert restored.optimization.result == output.optimization
    catalog[0].price += 1_000_000  # Changed live-like rows do not mutate saved payload.
    assert restored.catalog_snapshot[0].component.price != catalog[0].price
    payload = restored.model_dump(mode="json")
    payload["normalized_input"]["candidates"][0]["price"] += 1
    with pytest.raises(ValidationError, match="hash mismatch"):
        ReplaySnapshotV1.model_validate(payload)
    new_audit = restored.model_dump(mode="json")
    new_audit["run_id"] = str(uuid4())
    assert (
        ReplaySnapshotV1.model_validate(new_audit).optimization_input_hash
        == restored.optimization_input_hash
    )
