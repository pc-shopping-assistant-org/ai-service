"""Internal replay contracts. These do not execute replay or read live catalog."""

from __future__ import annotations

import hashlib
import json
from typing import Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    FiniteFloat,
    JsonValue,
    model_validator,
)

from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.assistant.conversation_core import (
    CompletedOptimizationV1,
    ConsultationStateV1,
    Digest,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    Text,
)
from ai_service.capabilities.pc_builder.application.recommendation_policy import (
    RecommendationReason,
)
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    ConstraintSource,
    PCBuildConstraints,
    UseCaseProfile,
)


def canonical_hash(value: BaseModel | JsonValue) -> str:
    """V1 SHA-256: UTF-8, sorted object keys, compact JSON, finite numbers.

    Callers preserve ordered candidate arrays and explicitly normalize set-like
    values. Audit timestamps/IDs must not be added to semantic hash inputs.
    """
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CatalogRowV1(Contract):
    variant_id: UUID
    sku: Text
    product_status: Literal["ACTIVE"]
    variant_status: Literal["ACTIVE"]
    quantity: NonNegativeInt
    component: ComponentSpec
    retrieved_at: AwareDatetime
    backend_revision: Text | None = None
    evidence_refs: list[Text] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> CatalogRowV1:
        if self.component.id != self.variant_id or self.component.price < 0:
            raise ValueError("Canonical component identity/price is invalid")
        canonical_hash(self.component)  # Reject nonfinite nested evidence/specs.
        return self


class SyntheticSourceV1(Contract):
    kind: Literal["INTEGRATED_GPU", "STOCK_COOLER"]
    category: ComponentCategory
    parent_variant_id: UUID
    evidence_refs: list[Text] = Field(min_length=1)


class OptimizerInputV1(Contract):
    constraints: PCBuildConstraints
    candidates: list[ComponentSpec]
    pinned_parts: dict[ComponentCategory, UUID] = Field(default_factory=dict)
    effective_spending_vnd: dict[UUID, NonNegativeInt]
    requested_objective: BuildObjective | None = None
    synthetic_sources: list[SyntheticSourceV1] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_explicit_constraints(self) -> OptimizerInputV1:
        for name in ("target_budget_vnd", "use_case"):
            if name not in self.constraints.model_fields_set:
                raise ValueError("Replay input cannot rely on optimizer defaults")
            if getattr(self.constraints, name).source == ConstraintSource.DEFAULT:
                raise ValueError(
                    "Budget/use case must come from validated requirements"
                )
        if self.constraints.target_budget_vnd.value < 0:
            raise ValueError("Core spending cap cannot be negative")
        if any(not part.exclude_from_budget for part in self.constraints.owned_parts):
            raise ValueError("V1 owned components cannot carry purchase spending")
        canonical_hash(self)
        return self


class OptimizerVersionsV1(Contract):
    graph: Text
    state_schema: Literal[1] = 1
    normalization: Text
    compatibility: Text
    optimizer: Text
    pruning: Text
    scoring: Text
    recommendation: Text
    code_artifact_digest: Digest
    python: Text
    dependency_lock_digest: Digest


class OptimizerPolicySnapshotV1(Contract):
    # Explicit engine config, never fetched from the current process during replay.
    weights: dict[UseCaseProfile, dict[Text, FiniteFloat]]
    thresholds: dict[Text, FiniteFloat]
    calibration_tables: dict[Text, JsonValue]
    search_limits: dict[Text, NonNegativeInt]
    tie_break_order: list[Text] = Field(min_length=1)
    default_objectives: dict[UseCaseProfile, BuildObjective]
    fallback_objectives: list[BuildObjective] = Field(min_length=1)
    random_seed: NonNegativeInt | None = None


class RecommendationSnapshotV1(Contract):
    objective: BuildObjective
    reason: RecommendationReason
    policy_version: Text
    # RankedBuild already holds all score/evidence and parts in the result.


class ReplaySnapshotV1(Contract):
    schema_version: Literal[1] = 1
    hash_format_version: Literal[1] = 1
    conversation_id: UUID
    run_id: UUID
    base_revision: NonNegativeInt
    constraints_revision: NonNegativeInt
    consultation: ConsultationStateV1
    catalog_snapshot: list[CatalogRowV1]
    normalized_input: OptimizerInputV1
    versions: OptimizerVersionsV1
    policy_snapshot: OptimizerPolicySnapshotV1
    optimization: CompletedOptimizationV1
    recommendation: RecommendationSnapshotV1 | None
    catalog_snapshot_hash: Digest
    normalized_input_hash: Digest
    optimization_input_hash: Digest
    optimization_result_hash: Digest
    recommendation_result_hash: Digest

    def expected_hashes(self) -> dict[str, str]:
        catalog: list[JsonValue] = [
            row.model_dump(mode="json", exclude={"retrieved_at"})
            for row in self.catalog_snapshot
        ]
        state = self.consultation.model_dump(
            mode="json",
            include={
                "constraints",
                "owned_parts",
                "pinned_parts",
                "accessories",
            },
        )
        return {
            "catalog_snapshot_hash": canonical_hash(catalog),
            "normalized_input_hash": canonical_hash(self.normalized_input),
            "optimization_input_hash": canonical_hash(
                {
                    "hash_format_version": self.hash_format_version,
                    "consultation": state,
                    "catalog": catalog,
                    "normalized_input": self.normalized_input.model_dump(mode="json"),
                    "versions": self.versions.model_dump(mode="json"),
                    "policy": self.policy_snapshot.model_dump(mode="json"),
                }
            ),
            "optimization_result_hash": canonical_hash(self.optimization),
            "recommendation_result_hash": canonical_hash(
                self.recommendation.model_dump(mode="json")
                if self.recommendation
                else None
            ),
        }

    @model_validator(mode="after")
    def validate_snapshot(self) -> ReplaySnapshotV1:
        if any(
            getattr(self, name) != digest
            for name, digest in self.expected_hashes().items()
        ):
            raise ValueError("Replay snapshot hash mismatch; never use live fallback")
        if self.constraints_revision != self.consultation.constraints_revision:
            raise ValueError("Snapshot constraint revision mismatch")
        use_case = self.normalized_input.constraints.use_case.value
        if (
            use_case != self.consultation.constraints.use_case.value
            or use_case != self.optimization.result.use_case
            or self.normalized_input.constraints.target_budget_vnd.value
            != self.optimization.result.target_budget_vnd
        ):
            raise ValueError("Saved constraints/input/result disagree")
        catalog_by_id = {row.variant_id: row for row in self.catalog_snapshot}
        if len(catalog_by_id) != len(self.catalog_snapshot):
            raise ValueError("Duplicate canonical catalog variant")
        owned_ids = {part.variant_id for part in self.consultation.owned_parts.values()}
        for candidate in self.normalized_input.candidates:
            if candidate.id is not None and (
                candidate.id not in catalog_by_id
                or candidate.category != catalog_by_id[candidate.id].component.category
            ):
                raise ValueError("Normalized candidate lacks canonical source")
            if candidate.id is not None:
                expected = 0 if candidate.id in owned_ids else candidate.price
                if (
                    self.normalized_input.effective_spending_vnd.get(candidate.id)
                    != expected
                ):
                    raise ValueError(
                        "Candidate spending must be explicit; owned is free, pinned paid"
                    )
        for source in self.normalized_input.synthetic_sources:
            parent = catalog_by_id.get(source.parent_variant_id)
            if parent is None or parent.component.category != ComponentCategory.CPU:
                raise ValueError("Synthetic source requires canonical CPU parent")
            if source.kind == "INTEGRATED_GPU":
                valid = (
                    source.category == ComponentCategory.GPU
                    and parent.component.has_integrated_graphics
                )
            else:
                valid = (
                    source.category == ComponentCategory.COOLER
                    and parent.component.includes_stock_cooler
                )
            if not valid:
                raise ValueError("Synthetic source disagrees with catalog evidence")
        for build in self.optimization.result.builds.values():
            cpu = build.parts.get(ComponentCategory.CPU)
            for slot, part in build.parts.items():
                if part.id is None and not any(
                    source.category == slot
                    and cpu is not None
                    and source.parent_variant_id == cpu.id
                    for source in self.normalized_input.synthetic_sources
                ):
                    raise ValueError("Synthetic output lacks parent/evidence snapshot")
        if self.recommendation is None:
            if self.optimization.outcome != "INFEASIBLE":
                raise ValueError("Built output requires a recommendation decision")
        elif (
            self.recommendation.objective not in self.optimization.result.builds
            or self.recommendation.policy_version != self.versions.recommendation
        ):
            raise ValueError("Recommendation must reference saved result/version")
        return self
