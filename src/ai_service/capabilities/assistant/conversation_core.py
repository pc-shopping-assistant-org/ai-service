"""Pure multi-turn state rules. No graph cursor, DB writes or LLM calls."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from ai_service.application.ports.hardware import ComponentCategory, ResolutionTier
from ai_service.capabilities.assistant.stateful_contracts import (
    AccessoryRequestV1,
    AccessoryType,
    BudgetScope,
    Contract,
    NonNegativeInt,
    OwnedPartRefV1,
    PinnedPartRefV1,
    Text,
    TurnPatchV1,
)
from ai_service.capabilities.pc_builder.schemas import (
    ConstraintSource,
    ConstraintValue,
    OptimizationResult,
    UseCaseProfile,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StageStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class StageVersionsV1(Contract):
    graph: Text
    implementation: Text
    policy: Text
    state_schema_version: Literal[1] = 1


class CompletionMarkerV1[T](Contract):
    status: StageStatus = StageStatus.NOT_STARTED
    input_hash: Digest | None = None
    result: T | None = None
    versions: StageVersionsV1 | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> CompletionMarkerV1[T]:
        fields = (self.input_hash, self.result, self.versions)
        if self.status == StageStatus.COMPLETED:
            if any(value is None for value in fields):
                raise ValueError(
                    "Completed stage requires validated result, hash and versions"
                )
        elif any(value is not None for value in fields):
            raise ValueError("Incomplete stage cannot carry reusable success data")
        return self

    def can_reuse(self, input_hash: str, versions: StageVersionsV1) -> bool:
        # Revalidate restored/mutated state, not just result != None.
        checked = type(self).model_validate(self.model_dump())
        return (
            checked.status == StageStatus.COMPLETED
            and checked.input_hash == input_hash
            and checked.versions == versions
        )


class CompletedOptimizationV1(Contract):
    outcome: Literal["BUILT", "INFEASIBLE"]
    result: OptimizationResult

    @model_validator(mode="after")
    def validate_outcome(self) -> CompletedOptimizationV1:
        if (self.outcome == "BUILT") != bool(self.result.builds):
            raise ValueError("Optimization outcome must match available builds")
        return self


class ConsultationConstraintsV1(Contract):
    budget_scope: ConstraintValue[BudgetScope | None] = Field(
        default_factory=lambda: ConstraintValue[BudgetScope | None](
            value=BudgetScope.BUILD_PC
        )
    )
    accessory_budget_vnd: ConstraintValue[NonNegativeInt | None] = Field(
        default_factory=lambda: ConstraintValue[NonNegativeInt | None](
            value=None, confidence=0.0
        )
    )
    # Unknown is explicit; never instantiate the optimizer's 20m/gaming defaults.
    target_budget_vnd: ConstraintValue[NonNegativeInt | None] = Field(
        default_factory=lambda: ConstraintValue[NonNegativeInt | None](
            value=None, confidence=0.0
        )
    )
    use_case: ConstraintValue[UseCaseProfile | None] = Field(
        default_factory=lambda: ConstraintValue[UseCaseProfile | None](
            value=None, confidence=0.0
        )
    )
    target_resolution: ConstraintValue[ResolutionTier | None] = Field(
        default_factory=lambda: ConstraintValue[ResolutionTier | None](
            value=None, confidence=0.0
        )
    )
    target_fps: ConstraintValue[
        Annotated[int, Field(strict=True, ge=30, le=1000)] | None
    ] = Field(
        default_factory=lambda: ConstraintValue[
            Annotated[int, Field(strict=True, ge=30, le=1000)] | None
        ](value=None, confidence=0.0)
    )
    preferred_gpu_brand: ConstraintValue[Text | None] = Field(
        default_factory=lambda: ConstraintValue[Text | None](value=None, confidence=0.0)
    )
    preferred_cpu_brand: ConstraintValue[Text | None] = Field(
        default_factory=lambda: ConstraintValue[Text | None](value=None, confidence=0.0)
    )
    form_factor_preference: ConstraintValue[Text | None] = Field(
        default_factory=lambda: ConstraintValue[Text | None](value=None, confidence=0.0)
    )


class ConsultationStateV1(Contract):
    schema_version: Literal[1] = 1
    constraints_revision: NonNegativeInt = 0
    constraints: ConsultationConstraintsV1 = Field(
        default_factory=ConsultationConstraintsV1
    )
    owned_parts: dict[ComponentCategory, OwnedPartRefV1] = Field(default_factory=dict)
    pinned_parts: dict[ComponentCategory, PinnedPartRefV1] = Field(default_factory=dict)
    accessories: dict[AccessoryType, AccessoryRequestV1] = Field(default_factory=dict)
    optimization: CompletionMarkerV1[CompletedOptimizationV1] = Field(
        default_factory=CompletionMarkerV1
    )
    explanation: CompletionMarkerV1[str] = Field(default_factory=CompletionMarkerV1)
    last_successful_build: OptimizationResult | None = None
    build_stale: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_schema_version(cls, value: object) -> object:
        return TurnPatchV1.require_integer_version(value)

    @model_validator(mode="after")
    def validate_bindings_and_history(self) -> ConsultationStateV1:
        total = self.constraints.target_budget_vnd.value
        accessory_cap = self.constraints.accessory_budget_vnd.value
        if total is not None and accessory_cap is not None and accessory_cap > total:
            raise ValueError("Accessory budget cannot exceed total spending cap")
        if self.owned_parts.keys() & self.pinned_parts.keys():
            raise ValueError("A slot cannot be both owned and pinned")
        if (
            self.last_successful_build is not None
            and not self.last_successful_build.builds
        ):
            raise ValueError("Historical successful build must contain builds")
        if self.build_stale and self.last_successful_build is None:
            raise ValueError("Stale build requires a historical successful build")
        return self


class SetupBudgetV1(Contract):
    """Arithmetic after canonical accessory spending is resolved, not allocation policy."""

    total_budget_vnd: NonNegativeInt
    accessory_spending_vnd: NonNegativeInt

    @model_validator(mode="after")
    def validate_spending(self) -> SetupBudgetV1:
        if self.accessory_spending_vnd > self.total_budget_vnd:
            raise ValueError("Accessory spending exceeds total setup cap")
        return self

    @property
    def core_budget_vnd(self) -> int:
        return self.total_budget_vnd - self.accessory_spending_vnd


class MergeConflict(ValueError):
    """Clarify/reject this turn; do not partially apply it."""


def apply_turn_patch(
    state: ConsultationStateV1,
    patch: TurnPatchV1,
    *,
    source: ConstraintSource,
    confirmed_fields: frozenset[str] = frozenset(),
    confidence: float = 1.0,
) -> ConsultationStateV1:
    """Source/confirmation are trusted application evidence, NOT client/LLM fields.

    Validation is atomic. This does not resolve SKU visibility/specs or enforce
    conversation ownership/revision CAS; those remain application transaction gates.
    """
    if source not in {ConstraintSource.USER, ConstraintSource.INFERRED}:
        raise MergeConflict("Turn patches cannot assign SYSTEM/default provenance")
    state = ConsultationStateV1.model_validate(state.model_dump())
    patch = TurnPatchV1.model_validate(patch.to_payload())
    payload = state.model_dump()
    if patch.constraints is not None:
        for name in patch.constraints.model_fields_set:
            operation = getattr(patch.constraints, name)
            value = operation.value if operation.op == "SET" else None
            current = getattr(state.constraints, name)
            if current.value == value and not (
                source == ConstraintSource.USER
                and current.source == ConstraintSource.INFERRED
            ):
                continue
            if current.source == ConstraintSource.SYSTEM:
                raise MergeConflict(f"Cannot override system constraint: {name}")
            if (
                source == ConstraintSource.INFERRED
                and current.source == ConstraintSource.USER
            ):
                raise MergeConflict(
                    f"Inference cannot override user constraint: {name}"
                )
            if current.locked and (
                source != ConstraintSource.USER or name not in confirmed_fields
            ):
                raise MergeConflict(f"Explicit confirmation required: {name}")
            payload["constraints"][name] = {
                "value": value,
                "source": source,
                "confidence": confidence,
                "locked": current.locked,
            }
    for group in ("owned_parts", "pinned_parts"):
        operations = getattr(patch, group)
        if not operations:
            continue
        if source != ConstraintSource.USER:
            raise MergeConflict("Owned/pinned bindings require explicit user evidence")
        for slot, operation in operations.items():
            if operation.op == "CLEAR":
                payload[group].pop(slot, None)
            else:
                payload[group][slot] = operation.value.model_dump()
    if patch.accessories:
        for slot, operation in patch.accessories.items():
            if source != ConstraintSource.USER and (
                operation.op == "CLEAR"
                or operation.value.kind != "RECOMMEND"
                or (
                    slot in state.accessories
                    and state.accessories[slot].kind != "RECOMMEND"
                )
            ):
                raise MergeConflict("Accessory bindings require explicit user evidence")
            if operation.op == "CLEAR":
                payload["accessories"].pop(slot, None)
            else:
                payload["accessories"][slot] = operation.value.model_dump()
    try:
        candidate = ConsultationStateV1.model_validate(payload)
    except ValueError as exc:
        raise MergeConflict(str(exc)) from exc
    if (
        candidate.constraints != state.constraints
        or candidate.owned_parts != state.owned_parts
        or candidate.pinned_parts != state.pinned_parts
        or candidate.accessories != state.accessories
    ):
        candidate.constraints_revision += 1
        candidate.optimization = CompletionMarkerV1[CompletedOptimizationV1]()
        candidate.explanation = CompletionMarkerV1[str]()
        candidate.build_stale = candidate.last_successful_build is not None
    return candidate
