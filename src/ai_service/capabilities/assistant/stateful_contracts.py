"""B0 contracts, independent of model, graph and persistence frameworks.

These schemas do not merge state, prove lineage/worker termination or authorize
a request. Those operations require the B1/B2/B4 adapters and transaction gates.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from ai_service.application.ports.hardware import ComponentCategory, ResolutionTier
from ai_service.capabilities.pc_builder.schemas import UseCaseProfile

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
Text = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SetValue[T](Contract):
    op: Literal["SET"]
    value: T


class ClearValue(Contract):
    op: Literal["CLEAR"]


type Operation[T] = Annotated[SetValue[T] | ClearValue, Field(discriminator="op")]


class SparseContract(Contract):
    @model_validator(mode="before")
    @classmethod
    def reject_explicit_null(cls, value: Any) -> Any:
        if isinstance(value, dict) and any(item is None for item in value.values()):
            raise ValueError("Explicit null is not absent or CLEAR")
        return value


class ConstraintsPatchV1(SparseContract):
    # No optimizer defaults here. Engine minimum budget remains a B0 policy issue.
    target_budget_vnd: Operation[NonNegativeInt] | None = None
    use_case: Operation[UseCaseProfile] | None = None
    target_resolution: Operation[ResolutionTier] | None = None
    target_fps: Operation[Annotated[StrictInt, Field(ge=30, le=1000)]] | None = None
    preferred_gpu_brand: Operation[Text] | None = None
    preferred_cpu_brand: Operation[Text] | None = None
    form_factor_preference: Operation[Text] | None = None


class PinnedPartRefV1(Contract):
    variant_id: UUID


class OwnedPartRefV1(Contract):
    variant_id: UUID
    exclude_from_budget: StrictBool = True


class TurnPatchV1(SparseContract):
    schema_version: Literal[1]
    constraints: ConstraintsPatchV1 | None = None
    owned_parts: dict[ComponentCategory, Operation[OwnedPartRefV1]] | None = None
    pinned_parts: dict[ComponentCategory, Operation[PinnedPartRefV1]] | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be integer 1")
        return value

    def to_payload(self) -> dict[str, Any]:
        """Preserve supplied operations; never serialize absent fields as null."""
        return self.model_dump(mode="json", exclude_unset=True)


def parse_turn_patch(payload: str) -> TurnPatchV1:
    """Use at raw JSON boundaries; normal json.loads silently loses duplicates."""

    def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Invalid JSON number: {value}")

    return TurnPatchV1.model_validate(
        json.loads(
            payload,
            object_pairs_hook=unique_keys,
            parse_constant=reject_nonfinite,
        )
    )


class TurnRequestV1(Contract):
    conversation_id: UUID
    request_id: UUID
    expected_revision: NonNegativeInt
    message: Annotated[Text, Field(max_length=4000)]


class CheckpointRefV1(Contract):
    """Opaque native checkpoint identity; internal only, never a client selector."""

    thread_id: UUID
    checkpoint_id: Annotated[str, StringConstraints(strict=True, min_length=1)]
    checkpoint_ns: str = ""


class ConversationHeadV1(Contract):
    conversation_id: UUID
    revision: NonNegativeInt
    accepted_checkpoint_ref: CheckpointRefV1 | None = None

    @property
    def thread_id(self) -> str:
        return str(self.conversation_id)

    @model_validator(mode="after")
    def validate_accepted_reference(self) -> ConversationHeadV1:
        ref = self.accepted_checkpoint_ref
        if (self.revision == 0) != (ref is None):
            raise ValueError(
                "Revision zero has no accepted head; published revisions require one"
            )
        if ref is not None and (
            ref.thread_id != self.conversation_id or ref.checkpoint_ns != ""
        ):
            raise ValueError(
                "Accepted head must belong to this conversation/root namespace"
            )
        return self


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    WAITING_INPUT = "WAITING_INPUT"
    COMPLETED = "COMPLETED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    CANCELLED = "CANCELLED"


class ExecutionStopEvidenceV1(Contract):
    """Shape validation only; trusted executor/coordinator must verify the proof."""

    kind: Literal["EXECUTOR_STOP_ACK", "PROCESS_TERMINATED"]
    execution_id: UUID
    worker_instance_id: UUID
    generation: PositiveInt
    stopped_at: AwareDatetime
    checkpoint_writes_drained: Literal[True]
    proof_ref: Text

    @field_validator("checkpoint_writes_drained", mode="before")
    @classmethod
    def require_true_boolean(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Checkpoint writes must be explicitly drained")
        return value

    @field_validator("stopped_at")
    @classmethod
    def require_utc(cls, value: AwareDatetime) -> AwareDatetime:
        offset = value.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ValueError("stopped_at must be UTC")
        return value
