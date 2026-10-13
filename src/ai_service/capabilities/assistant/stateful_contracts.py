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


class BudgetScope(StrEnum):
    BUILD_PC = "BUILD_PC"
    FULL_SETUP = "FULL_SETUP"


class ConstraintsPatchV1(SparseContract):
    # Money/use case remain unknown; BUILD_PC is the approved scope default only.
    target_budget_vnd: Operation[NonNegativeInt] | None = None
    budget_scope: Operation[BudgetScope] | None = None
    accessory_budget_vnd: Operation[NonNegativeInt] | None = None
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
    exclude_from_budget: Literal[True] = True

    @field_validator("exclude_from_budget", mode="before")
    @classmethod
    def require_true(cls, value: object) -> object:
        if value is not True:
            raise ValueError("V1 owned parts are always excluded from spending")
        return value


class AccessoryType(StrEnum):
    MONITOR = "MONITOR"
    MOUSE = "MOUSE"
    KEYBOARD = "KEYBOARD"
    HEADSET = "HEADSET"


class AccessoryRequestV1(Contract):
    kind: Literal["OWNED", "PINNED", "RECOMMEND"]
    variant_id: UUID | None = None
    quantity: Literal[1] = 1

    @field_validator("quantity", mode="before")
    @classmethod
    def require_one(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("V1 allows exactly one accessory per requested type")
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> AccessoryRequestV1:
        if (self.kind == "RECOMMEND") != (self.variant_id is None):
            raise ValueError("Owned/pinned require a variant; recommend has none yet")
        return self


class TurnPatchV1(SparseContract):
    schema_version: Literal[1]
    constraints: ConstraintsPatchV1 | None = None
    owned_parts: dict[ComponentCategory, Operation[OwnedPartRefV1]] | None = None
    pinned_parts: dict[ComponentCategory, Operation[PinnedPartRefV1]] | None = None
    accessories: dict[AccessoryType, Operation[AccessoryRequestV1]] | None = None

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


class ClarificationSelectionV1(Contract):
    item_index: NonNegativeInt
    variant_id: UUID


class ClarificationReplyV1(Contract):
    pending_message_id: UUID
    selections: list[ClarificationSelectionV1] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_items(self) -> ClarificationReplyV1:
        if len({item.item_index for item in self.selections}) != len(self.selections):
            raise ValueError("Duplicate clarification selection")
        return self


class TaskProductSelectionV1(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt
    product_id: UUID


class TaskClarificationReplyV1(Contract):
    pending_message_id: UUID
    constraints_revision: NonNegativeInt
    selections: list[TaskProductSelectionV1] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def unique_subjects(self) -> TaskClarificationReplyV1:
        keys = {(item.task_index, item.subject_index) for item in self.selections}
        if len(keys) != len(self.selections):
            raise ValueError("Duplicate task clarification selection")
        return self


QuantityUnit = Literal["VND", "GB", "TB", "GiB", "TiB", "W", "FPS"]


class UnitSelectionV1(Contract):
    item_index: NonNegativeInt
    unit: QuantityUnit
    start: NonNegativeInt
    end: StrictInt = Field(gt=0)
    quote: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def valid_span(self) -> UnitSelectionV1:
        if self.end <= self.start or self.end - self.start != len(self.quote):
            raise ValueError("Invalid unit confirmation evidence span")
        return self


class UnitClarificationReplyV1(Contract):
    pending_message_id: UUID
    constraints_revision: NonNegativeInt
    selections: list[UnitSelectionV1] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_items(self) -> UnitClarificationReplyV1:
        if len({item.item_index for item in self.selections}) != len(self.selections):
            raise ValueError("Duplicate unit clarification selection")
        return self


class PendingActionV1(Contract):
    action: Literal["DISCARD", "REPLACE"]
    pending_message_id: UUID
    constraints_revision: NonNegativeInt


class TurnRequestV1(Contract):
    conversation_id: UUID
    request_id: UUID
    expected_revision: NonNegativeInt
    message: Annotated[Text, Field(max_length=4000)]
    clarification_reply: ClarificationReplyV1 | None = None
    task_clarification_reply: TaskClarificationReplyV1 | None = None
    unit_clarification_reply: UnitClarificationReplyV1 | None = None
    pending_action: PendingActionV1 | None = None

    @model_validator(mode="after")
    def one_clarification_kind(self) -> TurnRequestV1:
        if self.pending_action is not None and any(
            reply is not None
            for reply in (
                self.clarification_reply,
                self.task_clarification_reply,
                self.unit_clarification_reply,
            )
        ):
            raise ValueError("Cannot combine pending action with a clarification reply")
        if (
            self.clarification_reply is not None
            or self.unit_clarification_reply is not None
        ) and self.task_clarification_reply is not None:
            raise ValueError("Choose one clarification reply kind")
        return self

    def execution_input(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "message": self.message,
            "message_id": str(self.request_id),
        }
        if self.clarification_reply is not None:
            payload["clarification_reply"] = self.clarification_reply.model_dump(
                mode="json"
            )
        if self.task_clarification_reply is not None:
            payload["task_clarification_reply"] = (
                self.task_clarification_reply.model_dump(mode="json")
            )
        if self.unit_clarification_reply is not None:
            payload["unit_clarification_reply"] = (
                self.unit_clarification_reply.model_dump(mode="json")
            )
        if self.pending_action is not None:
            payload["pending_action"] = self.pending_action.model_dump(mode="json")
        return payload


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
        if self.revision > 0 and ref is None:
            raise ValueError("Published revisions require an accepted head")
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
