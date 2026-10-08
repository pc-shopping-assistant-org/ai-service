"""Allowlisted stateful API/SSE contracts; no routes or internal execution refs."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, StrictBool, model_validator

from ai_service.application.ports.hardware import ComponentCategory, ResolutionTier
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.stateful_contracts import (
    AccessoryRequestV1,
    AccessoryType,
    BudgetScope,
    Contract,
    NonNegativeInt,
    OwnedPartRefV1,
    PinnedPartRefV1,
    PositiveInt,
    RunStatus,
    Text,
)
from ai_service.capabilities.pc_builder.schemas import BuildObjective, UseCaseProfile


class PublicConsultationV1(Contract):
    budget_scope: BudgetScope | None
    target_budget_vnd: NonNegativeInt | None
    accessory_budget_vnd: NonNegativeInt | None
    use_case: UseCaseProfile | None
    target_resolution: ResolutionTier | None
    target_fps: int | None
    preferred_gpu_brand: Text | None
    preferred_cpu_brand: Text | None
    form_factor_preference: Text | None
    owned_parts: dict[ComponentCategory, OwnedPartRefV1]
    pinned_parts: dict[ComponentCategory, PinnedPartRefV1]
    accessories: dict[AccessoryType, AccessoryRequestV1]
    build_stale: StrictBool


def to_public_consultation(state: ConsultationStateV1) -> PublicConsultationV1:
    checked = ConsultationStateV1.model_validate(state.model_dump())
    return PublicConsultationV1(
        **{
            name: getattr(checked.constraints, name).value
            for name in (
                "budget_scope",
                "target_budget_vnd",
                "accessory_budget_vnd",
                "use_case",
                "target_resolution",
                "target_fps",
                "preferred_gpu_brand",
                "preferred_cpu_brand",
                "form_factor_preference",
            )
        },
        owned_parts=checked.owned_parts,
        pinned_parts=checked.pinned_parts,
        accessories=checked.accessories,
        build_stale=checked.build_stale,
    )


class PublicBuildPartV1(Contract):
    slot: ComponentCategory | AccessoryType
    # None only for an explicitly labeled synthetic iGPU/stock cooler.
    variant_id: UUID | None
    synthetic_kind: Literal["INTEGRATED_GPU", "STOCK_COOLER"] | None = None
    name: Text
    spending_vnd: NonNegativeInt
    is_owned: StrictBool = False
    is_pinned: StrictBool = False

    @model_validator(mode="after")
    def validate_spending(self) -> PublicBuildPartV1:
        if self.is_owned and (self.is_pinned or self.spending_vnd != 0):
            raise ValueError("Owned is free and cannot also be pinned")
        if (self.variant_id is None) != (self.synthetic_kind is not None):
            raise ValueError("Missing SKU requires explicit synthetic kind")
        if self.synthetic_kind is not None and (
            self.is_pinned
            or self.spending_vnd != 0
            or (
                self.synthetic_kind == "INTEGRATED_GPU"
                and self.slot != ComponentCategory.GPU
            )
            or (
                self.synthetic_kind == "STOCK_COOLER"
                and self.slot != ComponentCategory.COOLER
            )
        ):
            raise ValueError("Synthetic parts have no purchase cost or pin")
        return self


class PublicBuildV1(Contract):
    objective: BuildObjective
    parts: list[PublicBuildPartV1] = Field(min_length=1)
    total_spending_vnd: NonNegativeInt
    stale: StrictBool

    @model_validator(mode="after")
    def validate_total(self) -> PublicBuildV1:
        if len({part.slot for part in self.parts}) != len(self.parts):
            raise ValueError("V1 permits one selected part per slot/type")
        if self.total_spending_vnd != sum(part.spending_vnd for part in self.parts):
            raise ValueError("Build total must match effective spending")
        return self


class RunViewV1(Contract):
    run_id: UUID
    conversation_id: UUID
    request_id: UUID
    status: RunStatus
    base_revision: NonNegativeInt


class TurnResultV1(Contract):
    run: RunViewV1
    revision: NonNegativeInt
    state: PublicConsultationV1
    answer: Text
    build: PublicBuildV1 | None = None

    @model_validator(mode="after")
    def validate_published_result(self) -> TurnResultV1:
        if self.run.status not in {RunStatus.COMPLETED, RunStatus.WAITING_INPUT}:
            raise ValueError("Failed/in-flight runs cannot be successful turn results")
        if self.revision != self.run.base_revision + 1:
            raise ValueError("Published turn must advance its base revision once")
        return self


class ConversationViewV1(Contract):
    conversation_id: UUID
    revision: NonNegativeInt
    title: Text | None
    updated_at: AwareDatetime
    state: PublicConsultationV1
    build: PublicBuildV1 | None = None


class ConversationCreateV1(Contract):
    title: Annotated[Text, Field(max_length=200)] | None = None


class PageV1[T](Contract):
    items: list[T]
    next_cursor: Text | None = None


class MessageViewV1(Contract):
    message_id: UUID
    run_id: UUID
    sequence: PositiveInt
    role: Literal["user", "assistant"]
    content: Text
    created_at: AwareDatetime


class StatefulErrorCode(StrEnum):
    # Stable codes only; request-specific details stay in envelope.errors.
    UNAUTHENTICATED = "AI_UNAUTHENTICATED"
    NOT_FOUND = "AI_CONVERSATION_NOT_FOUND"
    REVISION_CONFLICT = "AI_REVISION_CONFLICT"
    REQUEST_CONFLICT = "AI_REQUEST_CONFLICT"
    CONVERSATION_BUSY = "AI_CONVERSATION_BUSY"
    INPUT_REQUIRED = "AI_INPUT_REQUIRED"
    RUN_FAILED = "AI_RUN_FAILED"
    STREAM_INTERRUPTED = "AI_STREAM_INTERRUPTED"


class EventIdentityV1(Contract):
    conversation_id: UUID
    run_id: UUID
    sequence: PositiveInt


class RunStartedV1(EventIdentityV1):
    event: Literal["START"]
    run: RunViewV1

    @model_validator(mode="after")
    def validate_run_identity(self) -> RunStartedV1:
        if (self.run.run_id, self.run.conversation_id) != (
            self.run_id,
            self.conversation_id,
        ):
            raise ValueError("Start event must identify the same run/conversation")
        return self


class TextDeltaV1(EventIdentityV1):
    event: Literal["DELTA"]
    delta: Annotated[str, Field(strict=True, min_length=1)]


class RunCompletedV1(EventIdentityV1):
    event: Literal["COMPLETED"]
    result: TurnResultV1

    @model_validator(mode="after")
    def validate_run_identity(self) -> RunCompletedV1:
        if (self.result.run.run_id, self.result.run.conversation_id) != (
            self.run_id,
            self.conversation_id,
        ):
            raise ValueError("Completed event must identify the same run/conversation")
        return self


class RunErrorV1(EventIdentityV1):
    event: Literal["ERROR"]
    code: StatefulErrorCode


type StatefulEventV1 = Annotated[
    RunStartedV1 | TextDeltaV1 | RunCompletedV1 | RunErrorV1,
    Field(discriminator="event"),
]
