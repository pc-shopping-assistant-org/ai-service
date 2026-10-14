"""Explicit V2 request, never an implicit upgrade of stored V1 request bodies."""

from typing import Any
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from ai_service.capabilities.assistant.limits import (
    MAX_SELECTED_PRODUCTS,
    MAX_TURN_MESSAGE_CHARS,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    Contract,
    NonNegativeInt,
    TaskClarificationReplyV1,
    Text,
    TurnRequestV1,
    UnitClarificationReplyV1,
)

from .pending_commands import PendingCommandV2


class TurnRequestV2(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    conversation_id: UUID
    request_id: UUID
    expected_revision: NonNegativeInt
    message: Text = Field(max_length=MAX_TURN_MESSAGE_CHARS)
    selected_product_ids: list[UUID] = Field(
        default_factory=list,
        max_length=MAX_SELECTED_PRODUCTS,
        exclude_if=lambda value: not value,
    )
    confirmation: ProposalConfirmation | None = None
    reference_reply: ClarificationReplyV1 | None = None
    unit_reply: UnitClarificationReplyV1 | None = None
    task_reply: TaskClarificationReplyV1 | None = None
    pending_action: PendingCommandV2 | None = None

    @model_validator(mode="after")
    def one_command(self) -> "TurnRequestV2":
        if len(set(self.selected_product_ids)) != len(self.selected_product_ids):
            raise ValueError("Selected products must be distinct")
        if (
            sum(
                value is not None
                for value in (
                    self.confirmation,
                    self.reference_reply,
                    self.unit_reply,
                    self.task_reply,
                    self.pending_action,
                )
            )
            > 1
        ):
            raise ValueError("Choose one continuation command")
        return self

    def execution_input(self) -> dict[str, Any]:
        commands = self.model_dump(
            mode="json",
            exclude_none=True,
            include={
                "confirmation",
                "reference_reply",
                "unit_reply",
                "task_reply",
                "pending_action",
            },
        )
        return {"message": self.message, "message_id": str(self.request_id), **commands}


type TurnRequest = TurnRequestV1 | TurnRequestV2


def parse_turn_request(payload: dict[str, Any]) -> TurnRequest:
    if "schema_version" in payload:
        return TurnRequestV2.model_validate(payload)
    return TurnRequestV1.model_validate(payload)


def turn_request_payload(request: TurnRequest) -> dict[str, Any]:
    """Canonical durable payload; omit only absent optional commands, as V1 did."""
    return parse_turn_request(request.model_dump()).model_dump(
        mode="json", exclude_none=True
    )
