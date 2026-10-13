"""Model-interpreted choices remain source-bound proposals, never user consent."""

from uuid import UUID, uuid5

from pydantic import Field, model_validator

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.amendment import PendingAmendment
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
    ResolutionContinuation,
    amendment_source_id,
    continue_amendment,
    continue_reference,
    continue_unit,
    unit_reply_source_id,
)
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitChoiceDraft,
    unit_choice_reply,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    ChoiceDraft,
    choice_reply,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    Contract,
    TaskClarificationReplyV1,
    UnitClarificationReplyV1,
)

from .contracts import ToolContext
from .task_amendment import TaskAmendment, amend_pending_task
from .task_resolution import PendingTaskDraft, TaskChoiceDraft, task_choice_reply


class FreeResolutionReplyV2(Contract):
    run_id: UUID
    pending_message_id: UUID
    request_message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_choice: ChoiceDraft | None = None
    unit_choice: UnitChoiceDraft | None = None

    @model_validator(mode="after")
    def one_kind(self) -> "FreeResolutionReplyV2":
        if (self.reference_choice is None) == (self.unit_choice is None):
            raise ValueError("Exactly one interpreted choice kind is required")
        return self


def derive_resolution_reply(
    receipt: FreeResolutionReplyV2,
    message: str,
    pending: PendingResolution,
    run_id: UUID,
) -> tuple[ClarificationReplyV1 | None, UnitClarificationReplyV1 | None]:
    receipt = FreeResolutionReplyV2.model_validate(receipt.model_dump())
    if (
        receipt.run_id != run_id
        or receipt.pending_message_id != pending.draft.message_id
        or receipt.request_message_hash != canonical_hash(message)
    ):
        raise ValueError("Free resolution receipt source mismatch")
    if receipt.reference_choice is not None:
        return choice_reply(receipt.reference_choice, message, pending.draft), None
    assert receipt.unit_choice is not None
    return None, unit_choice_reply(receipt.unit_choice, pending.draft)


def apply_resolution_command(
    context: ToolContext,
    pending: PendingResolution,
    message: str,
    conversation_revision: int,
    reference: ClarificationReplyV1 | None,
    unit: UnitClarificationReplyV1 | None,
) -> ResolutionContinuation:
    if (reference is None) == (unit is None):
        raise ValueError("Resolution requires exactly one command")
    if reference is not None:
        return continue_reference(
            context.requirements,
            pending,
            reference,
            context.catalog,
            owner=context.account_id,
            conversation_id=context.conversation_id,
            conversation_revision=conversation_revision,
        )
    assert unit is not None
    return continue_unit(
        context.requirements,
        pending,
        unit,
        context.catalog,
        message_id=unit_reply_source_id(context.run_id, message, unit),
        message=message,
        owner=context.account_id,
        conversation_id=context.conversation_id,
        conversation_revision=conversation_revision,
    )


class FreeTaskReplyV2(Contract):
    run_id: UUID
    pending_message_id: UUID
    request_message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    choice: TaskChoiceDraft


def derive_task_reply(
    receipt: FreeTaskReplyV2, message: str, pending: PendingTaskDraft, run_id: UUID
) -> TaskClarificationReplyV1 | None:
    receipt = FreeTaskReplyV2.model_validate(receipt.model_dump())
    if (
        receipt.run_id != run_id
        or receipt.pending_message_id != pending.message_id
        or receipt.request_message_hash != canonical_hash(message)
    ):
        raise ValueError("Free task receipt source mismatch")
    return task_choice_reply(receipt.choice, message, pending)


class FreeAmendmentV2(Contract):
    run_id: UUID
    pending_message_id: UUID
    request_message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment: PendingAmendment


def apply_amendment_receipt(
    receipt: FreeAmendmentV2,
    context: ToolContext,
    pending: PendingResolution,
    message: str,
    conversation_revision: int,
) -> ResolutionContinuation:
    receipt = FreeAmendmentV2.model_validate(receipt.model_dump())
    if (
        receipt.run_id != context.run_id
        or receipt.pending_message_id != pending.draft.message_id
        or receipt.request_message_hash != canonical_hash(message)
    ):
        raise ValueError("Amendment receipt source mismatch")
    return continue_amendment(
        context.requirements,
        pending,
        receipt.amendment,
        context.catalog,
        message_id=amendment_source_id(context.run_id, message, receipt.amendment),
        message=message,
        owner=context.account_id,
        conversation_id=context.conversation_id,
        conversation_revision=conversation_revision,
    )


class FreeTaskAmendmentV2(Contract):
    run_id: UUID
    pending_message_id: UUID
    request_message_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    amendment: TaskAmendment


def apply_task_amendment_receipt(
    receipt: FreeTaskAmendmentV2,
    context: ToolContext,
    pending: PendingTaskDraft,
    message: str,
) -> PendingTaskDraft:
    receipt = FreeTaskAmendmentV2.model_validate(receipt.model_dump())
    if (
        receipt.run_id != context.run_id
        or receipt.pending_message_id != pending.message_id
        or receipt.request_message_hash != canonical_hash(message)
    ):
        raise ValueError("Task amendment receipt source mismatch")
    source = uuid5(
        context.run_id,
        "task-amendment:"
        + canonical_hash(
            {"message": message, "amendment": receipt.amendment.model_dump(mode="json")}
        ),
    )
    return amend_pending_task(
        pending,
        receipt.amendment,
        context.requirements,
        message_id=source,
        message=message,
    )
