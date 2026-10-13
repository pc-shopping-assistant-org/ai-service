"""Explicit pending-action bindings. Not authorization or publication authority."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.pending_action import (
    validate_pending_action,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    ProposalConfirmation,
    confirm_proposal,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    PendingActionV1,
)

from .published_state import FlexiblePublishedState


class DraftPendingCommandV2(Contract):
    target: Literal["REQUIREMENTS", "TASKS", "CLARIFICATION"]
    action: Literal["REPLACE", "DISCARD"]
    pending_message_id: UUID
    constraints_revision: NonNegativeInt


class ProposalPendingCommandV2(ProposalConfirmation):
    target: Literal["PROPOSAL"]
    action: Literal["REPLACE", "DISCARD"]


type PendingCommandV2 = Annotated[
    DraftPendingCommandV2 | ProposalPendingCommandV2, Field(discriminator="target")
]


def validate_pending_command(
    command: PendingCommandV2,
    pending: FlexiblePublishedState | None,
    state: CanonicalRequirementState,
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> None:
    # Task artifacts deliberately carry no owner field. The caller must load
    # the snapshot through the authorized repository/coordinator first.
    if (
        pending is None
        or pending.status != "WAITING_INPUT"
        or pending.conversation_id != conversation_id
    ):
        raise ValueError("Replacement requires an accepted pending artifact")
    if isinstance(command, ProposalPendingCommandV2):
        if pending.pending_proposal is None:
            raise ValueError("Replacement target is not a pending proposal")
        confirm_proposal(
            pending.pending_proposal,
            ProposalConfirmation(
                proposal_id=command.proposal_id,
                proposal_hash=command.proposal_hash,
                target_revision=command.target_revision,
                base_requirements_hash=command.base_requirements_hash,
            ),
            state,
            owner=owner,
            conversation_id=conversation_id,
            conversation_revision=conversation_revision,
        )
        return
    if command.target == "CLARIFICATION":
        if (
            pending.clarification is None
            or command.pending_message_id != pending.task_message_id
            or command.constraints_revision != state.revision
        ):
            raise ValueError("Pending clarification source/revision mismatch")
        return
    resolution = (
        pending.pending_resolution if command.target == "REQUIREMENTS" else None
    )
    tasks = pending.pending_task_resolution if command.target == "TASKS" else None
    if resolution:
        if resolution.owner != owner or resolution.conversation_id != conversation_id:
            raise ValueError("Pending action owner/conversation mismatch")
        if resolution.target_revision > conversation_revision:
            raise ValueError("Pending action target publication is not accepted")
    validate_pending_action(
        PendingActionV1(
            action=command.action,
            pending_message_id=command.pending_message_id,
            constraints_revision=command.constraints_revision,
        ),
        state,
        resolution.draft if resolution else None,
        tasks,
    )
