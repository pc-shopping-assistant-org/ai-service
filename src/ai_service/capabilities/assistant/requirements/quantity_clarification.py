"""Bind explicit unit choices to a frozen draft and both user sources."""

from typing import Protocol
from uuid import UUID

from pydantic import Field

from ai_service.capabilities.assistant.public_contracts import PublicClarificationV1
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    UnitClarificationReplyV1,
    UnitSelectionV1,
)

from .contracts import CanonicalRequirementState, SourceEvidence
from .merge import validate_evidence
from .resolver import PendingDraft
from .units import (
    MEASUREMENT,
    QuantityConfirmation,
    UnresolvedQuantityValue,
    confirmed_quantity,
    unresolved_amount,
    verify_quantity,
)


class UnitChoiceDraft(Contract):
    selections: list[UnitSelectionV1] = Field(default_factory=list, max_length=100)


class UnitClarificationExtractor(Protocol):
    async def choose_unit(
        self, message: str, options: PublicClarificationV1
    ) -> UnitChoiceDraft: ...


def unit_choice_reply(
    choice: UnitChoiceDraft, pending: PendingDraft
) -> UnitClarificationReplyV1 | None:
    choice = UnitChoiceDraft.model_validate(choice.model_dump())
    if not choice.selections:
        return None
    return UnitClarificationReplyV1(
        pending_message_id=pending.message_id,
        constraints_revision=pending.constraints_revision,
        selections=choice.selections,
    )


def confirmations_for_pending(
    pending: PendingDraft,
    reply: UnitClarificationReplyV1,
    state: CanonicalRequirementState,
    *,
    message_id: UUID,
    message: str,
) -> dict[int, QuantityConfirmation]:
    reply = UnitClarificationReplyV1.model_validate(reply.model_dump())
    if (
        pending.message_id != reply.pending_message_id
        or pending.constraints_revision != state.revision
        or reply.constraints_revision != state.revision
    ):
        raise ValueError("Stale unit clarification source/revision")
    offered = {
        item.item_index
        for item in pending.resolution.pending
        if item.reason == "QUANTITY_UNIT_REQUIRED"
    }
    confirmations = dict(pending.quantity_confirmations)
    for selection in reply.selections:
        if selection.item_index not in offered:
            raise ValueError("Invalid quantity clarification item")
        item = pending.resolution.draft.items[selection.item_index]
        if not isinstance(item.value, UnresolvedQuantityValue):
            raise TypeError("Invalid unresolved quantity")
        if (
            unresolved_amount(
                pending.source_for(selection.item_index).message[item.start : item.end]
            )
            != item.value.amount
        ):
            raise ValueError("Invalid original quantity evidence")
        evidence = SourceEvidence(
            message_id=message_id,
            start=selection.start,
            end=selection.end,
            quote=selection.quote,
        )
        validate_evidence(evidence, {message_id: message})
        value = confirmed_quantity(item.value.amount, selection.quote, selection.unit)
        # A unit-only reply must not discard an explicitly changed adjacent number.
        for measurement in MEASUREMENT.finditer(message):
            if (
                measurement.span(2) == (selection.start, selection.end)
                and verify_quantity(measurement.group(), value)[0] is None
            ):
                raise ValueError("Unit reply cannot change the original amount")
        confirmations[selection.item_index] = QuantityConfirmation(
            value=value,
            evidence=evidence,
            message=message,
        )
    return confirmations
