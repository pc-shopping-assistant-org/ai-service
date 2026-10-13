"""Source-bound edits to unmerged requirements, not a new expression language."""

from typing import Protocol
from uuid import UUID

from pydantic import Field, model_validator

from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
)

from .contracts import CanonicalRequirementState, SourceEvidence
from .merge import validate_evidence
from .resolver import DraftItem, DraftSource, PendingDraft


class ItemReplacement(Contract):
    item_index: NonNegativeInt
    item: DraftItem


class ItemDiscard(Contract):
    item_index: NonNegativeInt
    start: NonNegativeInt
    end: NonNegativeInt
    quote: str = Field(min_length=1, max_length=4000)


class PendingAmendment(Contract):
    replacements: list[ItemReplacement] = Field(default_factory=list, max_length=100)
    discards: list[ItemDiscard] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def unique_items(self) -> "PendingAmendment":
        indexes = [item.item_index for item in self.replacements + self.discards]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Duplicate draft amendment target")
        return self


class PendingAmender(Protocol):
    async def amend(self, message: str, pending: PendingDraft) -> PendingAmendment: ...


def amend_pending(
    pending: PendingDraft,
    amendment: PendingAmendment,
    state: CanonicalRequirementState,
    *,
    message_id: UUID,
    message: str,
) -> PendingDraft:
    pending = PendingDraft.model_validate(pending.model_dump())
    amendment = PendingAmendment.model_validate(amendment.model_dump())
    if pending.constraints_revision != state.revision:
        raise ValueError("Stale pending amendment revision")
    result = pending.model_copy(deep=True)
    source = DraftSource(message_id=message_id, message=message)
    operations: list[ItemReplacement | ItemDiscard] = [
        *amendment.replacements,
        *amendment.discards,
    ]
    for operation in operations:
        index = operation.item_index
        if (
            index >= len(result.resolution.draft.items)
            or index in result.discarded_items
        ):
            raise ValueError("Unavailable draft amendment item")
        original = result.resolution.draft.items[index]
        if isinstance(operation, ItemReplacement):
            item = operation.item
            if (item.target, item.subject, item.predicate, item.target_id) != (
                original.target,
                original.subject,
                original.predicate,
                original.target_id,
            ) or item.operation != "SET":
                raise ValueError("Replacement cannot change the draft target identity")
            if item.end <= item.start or item.end > len(message):
                raise ValueError("Invalid amendment evidence span")
            result.resolution.draft.items[index] = item
        else:
            validate_evidence(
                SourceEvidence(
                    message_id=message_id,
                    start=operation.start,
                    end=operation.end,
                    quote=operation.quote,
                ),
                {message_id: message},
            )
            result.discarded_items.append(index)
        result.item_sources[index] = source
        result.selections.pop(index, None)
        result.quantity_confirmations.pop(index, None)
    result.discarded_items.sort()
    result.resolution.patch = None
    return PendingDraft.model_validate(result.model_dump())
