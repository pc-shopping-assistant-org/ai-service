"""Owner/content-bound continuation of unresolved agent references."""

from uuid import UUID, uuid5

from pydantic import Field, StrictInt, model_validator

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    Contract,
    NonNegativeInt,
    UnitClarificationReplyV1,
)

from .admission import RequirementAdmission, propose_agent_patch
from .amendment import PendingAmendment, amend_pending
from .contracts import CanonicalRequirementState, requirements_hash
from .quantity_clarification import confirmations_for_pending
from .resolver import PendingDraft, resolve_draft, selection_for_pending


class PendingResolution(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    owner: UUID
    conversation_id: UUID
    target_revision: NonNegativeInt
    base_requirements_hash: Digest
    draft: PendingDraft

    @model_validator(mode="after")
    def unresolved_only(self) -> "PendingResolution":
        if self.draft.resolution.patch is not None or not self.draft.resolution.pending:
            raise ValueError("Pending resolution must contain unresolved draft items")
        return self


class ResolutionContinuation(Contract):
    pending: PendingResolution | None = None
    admission: RequirementAdmission | None = None

    @model_validator(mode="after")
    def one_result(self) -> "ResolutionContinuation":
        if (self.pending is None) == (self.admission is None):
            raise ValueError("Reference continuation requires one resulting artifact")
        return self


def unit_reply_source_id(
    run_id: UUID, message: str, reply: UnitClarificationReplyV1
) -> UUID:
    return uuid5(
        run_id,
        "unit:"
        + canonical_hash({"message": message, "reply": reply.model_dump(mode="json")}),
    )


def amendment_source_id(
    run_id: UUID, message: str, amendment: PendingAmendment
) -> UUID:
    return uuid5(
        run_id,
        "amendment:"
        + canonical_hash(
            {"message": message, "amendment": amendment.model_dump(mode="json")}
        ),
    )


def continue_amendment(
    state: CanonicalRequirementState,
    pending: PendingResolution,
    amendment: PendingAmendment,
    catalog: list[CatalogRowV1],
    *,
    message_id: UUID,
    message: str,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> ResolutionContinuation:
    state, pending = _bound_pending(
        state,
        pending,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
    modified = amend_pending(
        pending.draft, amendment, state, message_id=message_id, message=message
    )
    return _resolve_continuation(
        state,
        modified,
        catalog,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )


def _bound_pending(
    state: CanonicalRequirementState,
    pending: PendingResolution,
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> tuple[CanonicalRequirementState, PendingResolution]:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    pending = PendingResolution.model_validate(pending.model_dump())
    if (
        pending.owner != owner
        or pending.conversation_id != conversation_id
        or pending.target_revision > conversation_revision
        or pending.base_requirements_hash != requirements_hash(state)
    ):
        raise ValueError("Pending reference owner/publication/content mismatch")
    return state, pending


def _resolve_continuation(
    state: CanonicalRequirementState,
    draft: PendingDraft,
    catalog: list[CatalogRowV1],
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> ResolutionContinuation:
    resolution = resolve_draft(
        draft.resolution.draft,
        message_id=draft.message_id,
        message=draft.message,
        state=state,
        catalog=catalog,
        selections=draft.selections,
        quantity_confirmations=draft.quantity_confirmations,
        item_sources=draft.item_sources,
        discarded_items=draft.discarded_items,
    )
    if resolution.patch is None:
        updated = draft.model_copy(update={"resolution": resolution}, deep=True)
        return ResolutionContinuation(
            pending=PendingResolution(
                owner=owner,
                conversation_id=conversation_id,
                target_revision=conversation_revision + 1,
                base_requirements_hash=requirements_hash(state),
                draft=updated,
            )
        )
    messages = [(draft.message_id, draft.message)]
    messages.extend(
        (source.message_id, source.message) for source in draft.item_sources.values()
    )
    messages.extend(
        (confirmation.evidence.message_id, confirmation.message)
        for confirmation in draft.quantity_confirmations.values()
    )
    sources: dict[UUID, str] = {}
    for identifier, text in messages:
        if identifier in sources and sources[identifier] != text:
            raise ValueError("Conflicting text for the same source message identity")
        sources[identifier] = text
    admission = propose_agent_patch(
        state,
        resolution.patch,
        sources,
        owner=owner,
        conversation_id=conversation_id,
        target_revision=conversation_revision + 1,
    )
    return ResolutionContinuation(admission=admission)


def continue_reference(
    state: CanonicalRequirementState,
    pending: PendingResolution,
    reply: ClarificationReplyV1,
    catalog: list[CatalogRowV1],
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> ResolutionContinuation:
    state, pending = _bound_pending(
        state,
        pending,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
    reply = ClarificationReplyV1.model_validate(reply.model_dump())
    selections = selection_for_pending(pending.draft, reply, state)
    draft = pending.draft.model_copy(update={"selections": selections}, deep=True)
    return _resolve_continuation(
        state,
        draft,
        catalog,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )


def continue_unit(
    state: CanonicalRequirementState,
    pending: PendingResolution,
    reply: UnitClarificationReplyV1,
    catalog: list[CatalogRowV1],
    *,
    message_id: UUID,
    message: str,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> ResolutionContinuation:
    state, pending = _bound_pending(
        state,
        pending,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
    confirmations = confirmations_for_pending(
        pending.draft, reply, state, message_id=message_id, message=message
    )
    draft = pending.draft.model_copy(
        update={"quantity_confirmations": confirmations}, deep=True
    )
    return _resolve_continuation(
        state,
        draft,
        catalog,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
