"""Application-owned pending proposals. Confirmation returns a patch, never publishes.

Agent-origin evidence describes the immutable proposal, not original user intent.
F02 owns admission, semantic policy, single pending slot and actual merge/publication.
"""

from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, StrictInt, model_validator

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    Text,
)

from .contracts import CanonicalRequirementState, RequirementPatch, requirements_hash
from .merge import merge_requirements


class ProposalPayload(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    proposal_id: UUID
    owner: UUID
    conversation_id: UUID
    target_revision: NonNegativeInt
    base_requirements_hash: Digest
    origin: Literal["AGENT_PROPOSED"] = "AGENT_PROPOSED"
    patch: RequirementPatch
    source_messages: dict[UUID, Annotated[Text, Field(max_length=4000)]] = Field(
        min_length=1, max_length=8
    )


class PendingRequirementProposal(ProposalPayload):
    proposal_hash: Digest

    @model_validator(mode="after")
    def immutable_payload(self) -> "PendingRequirementProposal":
        payload = self.model_dump(mode="json", exclude={"proposal_hash"})
        if self.proposal_hash != canonical_hash(payload):
            raise ValueError("Proposal hash does not match immutable payload")
        return self


class ProposalConfirmation(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    proposal_id: UUID
    proposal_hash: Digest
    target_revision: NonNegativeInt
    base_requirements_hash: Digest


def issue_proposal(
    state: CanonicalRequirementState,
    patch: RequirementPatch,
    source_messages: dict[UUID, str],
    *,
    owner: UUID,
    conversation_id: UUID,
    target_revision: int,
) -> PendingRequirementProposal:
    # Validate source/CLEAR identities against a copy; never mutate accepted state.
    merge_requirements(state, patch, source_messages)
    payload = ProposalPayload(
        proposal_id=uuid4(),
        owner=owner,
        conversation_id=conversation_id,
        target_revision=target_revision,
        base_requirements_hash=requirements_hash(state),
        patch=patch.model_copy(deep=True),
        source_messages=dict(source_messages),
    )
    return PendingRequirementProposal(
        **payload.model_dump(),
        proposal_hash=canonical_hash(payload),
    )


def confirm_proposal(
    pending: PendingRequirementProposal,
    reply: ProposalConfirmation,
    state: CanonicalRequirementState,
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> RequirementPatch:
    pending = PendingRequirementProposal.model_validate(pending.model_dump())
    reply = ProposalConfirmation.model_validate(reply.model_dump())
    if pending.owner != owner or pending.conversation_id != conversation_id:
        raise ValueError("Proposal owner/conversation mismatch")
    if any(
        getattr(reply, field) != getattr(pending, field)
        for field in (
            "proposal_id",
            "proposal_hash",
            "target_revision",
            "base_requirements_hash",
        )
    ):
        raise ValueError("Confirmation does not match pending proposal")
    if pending.target_revision > conversation_revision:
        raise ValueError("Proposal target state is not published")
    if requirements_hash(state) != pending.base_requirements_hash:
        raise ValueError("Proposal base state changed")
    merge_requirements(state, pending.patch, pending.source_messages)
    return pending.patch.model_copy(deep=True)
