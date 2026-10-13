"""Application admission for agent-origin patches; never mutates accepted state.

Reference/unit resolution must precede this boundary. Exact source spans prove
binding, not user intent: a changed agent patch always needs confirmation here.
Explicit-user semantic admission is a separate trusted path, not a model flag.
"""

from typing import Literal
from uuid import UUID

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.stateful_contracts import Contract

from .contracts import CanonicalRequirementState, RequirementPatch, requirements_hash
from .merge import merge_requirements
from .proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
    confirm_proposal,
    issue_proposal,
)


class RequirementAdmission(Contract):
    disposition: Literal["ACCEPT", "NEEDS_CONFIRMATION"]
    requirements: CanonicalRequirementState
    previous_hash: Digest
    pending: PendingRequirementProposal | None = None

    @property
    def content_changed(self) -> bool:
        """Graph must invalidate dependent artifacts, not relabel its old ledger."""
        return self.previous_hash != requirements_hash(self.requirements)


def propose_agent_patch(
    state: CanonicalRequirementState,
    patch: RequirementPatch,
    source_messages: dict[UUID, str],
    *,
    owner: UUID,
    conversation_id: UUID,
    target_revision: int,
    pending: PendingRequirementProposal | None = None,
) -> RequirementAdmission:
    original = CanonicalRequirementState.model_validate(state.model_dump())
    before_hash = requirements_hash(original)
    if pending is not None:
        pending = PendingRequirementProposal.model_validate(pending.model_dump())
        if (
            pending.owner != owner
            or pending.conversation_id != conversation_id
            or pending.base_requirements_hash != before_hash
        ):
            raise ValueError("Pending proposal does not belong to current state")
    candidate = merge_requirements(original, patch, source_messages)
    if requirements_hash(candidate) == before_hash:
        return RequirementAdmission(
            disposition="ACCEPT",
            requirements=original,
            previous_hash=before_hash,
            pending=pending,
        )
    if pending is not None:
        raise ValueError("A proposal is already pending; cancel or confirm it first")
    proposal = issue_proposal(
        original,
        patch,
        source_messages,
        owner=owner,
        conversation_id=conversation_id,
        target_revision=target_revision,
    )
    return RequirementAdmission(
        disposition="NEEDS_CONFIRMATION",
        requirements=original,
        previous_hash=before_hash,
        pending=proposal,
    )


def confirm_agent_patch(
    state: CanonicalRequirementState,
    pending: PendingRequirementProposal | None,
    reply: ProposalConfirmation,
    *,
    owner: UUID,
    conversation_id: UUID,
    conversation_revision: int,
) -> RequirementAdmission:
    if pending is None:
        raise ValueError("No pending proposal to confirm")
    # Revalidate and copy before either confirmation or merge: nested frozen models
    # are not deeply immutable and must never be trusted from in-memory mutation.
    pending = PendingRequirementProposal.model_validate(pending.model_dump())
    original = CanonicalRequirementState.model_validate(state.model_dump())
    patch = confirm_proposal(
        pending,
        reply,
        original,
        owner=owner,
        conversation_id=conversation_id,
        conversation_revision=conversation_revision,
    )
    merged = merge_requirements(original, patch, pending.source_messages)
    return RequirementAdmission(
        disposition="ACCEPT",
        requirements=merged,
        previous_hash=requirements_hash(original),
        pending=None,
    )
