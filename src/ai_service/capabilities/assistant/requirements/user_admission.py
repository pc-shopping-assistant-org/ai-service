"""Application-owned admission after independent explicit-user extraction.

Matching drafts is a bounded semantic policy, not a mathematical intent proof.
Receipts are private native execution artifacts, never model actions/API commands.
The executor supplies the extractor result and the real run/message/state context;
publication must rederive the transition and still verify native lineage/stop/CAS.
"""

from typing import Literal
from uuid import UUID, uuid5

from pydantic import ConfigDict

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.react.actions import ProposeRequirementsAction
from ai_service.capabilities.assistant.stateful_contracts import Contract

from .contracts import CanonicalRequirementState, requirements_hash
from .merge import merge_requirements
from .resolver import Draft, resolve_draft


class UserRequirementReceipt(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: Literal["explicit-draft-match-v1"] = "explicit-draft-match-v1"
    run_id: UUID
    message_hash: Digest
    base_requirements_hash: Digest
    proposed: ProposeRequirementsAction
    extracted: Draft


def _same_items(proposed: Draft, extracted: Draft) -> bool:
    # Match the whole extraction, not a subset that silently drops user constraints.
    return sorted(canonical_hash(item) for item in proposed.items) == sorted(
        canonical_hash(item) for item in extracted.items
    )


def assess_user_draft(
    state: CanonicalRequirementState,
    proposed: ProposeRequirementsAction,
    extracted: Draft,
    *,
    run_id: UUID,
    message: str,
    catalog: list[CatalogRowV1],
) -> UserRequirementReceipt | None:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    proposed = ProposeRequirementsAction.model_validate(proposed.model_dump())
    extracted = Draft.model_validate(extracted.model_dump())
    for task in extracted.tasks:
        task.validate_source(message)
    if proposed.source_text is not None or not _same_items(proposed.draft, extracted):
        return None
    receipt = UserRequirementReceipt(
        run_id=run_id,
        message_hash=canonical_hash(message),
        base_requirements_hash=requirements_hash(state),
        proposed=proposed,
        extracted=extracted,
    )
    resolution = resolve_draft(
        proposed.draft,
        message_id=uuid5(run_id, "explicit-user-requirements"),
        message=message,
        state=state,
        catalog=catalog,
    )
    # No guessed units/UUIDs, partial merge or silent unsupported-item removal.
    return receipt if resolution.patch is not None else None


def apply_user_receipt(
    state: CanonicalRequirementState,
    receipt: UserRequirementReceipt,
    *,
    run_id: UUID,
    message: str,
    catalog: list[CatalogRowV1],
) -> CanonicalRequirementState:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    receipt = UserRequirementReceipt.model_validate(receipt.model_dump())
    if (
        receipt.run_id != run_id
        or receipt.message_hash != canonical_hash(message)
        or receipt.base_requirements_hash != requirements_hash(state)
        or receipt.proposed.source_text is not None
        or not _same_items(receipt.proposed.draft, receipt.extracted)
    ):
        raise ValueError("User admission receipt does not match current source/state")
    for task in receipt.extracted.tasks:
        task.validate_source(message)
    source = uuid5(run_id, "explicit-user-requirements")
    resolution = resolve_draft(
        receipt.proposed.draft,
        message_id=source,
        message=message,
        state=state,
        catalog=catalog,
    )
    if resolution.patch is None:
        raise ValueError("User admission requires a completely resolved draft")
    return merge_requirements(state, resolution.patch, {source: message})
