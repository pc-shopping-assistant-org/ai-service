"""Application-owned admission after independent explicit-user extraction.

Matching drafts is a bounded semantic policy, not a mathematical intent proof.
Receipts are private native execution artifacts, never model actions/API commands.
The executor supplies the extractor result and the real run/message/state context;
publication must rederive the transition and still verify native lineage/stop/CAS.
"""

from typing import Literal
from uuid import UUID, uuid5

from pydantic import ConfigDict, JsonValue, ValidationError

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    ProposeRequirementsAction,
    decode_flexible_decision,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract

from .contracts import CanonicalRequirementState, requirements_hash
from .merge import merge_requirements
from .resolver import Draft, RequirementDraft, resolve_draft


class UserRequirementReceipt(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: Literal["explicit-draft-match-v1"] = "explicit-draft-match-v1"
    run_id: UUID
    message_hash: Digest
    base_requirements_hash: Digest
    proposed: ProposeRequirementsAction
    extracted: RequirementDraft


class _LegacyUserRequirementReceipt(Contract):
    """Receipt shape persisted by the combined Draft requirement extractor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: Literal["explicit-draft-match-v1"] = "explicit-draft-match-v1"
    run_id: UUID
    message_hash: Digest
    base_requirements_hash: Digest
    proposed: dict[str, JsonValue]
    extracted: Draft


def decode_user_requirement_receipt(
    value: object, *, message: str
) -> UserRequirementReceipt:
    """Validate legacy source evidence, then project only requirement items."""
    try:
        return UserRequirementReceipt.model_validate(value)
    except ValidationError as current_error:
        try:
            legacy = _LegacyUserRequirementReceipt.model_validate(value)
        except ValidationError:
            raise current_error
        for task in legacy.extracted.tasks:
            task.validate_source(message)
        decision = decode_flexible_decision(
            {"schema_version": 2, "action": legacy.proposed}
        )
        if not isinstance(decision.action, ProposeRequirementsAction):
            raise TypeError("Legacy user receipt must contain a requirement proposal")
        return UserRequirementReceipt(
            policy_version=legacy.policy_version,
            run_id=legacy.run_id,
            message_hash=legacy.message_hash,
            base_requirements_hash=legacy.base_requirements_hash,
            proposed=decision.action,
            extracted=RequirementDraft(items=legacy.extracted.items),
        )


def _same_items(proposed: RequirementDraft, extracted: RequirementDraft) -> bool:
    # Match the whole extraction, not a subset that silently drops user constraints.
    return sorted(canonical_hash(item) for item in proposed.items) == sorted(
        canonical_hash(item) for item in extracted.items
    )


def assess_user_draft(
    state: CanonicalRequirementState,
    proposed: ProposeRequirementsAction,
    extracted: RequirementDraft,
    *,
    run_id: UUID,
    message: str,
    catalog: list[CatalogRowV1],
) -> UserRequirementReceipt | None:
    state = CanonicalRequirementState.model_validate(state.model_dump())
    proposed = ProposeRequirementsAction.model_validate(proposed.model_dump())
    extracted = RequirementDraft.model_validate(extracted.model_dump())
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
