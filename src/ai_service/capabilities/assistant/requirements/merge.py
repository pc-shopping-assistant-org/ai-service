"""Atomic evidence-bound merge. Callers must separately verify extraction semantics."""

from uuid import UUID

from .contracts import (
    CanonicalRequirementState,
    Evidence,
    Requirement,
    RequirementPatch,
    Resource,
    SourceEvidence,
)


def _apply[T: (Requirement, Resource)](
    values: dict[UUID, T],
    setting: list[T],
    clearing: list[UUID],
    messages: dict[UUID, str],
) -> None:
    for item in setting:
        validate_evidence(item.evidence, messages)
        values[item.id] = item
    for identifier in clearing:
        if identifier not in values:
            raise ValueError("CLEAR must target an existing identity")
        del values[identifier]


def validate_evidence(evidence: SourceEvidence, messages: dict[UUID, str]) -> None:
    message = messages.get(evidence.message_id)
    if (
        message is None
        or evidence.end > len(message)
        or message[evidence.start : evidence.end] != evidence.quote
    ):
        raise ValueError("Unbound or forged source evidence")
    if isinstance(evidence, Evidence):
        for supporting in evidence.supporting_evidence:
            validate_evidence(supporting, messages)


def merge_requirements(
    state: CanonicalRequirementState,
    patch: RequirementPatch,
    messages: dict[UUID, str],
) -> CanonicalRequirementState:
    original = CanonicalRequirementState.model_validate(state.model_dump())
    patch = RequirementPatch.model_validate(patch.model_dump())
    for evidence in patch.clear_evidence.values():
        validate_evidence(evidence, messages)
    merged = original.model_copy(deep=True)
    _apply(
        merged.requirements, patch.set_requirements, patch.clear_requirements, messages
    )
    _apply(merged.goals, patch.set_goals, patch.clear_goals, messages)
    _apply(
        merged.available_resources, patch.set_resources, patch.clear_resources, messages
    )
    changed_ids = {
        item.id
        for item in patch.set_requirements
        if original.requirements.get(item.id) != item
    } | set(patch.clear_requirements)
    for identifier in changed_ids:
        merged.resolved_bindings.pop(identifier, None)
    merged = CanonicalRequirementState.model_validate(merged.model_dump())
    if merged.model_dump() != original.model_dump():
        merged.revision += 1
    return merged
