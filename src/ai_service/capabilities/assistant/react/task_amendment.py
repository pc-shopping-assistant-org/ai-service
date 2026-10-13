"""Bounded, source-bound subject corrections; task kind and other bindings stay fixed."""

from typing import Protocol
from uuid import UUID

from pydantic import Field, model_validator

from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskEvidence
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
)

from .task_resolution import PendingTaskDraft, TaskMentionOverride


class TaskMentionEdit(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt
    evidence: TaskEvidence


class TaskAmendment(Contract):
    edits: list[TaskMentionEdit] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def unique_targets(self) -> "TaskAmendment":
        if len({(item.task_index, item.subject_index) for item in self.edits}) != len(
            self.edits
        ):
            raise ValueError("Duplicate task mention amendment")
        return self


class TaskAmender(Protocol):
    async def amend_task(
        self, message: str, pending: PendingTaskDraft
    ) -> TaskAmendment: ...


def amend_pending_task(
    pending: PendingTaskDraft,
    amendment: TaskAmendment,
    state: CanonicalRequirementState,
    *,
    message_id: UUID,
    message: str,
) -> PendingTaskDraft:
    pending = PendingTaskDraft.model_validate(pending.model_dump())
    amendment = TaskAmendment.model_validate(amendment.model_dump())
    if pending.constraints_revision != state.revision:
        raise ValueError("Stale task amendment revision")
    result = pending.model_copy(deep=True)
    overrides = {
        (item.task_index, item.subject_index): item for item in result.mention_overrides
    }
    for item in amendment.edits:
        item.evidence.validate_source(message)
        if (
            item.task_index >= len(result.tasks)
            or result.tasks[item.task_index].kind != "COMPARE"
            or item.subject_index >= len(result.tasks[item.task_index].subjects)
        ):
            raise ValueError("Invalid task amendment target")
        key = (item.task_index, item.subject_index)
        overrides[key] = TaskMentionOverride(
            task_index=item.task_index,
            subject_index=item.subject_index,
            evidence=item.evidence,
            message_id=message_id,
            message=message,
        )
        result.resolution.bindings = [
            binding
            for binding in result.resolution.bindings
            if (binding.task_index, binding.subject_index) != key
        ]
    result.mention_overrides = list(overrides.values())
    return PendingTaskDraft.model_validate(result.model_dump())
