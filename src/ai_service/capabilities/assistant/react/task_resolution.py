"""Resolve source-bound task proposals before the model can choose tools."""

from typing import Literal, Protocol
from uuid import UUID

from pydantic import Field, model_validator

from ai_service.capabilities.assistant.public_contracts import (
    PublicTaskClarificationV1,
    TaskClarificationItemV1,
    TaskProductOptionV1,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraft, TaskEvidence
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    TaskClarificationReplyV1,
    TaskProductSelectionV1,
    Text,
)

from .contracts import CompletionRequirement, ProductEvidence


class TaskCatalog(Protocol):
    async def resolve(self, mention: str) -> list[ProductEvidence]:
        """All bounded canonical matches, not a ranked/truncated top-k page."""
        ...


class PendingTask(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt | None = None
    reason: Literal[
        "NOT_FOUND",
        "AMBIGUOUS",
        "DISTINCT_PRODUCTS_REQUIRED",
        "SNAPSHOT_REQUIRED",
        "SELECTED_PRODUCT_UNAVAILABLE",
    ]
    candidates: list[ProductEvidence] = Field(default_factory=list, max_length=100)


class TaskResolution(Contract):
    requirements: list[CompletionRequirement] = Field(
        default_factory=list, max_length=8
    )
    pending: list[PendingTask] = Field(default_factory=list, max_length=40)
    bindings: list[TaskProductSelectionV1] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def consistent_resolution(self) -> "TaskResolution":
        keys = {(item.task_index, item.subject_index) for item in self.bindings}
        if len(keys) != len(self.bindings) or (self.pending and self.requirements):
            raise ValueError("Inconsistent task resolution")
        return self


class TaskMentionOverride(Contract):
    task_index: NonNegativeInt
    subject_index: NonNegativeInt
    evidence: TaskEvidence
    message_id: UUID
    message: Text = Field(max_length=4000)


def validate_mention_overrides(
    tasks: list[TaskDraft], overrides: list[TaskMentionOverride]
) -> None:
    keys = [(item.task_index, item.subject_index) for item in overrides]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task mention override")
    for item in overrides:
        if (
            item.task_index >= len(tasks)
            or tasks[item.task_index].kind != "COMPARE"
            or item.subject_index >= len(tasks[item.task_index].subjects)
        ):
            raise ValueError("Invalid task mention override target")
        item.evidence.validate_source(item.message)


class PendingTaskDraft(Contract):
    message_id: UUID
    message: Text = Field(max_length=4000)
    constraints_revision: NonNegativeInt
    tasks: list[TaskDraft] = Field(min_length=1, max_length=8)
    resolution: TaskResolution
    mention_overrides: list[TaskMentionOverride] = Field(
        default_factory=list, max_length=40, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def source_bound_tasks(self) -> "PendingTaskDraft":
        for task in self.tasks:
            task.validate_source(self.message)
        if not self.resolution.pending:
            raise ValueError("Pending task artifact must contain unresolved tasks")
        for item in self.resolution.pending:
            if item.task_index >= len(self.tasks) or (
                item.subject_index is not None
                and item.subject_index >= len(self.tasks[item.task_index].subjects)
            ):
                raise ValueError("Pending task subject is invalid")
        _validate_bindings(self.tasks, self.resolution.bindings)
        validate_mention_overrides(self.tasks, self.mention_overrides)
        return self


class TaskChoiceSelection(TaskProductSelectionV1):
    evidence: TaskEvidence


class TaskChoiceDraft(Contract):
    selections: list[TaskChoiceSelection] = Field(default_factory=list, max_length=40)


class TaskClarificationExtractor(Protocol):
    async def choose_task(
        self, message: str, options: PublicTaskClarificationV1
    ) -> TaskChoiceDraft: ...


def task_choice_reply(
    choice: TaskChoiceDraft, message: str, pending: PendingTaskDraft
) -> TaskClarificationReplyV1 | None:
    choice = TaskChoiceDraft.model_validate(choice.model_dump())
    for item in choice.selections:
        item.evidence.validate_source(message)
    if not choice.selections:
        return None
    return TaskClarificationReplyV1(
        pending_message_id=pending.message_id,
        constraints_revision=pending.constraints_revision,
        selections=[
            TaskProductSelectionV1(
                task_index=item.task_index,
                subject_index=item.subject_index,
                product_id=item.product_id,
            )
            for item in choice.selections
        ],
    )


def _validate_bindings(
    tasks: list[TaskDraft], bindings: list[TaskProductSelectionV1]
) -> None:
    if len({(item.task_index, item.subject_index) for item in bindings}) != len(
        bindings
    ):
        raise ValueError("Duplicate task product binding")
    for item in bindings:
        if (
            item.task_index >= len(tasks)
            or tasks[item.task_index].kind != "COMPARE"
            or item.subject_index >= len(tasks[item.task_index].subjects)
        ):
            raise ValueError("Invalid task product binding")


def selections_for_pending_task(
    pending: PendingTaskDraft,
    reply: TaskClarificationReplyV1,
    state: CanonicalRequirementState,
) -> list[TaskProductSelectionV1]:
    pending = PendingTaskDraft.model_validate(pending.model_dump())
    reply = TaskClarificationReplyV1.model_validate(reply.model_dump())
    if (
        reply.pending_message_id != pending.message_id
        or reply.constraints_revision != pending.constraints_revision
        or state.revision != pending.constraints_revision
    ):
        raise ValueError("Stale task clarification source/revision")
    offered = {
        (item.task_index, item.subject_index): item
        for item in pending.resolution.pending
    }
    selected = {
        (item.task_index, item.subject_index): item
        for item in pending.resolution.bindings
    }
    for item in reply.selections:
        candidate = offered.get((item.task_index, item.subject_index))
        if candidate is None or item.product_id not in {
            p.product_id for p in candidate.candidates
        }:
            raise ValueError("Invalid task clarification candidate")
        selected[item.task_index, item.subject_index] = item
    return list(selected.values())


def public_task_options(pending: PendingTaskDraft) -> PublicTaskClarificationV1:
    pending = PendingTaskDraft.model_validate(pending.model_dump())
    mentions = {
        (item.task_index, item.subject_index): item.evidence.quote
        for item in pending.mention_overrides
    }
    return PublicTaskClarificationV1(
        pending_message_id=pending.message_id,
        constraints_revision=pending.constraints_revision,
        items=[
            TaskClarificationItemV1(
                task_index=item.task_index,
                subject_index=item.subject_index,
                mention=mentions.get(
                    (item.task_index, item.subject_index),
                    pending.tasks[item.task_index].subjects[item.subject_index].quote,
                )
                if item.subject_index is not None
                else pending.tasks[item.task_index].quote,
                reason=item.reason,
                options=[
                    TaskProductOptionV1(product_id=p.product_id, name=p.name)
                    for p in item.candidates
                ],
            )
            for item in pending.resolution.pending
        ],
    )


async def resolve_tasks(
    tasks: list[TaskDraft],
    message: str,
    state: CanonicalRequirementState,
    catalog: TaskCatalog,
    *,
    saved_build_ref: str | None = None,
    selections: list[TaskProductSelectionV1] | None = None,
    mention_overrides: list[TaskMentionOverride] | None = None,
) -> TaskResolution:
    if not tasks or len(tasks) > 8:
        raise ValueError("TASK_DRAFT_REQUIRED")
    tasks = [TaskDraft.model_validate(task.model_dump()) for task in tasks]
    state = CanonicalRequirementState.model_validate(state.model_dump())
    # Validate the entire proposal before any outbound lookup.
    for task in tasks:
        task.validate_source(message)
    overrides = [
        TaskMentionOverride.model_validate(item.model_dump())
        for item in mention_overrides or []
    ]
    validate_mention_overrides(tasks, overrides)
    mentions = {
        (item.task_index, item.subject_index): item.evidence.quote for item in overrides
    }
    bindings = [
        TaskProductSelectionV1.model_validate(item.model_dump())
        for item in selections or []
    ]
    _validate_bindings(tasks, bindings)
    selected = {
        (item.task_index, item.subject_index): item.product_id for item in bindings
    }
    result = TaskResolution()
    for task_index, task in enumerate(tasks):
        if task.kind == "BUILD":
            result.requirements.append(
                CompletionRequirement(kind="BUILD", constraints_revision=state.revision)
            )
        elif task.kind == "ANSWER":
            result.requirements.append(CompletionRequirement(kind="ANSWER"))
        elif task.kind == "EXPLAIN_SAVED_BUILD":
            if saved_build_ref is None:
                result.pending.append(
                    PendingTask(task_index=task_index, reason="SNAPSHOT_REQUIRED")
                )
            else:
                result.requirements.append(
                    CompletionRequirement(
                        kind="EXPLAIN_SAVED_BUILD", saved_build_ref=saved_build_ref
                    )
                )
        else:
            identifiers: list[UUID] = []
            for subject_index, subject in enumerate(task.subjects):
                matches = [
                    ProductEvidence.model_validate(p.model_dump())
                    for p in await catalog.resolve(
                        mentions.get((task_index, subject_index), subject.quote)
                    )
                ]
                if len(matches) > 100 or len({p.product_id for p in matches}) != len(
                    matches
                ):
                    raise ValueError("Invalid canonical task candidate set")
                chosen = selected.get((task_index, subject_index))
                available = {p.product_id for p in matches}
                if chosen is not None:
                    # Preserve the binding even if unavailable, so we cannot silently
                    # select the one remaining match during a later continuation.
                    result.bindings.append(
                        TaskProductSelectionV1(
                            task_index=task_index,
                            subject_index=subject_index,
                            product_id=chosen,
                        )
                    )
                    if chosen in available:
                        identifiers.append(chosen)
                    else:
                        result.pending.append(
                            PendingTask(
                                task_index=task_index,
                                subject_index=subject_index,
                                reason="SELECTED_PRODUCT_UNAVAILABLE",
                                candidates=matches,
                            )
                        )
                elif len(matches) != 1:
                    result.pending.append(
                        PendingTask(
                            task_index=task_index,
                            subject_index=subject_index,
                            reason="AMBIGUOUS" if matches else "NOT_FOUND",
                            candidates=matches,
                        )
                    )
                else:
                    identifiers.append(matches[0].product_id)
                    result.bindings.append(
                        TaskProductSelectionV1(
                            task_index=task_index,
                            subject_index=subject_index,
                            product_id=matches[0].product_id,
                        )
                    )
            if len(identifiers) == len(task.subjects):
                if len(set(identifiers)) != len(identifiers):
                    result.pending.append(
                        PendingTask(
                            task_index=task_index, reason="DISTINCT_PRODUCTS_REQUIRED"
                        )
                    )
                else:
                    result.requirements.append(
                        CompletionRequirement(kind="COMPARE", product_ids=identifiers)
                    )
    if result.pending:
        result.requirements = []
    return result
