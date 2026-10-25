"""Persisted V2 task/pending context. No execution state reconstruction or counters."""

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from ai_service.capabilities.assistant.public_contracts import PublicClarificationV1
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

from .actions import AskClarificationAction
from .contracts import ConversationResponse
from .outcomes import FlexibleAnswer, TaskOutcome
from .task_resolution import PendingTaskDraft


class FlexiblePublishedState(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    conversation_id: UUID
    task_message_id: UUID
    task_message: Text = Field(max_length=4000)
    status: Literal["COMPLETED", "UNRESOLVED", "WAITING_INPUT", "DISCARDED"]
    administrative_message: Text | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    answer: FlexibleAnswer | None = None
    response: ConversationResponse | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    waiting_message: Text | None = None
    waiting_tasks: list[TaskOutcome] = Field(default_factory=list, max_length=8)
    pending_proposal: PendingRequirementProposal | None = None
    pending_resolution: PendingResolution | None = None
    clarification: AskClarificationAction | None = None
    pending_task_resolution: PendingTaskDraft | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    resolution_options: PublicClarificationV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def task_outcomes(self) -> list[TaskOutcome]:
        return self.answer.task_outcomes if self.answer else self.waiting_tasks

    @property
    def assistant_message(self) -> str:
        if self.response is not None:
            return self.response.text
        if self.status == "DISCARDED":
            assert self.administrative_message is not None
            return self.administrative_message
        if self.answer:
            return self.answer.text
        assert self.waiting_message is not None
        return self.waiting_message

    @model_validator(mode="after")
    def state_consistency(self) -> "FlexiblePublishedState":
        artifacts = [
            self.pending_proposal,
            self.pending_resolution,
            self.clarification,
            self.pending_task_resolution,
        ]
        if self.status == "DISCARDED":
            if (
                self.response is not None
                or self.answer is not None
                or self.administrative_message is None
                or self.waiting_message is not None
                or any(item is not None for item in artifacts)
                or self.resolution_options is not None
                or any(task.status != "UNRESOLVED" for task in self.waiting_tasks)
            ):
                raise ValueError(
                    "Discard cannot publish an answer, pending artifact or completed task"
                )
        elif self.administrative_message is not None:
            raise ValueError(
                "Only discarded state has an administrative acknowledgement"
            )
        elif self.status == "WAITING_INPUT":
            if (
                self.answer is not None
                or not self.waiting_message
                or sum(item is not None for item in artifacts) != 1
            ):
                raise ValueError(
                    "Waiting state requires one artifact and unresolved original tasks"
                )
            if any(task.status != "UNRESOLVED" for task in self.waiting_tasks):
                raise ValueError("Waiting state cannot complete requested tasks")
            if self.pending_task_resolution and (
                self.waiting_tasks
                or self.pending_task_resolution.message_id != self.task_message_id
                or self.pending_task_resolution.message != self.task_message
            ):
                raise ValueError(
                    "Unresolved task identity must preserve original source"
                )
        elif self.response is not None:
            if (
                self.status != "COMPLETED"
                or self.answer is not None
                or self.waiting_message is not None
                or self.waiting_tasks
                or any(item is not None for item in artifacts)
                or self.resolution_options is not None
            ):
                raise ValueError("A standalone response cannot assert task completion")
        elif (
            self.answer is None
            or self.answer.status != self.status
            or self.waiting_message is not None
            or self.waiting_tasks
            or any(item is not None for item in artifacts)
        ):
            raise ValueError(
                "Final state must contain one validated answer, not pending data"
            )
        for pending in (self.pending_proposal, self.pending_resolution):
            if pending and pending.conversation_id != self.conversation_id:
                raise ValueError("Pending artifact belongs to another conversation")
        if self.resolution_options and (
            self.pending_resolution is None
            or self.resolution_options.pending_message_id
            != self.pending_resolution.draft.message_id
            or self.resolution_options.constraints_revision
            != self.pending_resolution.draft.constraints_revision
        ):
            raise ValueError("Public resolution options must bind to the pending draft")
        return self
