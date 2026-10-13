"""Authenticated application records; no SQL or checkpoint framework types."""

from datetime import datetime
from typing import Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    MessageViewV1,
    PageV1,
    PublicBuildV1,
    PublicClarificationV1,
    PublicTaskClarificationV1,
    RunViewV1,
)
from ai_service.capabilities.assistant.react.published_state import (
    FlexiblePublishedState,
)
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    public_task_options,
)
from ai_service.capabilities.assistant.react.turn_contracts import TurnRequest
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    Contract,
)


class ConversationRecord(Contract):
    id: UUID
    title: str | None
    revision: int = Field(ge=0)
    accepted_checkpoint_id: str | None
    active_run_id: UUID | None
    created_at: AwareDatetime
    updated_at: AwareDatetime


class ConversationNotFound(Exception):
    """Missing and non-owned IDs have identical application behavior."""


class InvalidConversationCursor(ValueError):
    """Cursor does not identify a persisted message of this conversation."""


class AcceptedConversationSnapshot(Contract):
    consultation: ConsultationStateV1
    public_build: PublicBuildV1 | None = None
    public_clarification: PublicClarificationV1 | None = None
    public_task_clarification: PublicTaskClarificationV1 | None = None
    pending_tasks: PendingTaskDraft | None = None
    requirement_state: CanonicalRequirementState | None = None
    flexible: FlexiblePublishedState | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_public_snapshot_binding(self) -> "AcceptedConversationSnapshot":
        if self.flexible is not None:
            if self.requirement_state is None or any(
                value is not None
                for value in (
                    self.pending_tasks,
                    self.public_clarification,
                    self.public_task_clarification,
                )
            ):
                raise ValueError(
                    "V2 snapshot requires canonical state, not mixed V1 pending fields"
                )
            for pending in (
                self.flexible.pending_proposal,
                self.flexible.pending_resolution,
            ):
                if pending and pending.base_requirements_hash != requirements_hash(
                    self.requirement_state
                ):
                    raise ValueError(
                        "V2 pending artifact does not match requirement content"
                    )
            if any(
                task.task.kind == "BUILD"
                and task.task.constraints_revision != self.requirement_state.revision
                for task in self.flexible.task_outcomes
            ):
                raise ValueError("V2 task binding does not match working requirements")
            if self.flexible.pending_task_resolution and (
                self.flexible.pending_task_resolution.constraints_revision
                != self.requirement_state.revision
            ):
                raise ValueError("V2 pending task resolution is stale")
            if (
                self.flexible.answer
                and (
                    self.flexible.answer.outcome == "BUILT"
                    or self.flexible.answer.build_ref is not None
                )
                and (
                    self.public_build is None
                    or self.public_build.stale
                    or self.flexible.answer.build_ref is None
                )
            ):
                raise ValueError("V2 built answer requires a current public build")
        digest = self.consultation.published_build_hash
        if self.pending_tasks is not None:
            if (
                self.requirement_state is None
                or self.requirement_state.revision
                != self.pending_tasks.constraints_revision
            ):
                raise ValueError("Task clarification revision binding is inconsistent")
            if self.public_task_clarification != public_task_options(
                self.pending_tasks
            ):
                raise ValueError(
                    "Task clarification projection binding is inconsistent"
                )
        elif self.public_task_clarification is not None:
            raise ValueError("Task clarification has no pending artifact")
        if (
            self.public_clarification is not None
            and self.public_task_clarification is not None
        ):
            raise ValueError("Conflicting pending clarification kinds")
        if digest is not None and (
            self.public_build is None
            or digest != canonical_hash(self.public_build)
            or self.consultation.build_stale != self.public_build.stale
        ):
            raise ValueError("Public build snapshot binding is inconsistent")
        return self


class AcceptedCheckpointReader(Protocol):
    async def read(self, ref: CheckpointRefV1) -> AcceptedConversationSnapshot: ...


class AccessTokenVerifier(Protocol):
    async def authenticate(self, token: str) -> UUID: ...


class ConversationInitializer(Protocol):
    async def initialize(self, conversation_id: UUID) -> CheckpointRefV1: ...


class StatefulConversationStore(Protocol):
    async def messages(
        self, account_id: UUID, conversation_id: UUID, limit: int, after: UUID | None
    ) -> PageV1[MessageViewV1]: ...

    async def create(
        self, account_id: UUID, title: str | None
    ) -> ConversationRecord: ...

    async def get(
        self, account_id: UUID, conversation_id: UUID
    ) -> ConversationRecord: ...

    async def initialize(
        self,
        account_id: UUID,
        conversation_id: UUID,
        initializer: ConversationInitializer,
    ) -> ConversationRecord: ...

    async def list_owned(
        self,
        account_id: UUID,
        limit: int = 20,
        before: tuple[datetime, UUID] | None = None,
    ) -> list[ConversationRecord]: ...


class RunAdmissionError(ValueError):
    """Static rejection key; no internal identity or SQL detail."""


class StatefulRunStore(Protocol):
    async def submit(self, owner: UUID, request: TurnRequest) -> RunViewV1: ...

    async def get(
        self, owner: UUID, conversation_id: UUID, run_id: UUID
    ) -> RunViewV1: ...
