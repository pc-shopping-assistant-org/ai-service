"""Encode validated preparation with original task source; does not publish/CAS."""

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from pydantic import Field, StrictBool, model_validator

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1
from ai_service.capabilities.assistant.requirements.pending_action import (
    DISCARDED_ANSWER,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ClarificationReplyV1,
    Contract,
    TaskClarificationReplyV1,
    Text,
    UnitClarificationReplyV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

from .actions import AskClarificationAction
from .contracts import CompletionRequirement, ToolContext
from .pending_commands import validate_pending_command
from .public_views import public_resolution_options
from .publication import PreparedAnswerPublication, prepare_answer_publication
from .published_state import FlexiblePublishedState
from .resolution_reply import (
    FreeAmendmentV2,
    FreeResolutionReplyV2,
    FreeTaskAmendmentV2,
    FreeTaskReplyV2,
    apply_amendment_receipt,
    apply_resolution_command,
    apply_task_amendment_receipt,
    derive_resolution_reply,
    derive_task_reply,
)
from .task_resolution import (
    PendingTaskDraft,
    TaskResolution,
    selections_for_pending_task,
)
from .turn_contracts import TurnRequestV2
from .waiting_publication import PreparedWaitingPublication, prepare_waiting_publication


class FlexiblePublicationInput(Contract):
    """Trusted application records, not a model action or HTTP payload."""

    accepted: AcceptedConversationSnapshot
    task_message_id: UUID
    task_message: Text = Field(max_length=4000)
    request_message: Text = Field(max_length=4000)
    confirmation: ProposalConfirmation | None = None
    reference_reply: ClarificationReplyV1 | None = None
    unit_reply: UnitClarificationReplyV1 | None = None
    task_reply: TaskClarificationReplyV1 | None = None
    discard_pending: StrictBool = False

    @classmethod
    def for_request(
        cls,
        accepted: AcceptedConversationSnapshot,
        request: TurnRequestV2,
        *,
        owner: UUID | None = None,
    ) -> "FlexiblePublicationInput":
        accepted = AcceptedConversationSnapshot.model_validate(accepted.model_dump())
        request = TurnRequestV2.model_validate(request.model_dump())
        discard = False
        if request.pending_action is not None:
            if owner is None or accepted.requirement_state is None:
                raise ValueError("Replacement requires authorized snapshot context")
            validate_pending_command(
                request.pending_action,
                accepted.flexible,
                accepted.requirement_state,
                owner=owner,
                conversation_id=request.conversation_id,
                conversation_revision=request.expected_revision,
            )
            discard = request.pending_action.action == "DISCARD"
            if not discard:
                accepted = accepted.model_copy(update={"flexible": None})
        if (
            accepted.flexible
            and accepted.flexible.conversation_id != request.conversation_id
        ):
            raise ValueError(
                "Request and accepted snapshot belong to different conversations"
            )
        pending = (
            accepted.flexible
            if accepted.flexible and accepted.flexible.status == "WAITING_INPUT"
            else None
        )
        return cls(
            discard_pending=discard,
            accepted=accepted,
            task_message_id=pending.task_message_id if pending else request.request_id,
            task_message=pending.task_message if pending else request.message,
            request_message=request.message,
            confirmation=request.confirmation,
            reference_reply=request.reference_reply,
            unit_reply=request.unit_reply,
            task_reply=request.task_reply,
        )

    def turn_input(
        self, context: ToolContext, tasks: list[CompletionRequirement]
    ) -> dict:
        if self.accepted.requirement_state != context.requirements:
            raise ValueError("Publication input differs from accepted requirements")
        pending = self.accepted.flexible
        if pending:
            if pending.conversation_id != context.conversation_id:
                raise ValueError("Accepted conversation identity mismatch")
            if (
                pending.status == "WAITING_INPUT"
                and not self.discard_pending
                and (
                    self.task_message_id != pending.task_message_id
                    or self.task_message != pending.task_message
                    or tasks != [item.task for item in pending.task_outcomes]
                )
            ):
                raise ValueError("Continuation cannot replace the original task")
        if (
            sum(
                value is not None
                for value in (
                    self.confirmation,
                    self.reference_reply,
                    self.unit_reply,
                    self.task_reply,
                )
            )
            > 1
        ):
            raise ValueError("Cannot mix continuation commands")
        if self.task_reply:
            if pending is None or pending.pending_task_resolution is None:
                raise ValueError("Task selection requires an accepted pending task")
            selections_for_pending_task(
                pending.pending_task_resolution,
                self.task_reply,
                context.requirements,
            )
        return {
            "working_requirements": context.requirements.model_dump(mode="json"),
            "pending_proposal": pending.pending_proposal.model_dump(mode="json")
            if pending and pending.pending_proposal
            else None,
            "pending_draft": pending.pending_resolution.model_dump(mode="json")
            if pending and pending.pending_resolution
            else None,
            "confirmation": self.confirmation.model_dump(mode="json")
            if self.confirmation
            else None,
            "reference_reply": self.reference_reply.model_dump(mode="json")
            if self.reference_reply
            else None,
            "unit_reply": self.unit_reply.model_dump(mode="json")
            if self.unit_reply
            else None,
            "accepted_snapshot": None,
        }

    def prepare(
        self,
        values: Mapping[str, Any],
        context: ToolContext,
        tasks: list[CompletionRequirement],
        service: PCBuildApplicationService,
        conversation_revision: int,
    ) -> AcceptedConversationSnapshot:
        if values.get("message") != self.request_message:
            raise ValueError("Graph request differs from the stored request")
        pending = self.accepted.flexible
        if self.discard_pending:
            if (
                pending is None
                or values.get("phase") != "READY_TO_PUBLISH"
                or values.get("working_requirements")
                != context.requirements.model_dump(mode="json")
                or context.requirements != self.accepted.requirement_state
                or any(
                    values.get(key) is not None
                    for key in ("pending_draft", "pending_proposal", "clarification")
                )
                or values.get("answer")
                or tasks
            ):
                raise ValueError(
                    "Discard must preserve accepted state without answering tasks"
                )
            discarded = FlexiblePublishedState(
                conversation_id=context.conversation_id,
                task_message_id=pending.task_message_id,
                task_message=pending.task_message,
                status="DISCARDED",
                administrative_message=DISCARDED_ANSWER,
                waiting_tasks=pending.task_outcomes,
            )
            return AcceptedConversationSnapshot.model_validate(
                self.accepted.model_dump() | {"flexible": discarded.model_dump()}
            )
        if values.get("free_task_amendment") is not None:
            if (
                pending is None
                or pending.pending_task_resolution is None
                or any(
                    command is not None
                    for command in (
                        self.task_reply,
                        self.reference_reply,
                        self.unit_reply,
                        self.confirmation,
                    )
                )
            ):
                raise ValueError(
                    "Task amendment requires accepted source and no stored command"
                )
            if (
                values.get("free_task_reply") is not None
                and derive_task_reply(
                    FreeTaskReplyV2.model_validate(values["free_task_reply"]),
                    self.request_message,
                    pending.pending_task_resolution,
                    context.run_id,
                )
                is not None
            ):
                raise ValueError("Cannot combine task choice and amendment")
            amended = apply_task_amendment_receipt(
                FreeTaskAmendmentV2.model_validate(values["free_task_amendment"]),
                context,
                pending.pending_task_resolution,
                self.request_message,
            )
            working_base = self.accepted.model_copy(
                update={
                    "flexible": pending.model_copy(
                        update={"pending_task_resolution": amended}
                    )
                }
            )
            return self.model_copy(update={"accepted": working_base}).prepare(
                dict(values) | {"free_task_amendment": None, "free_task_reply": None},
                context,
                tasks,
                service,
                conversation_revision,
            )
        if values.get("free_amendment") is not None:
            if (
                pending is None
                or pending.pending_resolution is None
                or any(
                    command is not None
                    for command in (
                        self.confirmation,
                        self.reference_reply,
                        self.unit_reply,
                        self.task_reply,
                    )
                )
            ):
                raise ValueError(
                    "Amendment requires accepted source and no stored command"
                )
            if values.get("free_resolution_reply") is not None:
                reference, unit = derive_resolution_reply(
                    FreeResolutionReplyV2.model_validate(
                        values["free_resolution_reply"]
                    ),
                    self.request_message,
                    pending.pending_resolution,
                    context.run_id,
                )
                if reference or unit:
                    raise ValueError("Cannot combine choice and amendment")
            amendment_receipt = FreeAmendmentV2.model_validate(values["free_amendment"])
            continued = apply_amendment_receipt(
                amendment_receipt,
                context,
                pending.pending_resolution,
                self.request_message,
                conversation_revision,
            )
            if (
                continued.pending is None
                and continued.admission
                and continued.admission.pending is None
            ):
                if (
                    continued.admission.disposition != "ACCEPT"
                    or continued.admission.content_changed
                    or continued.admission.requirements != context.requirements
                ):
                    raise ValueError(
                        "No-op amendment cannot change accepted requirements"
                    )
                return self.prepare(
                    dict(values)
                    | {"free_amendment": None, "free_resolution_reply": None},
                    context,
                    tasks,
                    service,
                    conversation_revision,
                )
            amended_publication = prepare_waiting_publication(
                values,
                self.accepted,
                context,
                tasks,
                source_message=self.request_message,
                conversation_revision=conversation_revision,
                accepted_resolution=pending.pending_resolution,
                amendment=amendment_receipt.amendment,
            )
            return accepted_snapshot_for(
                amended_publication,
                conversation_id=context.conversation_id,
                task_message_id=self.task_message_id,
                task_message=self.task_message,
                catalog=context.catalog,
            )
        if values.get("free_task_reply") is not None:
            if (
                pending is None
                or pending.pending_task_resolution is None
                or any(
                    command is not None
                    for command in (
                        self.task_reply,
                        self.reference_reply,
                        self.unit_reply,
                        self.confirmation,
                    )
                )
            ):
                raise ValueError(
                    "Free task choice requires accepted source and no stored command"
                )
            reply = derive_task_reply(
                FreeTaskReplyV2.model_validate(values["free_task_reply"]),
                self.request_message,
                pending.pending_task_resolution,
                context.run_id,
            )
            if reply:
                selections_for_pending_task(
                    pending.pending_task_resolution, reply, context.requirements
                )
            return self.model_copy(update={"task_reply": reply}).prepare(
                dict(values) | {"free_task_reply": None},
                context,
                tasks,
                service,
                conversation_revision,
            )
        if values.get("free_resolution_reply") is not None:
            if (
                pending is None
                or pending.pending_resolution is None
                or any(
                    command is not None
                    for command in (
                        self.confirmation,
                        self.reference_reply,
                        self.unit_reply,
                        self.task_reply,
                    )
                )
            ):
                raise ValueError(
                    "Free choice requires accepted resolution and no stored command"
                )
            receipt = FreeResolutionReplyV2.model_validate(
                values["free_resolution_reply"]
            )
            reference, unit = derive_resolution_reply(
                receipt,
                self.request_message,
                pending.pending_resolution,
                context.run_id,
            )
            if reference is not None or unit is not None:
                return self.model_copy(
                    update={"reference_reply": reference, "unit_reply": unit}
                ).prepare(
                    dict(values) | {"free_resolution_reply": None},
                    context,
                    tasks,
                    service,
                    conversation_revision,
                )
            if (
                values.get("phase") != "WAITING_INPUT"
                or values.get("pending_draft")
                != pending.pending_resolution.model_dump(mode="json")
                or values.get("pending_proposal") is not None
                or values.get("answer")
                or values.get("working_requirements")
                != context.requirements.model_dump(mode="json")
            ):
                raise ValueError("Unresolved free choice cannot change accepted state")
            waiting = pending.model_copy(
                update={
                    "waiting_message": "Chưa xác định được lựa chọn. Vui lòng chọn một tùy chọn đã nêu."
                }
            )
            return AcceptedConversationSnapshot.model_validate(
                self.accepted.model_dump() | {"flexible": waiting.model_dump()}
            )
        task_resolution = (
            TaskResolution.model_validate(values["task_resolution"])
            if values.get("task_resolution")
            else None
        )
        if task_resolution and task_resolution.pending:
            if (
                values.get("phase") != "WAITING_INPUT"
                or any(
                    values.get(key)
                    for key in ("pending_proposal", "pending_draft", "answer")
                )
                or values.get("working_requirements")
                != context.requirements.model_dump(mode="json")
                or self.accepted.requirement_state != context.requirements
                or self.confirmation
                or self.reference_reply
                or self.unit_reply
            ):
                raise ValueError(
                    "Unresolved tasks require a source-bound clarification"
                )
            draft = TaskDraftBatch.model_validate(values["task_draft"])
            draft.validate_source(self.task_message)
            question = AskClarificationAction.model_validate(
                values.get("clarification")
            )
            reasons = {item.reason for item in task_resolution.pending}
            if not set(question.missing_fields) <= reasons:
                raise ValueError("Question is not bound to unresolved task subjects")
            unresolved = PendingTaskDraft(
                message_id=self.task_message_id,
                message=self.task_message,
                constraints_revision=context.requirements.revision,
                tasks=draft.tasks,
                resolution=task_resolution,
                mention_overrides=pending.pending_task_resolution.mention_overrides
                if pending and pending.pending_task_resolution
                else [],
            )
            return AcceptedConversationSnapshot.model_validate(
                self.accepted.model_dump()
                | {
                    "flexible": FlexiblePublishedState(
                        conversation_id=context.conversation_id,
                        task_message_id=self.task_message_id,
                        task_message=self.task_message,
                        status="WAITING_INPUT",
                        waiting_message="Cần xác định rõ sản phẩm hoặc cấu hình để hoàn thành yêu cầu.",
                        pending_task_resolution=unresolved,
                    ).model_dump()
                }
            )
        proposal = pending.pending_proposal if pending else None
        resolution = pending.pending_resolution if pending else None
        # Artifact without command is not evidence of an approved transition.
        proposal = proposal if self.confirmation else None
        prepared: PreparedWaitingPublication | PreparedAnswerPublication
        user_receipts = [
            UserRequirementReceipt.model_validate(row)
            for row in values.get("user_receipts", [])
        ]
        if values.get("phase") == "WAITING_INPUT":
            prepared = prepare_waiting_publication(
                values,
                self.accepted,
                context,
                tasks,
                source_message=self.request_message,
                conversation_revision=conversation_revision,
                accepted_proposal=proposal,
                confirmation=self.confirmation,
                accepted_resolution=resolution
                if self.reference_reply or self.unit_reply
                else None,
                reference_reply=self.reference_reply,
                unit_reply=self.unit_reply,
                user_receipts=user_receipts,
            )
        else:
            if self.reference_reply or self.unit_reply:
                if resolution is None:
                    raise ValueError(
                        "No-op resolution requires exact accepted artifact"
                    )
                continued = apply_resolution_command(
                    context,
                    resolution,
                    self.request_message,
                    conversation_revision,
                    self.reference_reply,
                    self.unit_reply,
                )
                if (
                    continued.pending is not None
                    or continued.admission is None
                    or continued.admission.disposition != "ACCEPT"
                    or continued.admission.pending is not None
                    or continued.admission.content_changed
                    or continued.admission.requirements != context.requirements
                ):
                    raise ValueError(
                        "Resolution continuation must retain pending consent"
                    )
            prepared = prepare_answer_publication(
                values,
                self.accepted,
                context,
                tasks,
                service,
                accepted_proposal=proposal,
                confirmation=self.confirmation,
                conversation_revision=conversation_revision,
                user_receipts=user_receipts,
                user_message=self.request_message,
            )
        return accepted_snapshot_for(
            prepared,
            conversation_id=context.conversation_id,
            task_message_id=self.task_message_id,
            task_message=self.task_message,
            catalog=context.catalog,
        )


class FlexibleExecutionContext(Contract):
    """Application-written begin-turn channel; model actions cannot update it."""

    context: ToolContext
    required: list[CompletionRequirement] = Field(max_length=8)
    registered: StrictBool
    discard_only: StrictBool = Field(default=False, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def task_mode(self) -> "FlexibleExecutionContext":
        if self.discard_only:
            if self.registered or self.required:
                raise ValueError("Discard is not a requested-task execution")
        elif bool(self.required) == self.registered:
            raise ValueError("Use either fixed accepted tasks or dynamic registration")
        return self


def revalidate_flexible_publication(
    values: Mapping[str, Any],
    accepted: AcceptedConversationSnapshot,
    request: TurnRequestV2,
    *,
    owner: UUID,
    run_id: UUID,
    base_ref: CheckpointRefV1,
    service: PCBuildApplicationService,
) -> AcceptedConversationSnapshot:
    execution = FlexibleExecutionContext.model_validate(
        values.get("publication_context")
    )
    ctx = execution.context
    if (
        ctx.account_id != owner
        or ctx.run_id != run_id
        or ctx.conversation_id != request.conversation_id
        or base_ref.thread_id != request.conversation_id
        or not ctx.read_catalog_allowed
        or not ctx.execution_allowed
        or ctx.requirements != accepted.requirement_state
    ):
        raise ValueError("Publication context does not match authorized accepted run")
    if ctx.saved_build and (
        ctx.saved_build.source_ref != base_ref
        or ctx.saved_build.snapshot != accepted.public_build
    ):
        raise ValueError("Saved build must come from the exact accepted checkpoint")
    binding = FlexiblePublicationInput.for_request(accepted, request, owner=owner)
    binding.turn_input(ctx, execution.required)
    tasks = execution.required
    if execution.discard_only:
        if not binding.discard_pending:
            raise ValueError("Discard context does not match stored command")
    elif binding.discard_pending:
        raise ValueError("Discard command requires a dedicated application context")
    elif execution.registered:
        draft = TaskDraftBatch.model_validate(values.get("task_draft"))
        draft.validate_source(binding.task_message)
        resolution = TaskResolution.model_validate(values.get("task_resolution"))
        if not resolution.pending and (
            len(resolution.requirements) != len(draft.tasks)
            or [task.kind for task in resolution.requirements]
            != [task.kind for task in draft.tasks]
        ):
            raise ValueError("Registration cannot change requested outcomes")
        previous = (
            binding.accepted.flexible.pending_task_resolution
            if binding.accepted.flexible
            else None
        )
        if values.get("free_task_amendment") is not None:
            if previous is None or any(
                command is not None
                for command in (
                    request.task_reply,
                    request.reference_reply,
                    request.unit_reply,
                    request.confirmation,
                )
            ):
                raise ValueError("Task amendment conflicts with stored request")
            previous = apply_task_amendment_receipt(
                FreeTaskAmendmentV2.model_validate(values["free_task_amendment"]),
                ctx,
                previous,
                request.message,
            )
        if previous and draft.tasks != previous.tasks:
            raise ValueError("Continuation cannot replace accepted task draft")
        task_reply = request.task_reply
        if values.get("free_task_reply") is not None:
            if previous is None or any(
                command is not None
                for command in (
                    request.task_reply,
                    request.reference_reply,
                    request.unit_reply,
                    request.confirmation,
                )
            ):
                raise ValueError("Free task choice conflicts with stored request")
            task_reply = derive_task_reply(
                FreeTaskReplyV2.model_validate(values["free_task_reply"]),
                request.message,
                previous,
                ctx.run_id,
            )
        if previous:
            selected = (
                selections_for_pending_task(previous, task_reply, ctx.requirements)
                if task_reply
                else previous.resolution.bindings
            )
            expected = {
                (item.task_index, item.subject_index): item.product_id
                for item in selected
            }
            actual = {
                (item.task_index, item.subject_index): item.product_id
                for item in resolution.bindings
            }
            if any(actual.get(key) != identity for key, identity in expected.items()):
                raise ValueError("Registration dropped an accepted product selection")
        tasks = resolution.requirements
    elif (
        binding.accepted.flexible is None
        or binding.accepted.flexible.status != "WAITING_INPUT"
    ):
        raise ValueError("Fresh tasks require source-bound registration")
    prepared = binding.prepare(values, ctx, tasks, service, request.expected_revision)
    persisted = AcceptedConversationSnapshot.model_validate(
        values.get("accepted_snapshot")
    )
    if prepared != persisted:
        raise ValueError("Terminal snapshot differs from application revalidation")
    return prepared


def accepted_snapshot_for(
    prepared: PreparedAnswerPublication | PreparedWaitingPublication,
    *,
    conversation_id: UUID,
    task_message_id: UUID,
    task_message: str,
    catalog: list[CatalogRowV1] | None = None,
) -> AcceptedConversationSnapshot:
    if isinstance(prepared, PreparedAnswerPublication):
        published = FlexiblePublishedState(
            conversation_id=conversation_id,
            task_message_id=task_message_id,
            task_message=task_message,
            status=prepared.answer.status,
            answer=prepared.answer,
        )
    else:
        published = FlexiblePublishedState(
            conversation_id=conversation_id,
            task_message_id=task_message_id,
            task_message=task_message,
            status="WAITING_INPUT",
            waiting_message=prepared.assistant_message,
            waiting_tasks=prepared.task_outcomes,
            pending_proposal=prepared.pending_proposal,
            pending_resolution=prepared.pending_resolution,
            clarification=prepared.clarification,
            resolution_options=public_resolution_options(
                prepared.pending_resolution.draft, catalog or []
            )
            if prepared.pending_resolution
            else None,
        )
    return AcceptedConversationSnapshot.model_validate(
        prepared.snapshot.model_dump() | {"flexible": published.model_dump()}
    )
