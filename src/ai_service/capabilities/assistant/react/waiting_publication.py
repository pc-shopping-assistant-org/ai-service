"""Prepare new waiting artifacts; no merge, checkpoint writes or CAS authority.

Existing pending commands/continuations need a separate accepted-artifact lane.
The coordinator supplies exact accepted state, stored request text and base revision.
"""

from collections.abc import Mapping
from typing import Any, Literal
from uuid import uuid5

from pydantic import Field, StrictInt, model_validator

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.amendment import PendingAmendment
from ai_service.capabilities.assistant.requirements.catalog_policy import (
    build_clarification_reasons,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
    amendment_source_id,
    continue_amendment,
    continue_reference,
    continue_unit,
    unit_reply_source_id,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    PendingDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    Contract,
    Text,
    UnitClarificationReplyV1,
)

from .actions import (
    AskClarificationAction,
    ProposeRequirementsAction,
    decode_flexible_decision,
)
from .contracts import CompletionRequirement, ToolContext
from .outcomes import TaskOutcome
from .publication import bind_publication_context, snapshot_for_requirements


class PreparedWaitingPublication(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    status: Literal["WAITING_INPUT"] = "WAITING_INPUT"
    snapshot: AcceptedConversationSnapshot
    assistant_message: Text
    pending_proposal: PendingRequirementProposal | None = None
    pending_resolution: PendingResolution | None = None
    clarification: AskClarificationAction | None = None
    task_outcomes: list[TaskOutcome] = Field(max_length=8)

    @model_validator(mode="after")
    def waiting_only(self) -> "PreparedWaitingPublication":
        if (
            sum(
                item is not None
                for item in (
                    self.pending_proposal,
                    self.pending_resolution,
                    self.clarification,
                )
            )
            != 1
        ):
            raise ValueError("Waiting publication requires exactly one pending kind")
        if any(task.status != "UNRESOLVED" for task in self.task_outcomes):
            raise ValueError("Waiting publication cannot complete requested tasks")
        return self


def prepare_waiting_publication(
    values: Mapping[str, Any],
    accepted: AcceptedConversationSnapshot,
    context: ToolContext,
    required: list[CompletionRequirement],
    *,
    source_message: str,
    conversation_revision: int,
    accepted_resolution: PendingResolution | None = None,
    reference_reply: ClarificationReplyV1 | None = None,
    unit_reply: UnitClarificationReplyV1 | None = None,
    amendment: PendingAmendment | None = None,
    accepted_proposal: PendingRequirementProposal | None = None,
    confirmation: ProposalConfirmation | None = None,
    user_receipts: list[UserRequirementReceipt] | None = None,
) -> PreparedWaitingPublication:
    accepted = AcceptedConversationSnapshot.model_validate(accepted.model_dump())
    context = ToolContext.model_validate(context.model_dump())
    tasks = [
        CompletionRequirement.model_validate(task.model_dump()) for task in required
    ]
    if (
        values.get("phase") != "WAITING_INPUT"
        or values.get("message") != source_message
        or not source_message.strip()
        or len(source_message) > 4000
        or conversation_revision < 0
        or values.get("answer")
        or not context.execution_allowed
        or not context.read_catalog_allowed
        or accepted.requirement_state is None
        or accepted.public_clarification is not None
        or accepted.pending_tasks is not None
    ):
        raise ValueError(
            "New waiting publication requires stored request and non-pending accepted base"
        )
    if (accepted_proposal is not None or confirmation is not None) and any(
        value is not None
        for value in (accepted_resolution, reference_reply, unit_reply, amendment)
    ):
        raise ValueError("Cannot mix confirmation and resolution commands")
    context, tasks = bind_publication_context(
        accepted,
        context,
        tasks,
        accepted_proposal=accepted_proposal,
        confirmation=confirmation,
        conversation_revision=conversation_revision,
        user_receipts=user_receipts,
        user_message=source_message,
    )
    state = CanonicalRequirementState.model_validate(values.get("working_requirements"))
    if state != context.requirements:
        raise ValueError(
            "Waiting artifact differs from the accepted command transition"
        )
    keys = ("pending_proposal", "pending_draft", "clarification")
    if sum(values.get(key) is not None for key in keys) != 1:
        raise ValueError("Conflicting or missing waiting artifact")
    task_outcomes = [TaskOutcome(task=task, status="UNRESOLVED") for task in tasks]
    snapshot = snapshot_for_requirements(accepted, context)
    if any(
        value is not None
        for value in (accepted_resolution, reference_reply, unit_reply, amendment)
    ):
        if (
            accepted_resolution is None
            or sum(
                item is not None for item in (reference_reply, unit_reply, amendment)
            )
            != 1
        ):
            raise ValueError(
                "Waiting continuation requires exact accepted artifact and one stored reply"
            )
        if amendment is not None:
            continued = continue_amendment(
                state,
                accepted_resolution,
                amendment,
                context.catalog,
                message_id=amendment_source_id(
                    context.run_id, source_message, amendment
                ),
                message=source_message,
                owner=context.account_id,
                conversation_id=context.conversation_id,
                conversation_revision=conversation_revision,
            )
        elif reference_reply is not None:
            continued = continue_reference(
                state,
                accepted_resolution,
                reference_reply,
                context.catalog,
                owner=context.account_id,
                conversation_id=context.conversation_id,
                conversation_revision=conversation_revision,
            )
        else:
            assert unit_reply is not None
            continued = continue_unit(
                state,
                accepted_resolution,
                unit_reply,
                context.catalog,
                message_id=unit_reply_source_id(
                    context.run_id, source_message, unit_reply
                ),
                message=source_message,
                owner=context.account_id,
                conversation_id=context.conversation_id,
                conversation_revision=conversation_revision,
            )
        if continued.pending is not None:
            actual_resolution = PendingResolution.model_validate(
                values.get("pending_draft")
            )
            if actual_resolution != continued.pending:
                raise ValueError(
                    "Continuation pending result differs from accepted command"
                )
            return PreparedWaitingPublication(
                snapshot=snapshot,
                task_outcomes=task_outcomes,
                pending_resolution=actual_resolution,
                assistant_message="Còn yêu cầu chưa được làm rõ.",
            )
        expected_proposal = continued.admission.pending if continued.admission else None
        actual_proposal = PendingRequirementProposal.model_validate(
            values.get("pending_proposal")
        )
        if expected_proposal is None or actual_proposal.model_dump(
            exclude={"proposal_id", "proposal_hash"}
        ) != expected_proposal.model_dump(exclude={"proposal_id", "proposal_hash"}):
            raise ValueError(
                "Continuation proposal does not match accepted command/source"
            )
        # Newly issued IDs may differ when independently re-deriving the payload;
        # actual proposal's immutable hash was validated above. No accepted ID is replaced.
        return PreparedWaitingPublication(
            snapshot=snapshot,
            task_outcomes=task_outcomes,
            pending_proposal=actual_proposal,
            assistant_message="Yêu cầu đã được làm rõ; đề xuất đang chờ bạn xác nhận trước khi thay đổi state.",
        )
    action = decode_flexible_decision(values.get("decision")).action
    if isinstance(action, AskClarificationAction):
        checked = AskClarificationAction.model_validate(values.get("clarification"))
        missing = build_clarification_reasons(
            state, context.catalog, catalog_ready=context.catalog_ready
        )
        if (
            checked != action
            or not any(task.kind == "BUILD" for task in tasks)
            or not set(checked.missing_fields) <= missing
        ):
            raise ValueError("Question is not bound to actual missing prerequisites")
        return PreparedWaitingPublication(
            snapshot=snapshot,
            task_outcomes=task_outcomes,
            clarification=checked,
            assistant_message="Cần bổ sung: " + ", ".join(checked.missing_fields) + ".",
        )
    if action.kind != "PROPOSE_REQUIREMENTS":
        raise ValueError("Waiting artifact requires a proposal or validated question")
    action = ProposeRequirementsAction.model_validate(action.model_dump())
    text = action.source_text if action.source_text is not None else source_message
    source_id = uuid5(context.run_id, canonical_hash(action))
    resolution = resolve_draft(
        action.draft,
        message_id=source_id,
        message=text,
        state=state,
        catalog=context.catalog,
    )
    if resolution.patch is None:
        expected = PendingResolution(
            owner=context.account_id,
            conversation_id=context.conversation_id,
            target_revision=conversation_revision + 1,
            base_requirements_hash=requirements_hash(state),
            draft=PendingDraft(
                message_id=source_id,
                message=text,
                constraints_revision=state.revision,
                resolution=resolution,
            ),
        )
        actual = PendingResolution.model_validate(values.get("pending_draft"))
        if actual != expected:
            raise ValueError("Unresolved artifact differs from canonical resolution")
        return PreparedWaitingPublication(
            snapshot=snapshot,
            task_outcomes=task_outcomes,
            pending_resolution=actual,
            assistant_message="Cần làm rõ: "
            + ", ".join(item.reason for item in resolution.pending)
            + ".",
        )
    proposal = PendingRequirementProposal.model_validate(values.get("pending_proposal"))
    if (
        proposal.owner != context.account_id
        or proposal.conversation_id != context.conversation_id
        or proposal.target_revision != conversation_revision + 1
        or proposal.base_requirements_hash != requirements_hash(state)
        or proposal.patch != resolution.patch
        or proposal.source_messages != {source_id: text}
        or requirements_hash(
            merge_requirements(state, proposal.patch, proposal.source_messages)
        )
        == requirements_hash(state)
    ):
        raise ValueError(
            "Proposal does not match current action/source/publication target"
        )
    return PreparedWaitingPublication(
        snapshot=snapshot,
        task_outcomes=task_outcomes,
        pending_proposal=proposal,
        assistant_message="Đề xuất thay đổi yêu cầu đang chờ bạn xác nhận; yêu cầu đã công bố chưa thay đổi.",
    )
