"""Conversation response policy; business completion stays in outcomes/tools."""

from typing import TYPE_CHECKING
from uuid import UUID

from .actions import RespondAction
from .contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ConsultationExplanation,
    ConversationResponse,
    PendingConversationContext,
    ToolContext,
    ToolRecord,
)
from .published_state import FlexiblePublishedState
from .task_resolution import public_task_options
from .tools import project_current_tools, validate_answer

if TYPE_CHECKING:
    from ai_service.application.ports.stateful_conversation import (
        AcceptedConversationSnapshot,
    )


def pending_context(
    pending: FlexiblePublishedState | None,
) -> PendingConversationContext | None:
    if pending is None or pending.status != "WAITING_INPUT":
        return None
    proposal = pending.pending_proposal
    return PendingConversationContext(
        question=pending.waiting_message or pending.task_message,
        proposal=proposal.patch if proposal else None,
        proposal_sources=list(proposal.source_messages.values()) if proposal else [],
        resolution=pending.resolution_options,
        task_selection=public_task_options(pending.pending_task_resolution)
        if pending.pending_task_resolution
        else None,
    )


def validate_response(
    action: RespondAction,
    context: ToolContext,
    required: list[CompletionRequirement],
    records: list[ToolRecord],
    *,
    continuing_pending: bool = False,
) -> ConversationResponse:
    action = RespondAction.model_validate(action.model_dump())
    if required and not continuing_pending:
        raise ValueError("A response cannot replace a requested business result")
    if not context.execution_allowed or not context.read_catalog_allowed:
        raise ValueError("Conversation execution is not authorized")
    current = project_current_tools(records, context)
    if (
        action.evidence_refs
        or action.build_ref is not None
        or action.web_excerpts
        or action.review is not None
        or action.explanation is not None
    ):
        # Reuse canonical renderers without claiming the pending task completed.
        candidate = AnswerCandidate.model_validate(action.model_dump(exclude={"kind"}))
        if (
            candidate.explanation is None
            and not candidate.web_excerpts
            and candidate.review is None
        ):
            candidate.explanation = ConsultationExplanation(
                text=action.text, evidence_refs=action.evidence_refs
            )
        rendered = validate_answer(
            candidate,
            [row.result for row in current if row.result],
            context,
            [CompletionRequirement(kind="ANSWER")],
        )
        return ConversationResponse(
            text=rendered.text, evidence_refs=rendered.evidence_refs
        )
    return ConversationResponse(text=action.text)


def response_snapshot(
    accepted: "AcceptedConversationSnapshot",
    response: ConversationResponse,
    *,
    conversation_id: UUID,
    message_id: UUID,
    message: str,
) -> "AcceptedConversationSnapshot":
    from ai_service.application.ports.stateful_conversation import (
        AcceptedConversationSnapshot,
    )

    previous = accepted.flexible
    if previous and previous.status == "WAITING_INPUT":
        published = previous.model_copy(update={"response": response}, deep=True)
    else:
        published = FlexiblePublishedState(
            conversation_id=conversation_id,
            task_message_id=message_id,
            task_message=message,
            status="COMPLETED",
            response=response,
        )
    return AcceptedConversationSnapshot.model_validate(
        accepted.model_dump() | {"flexible": published.model_dump()}
    )
