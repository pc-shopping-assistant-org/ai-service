"""Pure answer preparation, not authority to publish a native checkpoint.

The coordinator supplies an exact authorized accepted base/pending proposal and
stored confirmation command, and separately verifies stop/lineage/fencing/CAS.
Waiting-input publication still needs its own proposal/clarification binding lane.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import Field, StrictInt

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.requirements.admission import confirm_agent_patch
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.requirements.publication import public_projection
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
    apply_user_receipt,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

from .actions import FinalAnswerAction, FlexibleDecision
from .contracts import BuildResult, CompletionRequirement, ToolContext, ToolRecord
from .outcomes import FlexibleAnswer, validate_final_answer
from .tools import project_current_tools


class PreparedAnswerPublication(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    snapshot: AcceptedConversationSnapshot
    answer: FlexibleAnswer


def bind_publication_context(
    accepted: AcceptedConversationSnapshot,
    context: ToolContext,
    required: list[CompletionRequirement],
    *,
    accepted_proposal: PendingRequirementProposal | None = None,
    confirmation: ProposalConfirmation | None = None,
    conversation_revision: int | None = None,
    user_receipts: list[UserRequirementReceipt] | None = None,
    user_message: str | None = None,
) -> tuple[ToolContext, list[CompletionRequirement]]:
    """Shared answer/waiting transition; inputs come from the accepted store/request."""
    context = ToolContext.model_validate(context.model_dump())
    if (
        accepted.requirement_state is None
        or context.requirements != accepted.requirement_state
    ):
        raise ValueError(
            "Publication context must start from the accepted requirements"
        )
    expected = CanonicalRequirementState.model_validate(
        accepted.requirement_state.model_dump()
    )
    tasks = [
        CompletionRequirement.model_validate(task.model_dump()) for task in required
    ]
    if accepted_proposal is not None or confirmation is not None:
        if (
            accepted_proposal is None
            or confirmation is None
            or conversation_revision is None
        ):
            raise ValueError(
                "Confirmation publication requires exact accepted proposal and revision"
            )
        expected = confirm_agent_patch(
            expected,
            accepted_proposal,
            confirmation,
            owner=context.account_id,
            conversation_id=context.conversation_id,
            conversation_revision=conversation_revision,
        ).requirements
    if user_receipts:
        if user_message is None:
            raise ValueError(
                "User admission replay requires the stored request message"
            )
        for receipt in user_receipts:
            expected = apply_user_receipt(
                expected,
                receipt,
                run_id=context.run_id,
                message=user_message,
                catalog=context.catalog,
            )
    if confirmation is not None or user_receipts:
        tasks = [
            task.model_copy(update={"constraints_revision": expected.revision})
            if task.kind == "BUILD"
            else task
            for task in tasks
        ]
    return ToolContext.model_validate(
        context.model_dump() | {"requirements": expected.model_dump()}
    ), tasks


def snapshot_for_requirements(
    accepted: AcceptedConversationSnapshot,
    context: ToolContext,
) -> AcceptedConversationSnapshot:
    if context.requirements == accepted.requirement_state:
        return accepted.model_copy(deep=True)
    consultation, build = public_projection(
        context.requirements,
        None,
        accepted.public_build,
        compilation=compile_requirements(context.requirements, context.catalog),
    )
    return AcceptedConversationSnapshot(
        consultation=consultation,
        public_build=build,
        requirement_state=context.requirements,
    )


def prepare_answer_publication(
    values: Mapping[str, Any],
    accepted: AcceptedConversationSnapshot,
    context: ToolContext,
    required: list[CompletionRequirement],
    service: PCBuildApplicationService,
    *,
    accepted_proposal: PendingRequirementProposal | None = None,
    confirmation: ProposalConfirmation | None = None,
    conversation_revision: int | None = None,
    user_receipts: list[UserRequirementReceipt] | None = None,
    user_message: str | None = None,
) -> PreparedAnswerPublication:
    accepted = AcceptedConversationSnapshot.model_validate(accepted.model_dump())
    context = ToolContext.model_validate(context.model_dump())
    if (
        values.get("phase") != "READY_TO_PUBLISH"
        or not context.execution_allowed
        or not context.read_catalog_allowed
        or any(
            values.get(field)
            for field in ("pending_proposal", "pending_draft", "clarification")
        )
        or accepted.requirement_state is None
        or accepted.public_clarification is not None
        or accepted.pending_tasks is not None
    ):
        raise ValueError(
            "Answer preparation requires a terminal non-pending accepted base"
        )
    context, required = bind_publication_context(
        accepted,
        context,
        required,
        accepted_proposal=accepted_proposal,
        confirmation=confirmation,
        conversation_revision=conversation_revision,
        user_receipts=user_receipts,
        user_message=user_message,
    )
    working = CanonicalRequirementState.model_validate(
        values.get("working_requirements")
    )
    if working != context.requirements:
        raise ValueError(
            "Requirement change requires explicit command-bound publication"
        )
    action = FinalAnswerAction.model_validate(
        FlexibleDecision.model_validate(values.get("decision")).action.model_dump()
    )
    records = [ToolRecord.model_validate(row) for row in values.get("ledger", [])]
    if len({(row.origin_run_id, row.call.call_id) for row in records}) != len(
        records
    ) or any(row.result is None for row in records):
        raise ValueError("Publication ledger is unresolved or contains duplicate calls")
    current = project_current_tools(records, context)
    answer = validate_final_answer(
        action,
        [row.result for row in current if row.result],
        context,
        required,
        service,
    )
    if answer != FlexibleAnswer.model_validate(values.get("answer")):
        raise ValueError(
            "Terminal answer does not match revalidated canonical outcomes"
        )
    snapshot = snapshot_for_requirements(accepted, context)
    if answer.build_ref is not None:
        payload = next(
            (
                row.result.result
                for row in current
                if row.result
                and isinstance(row.result.result, BuildResult)
                and row.result.result.build_ref == answer.build_ref
            ),
            None,
        )
        if not isinstance(payload, BuildResult):
            raise ValueError("Public build lacks matching verified receipt")
        consultation, build = public_projection(
            working, payload.receipt, accepted.public_build
        )
        snapshot = AcceptedConversationSnapshot(
            consultation=consultation, public_build=build, requirement_state=working
        )
    return PreparedAnswerPublication(snapshot=snapshot, answer=answer)
