"""Requirement proposal admission; native graph owns checkpoints and execution."""

from uuid import uuid5

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.admission import (
    RequirementAdmission,
    propose_agent_patch,
)
from ai_service.capabilities.assistant.requirements.contracts import requirements_hash
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    PendingDraft,
    resolve_draft,
)
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
    admit_source_draft,
)

from .actions import ProposeRequirementsAction
from .contracts import ToolContext


async def resolve_requirement_proposal(
    action: ProposeRequirementsAction,
    context: ToolContext,
    message: str,
    conversation_revision: int,
) -> RequirementAdmission | PendingResolution | UserRequirementReceipt:
    if action.source_text is None:
        receipt = admit_source_draft(
            context.requirements,
            action,
            run_id=context.run_id,
            message=message,
            catalog=context.catalog,
        )
        if receipt is not None:
            return receipt
    text = action.source_text if action.source_text is not None else message
    source = uuid5(context.run_id, canonical_hash(action))
    resolution = resolve_draft(
        action.draft,
        message_id=source,
        message=text,
        state=context.requirements,
        catalog=context.catalog,
    )
    if resolution.patch is None:
        return PendingResolution(
            owner=context.account_id,
            conversation_id=context.conversation_id,
            target_revision=conversation_revision + 1,
            base_requirements_hash=requirements_hash(context.requirements),
            draft=PendingDraft(
                message_id=source,
                message=text,
                constraints_revision=context.requirements.revision,
                resolution=resolution,
            ),
        )
    return propose_agent_patch(
        context.requirements,
        resolution.patch,
        {source: text},
        owner=context.account_id,
        conversation_id=context.conversation_id,
        target_revision=conversation_revision + 1,
    )
