"""Native control contracts; application transitions are not business tool results."""

from enum import StrEnum

from ai_service.application.ports.web_search import WebPageReadStatus
from ai_service.capabilities.assistant.stateful_contracts import Contract
from ai_service.capabilities.pc_builder.schemas import UseCaseProfile

from .actions import (
    ActionKind,
    AskClarificationAction,
    FlexibleDecision,
    ProposeRequirementsAction,
    RepairFeedback,
    RespondAction,
)
from .contracts import ToolCall, ToolContext, ToolRecord, result_web_pages
from .model_contracts import (
    EvidenceSubmissionInput,
    RequirementProposalInput,
    ResultSubmissionInput,
)
from .submissions import derive_final_action, selected_answer


class ControlName(StrEnum):
    PROPOSE_REQUIREMENTS = "propose_requirements"
    ASK_CLARIFICATION = "ask_clarification"
    SUBMIT_RESULT = "submit_result"


CONTROL_SCHEMAS: dict[ControlName, type[Contract]] = {
    ControlName.PROPOSE_REQUIREMENTS: RequirementProposalInput,
    ControlName.ASK_CLARIFICATION: AskClarificationAction,
    ControlName.SUBMIT_RESULT: ResultSubmissionInput,
}
CONTROL_ACTIONS = {
    ControlName.PROPOSE_REQUIREMENTS: ActionKind.PROPOSE_REQUIREMENTS,
    ControlName.ASK_CLARIFICATION: ActionKind.ASK_CLARIFICATION,
    ControlName.SUBMIT_RESULT: ActionKind.FINAL_ANSWER,
}


def control_input(name: ControlName, records: list[ToolRecord]) -> type[Contract]:
    if name == ControlName.SUBMIT_RESULT and not any(
        row.result
        and any(
            page.status == WebPageReadStatus.READABLE
            for page in result_web_pages(row.result.result)
        )
        for row in records
    ):
        return EvidenceSubmissionInput
    return CONTROL_SCHEMAS[name]


CONTROL_DESCRIPTIONS = {
    ControlName.PROPOSE_REQUIREMENTS: (
        "Extract stated requirements once with exact quotes. Numeric value + unit; missing unit stays unresolved. "
        'Budget: subject="SETUP", predicate="budget", target="GOAL", operator="LTE", numeric value in VND. '
        'Workload: subject="WORKLOAD", predicate="use_case", target="GOAL", operator="EQ", value one of '
        + ", ".join(profile.value for profile in UseCaseProfile)
        + '; never guess a profile. RAM/storage: subject="RAM"/"STORAGE", predicate="capacity", numeric value with GB/TB; '
        'SSD/HDD: subject="STORAGE", predicate="media", text value. Capacity is HARD; preferences SOFT. Exact identity text resolves through catalog. '
        "Owned uses RESOURCE; paid pins use identity requirements. "
        "If subject/predicate already exists, SET/CLEAR must copy its target_id from context; do not add a second budget. "
        "source_text is only your own suggestion (requires consent)."
    ),
    ControlName.ASK_CLARIFICATION: "Ask for missing business input using actual missing field names or unresolved selection reasons. General conversation can be plain text.",
    ControlName.SUBMIT_RESULT: "Select refs/build_ref or supporting calls. task_index=null answers without completing pending work. Use explanation for advice, quotes for exact source excerpts. Application verifies outcomes and renders facts.",
}


def allowed_controls(
    records: list[ToolRecord],
    context: ToolContext | None,
    repair: RepairFeedback | None,
) -> tuple[ControlName, ...]:
    controls = []
    if context is None or context.pending is None or context.pending.proposal is None:
        controls.append(ControlName.PROPOSE_REQUIREMENTS)
    if (
        context is None
        or context.completion_requirements
        or context.pending
        or context.requirements.requirements
        or context.requirements.goals
        or context.requirements.available_resources
        or (repair and ActionKind.ASK_CLARIFICATION in repair.allowed_next_actions)
    ):
        controls.append(ControlName.ASK_CLARIFICATION)
    if records or (context and context.saved_build):
        controls.append(ControlName.SUBMIT_RESULT)
    if repair:
        controls = [
            name
            for name in controls
            if CONTROL_ACTIONS[name] in repair.allowed_next_actions
        ]
    return tuple(controls)


def normalize_control(
    call: ToolCall,
    records: list[ToolRecord],
    message: str,
    context: ToolContext | None = None,
) -> FlexibleDecision:
    name = ControlName(call.name)
    if name == ControlName.PROPOSE_REQUIREMENTS:
        quoted = RequirementProposalInput.model_validate(call.arguments)
        draft = quoted.quoted().bind(quoted.source_text or message)
        # Existing IDs are globally unique; their trusted collection determines
        # scope, even when the provider repeats the default REQUIREMENT target.
        if context is not None:
            for item in draft.items:
                if item.target_id in context.requirements.goals:
                    item.target = "GOAL"
                elif item.target_id in context.requirements.requirements:
                    item.target = "REQUIREMENT"
        action = ProposeRequirementsAction(
            draft=draft,
            source_text=quoted.source_text,
        )
        return FlexibleDecision(action=action, control_call=call)
    if name == ControlName.ASK_CLARIFICATION:
        question = AskClarificationAction.model_validate(call.arguments)
        return FlexibleDecision(action=question, control_call=call)
    validated = control_input(name, records).model_validate(call.arguments)
    submitted = ResultSubmissionInput.model_validate(
        validated.model_dump()
    ).submission()
    if len(submitted.results) == 1 and submitted.results[0].task_index is None:
        response = RespondAction.model_validate(
            selected_answer(submitted.results[0]).model_dump()
        )
        return FlexibleDecision(action=response, control_call=call)
    return FlexibleDecision(
        action=derive_final_action(submitted, records), control_call=call
    )
