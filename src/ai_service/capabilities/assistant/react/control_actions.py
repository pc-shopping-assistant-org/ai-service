"""Native control contracts; application transitions are not business tool results."""

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ai_service.capabilities.assistant.source_quotes import QuotedRequirementDraft
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

from .actions import (
    ActionKind,
    AskClarificationAction,
    FlexibleDecision,
    ProposeRequirementsAction,
    RepairFeedback,
    RespondAction,
)
from .contracts import ToolCall, ToolContext, ToolRecord
from .submissions import SubmitResults, derive_final_action, selected_answer


class ControlName(StrEnum):
    PROPOSE_REQUIREMENTS = "propose_requirements"
    ASK_CLARIFICATION = "ask_clarification"
    SUBMIT_RESULT = "submit_result"


class QuotedProposeRequirementsAction(Contract):
    kind: Literal["PROPOSE_REQUIREMENTS"] = "PROPOSE_REQUIREMENTS"
    draft: QuotedRequirementDraft
    source_text: Text | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def requirement_only(self) -> "QuotedProposeRequirementsAction":
        if not self.draft.items:
            raise ValueError("Requirement proposal cannot replace requested tasks")
        return self


CONTROL_SCHEMAS: dict[ControlName, type[Contract]] = {
    ControlName.PROPOSE_REQUIREMENTS: QuotedProposeRequirementsAction,
    ControlName.ASK_CLARIFICATION: AskClarificationAction,
    ControlName.SUBMIT_RESULT: SubmitResults,
}
CONTROL_ACTIONS = {
    ControlName.PROPOSE_REQUIREMENTS: ActionKind.PROPOSE_REQUIREMENTS,
    ControlName.ASK_CLARIFICATION: ActionKind.ASK_CLARIFICATION,
    ControlName.SUBMIT_RESULT: ActionKind.FINAL_ANSWER,
}
CONTROL_DESCRIPTIONS = {
    ControlName.PROPOSE_REQUIREMENTS: "Propose requirements explicitly sourced from this turn, or supply source_text for your own suggestion. Application resolves and admits the change; suggestions require consent.",
    ControlName.ASK_CLARIFICATION: "Ask for missing business input using actual missing field names or unresolved selection reasons. General conversation can be plain text.",
    ControlName.SUBMIT_RESULT: "Select evidence for requested task indices; task_index=null answers this turn without completing pending work. Application derives outcomes and renders facts. Use explanation for qualitative interpretation, web_excerpts for exact source quotes. Never provide status or outcome.",
}


def allowed_controls(
    records: list[ToolRecord],
    context: ToolContext | None,
    repair: RepairFeedback | None,
) -> tuple[ControlName, ...]:
    controls = []
    if context is None or context.pending is None or context.pending.proposal is None:
        controls.append(ControlName.PROPOSE_REQUIREMENTS)
    if context is None or context.completion_requirements or context.pending:
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
    call: ToolCall, records: list[ToolRecord], message: str
) -> FlexibleDecision:
    name = ControlName(call.name)
    if name == ControlName.PROPOSE_REQUIREMENTS:
        quoted = QuotedProposeRequirementsAction.model_validate(call.arguments)
        action = ProposeRequirementsAction(
            draft=quoted.draft.bind(quoted.source_text or message),
            source_text=quoted.source_text,
        )
        return FlexibleDecision(action=action, control_call=call)
    if name == ControlName.ASK_CLARIFICATION:
        question = AskClarificationAction.model_validate(call.arguments)
        return FlexibleDecision(action=question, control_call=call)
    submitted = SubmitResults.model_validate(call.arguments)
    if len(submitted.results) == 1 and submitted.results[0].task_index is None:
        response = RespondAction.model_validate(
            selected_answer(submitted.results[0]).model_dump()
        )
        return FlexibleDecision(action=response, control_call=call)
    return FlexibleDecision(
        action=derive_final_action(submitted, records), control_call=call
    )
