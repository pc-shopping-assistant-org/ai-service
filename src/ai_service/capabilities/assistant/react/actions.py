"""Flexible model proposals, separate from the V1 runtime and application authority."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictInt, ValidationError, model_validator

from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    RequirementDraft,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

from .contracts import MAX_NATIVE_TOOL_CALLS, AnswerCandidate, ToolCall


class ActionKind(StrEnum):
    CALL_TOOL = "CALL_TOOL"
    CALL_TOOLS = "CALL_TOOLS"
    PROPOSE_REQUIREMENTS = "PROPOSE_REQUIREMENTS"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    FINAL_ANSWER = "FINAL_ANSWER"
    RESPOND = "RESPOND"


class CallToolAction(Contract):
    kind: Literal["CALL_TOOL"] = "CALL_TOOL"
    call: ToolCall


class CallToolsAction(Contract):
    kind: Literal["CALL_TOOLS"] = "CALL_TOOLS"
    calls: list[ToolCall] = Field(min_length=1, max_length=MAX_NATIVE_TOOL_CALLS)

    @model_validator(mode="after")
    def unique_ids(self) -> "CallToolsAction":
        if len({call.call_id for call in self.calls}) != len(self.calls):
            raise ValueError("Native tool call IDs must be unique")
        return self


def action_calls(action: CallToolAction | CallToolsAction) -> list[ToolCall]:
    """Single-call actions remain readable from previously stored checkpoints."""
    return action.calls if isinstance(action, CallToolsAction) else [action.call]


class ProposeRequirementsAction(Contract):
    kind: Literal["PROPOSE_REQUIREMENTS"] = "PROPOSE_REQUIREMENTS"
    draft: RequirementDraft
    # If supplied, this is agent-authored proposal text, never USER provenance.
    source_text: Text | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def requirement_only(self) -> "ProposeRequirementsAction":
        if not self.draft.items:
            raise ValueError("Requirement proposal cannot replace requested tasks")
        return self


class _LegacyProposeRequirementsAction(Contract):
    """Validated shape emitted by checkpoints created before RequirementDraft v2."""

    kind: Literal["PROPOSE_REQUIREMENTS"] = "PROPOSE_REQUIREMENTS"
    draft: Draft
    source_text: Text | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def requirement_only(self) -> "_LegacyProposeRequirementsAction":
        if not self.draft.items or self.draft.tasks:
            raise ValueError("Requirement proposal cannot replace requested tasks")
        return self


class AskClarificationAction(Contract):
    kind: Literal["ASK_CLARIFICATION"] = "ASK_CLARIFICATION"
    question: Text = Field(max_length=2000)
    missing_fields: list[Text] = Field(min_length=1, max_length=20)


class RespondAction(AnswerCandidate):
    kind: Literal["RESPOND"] = "RESPOND"


type FinalOutcome = Literal[
    "ANSWERED",
    "BUILT",
    "INFEASIBLE",
    "SEARCH_INCOMPLETE",
    "INSUFFICIENT_EVIDENCE",
    "DEPENDENCY_UNAVAILABLE",
]


class FinalAnswerPayload(Contract):
    answer: AnswerCandidate
    # References to canonical outcomes, including failed/empty calls with no facts.
    supporting_tool_call_ids: list[Text] = Field(default_factory=list, max_length=20)


class TaskFinalCandidate(FinalAnswerPayload):
    outcome: FinalOutcome = "ANSWERED"
    task_index: StrictInt = Field(ge=0, le=7)


class FinalAnswerAction(FinalAnswerPayload):
    kind: Literal["FINAL_ANSWER"] = "FINAL_ANSWER"
    outcome: FinalOutcome | Literal["MULTI_TASK"] = "ANSWERED"
    task_answers: list[TaskFinalCandidate] = Field(
        default_factory=list,
        max_length=8,
        exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def one_final_representation(self) -> "FinalAnswerAction":
        if (self.outcome == "MULTI_TASK") != bool(self.task_answers):
            raise ValueError(
                "MULTI_TASK requires task answers; single outcome forbids them"
            )
        if self.task_answers and (
            self.supporting_tool_call_ids
            or self.answer.evidence_refs
            or self.answer.build_ref
            or self.answer.web_excerpts
            or self.answer.review
            or self.answer.clarification_fields
            or self.answer.explanation is not None
        ):
            raise ValueError(
                "Task answers own their evidence, not the aggregate candidate"
            )
        return self


Action = Annotated[
    CallToolAction
    | CallToolsAction
    | ProposeRequirementsAction
    | AskClarificationAction
    | FinalAnswerAction
    | RespondAction,
    # A response is not an assertion that any registered task is completed.
    Field(discriminator="kind"),
]


class FlexibleDecision(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    action: Action
    control_call: ToolCall | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    registration: TaskDraftBatch | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    registration_call: ToolCall | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class _StoredDecisionEnvelope(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    action: dict[str, JsonValue]


def decode_flexible_decision(value: object) -> FlexibleDecision:
    """Read stored V1 proposals without widening the live model schema."""
    try:
        return FlexibleDecision.model_validate(value)
    except ValidationError:
        legacy = _StoredDecisionEnvelope.model_validate(value)
        if legacy.action.get("kind") != "PROPOSE_REQUIREMENTS":
            return FlexibleDecision.model_validate(value)
        old_action = _LegacyProposeRequirementsAction.model_validate(legacy.action)
        return FlexibleDecision.model_validate(
            {
                "schema_version": legacy.schema_version,
                "action": {
                    "kind": old_action.kind,
                    "source_text": old_action.source_text,
                    "draft": RequirementDraft(items=old_action.draft.items).model_dump(
                        mode="json"
                    ),
                },
            }
        )


class RepairFeedback(Contract):
    code: Literal[
        "INVALID_ACTION",
        "BAD_TOOL_ARGUMENTS",
        "TOOL_NOT_ALLOWED",
        "PROPOSAL_CONFIRMATION_REQUIRED",
        "STALE_REQUIREMENTS",
        "INVALID_CONFIRMATION",
        "MISSING_EVIDENCE",
        "TASK_INCOMPLETE",
        "SEARCH_PROOF_REQUIRED",
    ]
    missing_fields: list[Text] = Field(default_factory=list, max_length=20)
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)
    task_indices: list[Annotated[StrictInt, Field(ge=0, le=7)]] = Field(
        default_factory=list,
        max_length=8,
        exclude_if=lambda value: not value,
    )
    allowed_next_actions: list[ActionKind] = Field(
        min_length=1, max_length=len(ActionKind)
    )
