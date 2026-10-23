"""Model selects results; application derives outcomes from canonical evidence."""

from uuid import UUID

from pydantic import Field, StrictInt

from ai_service.application.ports.web_search import WebFactStatus
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

from .actions import FinalAnswerAction, FinalOutcome, TaskFinalCandidate
from .contracts import (
    AnswerCandidate,
    BuildResult,
    ConsultationExplanation,
    ProductReview,
    ToolRecord,
    WebExcerpt,
    result_web_pages,
)

MAX_TASK_RESULTS = 8


class ResultSelection(Contract):
    task_index: StrictInt | None = Field(default=0, ge=0, lt=MAX_TASK_RESULTS)
    text: Text = Field(max_length=8000)
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)
    build_ref: UUID | None = None
    supporting_tool_call_ids: list[Text] = Field(default_factory=list, max_length=20)
    web_excerpts: list[WebExcerpt] = Field(default_factory=list, max_length=8)
    explanation: Text | None = Field(default=None, max_length=3000)
    explanation_evidence_refs: list[Text] | None = Field(default=None, max_length=20)
    review: ProductReview | None = None


class SubmitResults(Contract):
    """Submit selected evidence for requested tasks. Never assert completion status."""

    results: list[ResultSelection] = Field(min_length=1, max_length=MAX_TASK_RESULTS)


def selected_answer(selection: ResultSelection) -> AnswerCandidate:
    return AnswerCandidate(
        text=selection.text,
        evidence_refs=selection.evidence_refs,
        build_ref=selection.build_ref,
        web_excerpts=selection.web_excerpts,
        review=selection.review,
        explanation=ConsultationExplanation(
            text=selection.explanation,
            evidence_refs=selection.explanation_evidence_refs
            if selection.explanation_evidence_refs is not None
            else selection.evidence_refs,
        )
        if selection.explanation
        else None,
    )


def _derive_part(
    selection: ResultSelection, records: list[ToolRecord]
) -> TaskFinalCandidate:
    if selection.task_index is None:
        raise ValueError("Conversation response is not a business-task outcome")
    by_id = {record.call.call_id: record for record in records}
    ids = selection.supporting_tool_call_ids
    if (
        len(by_id) != len(records)
        or len(ids) != len(set(ids))
        or not set(ids) <= by_id.keys()
    ):
        raise ValueError("Submission requires canonical unique tool selections")
    refs = set(selection.evidence_refs)
    selected = [
        record
        for record in records
        if record.result
        and (
            record.call.call_id in ids
            or refs.intersection(record.result.evidence_refs)
            or (
                isinstance(record.result.result, BuildResult)
                and record.result.result.build_ref == selection.build_ref
            )
        )
    ]
    outcome: FinalOutcome = "ANSWERED"
    supporting = list(ids)
    build_ref = selection.build_ref
    for record in selected:
        result = record.result
        assert result is not None
        payload = result.result
        if (
            isinstance(payload, BuildResult)
            and payload.search_evidence is not None
            and payload.search_evidence.outcome != "FEASIBLE"
        ):
            outcome = payload.search_evidence.outcome
            supporting = [record.call.call_id]
            build_ref = None
            break
        if isinstance(payload, BuildResult) and payload.receipt.outcome == "BUILT":
            outcome = "BUILT"
            build_ref = payload.build_ref
        if any(
            fact.evidence_ref in refs
            and fact.fact.fact_status != WebFactStatus.SUPPORTED
            for page in result_web_pages(payload)
            for fact in page.facts
        ):
            outcome = "INSUFFICIENT_EVIDENCE"
            supporting.append(record.call.call_id)
    if selected and outcome == "ANSWERED" and supporting and not refs:
        outcome = (
            "DEPENDENCY_UNAVAILABLE"
            if all(
                record.result and record.result.status == "FAILED"
                for record in selected
            )
            else "INSUFFICIENT_EVIDENCE"
        )
    candidate = selected_answer(selection).model_copy(update={"build_ref": build_ref})
    return TaskFinalCandidate(
        task_index=selection.task_index,
        outcome=outcome,
        answer=candidate,
        supporting_tool_call_ids=list(dict.fromkeys(supporting)),
    )


def derive_final_action(
    submission: SubmitResults, records: list[ToolRecord]
) -> FinalAnswerAction:
    submission = SubmitResults.model_validate(submission.model_dump())
    parts = [_derive_part(selection, records) for selection in submission.results]
    if len(parts) == 1:
        if parts[0].task_index != 0:
            raise ValueError("A single result must target the first requested task")
        return FinalAnswerAction.model_validate(
            parts[0].model_dump(exclude={"task_index"})
        )
    return FinalAnswerAction(
        outcome="MULTI_TASK",
        answer=AnswerCandidate(text="Task results"),
        task_answers=parts,
    )
