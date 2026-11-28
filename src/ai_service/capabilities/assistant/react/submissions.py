"""Model selects results; application derives outcomes from canonical evidence."""

from uuid import UUID

from pydantic import Field, StrictInt

from ai_service.application.ports.web_search import WebFactStatus
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text

from .actions import (
    AskClarificationAction,
    FinalAnswerAction,
    FinalOutcome,
    TaskFinalCandidate,
    decode_flexible_decision,
)
from .contracts import (
    AnswerCandidate,
    BuildResult,
    CatalogResult,
    ComparisonResult,
    CompatibilityCheckResult,
    CompletionRequirement,
    ConsultationExplanation,
    DetailsResult,
    ProductReview,
    ToolContext,
    ToolName,
    ToolRecord,
    WebExcerpt,
    result_web_pages,
)

MAX_TASK_RESULTS = 8


def evidence_requirements(
    value: object, records: list[ToolRecord], context: ToolContext
) -> list[CompletionRequirement]:
    """Derive completion checks from canonical operations, never a task LLM call.

    Fixed UI/pending tasks take precedence at the caller. Internal searches do not
    become separate tasks. A started build remains an obligation through repairs;
    a later plain response cannot bypass its result-completion check.
    """
    started_build = any(
        row.call.name == ToolName.BUILD_PC and row.origin_run_id == context.run_id
        for row in records
    )
    build_task = CompletionRequirement(
        kind="BUILD", constraints_revision=context.requirements.revision
    )
    if not value:
        return [build_task] if started_build else []
    action = decode_flexible_decision(value).action
    if isinstance(action, AskClarificationAction):
        return [
            CompletionRequirement(
                kind="BUILD", constraints_revision=context.requirements.revision
            )
        ]
    if not isinstance(action, FinalAnswerAction):
        return [build_task] if started_build else []
    parts: list[TaskFinalCandidate | FinalAnswerAction] = (
        list(action.task_answers) if action.task_answers else [action]
    )
    tasks = []
    for part in parts:
        refs = set(part.answer.evidence_refs)
        selected = [
            row.result.result
            for row in records
            if row.result
            and (
                row.call.call_id in part.supporting_tool_call_ids
                or refs.intersection(row.result.evidence_refs)
                or (
                    isinstance(row.result.result, BuildResult)
                    and row.result.result.build_ref == part.answer.build_ref
                )
            )
        ]
        if any(isinstance(result, BuildResult) for result in selected):
            task = CompletionRequirement(
                kind="BUILD", constraints_revision=context.requirements.revision
            )
        elif comparison := next(
            (result for result in selected if isinstance(result, ComparisonResult)),
            None,
        ):
            task = CompletionRequirement(
                kind="COMPARE", product_ids=[p.product_id for p in comparison.products]
            )
        elif context.saved_build and context.saved_build.evidence_ref in refs:
            task = CompletionRequirement(
                kind="EXPLAIN_SAVED_BUILD",
                saved_build_ref=context.saved_build.evidence_ref,
            )
        else:
            task = CompletionRequirement(kind="ANSWER")
        tasks.append(task)
    if started_build and not any(task.kind == "BUILD" for task in tasks):
        tasks.append(build_task)
    return tasks


class ResultSelection(Contract):
    task_index: StrictInt | None = Field(default=0, ge=0, lt=MAX_TASK_RESULTS)
    text: Text = Field(max_length=8000)
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)
    build_ref: UUID | None = None
    supporting_tool_call_ids: list[Text] = Field(default_factory=list, max_length=20)
    web_excerpts: list[WebExcerpt] = Field(default_factory=list, max_length=8)
    explanation: Text | None = Field(default=None, max_length=4000)
    explanation_evidence_refs: list[Text] | None = Field(default=None, max_length=100)
    review: ProductReview | None = None


class SubmitResults(Contract):
    """Submit selected evidence for requested tasks. Never assert completion status."""

    results: list[ResultSelection] = Field(min_length=1, max_length=MAX_TASK_RESULTS)


def selected_answer(selection: ResultSelection) -> AnswerCandidate:
    explanation_refs = (
        selection.explanation_evidence_refs
        if selection.explanation_evidence_refs
        else selection.evidence_refs
    )
    return AnswerCandidate(
        text=selection.text,
        evidence_refs=selection.evidence_refs,
        build_ref=selection.build_ref,
        web_excerpts=selection.web_excerpts,
        review=selection.review,
        explanation=ConsultationExplanation(
            text=selection.explanation,
            evidence_refs=explanation_refs,
        )
        if selection.explanation and explanation_refs
        else None,
    )


def resolve_result_selection(
    selection: ResultSelection, records: list[ToolRecord]
) -> tuple[ResultSelection, list[ToolRecord]]:
    """Resolve model-selected calls into canonical evidence for every answer lane."""
    by_id = {record.call.call_id: record for record in records}
    by_name: dict[str, list[ToolRecord]] = {}
    for record in records:
        by_name.setdefault(record.call.name, []).append(record)
    ids: list[str] = []
    for cid in selection.supporting_tool_call_ids:
        if cid in by_id:
            ids.append(cid)
        elif cid in by_name and len(by_name[cid]) == 1:
            ids.append(by_name[cid][0].call.call_id)
        elif not (selection.build_ref is not None or cid.startswith("build-")):
            ids.append(cid)
    if (
        len(by_id) != len(records)
        or len(ids) != len(set(ids))
        or not set(ids) <= by_id.keys()
    ):
        raise ValueError("Submission requires canonical unique tool selections")
    build_by_uuid = {
        str(record.result.result.build_ref): record.result.result.build_ref
        for record in records
        if record.result and isinstance(record.result.result, BuildResult)
    }
    normalized_refs = [
        f"build:{ref}" if ref in build_by_uuid else ref
        for ref in selection.evidence_refs
    ]
    normalized_explanation_refs = (
        [
            f"build:{ref}" if ref in build_by_uuid else ref
            for ref in selection.explanation_evidence_refs
        ]
        if selection.explanation_evidence_refs is not None
        else None
    )
    resolved_build_ref = selection.build_ref
    if resolved_build_ref is None:
        matched_builds = {
            build_by_uuid[ref[6:]]
            for ref in normalized_refs
            if ref.startswith("build:") and ref[6:] in build_by_uuid
        }
        if len(matched_builds) == 1:
            resolved_build_ref = next(iter(matched_builds))
    refs = set(normalized_refs) | set(normalized_explanation_refs or [])
    if not ids and not refs and resolved_build_ref is None:
        # One completed canonical operation has no selection ambiguity. Final
        # task/identity validation still decides whether it fulfills this request.
        candidates = [
            record
            for record in records
            if record.result
            and record.result.status == "COMPLETED"
            and isinstance(
                record.result.result,
                (
                    BuildResult,
                    CatalogResult,
                    DetailsResult,
                    ComparisonResult,
                    CompatibilityCheckResult,
                ),
            )
        ]
        if len(candidates) == 1:
            ids = [candidates[0].call.call_id]
    selected = [
        record
        for record in records
        if record.result
        and (
            record.call.call_id in ids
            or refs.intersection(record.result.evidence_refs)
            or (
                isinstance(record.result.result, BuildResult)
                and record.result.result.build_ref == resolved_build_ref
            )
        )
    ]
    # Selecting a canonical result by call ID/build ref also selects its evidence.
    # Do not require the model to copy the same reference into a second field.
    updates: dict[str, object] = {}
    if ids != selection.supporting_tool_call_ids:
        updates["supporting_tool_call_ids"] = ids
    if normalized_refs != selection.evidence_refs:
        updates["evidence_refs"] = normalized_refs
    if normalized_explanation_refs != selection.explanation_evidence_refs:
        updates["explanation_evidence_refs"] = normalized_explanation_refs
    if resolved_build_ref != selection.build_ref:
        updates["build_ref"] = resolved_build_ref
    resolved_selection = (
        selection.model_copy(update=updates) if updates else selection
    )
    if not resolved_selection.evidence_refs and any(
        record.result
        and isinstance(
            record.result.result,
            (
                BuildResult,
                CatalogResult,
                DetailsResult,
                ComparisonResult,
                CompatibilityCheckResult,
            ),
        )
        for record in selected
    ):
        resolved_selection = resolved_selection.model_copy(
            update={
                "evidence_refs": list(
                    dict.fromkeys(
                        ref
                        for record in selected
                        if record.result
                        and isinstance(
                            record.result.result,
                            (
                                BuildResult,
                                CatalogResult,
                                DetailsResult,
                                ComparisonResult,
                                CompatibilityCheckResult,
                            ),
                        )
                        for ref in record.result.evidence_refs
                    )
                )
            }
        )
    return resolved_selection, selected


def _derive_part(
    selection: ResultSelection, records: list[ToolRecord]
) -> TaskFinalCandidate:
    if selection.task_index is None:
        raise ValueError("Conversation response is not a business-task outcome")
    resolved_selection, selected = resolve_result_selection(selection, records)
    outcome: FinalOutcome = "ANSWERED"
    refs = set(resolved_selection.evidence_refs)
    supporting = list(resolved_selection.supporting_tool_call_ids)
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
            resolved_selection = resolved_selection.model_copy(
                update={
                    "evidence_refs": list(result.evidence_refs),
                    "explanation_evidence_refs": None,
                }
            )
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
    candidate = selected_answer(resolved_selection).model_copy(
        update={"build_ref": build_ref}
    )
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
