"""V2 result completion. Negative tool outcomes never prove absent PC builds."""

from typing import Literal
from uuid import UUID

from pydantic import Field, StrictInt, model_validator

from ai_service.application.ports.web_search import WebFactStatus, WebPageReadStatus
from ai_service.capabilities.assistant.requirements.search_report import search_report
from ai_service.capabilities.assistant.stateful_contracts import Contract, Text
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

from .actions import FinalAnswerAction, FinalOutcome
from .contracts import (
    BuildResult,
    CatalogResult,
    CompletionRequirement,
    ReadWebPageResult,
    SearchWebResult,
    ToolContext,
    ToolError,
    ToolResult,
    ValidatedAnswer,
)
from .search_evidence import BuildSearchEvidence

type AnswerReason = Literal[
    "NO_RESULTS",
    "SOURCE_UNAVAILABLE",
    "DEPENDENCY_UNAVAILABLE",
    "SCOPED_EXHAUSTION",
    "INCOMPLETE_SEARCH",
    "FACT_UNKNOWN",
    "FACT_CONFLICT",
]


class InvalidTaskAnswer(ValueError):
    """Static failure plus trusted task indices; no model/adapter exception prose."""

    def __init__(self, task_indices: list[int]):
        super().__init__("Invalid registered task answer")
        self.task_indices = task_indices


class TaskOutcome(Contract):
    task: CompletionRequirement
    status: Literal["COMPLETED", "UNRESOLVED"]
    outcome: FinalOutcome | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    reason: AnswerReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class FlexibleAnswer(Contract):
    schema_version: StrictInt = Field(default=2, ge=2, le=2)
    text: Text
    status: Literal["COMPLETED", "UNRESOLVED"]
    outcome: Literal[
        "ANSWERED",
        "BUILT",
        "INSUFFICIENT_EVIDENCE",
        "DEPENDENCY_UNAVAILABLE",
        "INFEASIBLE",
        "SEARCH_INCOMPLETE",
        "MULTI_TASK",
    ]
    reason: AnswerReason | None = None
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)
    supporting_tool_call_ids: list[Text] = Field(default_factory=list, max_length=20)
    build_ref: UUID | None = None
    task_outcomes: list[TaskOutcome] = Field(min_length=1, max_length=8)
    search_evidence: BuildSearchEvidence | None = None

    @model_validator(mode="after")
    def completion_consistency(self) -> "FlexibleAnswer":
        completed = all(task.status == "COMPLETED" for task in self.task_outcomes)
        if completed != (self.status == "COMPLETED"):
            raise ValueError("Answer status must reflect trusted task completion")
        negative = self.outcome in {"INSUFFICIENT_EVIDENCE", "DEPENDENCY_UNAVAILABLE"}
        if self.outcome == "MULTI_TASK":
            if (
                len(self.task_outcomes) < 2
                or self.reason is not None
                or any(row.outcome is None for row in self.task_outcomes)
            ):
                raise ValueError(
                    "Multi-task answer requires individual validated outcomes"
                )
            if bool(self.build_ref) != any(
                row.outcome == "BUILT" for row in self.task_outcomes
            ):
                raise ValueError(
                    "Multi-task build reference must match a completed build outcome"
                )
            searched = [
                row
                for row in self.task_outcomes
                if row.outcome in {"INFEASIBLE", "SEARCH_INCOMPLETE"}
            ]
            if bool(searched) != bool(self.search_evidence) or any(
                row.task.kind != "BUILD"
                or self.search_evidence is None
                or row.outcome != self.search_evidence.outcome
                or row.status
                != ("COMPLETED" if row.outcome == "INFEASIBLE" else "UNRESOLVED")
                for row in searched
            ):
                raise ValueError(
                    "Multi-task search outcome lacks matching scoped evidence"
                )
            return self
        if self.outcome in {"INFEASIBLE", "SEARCH_INCOMPLETE"}:
            if (
                self.search_evidence is None
                or self.search_evidence.outcome != self.outcome
                or self.build_ref is not None
            ):
                raise ValueError(
                    "Search answer requires matching scoped execution evidence"
                )
            expected_reason = (
                "SCOPED_EXHAUSTION"
                if self.outcome == "INFEASIBLE"
                else "INCOMPLETE_SEARCH"
            )
            if self.reason != expected_reason:
                raise ValueError("Search answer reason does not match outcome")
            for task in self.task_outcomes:
                expected = (
                    "COMPLETED"
                    if task.task.kind == "BUILD" and self.outcome == "INFEASIBLE"
                    else "UNRESOLVED"
                )
                if task.status != expected:
                    raise ValueError(
                        "Search answer cannot complete unrelated or incomplete tasks"
                    )
        elif self.search_evidence is not None:
            raise ValueError("Non-search answer cannot carry no-solution evidence")
        elif negative:
            if completed or self.reason is None or self.build_ref is not None:
                raise ValueError(
                    "Negative data outcome cannot complete tasks or claim a build"
                )
        elif not completed or self.reason is not None:
            raise ValueError("Successful answer requires completed tasks")
        return self


def completed_answer(
    answer: ValidatedAnswer,
    required: list[CompletionRequirement],
) -> FlexibleAnswer:
    answer = ValidatedAnswer.model_validate(answer.model_dump())
    if answer.status != "COMPLETED":
        raise ValueError("Clarification is not a completed final answer")
    return FlexibleAnswer(
        text=answer.text,
        status="COMPLETED",
        outcome="BUILT" if answer.build_ref else "ANSWERED",
        evidence_refs=answer.evidence_refs,
        build_ref=answer.build_ref,
        task_outcomes=[TaskOutcome(task=task, status="COMPLETED") for task in required],
    )


def validate_negative_answer(
    action: FinalAnswerAction,
    ledger: list[ToolResult],
    context: ToolContext,
    required: list[CompletionRequirement],
) -> FlexibleAnswer:
    """Validate only actual unavailable/empty results, not claims in free prose.

    Caller first verifies context hashes against immutable call/result pairs.
    This is an unresolved response, not successful completion of BUILD/COMPARE.
    Missing facts need canonical UNKNOWN/CONFLICT references; general readable
    prose does not certify absence. Optimizer reports use a separate boundary.
    """
    action = FinalAnswerAction.model_validate(action.model_dump())
    context = ToolContext.model_validate(context.model_dump())
    ledger = [ToolResult.model_validate(item.model_dump()) for item in ledger]
    tasks = [
        CompletionRequirement.model_validate(task.model_dump()) for task in required
    ]
    ids = action.supporting_tool_call_ids
    by_id = {item.tool_call_id: item for item in ledger}
    if (
        not tasks
        or not ids
        or len(ids) != len(set(ids))
        or len(by_id) != len(ledger)
        or not set(ids) <= by_id.keys()
        or action.answer.build_ref is not None
        or action.answer.web_excerpts
        or action.answer.review is not None
        or action.answer.clarification_fields
    ):
        raise ValueError("Negative answer requires exact canonical tool outcomes")
    selected = [by_id[identifier] for identifier in ids]
    if any(
        item.constraints_revision != context.requirements.revision for item in selected
    ):
        raise ValueError("Negative answer uses stale requirements")
    if action.answer.evidence_refs:
        from .tools import validate_answer

        refs = set(action.answer.evidence_refs)
        statuses: list[WebFactStatus] = []
        found: set[str] = set()
        for item in selected:
            page = item.result
            if (
                action.outcome != "INSUFFICIENT_EVIDENCE"
                or item.status != "COMPLETED"
                or not isinstance(page, ReadWebPageResult)
                or page.status != WebPageReadStatus.READABLE
            ):
                raise ValueError("Missing fact requires a completed structured page")
            facts = [fact for fact in page.facts if fact.evidence_ref in refs]
            if not facts or any(
                fact.fact.fact_status == WebFactStatus.SUPPORTED for fact in facts
            ):
                raise ValueError("Missing fact requires canonical UNKNOWN/CONFLICT")
            statuses.extend(fact.fact.fact_status for fact in facts)
            found.update(fact.evidence_ref for fact in facts)
        if found != refs:
            raise ValueError("Unknown missing-fact reference")
        # Reuse the grounded fact/citation renderer, not LLM prose. Its local ANSWER
        # rendering check does not complete any of the actual requested tasks.
        rendered = validate_answer(
            action.answer, selected, context, [CompletionRequirement(kind="ANSWER")]
        )
        return FlexibleAnswer(
            text=rendered.text + "\n\nChưa đủ bằng chứng để hoàn thành yêu cầu.",
            status="UNRESOLVED",
            outcome="INSUFFICIENT_EVIDENCE",
            reason="FACT_CONFLICT"
            if WebFactStatus.CONFLICT in statuses
            else "FACT_UNKNOWN",
            evidence_refs=action.answer.evidence_refs,
            supporting_tool_call_ids=ids,
            task_outcomes=[
                TaskOutcome(task=task, status="UNRESOLVED") for task in tasks
            ],
        )
    if action.outcome == "DEPENDENCY_UNAVAILABLE":
        if not all(
            item.status == "FAILED"
            and item.error
            in {
                ToolError.CATALOG_UNAVAILABLE,
                ToolError.WEB_SEARCH_UNAVAILABLE,
            }
            for item in selected
        ):
            raise ValueError("Dependency failure requires actual adapter failure")
        reason: Literal[
            "NO_RESULTS", "SOURCE_UNAVAILABLE", "DEPENDENCY_UNAVAILABLE"
        ] = "DEPENDENCY_UNAVAILABLE"
        text = "Nguồn dữ liệu hiện không khả dụng. Yêu cầu chưa hoàn thành; vui lòng thử lại sau."
    elif action.outcome == "INSUFFICIENT_EVIDENCE":
        reasons = []
        for item in selected:
            payload = item.result
            if item.status != "COMPLETED":
                raise ValueError("Insufficient evidence requires a typed data outcome")
            if (isinstance(payload, CatalogResult) and not payload.products) or (
                isinstance(payload, SearchWebResult) and not payload.results
            ):
                reasons.append("NO_RESULTS")
            elif (
                isinstance(payload, ReadWebPageResult)
                and payload.status != WebPageReadStatus.READABLE
            ):
                reasons.append("SOURCE_UNAVAILABLE")
            else:
                raise ValueError("Tool result does not support the missing-data claim")
        reason = (
            "SOURCE_UNAVAILABLE" if "SOURCE_UNAVAILABLE" in reasons else "NO_RESULTS"
        )
        text = (
            "Chưa đọc được nội dung nguồn đã chọn. Yêu cầu chưa hoàn thành."
            if reason == "SOURCE_UNAVAILABLE"
            else "Lượt tìm kiếm đã chọn không trả kết quả. Chưa đủ dữ liệu để hoàn thành yêu cầu; điều này không chứng minh không có cấu hình phù hợp."
        )
    else:
        raise ValueError("Tool data failure is not optimizer search evidence")
    return FlexibleAnswer(
        text=text,
        status="UNRESOLVED",
        outcome=action.outcome,
        reason=reason,
        supporting_tool_call_ids=ids,
        task_outcomes=[TaskOutcome(task=task, status="UNRESOLVED") for task in tasks],
    )


def validate_search_answer(
    action: FinalAnswerAction,
    ledger: list[ToolResult],
    context: ToolContext,
    required: list[CompletionRequirement],
    service: PCBuildApplicationService,
) -> FlexibleAnswer:
    """Re-derive the selected report; prose/legacy receipt outcome is never proof."""
    action = FinalAnswerAction.model_validate(action.model_dump())
    context = ToolContext.model_validate(context.model_dump())
    tasks = [
        CompletionRequirement.model_validate(task.model_dump()) for task in required
    ]
    ledger = [ToolResult.model_validate(item.model_dump()) for item in ledger]
    if (
        action.outcome not in {"INFEASIBLE", "SEARCH_INCOMPLETE"}
        or len(action.supporting_tool_call_ids) != 1
        or not any(task.kind == "BUILD" for task in tasks)
        or action.answer.build_ref is not None
        or action.answer.web_excerpts
        or action.answer.review is not None
        or len({item.tool_call_id for item in ledger}) != len(ledger)
    ):
        raise ValueError(
            "Search answer requires one canonical optimizer outcome and BUILD task"
        )
    selected = next(
        (
            item
            for item in ledger
            if item.tool_call_id == action.supporting_tool_call_ids[0]
        ),
        None,
    )
    if (
        selected is None
        or selected.status != "COMPLETED"
        or not isinstance(selected.result, BuildResult)
    ):
        raise ValueError("Missing completed optimizer outcome")
    if selected.constraints_revision != context.requirements.revision or any(
        task.kind == "BUILD"
        and task.constraints_revision != context.requirements.revision
        for task in tasks
    ):
        raise ValueError("Search outcome/task is stale")
    if not set(action.answer.evidence_refs) <= set(selected.evidence_refs):
        raise ValueError("Search answer cites unrelated evidence")
    payload = selected.result
    report = search_report(
        context.requirements,
        context.catalog,
        service,
        payload.receipt,
        build_ref=payload.build_ref,
    )
    if (
        report is None
        or payload.search_evidence != report
        or report.outcome != action.outcome
    ):
        raise ValueError(
            "Search report is missing, changed or does not support requested outcome"
        )
    proven = report.outcome == "INFEASIBLE"
    task_outcomes = [
        TaskOutcome(
            task=task,
            status="COMPLETED" if proven and task.kind == "BUILD" else "UNRESOLVED",
        )
        for task in tasks
    ]
    completed = all(task.status == "COMPLETED" for task in task_outcomes)
    text = (
        "Trong tập ứng viên đã cung cấp, không có cấu hình thỏa ngân sách và các ràng buộc được hỗ trợ. Kết luận không áp dụng cho toàn catalog hoặc các linh kiện ngoài tập này."
        if proven
        else "Chưa tìm được cấu hình trong tập ứng viên đã cung cấp. Dữ liệu hoặc phạm vi tìm kiếm chưa đủ để kết luận vô nghiệm."
    )
    if not completed:
        text += " Yêu cầu chưa hoàn thành đầy đủ."
    return FlexibleAnswer(
        text=text,
        status="COMPLETED" if completed else "UNRESOLVED",
        outcome=action.outcome,
        reason="SCOPED_EXHAUSTION" if proven else "INCOMPLETE_SEARCH",
        evidence_refs=selected.evidence_refs,
        supporting_tool_call_ids=action.supporting_tool_call_ids,
        task_outcomes=task_outcomes,
        search_evidence=report,
    )


def validate_final_answer(
    action: FinalAnswerAction,
    ledger: list[ToolResult],
    context: ToolContext,
    required: list[CompletionRequirement],
    service: PCBuildApplicationService,
) -> FlexibleAnswer:
    """Canonical final-action policy shared by graph and publication preparation."""
    from .tools import validate_answer

    action = FinalAnswerAction.model_validate(action.model_dump())
    ledger = [ToolResult.model_validate(item.model_dump()) for item in ledger]
    context = ToolContext.model_validate(context.model_dump())
    if action.outcome == "MULTI_TASK":
        return validate_task_answers(action, ledger, context, required, service)
    if action.outcome in {"INFEASIBLE", "SEARCH_INCOMPLETE"}:
        return validate_search_answer(action, ledger, context, required, service)
    if action.outcome in {"INSUFFICIENT_EVIDENCE", "DEPENDENCY_UNAVAILABLE"}:
        return validate_negative_answer(action, ledger, context, required)
    selected = set(action.answer.evidence_refs)
    if any(
        isinstance(item.result, ReadWebPageResult)
        and any(
            fact.evidence_ref in selected
            and fact.fact.fact_status != WebFactStatus.SUPPORTED
            for fact in item.result.facts
        )
        for item in ledger
    ):
        raise ValueError(
            "Unknown/conflicting fact requires unresolved insufficient evidence"
        )
    if any(
        isinstance(item.result, BuildResult)
        and item.result.receipt.outcome != "BUILT"
        and selected.intersection(item.evidence_refs)
        for item in ledger
    ):
        raise ValueError("No-solution receipt requires scoped search validation")
    if action.outcome == "BUILT" and action.answer.build_ref is None:
        raise ValueError("Built answer requires an optimizer build reference")
    validated = validate_answer(action.answer, ledger, context, required)
    return completed_answer(validated, required)


def validate_task_answers(
    action: FinalAnswerAction,
    ledger: list[ToolResult],
    context: ToolContext,
    required: list[CompletionRequirement],
    service: PCBuildApplicationService,
) -> FlexibleAnswer:
    """Aggregate independently proven task results; model cannot supply task kinds/statuses."""
    tasks = [
        CompletionRequirement.model_validate(task.model_dump()) for task in required
    ]
    parts = {part.task_index: part for part in action.task_answers}
    if (
        len(tasks) < 2
        or len(parts) != len(action.task_answers)
        or set(parts) != set(range(len(tasks)))
    ):
        raise InvalidTaskAnswer(
            [
                index
                for index in range(len(tasks))
                if sum(part.task_index == index for part in action.task_answers) != 1
            ]
        )
    answers = []
    for index, task in enumerate(tasks):
        part = parts[index]
        try:
            if part.outcome == "BUILT" and task.kind != "BUILD":
                raise ValueError(
                    "BUILT candidate must target the registered BUILD task"
                )
            single = FinalAnswerAction(**part.model_dump(exclude={"task_index"}))
            result = validate_final_answer(single, ledger, context, [task], service)
            if result.outcome != part.outcome:
                raise ValueError("Task outcome differs from the proven result")
        except ValueError as exc:
            raise InvalidTaskAnswer([index]) from exc
        answers.append(result)
    builds = {answer.build_ref for answer in answers if answer.build_ref is not None}
    reports = [answer.search_evidence for answer in answers if answer.search_evidence]
    if len(builds) > 1 or any(report != reports[0] for report in reports):
        raise ValueError("Multi-task result has conflicting build/search snapshots")
    return FlexibleAnswer(
        outcome="MULTI_TASK",
        status="COMPLETED"
        if all(answer.status == "COMPLETED" for answer in answers)
        else "UNRESOLVED",
        text="\n\n".join(
            f"{index + 1}. {tasks[index].kind}\n{answer.text}"
            for index, answer in enumerate(answers)
        ),
        evidence_refs=list(
            dict.fromkeys(ref for answer in answers for ref in answer.evidence_refs)
        ),
        supporting_tool_call_ids=list(
            dict.fromkeys(
                ref for answer in answers for ref in answer.supporting_tool_call_ids
            )
        ),
        build_ref=next(iter(builds), None),
        search_evidence=reports[0] if reports else None,
        task_outcomes=[
            TaskOutcome(
                task=tasks[index],
                status=answer.status,
                outcome=parts[index].outcome,
                reason=answer.reason,
            )
            for index, answer in enumerate(answers)
        ],
    )
