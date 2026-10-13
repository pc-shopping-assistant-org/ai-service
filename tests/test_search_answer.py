import pytest
from test_react_tools import Catalog, context
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.react.actions import FinalAnswerAction
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolCall,
)
from ai_service.capabilities.assistant.react.outcomes import validate_search_answer
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer


async def no_result(realistic_catalog, *, incomplete=False):
    canonical = state()
    if incomplete:
        parts = [
            part.model_copy(update={"performance_score": None})
            if part.category == "CPU"
            else part
            for part in realistic_catalog
        ]
    else:
        next(iter(canonical.requirements.values())).value.amount = 1
        parts = realistic_catalog
    ctx = context(canonical, rows(parts))
    service = PCBuildApplicationService(DeterministicPCOptimizer(enable_pruning=False))
    result = await ToolDispatcher(Catalog(), service).execute(
        ToolCall(call_id="build", name="build_pc", arguments={}), ctx
    )
    return ctx, service, result


@pytest.mark.asyncio
@pytest.mark.parametrize("incomplete", [False, True])
async def test_real_scoped_report_controls_build_task_completion(
    realistic_catalog, incomplete
):
    ctx, service, result = await no_result(realistic_catalog, incomplete=incomplete)
    outcome = "SEARCH_INCOMPLETE" if incomplete else "INFEASIBLE"
    action = FinalAnswerAction(
        outcome=outcome,
        supporting_tool_call_ids=["build"],
        answer={"text": "No build exists in the world"},
    )
    answer = validate_search_answer(
        action,
        [result],
        ctx,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=ctx.requirements.revision
            )
        ],
        service,
    )
    status = "UNRESOLVED" if incomplete else "COMPLETED"
    assert answer.status == answer.task_outcomes[0].status == status
    assert answer.search_evidence == result.result.search_evidence
    assert answer.build_ref is None
    assert "world" not in answer.text and "ứng viên" in answer.text


@pytest.mark.asyncio
async def test_proven_search_does_not_complete_comparison_task(realistic_catalog):
    ctx, service, result = await no_result(realistic_catalog)
    from uuid import uuid4

    tasks = [
        CompletionRequirement(kind="BUILD", constraints_revision=0),
        CompletionRequirement(kind="COMPARE", product_ids=[uuid4(), uuid4()]),
    ]
    answer = validate_search_answer(
        FinalAnswerAction(
            outcome="INFEASIBLE",
            supporting_tool_call_ids=["build"],
            answer={"text": "Done"},
        ),
        [result],
        ctx,
        tasks,
        service,
    )
    assert answer.status == "UNRESOLVED"
    assert [task.status for task in answer.task_outcomes] == ["COMPLETED", "UNRESOLVED"]


@pytest.mark.asyncio
async def test_incomplete_or_legacy_result_cannot_be_relabelled_infeasible(
    realistic_catalog,
):
    ctx, service, result = await no_result(realistic_catalog, incomplete=True)
    action = FinalAnswerAction(
        outcome="INFEASIBLE",
        supporting_tool_call_ids=["build"],
        answer={"text": "No solution"},
    )
    tasks = [CompletionRequirement(kind="BUILD", constraints_revision=0)]
    with pytest.raises(ValueError):
        validate_search_answer(action, [result], ctx, tasks, service)
    legacy = result.model_copy(deep=True)
    legacy.result.search_evidence = None
    with pytest.raises(ValueError):
        validate_search_answer(action, [legacy], ctx, tasks, service)
