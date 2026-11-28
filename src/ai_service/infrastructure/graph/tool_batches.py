"""Execute a frozen native batch; LangGraph persists each completed call."""

import asyncio
import logging
from typing import Any

from langgraph.func import task

from ai_service.application.activity import ActivitySink
from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.capabilities.assistant.public_contracts import AgentActivityStage
from ai_service.capabilities.assistant.react.contracts import (
    BuildResult,
    ReactLimits,
    ToolCall,
    ToolContext,
    ToolName,
    ToolRecord,
    ToolResult,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher, call_hash
from ai_service.infrastructure.observability import step

logger = logging.getLogger(__name__)

_TOOL_STAGES: dict[str, AgentActivityStage] = {
    ToolName.SEARCH_CATALOG.value: AgentActivityStage.SEARCHING_CATALOG,
    ToolName.GET_PRODUCT_DETAILS.value: AgentActivityStage.LOADING_DETAILS,
    ToolName.COMPARE_PRODUCTS.value: AgentActivityStage.COMPARING,
    ToolName.CHECK_BUILD_COMPATIBILITY.value: AgentActivityStage.CHECKING_COMPATIBILITY,
    ToolName.BUILD_PC.value: AgentActivityStage.BUILDING_PC,
    ToolName.SEARCH_WEB.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.READ_WEB_PAGE.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.RESEARCH_HARDWARE.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.RESEARCH_COMPATIBILITY.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.RESEARCH_WORKLOAD.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.RESEARCH_PC_BUILD.value: AgentActivityStage.SEARCHING_WEB,
    ToolName.RESEARCH_DRIVER.value: AgentActivityStage.SEARCHING_WEB,
}


def tool_activity_meta(call: ToolCall) -> tuple[AgentActivityStage, str | None]:
    stage = _TOOL_STAGES.get(call.name, AgentActivityStage.THINKING)
    args = call.arguments
    detail: str | None = None
    if call.name == ToolName.SEARCH_CATALOG.value:
        query = args.get("query")
        brand = args.get("brand")
        category = args.get("category")
        if isinstance(query, str) and query.strip() and query.strip() != "*":
            detail = query.strip()
        elif isinstance(brand, str) and brand.strip():
            detail = brand.strip()
        elif isinstance(category, str) and category.strip():
            detail = category.strip()
    elif call.name == ToolName.COMPARE_PRODUCTS.value:
        queries = args.get("queries")
        if isinstance(queries, list):
            parts = [str(item).strip() for item in queries if isinstance(item, str) and item.strip()]
            if parts:
                detail = " vs ".join(parts[:3])
    elif stage == AgentActivityStage.SEARCHING_WEB:
        for key in ("query", "subject", "workload"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                detail = value.strip()
                break
        subjects = args.get("subjects")
        if detail is None and isinstance(subjects, list):
            parts = [
                str(item).strip()
                for item in subjects
                if isinstance(item, str) and item.strip()
            ]
            if parts:
                detail = ", ".join(parts[:2])
    return stage, detail[:120] if detail else None


async def execute_tool_batch(
    entries: list[ToolRecord],
    context: ToolContext,
    model_step: int,
    dispatcher: ToolDispatcher,
    attempts: AttemptBoundary,
    limits: ReactLimits,
    *,
    activity_sink: ActivitySink | None = None,
) -> None:
    batch = [
        entry
        for entry in entries
        if entry.origin_run_id == context.run_id and entry.model_step == model_step
    ]
    previous = [entry for entry in entries if entry not in batch]

    async def dispatch(value: dict[str, Any]) -> dict[str, Any]:
        call = ToolCall.model_validate(value)
        if activity_sink is not None:
            stage, detail = tool_activity_meta(call)
            activity_sink(stage, call.name, detail)
        with step(f"tool.{call.name}") as observation:
            result = await attempts.invoke(
                AttemptKind.TOOL_IO,
                f"flexible:tool:{call.call_id}",
                call_hash(call, context),
                lambda: dispatcher.execute(call, context, records=previous),
            )
            observation.span.set_attribute("ai.tool.result_type", result.tool)
            observation.result_status = result.status
            if isinstance(result.result, BuildResult):
                build_outcome = (
                    result.result.search_evidence.outcome
                    if result.result.search_evidence
                    and result.result.receipt.outcome != "BUILT"
                    else result.result.receipt.outcome
                )
                observation.span.set_attribute("ai.build.outcome", build_outcome)
                logger.info(
                    "tool.build_pc.completed outcome=%s build_ref=%s",
                    build_outcome,
                    result.result.build_ref,
                )
        byte_limit = (
            limits.max_build_result_bytes
            if isinstance(result.result, BuildResult)
            else limits.max_tool_bytes
        )
        if len(result.model_dump_json().encode()) > byte_limit:
            raise ValueError("REACT_TOOL_RESULT_TOO_LARGE")
        return result.model_dump(mode="json")

    dispatch_task = task(dispatch)
    reads = [
        entry
        for entry in batch
        if entry.result is None and entry.call.name != ToolName.BUILD_PC
    ]
    # Join every child before propagating a fatal interruption. Successful native
    # task writes are then available to the authorized recovery execution.
    outcomes = await asyncio.gather(
        *(dispatch_task(entry.call.model_dump(mode="json")) for entry in reads),
        return_exceptions=True,
    )
    for entry, outcome in zip(reads, outcomes, strict=True):
        if not isinstance(outcome, BaseException):
            entry.result = ToolResult.model_validate(outcome)
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    # Optimizer executions share the frozen requirements but run sequentially.
    # Reads requiring a new build/ref belong in a later model response.
    for entry in batch:
        if entry.result is None:
            entry.result = ToolResult.model_validate(
                await dispatch_task(entry.call.model_dump(mode="json"))
            )
