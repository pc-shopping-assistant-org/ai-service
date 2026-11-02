"""Execute a frozen native batch; LangGraph persists each completed call."""

import asyncio
from typing import Any

from langgraph.func import task

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
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


async def execute_tool_batch(
    entries: list[ToolRecord],
    context: ToolContext,
    model_step: int,
    dispatcher: ToolDispatcher,
    attempts: AttemptBoundary,
    limits: ReactLimits,
) -> None:
    batch = [
        entry
        for entry in entries
        if entry.origin_run_id == context.run_id and entry.model_step == model_step
    ]
    previous = [entry for entry in entries if entry not in batch]

    async def dispatch(value: dict[str, Any]) -> dict[str, Any]:
        call = ToolCall.model_validate(value)
        with step(f"tool.{call.name}") as observation:
            result = await attempts.invoke(
                AttemptKind.TOOL_IO,
                f"flexible:tool:{call.call_id}",
                call_hash(call, context),
                lambda: dispatcher.execute(call, context, records=previous),
            )
            observation.span.set_attribute("ai.tool.result_type", result.tool)
            observation.result_status = result.status
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
