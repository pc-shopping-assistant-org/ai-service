"""Core ReAct loop nodes (prepare, agent, validate, execute, resolve) for the flexible graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    ActionKind,
    AskClarificationAction,
    CallToolAction,
    CallToolsAction,
    FinalAnswerAction,
    FlexibleDecision,
    ProposeRequirementsAction,
    RepairFeedback,
    RespondAction,
    action_calls,
    decode_flexible_decision,
)
from ai_service.capabilities.assistant.react.contracts import (
    TOOL_SCHEMAS,
    ToolCall,
    ToolError,
    ToolName,
    ToolRecord,
    ToolResult,
)
from ai_service.capabilities.assistant.react.dialogue import validate_response
from ai_service.capabilities.assistant.react.outcomes import (
    InvalidTaskAnswer,
    validate_final_answer,
)
from ai_service.capabilities.assistant.react.proposal_resolution import (
    resolve_requirement_proposal,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    call_hash,
    project_current_tools,
)
from ai_service.capabilities.assistant.requirements.catalog_policy import (
    build_clarification_reasons,
    build_needs_catalog,
    draft_needs_catalog,
)
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
)
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
    apply_user_receipt,
)
from ai_service.infrastructure.graph.flexible_state import (
    FLEXIBLE_GRAPH_VERSION,
    AsyncNode,
    FlexibleModel,
    FlexibleRuntime,
    FlexibleState,
)
from ai_service.infrastructure.graph.tool_batches import execute_tool_batch


@dataclass(frozen=True, slots=True)
class LoopNodes:
    prepare_agent: AsyncNode
    agent: AsyncNode
    validate: AsyncNode
    execute: AsyncNode
    resolve: AsyncNode


def build_loop_nodes(
    runtime: FlexibleRuntime,
    *,
    model: FlexibleModel,
    dispatcher: ToolDispatcher,
    attempts: AttemptBoundary,
) -> LoopNodes:
    async def prepare_agent(state: FlexibleState) -> dict[str, Any]:
        if (
            runtime.publication
            and runtime.publication.dialogue_continuation
            and (runtime.current(state).requirements == runtime.trusted.requirements)
        ):
            return {}
        if state.get("phase") == "DECIDED":
            return {}
        ctx = runtime.current(state)
        tasks_for_turn = runtime.registered_tasks(state)
        entries = runtime.records(state)
        should_build = (
            not ctx.read_only_turn
            and ctx.execution_allowed
            and ctx.read_catalog_allowed
            and not state.get("pending_proposal")
            and not state.get("pending_draft")
            and build_needs_catalog(ctx.requirements)
            and (
                any(task.kind == "BUILD" for task in tasks_for_turn)
                if tasks_for_turn
                else (
                    ctx.requirements != runtime.trusted.requirements
                    or ctx.saved_build is None
                )
            )
            and not any(
                entry.call.name == ToolName.BUILD_PC
                and entry.origin_run_id == ctx.run_id
                and entry.result is not None
                and entry.result.constraints_revision == ctx.requirements.revision
                for entry in entries
            )
        )
        if not should_build:
            return {}
        loaded = await runtime.load_catalog(state) if not ctx.catalog_ready else {}
        merged = cast(FlexibleState, dict(state) | loaded)
        ctx = runtime.current(merged)
        if not (
            ctx.catalog_ready
            and compile_requirements(ctx.requirements, ctx.catalog).ready
        ):
            return loaded
        if merged.get("tool_calls", 0) >= runtime.limits.max_tool_calls:
            return loaded
        used = {
            entry.call.call_id
            for entry in entries
            if entry.origin_run_id == ctx.run_id
        } | {entry.call.call_id for entry in ctx.control_history}
        call_id = (
            "build" if "build" not in used else f"build_{ctx.requirements.revision}"
        )
        entries.append(
            ToolRecord(
                call=ToolCall(call_id=call_id, name=ToolName.BUILD_PC, arguments={}),
                result=None,
                origin_run_id=ctx.run_id,
                model_step=merged.get("model_calls", 0),
            )
        )
        await execute_tool_batch(
            entries,
            ctx,
            merged.get("model_calls", 0),
            dispatcher,
            attempts,
            runtime.limits,
        )
        return runtime.bounded(
            merged,
            loaded
            | {
                "ledger": [item.model_dump(mode="json") for item in entries],
                "tool_calls": merged.get("tool_calls", 0) + 1,
            },
        )

    async def agent(state: FlexibleState) -> dict[str, Any]:
        ctx = runtime.current(state)
        entries = runtime.records(state)
        transcript = project_current_tools(entries, ctx)
        count = state.get("model_calls", 0)
        if count >= runtime.limits.max_model_calls:
            raise ValueError("REACT_MODEL_LIMIT")
        feedback = (
            RepairFeedback.model_validate(state["repair_feedback"])
            if state.get("repair_feedback")
            else None
        )
        # Proactively stop exposing tools once the tool call budget is exhausted so
        # models synthesize from existing evidence instead of crashing with REACT_TOOL_LIMIT.
        exposed = (
            ()
            if state.get("tool_calls", 0) >= runtime.limits.max_tool_calls
            else allowed_tools(ctx, transcript)
        )
        payload: dict[str, JsonValue] = {
            "message": state["message"],
            "transcript": [r.model_dump(mode="json") for r in transcript],
            "allowed": list(exposed),
            "feedback": feedback.model_dump(mode="json") if feedback else None,
            "context": runtime.context_digest_payload(ctx),
            "graph_version": FLEXIBLE_GRAPH_VERSION,
        }
        decision = await attempts.invoke(
            AttemptKind.MODEL,
            f"flexible:agent:{count}",
            canonical_hash(payload),
            lambda: model.decide(state["message"], transcript, exposed, feedback, ctx),
        )
        return runtime.bounded(
            state,
            {
                "model_calls": count + 1,
                "decision": FlexibleDecision.model_validate(
                    decision.model_dump()
                ).model_dump(mode="json")
                if decision
                else None,
                "phase": "REGISTER_REQUEST"
                if decision and decision.registration is not None
                else "DECIDED",
            },
        )

    async def validate(state: FlexibleState) -> dict[str, Any]:
        if state.get("decision") is None:
            return runtime.repair(state, "INVALID_ACTION")
        decision = decode_flexible_decision(state["decision"])
        action = decision.action
        ctx = runtime.current(state)
        entries = runtime.records(state)
        if decision.control_call and decision.control_call.call_id in {
            entry.call.call_id for entry in entries
        } | {entry.call.call_id for entry in ctx.control_history}:
            return runtime.repair(state, "INVALID_ACTION")
        if isinstance(action, (CallToolAction, CallToolsAction)):
            calls = action_calls(action)
            if len(calls) > runtime.limits.max_tool_calls_per_batch:
                return runtime.repair(
                    state, "BAD_TOOL_ARGUMENTS", ["max_tool_calls_per_batch"]
                )
            if state.get("tool_calls", 0) + len(calls) > runtime.limits.max_tool_calls:
                raise ValueError("REACT_TOOL_LIMIT")
            loaded: dict[str, Any] = {}
            if (
                not ctx.catalog_ready
                and runtime.catalog_loader is not None
                and any(call.name == ToolName.BUILD_PC for call in calls)
            ):
                loaded = await runtime.load_catalog(state)
                if loaded:
                    state = cast(FlexibleState, dict(state) | loaded)
                    ctx = runtime.current(state)
            used = {
                entry.call.call_id
                for entry in entries
                if entry.origin_run_id == ctx.run_id
            } | {entry.call.call_id for entry in ctx.control_history}
            if len({call.call_id for call in calls}) != len(calls) or any(
                call.call_id in used for call in calls
            ):
                return runtime.repair(state, "INVALID_ACTION")
            transcript = project_current_tools(entries, ctx)
            current_allowed = allowed_tools(ctx, transcript)
            for call in calls:
                error = None
                try:
                    name = ToolName(call.name)
                except ValueError:
                    error = ToolError.UNKNOWN_TOOL
                else:
                    if name not in current_allowed:
                        error = ToolError.PREREQUISITE_MISSING
                    else:
                        try:
                            TOOL_SCHEMAS[name].model_validate(call.arguments)
                        except ValidationError:
                            error = ToolError.BAD_ARGUMENTS
                rejected = (
                    ToolResult(
                        tool_call_id=call.call_id,
                        tool=call.name,
                        input_hash=call_hash(call, ctx),
                        constraints_revision=ctx.requirements.revision,
                        status="REJECTED",
                        error=error,
                    )
                    if error
                    else None
                )
                entries.append(
                    ToolRecord(
                        call=call,
                        result=rejected,
                        origin_run_id=ctx.run_id,
                        model_step=state.get("model_calls", 0),
                    )
                )
            return runtime.bounded(
                state,
                loaded
                | {
                    "ledger": [entry.model_dump(mode="json") for entry in entries],
                    "tool_calls": state.get("tool_calls", 0) + len(calls),
                    "repair_feedback": None,
                    "phase": "TOOL_PENDING",
                },
            )
        if isinstance(action, ProposeRequirementsAction):
            return {"phase": "RESOLVE_PROPOSAL"}
        if isinstance(action, RespondAction):
            continuing = bool(
                runtime.publication
                and runtime.publication.dialogue_continuation
                and ctx.requirements == runtime.trusted.requirements
                and not state.get("user_receipts")
            )
            unresolved = runtime.unresolved_tasks(state)
            if unresolved and not continuing:
                return runtime.repair(state, "TASK_INCOMPLETE", unresolved)
            try:
                response = validate_response(
                    action,
                    ctx,
                    ctx.completion_requirements,
                    entries,
                    continuing_pending=continuing,
                )
            except ValueError:
                transcript = project_current_tools(entries, ctx)
                if ctx.completion_requirements and any(
                    entry.result
                    and entry.result.status == "COMPLETED"
                    and entry.result.evidence_refs
                    for entry in transcript
                ):
                    return runtime.repair(
                        state,
                        "TASK_INCOMPLETE",
                        ["submit_result.results"],
                        next_actions=[ActionKind.FINAL_ANSWER],
                    )
                return runtime.repair(state, "TASK_INCOMPLETE")
            return runtime.bounded(
                state,
                {
                    "response": response.model_dump(mode="json"),
                    "repair_feedback": None,
                    "phase": "READY_TO_PUBLISH",
                },
            )
        if isinstance(action, AskClarificationAction):
            pending_fields = runtime.unresolved_tasks(state)
            if pending_fields:
                if not set(action.missing_fields) <= set(pending_fields):
                    return runtime.repair(state, "INVALID_ACTION", pending_fields)
                return runtime.bounded(
                    state,
                    {
                        "clarification": action.model_dump(mode="json"),
                        "phase": "WAITING_INPUT",
                    },
                )
            missing = build_clarification_reasons(
                ctx.requirements, ctx.catalog, catalog_ready=ctx.catalog_ready
            )
            if (
                not any(task.kind == "BUILD" for task in ctx.completion_requirements)
                or not set(action.missing_fields) <= missing
            ):
                return runtime.repair(state, "INVALID_ACTION", sorted(missing))
            return runtime.bounded(
                state,
                {
                    "clarification": action.model_dump(mode="json"),
                    "phase": "WAITING_INPUT",
                },
            )
        assert isinstance(action, FinalAnswerAction)
        unresolved = runtime.unresolved_tasks(state)
        if unresolved:
            return runtime.repair(state, "TASK_INCOMPLETE", unresolved)
        transcript = project_current_tools(entries, ctx)
        try:
            answer = validate_final_answer(
                action,
                [entry.result for entry in transcript if entry.result],
                ctx,
                ctx.completion_requirements,
                dispatcher.service,
            )
        except InvalidTaskAnswer as exc:
            return runtime.repair(
                state,
                "TASK_INCOMPLETE",
                ["task_answers"],
                task_indices=exc.task_indices,
            )
        except ValueError:
            if action.outcome in {"INFEASIBLE", "SEARCH_INCOMPLETE"}:
                return runtime.repair(
                    state,
                    "SEARCH_PROOF_REQUIRED",
                    ["supporting_tool_call_ids", "search_evidence"],
                )
            if action.outcome in {"INSUFFICIENT_EVIDENCE", "DEPENDENCY_UNAVAILABLE"}:
                return runtime.repair(
                    state, "MISSING_EVIDENCE", ["supporting_tool_call_ids"]
                )
            return runtime.repair(state, "TASK_INCOMPLETE")
        return runtime.bounded(
            state,
            {
                "answer": answer.model_dump(mode="json"),
                "repair_feedback": None,
                "phase": "READY_TO_PUBLISH",
            },
        )

    async def execute(state: FlexibleState) -> dict[str, Any]:
        ctx, entries = runtime.current(state), runtime.records(state)
        await execute_tool_batch(
            entries,
            ctx,
            state.get("model_calls", 0),
            dispatcher,
            attempts,
            runtime.limits,
        )
        return runtime.bounded(
            state,
            {
                "ledger": [item.model_dump(mode="json") for item in entries],
                "phase": "TOOL_COMMITTED",
            },
        )

    async def resolve(state: FlexibleState) -> dict[str, Any]:
        action = decode_flexible_decision(state["decision"]).action
        assert isinstance(action, ProposeRequirementsAction)
        loaded: dict[str, Any] = {}
        if draft_needs_catalog(action.draft):
            loaded = await runtime.load_catalog(state)
            if loaded:
                state = cast(FlexibleState, dict(state) | loaded)
        ctx = runtime.current(state)
        if state.get("pending_proposal"):
            return runtime.repair(state, "PROPOSAL_CONFIRMATION_REQUIRED")
        if state.get("pending_draft"):
            return runtime.repair(state, "INVALID_ACTION", ["pending_draft"])
        try:
            result = await resolve_requirement_proposal(
                action,
                ctx,
                state["message"],
                runtime.conversation_revision,
            )
            update: dict[str, Any] = {
                "repair_feedback": None,
            }
            if isinstance(result, PendingResolution):
                update.update(
                    pending_draft=result.model_dump(mode="json"), phase="WAITING_INPUT"
                )
            elif isinstance(result, UserRequirementReceipt):
                changed = apply_user_receipt(
                    ctx.requirements,
                    result,
                    run_id=ctx.run_id,
                    message=state["message"],
                    catalog=ctx.catalog,
                )
                update.update(
                    working_requirements=changed.model_dump(mode="json"),
                    user_receipts=[
                        *state.get("user_receipts", []),
                        result.model_dump(mode="json"),
                    ],
                    phase="REQUIREMENTS_ACCEPTED",
                )
            else:
                update.update(
                    working_requirements=result.requirements.model_dump(mode="json"),
                    pending_proposal=result.pending.model_dump(mode="json")
                    if result.pending
                    else None,
                    phase="WAITING_INPUT"
                    if result.disposition == "NEEDS_CONFIRMATION"
                    else "REQUIREMENTS_UNCHANGED",
                )
        except ValueError:
            return runtime.bounded(
                state, loaded | runtime.repair(state, "INVALID_ACTION")
            )
        return runtime.bounded(state, loaded | runtime.acknowledge_control(state, update))

    return LoopNodes(
        prepare_agent=prepare_agent,
        agent=agent,
        validate=validate,
        execute=execute,
        resolve=resolve,
    )
