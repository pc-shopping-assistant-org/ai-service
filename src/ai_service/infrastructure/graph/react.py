"""Bounded internal subflow; not a detached HTTP task or publication authority."""

import json
from collections.abc import Callable
from typing import Any, Protocol, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import Field, StrictInt

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
    ToolContext,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    call_hash,
    validate_answer,
)
from ai_service.capabilities.assistant.stateful_contracts import Contract


class ReactDecision(Contract):
    calls: list[ToolCall] = Field(default_factory=list, max_length=20)
    answer: AnswerCandidate | None = None


class ReactModel(Protocol):
    async def decide(
        self,
        message: str,
        transcript: list[ToolRecord],
        allowed: tuple[ToolName, ...],
        repair: bool,
        context: ToolContext | None = None,
    ) -> ReactDecision: ...


class ReactLimits(Contract):
    max_model_calls: StrictInt = Field(default=12, ge=1, le=100)
    max_tool_calls: StrictInt = Field(default=8, ge=1, le=50)
    max_repairs: StrictInt = Field(default=2, ge=0, le=10)
    max_state_bytes: StrictInt = Field(default=512_000, ge=1)
    max_tool_bytes: StrictInt = Field(default=128_000, ge=1)


class ReactState(TypedDict, total=False):
    message: str
    ledger: list[dict[str, Any]]
    decision: dict[str, Any]
    model_calls: int
    repairs: int
    repair: bool
    phase: str
    answer: dict[str, Any]
    react_answer: dict[str, Any]


def add_react_nodes(
    builder: StateGraph,
    model: ReactModel,
    dispatcher: ToolDispatcher,
    context: Callable[[ReactState], ToolContext],
    required: Callable[[ReactState], list[CompletionRequirement]],
    attempts: AttemptBoundary,
    *,
    limits: ReactLimits | None = None,
    done_node: str = END,
    answer_channel: str = "answer",
) -> None:
    limits = limits or ReactLimits()

    def trusted_context(state: ReactState) -> ToolContext:
        return ToolContext.model_validate(context(state).model_dump())

    def bounded(state: ReactState, update: dict[str, Any]) -> dict[str, Any]:
        size = len(
            json.dumps(
                dict(state) | update, ensure_ascii=False, allow_nan=False
            ).encode()
        )
        if size > limits.max_state_bytes:
            raise ValueError("REACT_STATE_TOO_LARGE")
        return update

    def records(state: ReactState) -> list[ToolRecord]:
        current = trusted_context(state)
        entries = [ToolRecord.model_validate(row) for row in state.get("ledger", [])]
        if len({entry.call.call_id for entry in entries}) != len(entries):
            raise ValueError("REACT_DUPLICATE_CALL_ID")
        for entry in entries:
            if entry.result and entry.result.input_hash != call_hash(
                entry.call, current
            ):
                raise ValueError("REACT_TOOL_CONTEXT_CHANGED")
        return entries

    def repair(state: ReactState) -> dict[str, Any]:
        repairs = state.get("repairs", 0) + 1
        if repairs > limits.max_repairs:
            raise ValueError("REACT_VALIDATION_EXHAUSTED")
        return bounded(
            state,
            {"repairs": repairs, "repair": True, "phase": "REPAIR", answer_channel: {}},
        )

    async def agent(state: ReactState) -> dict[str, Any]:
        current = trusted_context(state)
        if not current.execution_allowed or not current.read_catalog_allowed:
            raise ValueError("REACT_EXECUTION_NOT_AUTHORIZED")
        if not state.get("message", "").strip() or len(state["message"]) > 4000:
            raise ValueError("REACT_MESSAGE_INVALID")
        bounded(state, {})
        transcript = records(state)
        if any(entry.result is None for entry in transcript):
            raise ValueError("REACT_UNRESOLVED_TOOL_PROTOCOL")
        count = state.get("model_calls", 0)
        if count >= limits.max_model_calls:
            raise ValueError("REACT_MODEL_LIMIT")
        exposed = allowed_tools(current)
        request_hash = canonical_hash(
            {
                "message": state["message"],
                "transcript": [r.model_dump(mode="json") for r in transcript],
                "allowed": list(exposed),
                "repair": state.get("repair", False),
                "requirements": current.requirements.model_dump(mode="json"),
                "task_message": current.task_message,
                "completion_requirements": [
                    task.model_dump(mode="json")
                    for task in current.completion_requirements
                ],
                "saved_build": current.saved_build.model_dump(mode="json")
                if current.saved_build
                else None,
            }
        )
        decision = await attempts.invoke(
            AttemptKind.MODEL,
            f"react:agent:{count}",
            request_hash,
            lambda: model.decide(
                state["message"],
                transcript,
                exposed,
                state.get("repair", False),
                current,
            ),
        )
        decision = ReactDecision.model_validate(decision.model_dump())
        return bounded(
            state,
            {
                "decision": decision.model_dump(mode="json"),
                "model_calls": count + 1,
                answer_channel: {},
                "phase": "DECIDED",
            },
        )

    async def validate_decision(state: ReactState) -> dict[str, Any]:
        decision = ReactDecision.model_validate(state["decision"])
        transcript = records(state)
        # Reject the entire malformed response. No partial assistant/tool pair
        # enters canonical history; repair sees only the last valid transcript.
        if len(decision.calls) > 1 or bool(decision.calls) == (
            decision.answer is not None
        ):
            return repair(state)
        if decision.calls:
            if len(transcript) >= limits.max_tool_calls:
                raise ValueError("REACT_TOOL_LIMIT")
            call = decision.calls[0]
            if call.call_id in {r.call.call_id for r in transcript}:
                return repair(state)
            transcript.append(ToolRecord(call=call))
            return bounded(
                state,
                {
                    "ledger": [r.model_dump(mode="json") for r in transcript],
                    "phase": "TOOL_PENDING",
                    "repair": False,
                },
            )
        assert decision.answer is not None
        current = trusted_context(state)
        tasks = [
            CompletionRequirement.model_validate(t.model_dump())
            for t in required(state)
        ]
        try:
            answer = validate_answer(
                decision.answer,
                [r.result for r in transcript if r.result],
                current,
                tasks,
            )
        except ValueError:
            return repair(state)
        return bounded(
            state,
            {
                answer_channel: answer.model_dump(mode="json"),
                "phase": "READY_TO_PUBLISH",
                "repair": False,
            },
        )

    async def execute(state: ReactState) -> dict[str, Any]:
        current = trusted_context(state)
        transcript = records(state)
        entry = transcript[-1]
        if entry.result is None:
            if entry.call.name in {ToolName.SEARCH_WEB, ToolName.READ_WEB_PAGE}:
                entry.result = await attempts.invoke(
                    AttemptKind.TOOL_IO,
                    f"react:tool:{entry.call.call_id}",
                    call_hash(entry.call, current),
                    lambda: dispatcher.execute(entry.call, current),
                )
            else:
                entry.result = await dispatcher.execute(entry.call, current)
        if len(entry.result.model_dump_json().encode()) > limits.max_tool_bytes:
            raise ValueError("REACT_TOOL_RESULT_TOO_LARGE")
        return bounded(
            state,
            {
                "ledger": [r.model_dump(mode="json") for r in transcript],
                "phase": "TOOL_COMMITTED",
            },
        )

    async def observe(state: ReactState) -> dict[str, Any]:
        # Transcript is a pure projection of validated records; no independently
        # writable messages channel and no model call before all pairs resolve.
        if any(r.result is None for r in records(state)):
            raise ValueError("REACT_UNRESOLVED_TOOL_PROTOCOL")
        return {"phase": "OBSERVED"}

    # Use the owning graph's full state schema; automatic annotation inference
    # would filter out requirements/catalog channels when these nodes are embedded.
    builder.add_node("agent", agent, input_schema=builder.state_schema)
    builder.add_node(
        "validate_decision", validate_decision, input_schema=builder.state_schema
    )
    builder.add_node("execute_tool", execute, input_schema=builder.state_schema)
    builder.add_node("observe", observe, input_schema=builder.state_schema)
    builder.add_edge("agent", "validate_decision")
    builder.add_conditional_edges(
        "validate_decision",
        lambda s: s["phase"],
        {
            "REPAIR": "agent",
            "TOOL_PENDING": "execute_tool",
            "READY_TO_PUBLISH": done_node,
        },
    )
    builder.add_edge("execute_tool", "observe")
    builder.add_edge("observe", "agent")


def build_react_graph(
    model: ReactModel,
    dispatcher: ToolDispatcher,
    context: ToolContext,
    required: list[CompletionRequirement],
    attempts: AttemptBoundary,
    *,
    limits: ReactLimits | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    context = ToolContext.model_validate(context.model_dump())
    required = [CompletionRequirement.model_validate(t.model_dump()) for t in required]
    if not required:
        raise ValueError("Trusted completion requirements are required")
    builder = StateGraph(ReactState)
    add_react_nodes(
        builder,
        model,
        dispatcher,
        lambda state: context,
        lambda state: required,
        attempts,
        limits=limits,
    )
    builder.add_edge(START, "agent")
    return builder.compile(checkpointer=checkpointer)
