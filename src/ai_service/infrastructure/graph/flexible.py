"""Internal V2 loop; publication and execution ownership remain application-owned.

Used by the managed V2 runtime. Inputs are trusted accepted/checkpoint state,
never raw request dictionaries.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ReactLimits,
    ToolContext,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.task_amendment import TaskAmender
from ai_service.capabilities.assistant.react.task_resolution import (
    TaskCatalog,
    TaskClarificationExtractor,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.amendment import PendingAmender
from ai_service.capabilities.assistant.requirements.build import (
    ClarificationExtractor,
    RequirementCatalog,
)
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitClarificationExtractor,
)
from ai_service.infrastructure.graph.flexible_clarification import (
    build_clarification_nodes,
)
from ai_service.infrastructure.graph.flexible_loop import build_loop_nodes
from ai_service.infrastructure.graph.flexible_state import (
    FLEXIBLE_GRAPH_VERSION,
    FlexibleModel,
    FlexibleRuntime,
    FlexibleState,
)
from ai_service.infrastructure.observability import observed_async, observed_sync


def build_flexible_graph(
    model: FlexibleModel,
    dispatcher: ToolDispatcher,
    context: ToolContext,
    required: list[CompletionRequirement],
    attempts: AttemptBoundary,
    *,
    conversation_revision: int,
    limits: ReactLimits | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    publication: FlexiblePublicationInput | None = None,
    task_catalog: TaskCatalog | None = None,
    clarifier: ClarificationExtractor | None = None,
    unit_clarifier: UnitClarificationExtractor | None = None,
    task_clarifier: TaskClarificationExtractor | None = None,
    amender: PendingAmender | None = None,
    task_amender: TaskAmender | None = None,
    catalog_loader: RequirementCatalog | None = None,
) -> CompiledStateGraph:
    trusted = ToolContext.model_validate(context.model_dump())
    if publication and publication.read_only_turn:
        trusted.read_only_turn = True
    tasks = [CompletionRequirement.model_validate(t.model_dump()) for t in required]
    registering = task_catalog is not None
    if (
        publication
        and publication.accepted.flexible
        and publication.accepted.flexible.status == "WAITING_INPUT"
        and publication.accepted.flexible.pending_task_resolution is None
        and not publication.discard_pending
        and not publication.read_only_turn
    ):
        tasks = [
            row.task.model_copy(deep=True)
            for row in publication.accepted.flexible.task_outcomes
        ]
        registering = False
    if publication and publication.task_reply and not registering:
        raise ValueError("Task selection requires canonical task re-resolution")
    if registering and (tasks or publication is None):
        raise ValueError("Dynamic registration requires stored source, not fixed tasks")
    if (
        not tasks and not registering and publication is None
    ) or conversation_revision < 0:
        raise ValueError("Trusted tasks and publication revision are required")

    resolved_limits = limits or ReactLimits()
    validated_publication = (
        FlexiblePublicationInput.model_validate(publication.model_dump())
        if publication
        else None
    )
    bootstrap_tasks = tasks
    if registering and validated_publication and validated_publication.accepted.flexible:
        bootstrap_tasks = [
            row.task for row in validated_publication.accepted.flexible.task_outcomes
        ]
    turn_input = (
        validated_publication.turn_input(trusted, bootstrap_tasks)
        if validated_publication
        else {}
    )

    runtime = FlexibleRuntime(
        trusted=trusted,
        tasks=tasks,
        registering=registering,
        limits=resolved_limits,
        publication=validated_publication,
        conversation_revision=conversation_revision,
        turn_input=turn_input,
        catalog_loader=catalog_loader,
        clarifier=clarifier,
        unit_clarifier=unit_clarifier,
    )
    clarification = build_clarification_nodes(
        runtime,
        task_catalog=task_catalog,
        task_clarifier=task_clarifier,
        amender=amender,
        task_amender=task_amender,
    )
    loop = build_loop_nodes(
        runtime,
        model=model,
        dispatcher=dispatcher,
        attempts=attempts,
    )

    def begin(state: FlexibleState) -> dict[str, Any]:
        return runtime.initialize(state, runtime.turn_input)

    builder = StateGraph(FlexibleState)
    builder.add_node("begin_turn", observed_sync("begin_turn", begin))
    builder.add_node("agent", observed_async("agent", loop.agent))
    builder.add_node("validate_action", observed_async("validate_action", loop.validate))
    builder.add_node("execute_tool", observed_async("execute_tool", loop.execute))
    builder.add_node("resolve_proposal", observed_async("resolve_proposal", loop.resolve))
    builder.add_node("prepare_agent", observed_async("prepare_agent", loop.prepare_agent))
    builder.add_conditional_edges(
        "prepare_agent",
        lambda s: "admitted" if s.get("phase") == "DECIDED" else "ready",
        {"admitted": "validate_action", "ready": "agent"},
    )
    builder.add_node(
        "interpret_reply", observed_async("interpret_reply", clarification.interpret_reply)
    )
    builder.add_node(
        "amend_pending", observed_async("amend_pending", clarification.amend_reply)
    )

    terminal = END
    if validated_publication:

        def prepare(state: FlexibleState) -> dict[str, Any]:
            assert validated_publication is not None
            snapshot = validated_publication.prepare(
                state,
                runtime.base_context(state),
                runtime.registered_tasks(state),
                dispatcher.service,
                conversation_revision,
            )
            return runtime.bounded(
                state, {"accepted_snapshot": snapshot.model_dump(mode="json")}
            )

        terminal = "prepare_publish"
        builder.add_node(terminal, observed_sync(terminal, prepare))
        builder.add_edge(terminal, END)

    builder.add_edge(START, "begin_turn")
    start_agent = "prepare_agent"
    if registering:
        registration_node = "register_request"
        builder.add_node(
            registration_node, observed_async(registration_node, clarification.register)
        )
        builder.add_node(
            "resolve_registered_tasks",
            observed_async("resolve_registered_tasks", clarification.resolve_registered),
        )
        if task_clarifier:
            builder.add_node(
                "interpret_task_reply",
                observed_async(
                    "interpret_task_reply", clarification.interpret_task_reply
                ),
            )
            builder.add_conditional_edges(
                registration_node,
                lambda s: "new" if s.get("decision") else "pending",
                {"new": "resolve_registered_tasks", "pending": "interpret_task_reply"},
            )
            if task_amender:
                builder.add_node(
                    "amend_task",
                    observed_async("amend_task", clarification.amend_task_reply),
                )
                builder.add_conditional_edges(
                    "interpret_task_reply",
                    lambda s: (
                        "AMEND_TASK" if s.get("phase") == "AMEND_TASK" else "STARTED"
                    ),
                    {"AMEND_TASK": "amend_task", "STARTED": "resolve_registered_tasks"},
                )
                builder.add_edge("amend_task", "resolve_registered_tasks")
            else:
                builder.add_edge("interpret_task_reply", "resolve_registered_tasks")
        else:
            builder.add_edge(registration_node, "resolve_registered_tasks")
        builder.add_edge("resolve_registered_tasks", "prepare_agent")
        if (
            validated_publication
            and validated_publication.accepted.flexible
            and validated_publication.accepted.flexible.pending_task_resolution
        ):
            start_agent = registration_node

    builder.add_conditional_edges(
        "begin_turn",
        lambda s: s["phase"],
        {
            "STARTED": start_agent,
            "WAITING_INPUT": terminal,
            "INTERPRET_REPLY": "interpret_reply",
            "READY_TO_PUBLISH": terminal,
        },
    )
    builder.add_conditional_edges(
        "interpret_reply",
        lambda s: s["phase"],
        {
            "WAITING_INPUT": terminal,
            "STARTED": start_agent,
            "AMEND_PENDING": "amend_pending",
        },
    )
    builder.add_conditional_edges(
        "amend_pending",
        lambda s: s["phase"],
        {"WAITING_INPUT": terminal, "STARTED": start_agent},
    )
    builder.add_conditional_edges(
        "agent",
        lambda s: s["phase"],
        {
            "DECIDED": "validate_action",
            **({"REGISTER_REQUEST": "register_request"} if registering else {}),
        },
    )
    builder.add_conditional_edges(
        "validate_action",
        lambda s: s["phase"],
        {
            "REPAIR": "agent",
            "TOOL_PENDING": "execute_tool",
            "RESOLVE_PROPOSAL": "resolve_proposal",
            "WAITING_INPUT": terminal,
            "READY_TO_PUBLISH": terminal,
        },
    )
    builder.add_edge("execute_tool", "agent")
    builder.add_conditional_edges(
        "resolve_proposal",
        lambda s: s["phase"],
        {
            "REPAIR": "agent",
            "WAITING_INPUT": terminal,
            "REQUIREMENTS_UNCHANGED": "prepare_agent",
            "REQUIREMENTS_ACCEPTED": "prepare_agent",
        },
    )
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "FLEXIBLE_GRAPH_VERSION",
    "FlexibleModel",
    "FlexibleState",
    "build_flexible_graph",
]
