"""Internal P3 build graph. P4 exclusively owns execution and publication.

Opt-in guided/ReAct execution uses this graph; SSE subscribers do not own it.
A checkpointer can resume node work;
it does not grant worker ownership, call budgets or accepted-head authority.
"""

import json
from dataclasses import dataclass
from typing import Any, TypedDict, cast
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.application.cpu_work import run_cpu
from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    PublicBuildV1,
    PublicClarificationV1,
)
from ai_service.capabilities.assistant.react.contracts import (
    BuildResult,
    CompletionRequirement,
    ToolContext,
    ValidatedAnswer,
)
from ai_service.capabilities.assistant.react.public_views import (
    public_resolution_options,
)
from ai_service.capabilities.assistant.react.task_amendment import (
    TaskAmender,
    amend_pending_task,
)
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    TaskCatalog,
    TaskClarificationExtractor,
    TaskResolution,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
    task_choice_reply,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.assistant.requirements.amendment import (
    PendingAmender,
    amend_pending,
)
from ai_service.capabilities.assistant.requirements.build import (
    BuildExplainer,
    BuildReceipt,
    ClarificationExtractor,
    RequirementCatalog,
    RequirementExtractor,
    grounded_summary,
    optimize_requirements,
    verify_receipt,
)
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.merge import merge_requirements
from ai_service.capabilities.assistant.requirements.pending_action import (
    DISCARDED_ANSWER,
    DISCARDED_OUTCOME,
    validate_pending_action,
)
from ai_service.capabilities.assistant.requirements.publication import public_projection
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitClarificationExtractor,
    confirmations_for_pending,
    unit_choice_reply,
)
from ai_service.capabilities.assistant.requirements.resolver import (
    Draft,
    PendingDraft,
    Resolution,
    choice_reply,
    resolve_draft,
    selection_for_pending,
)
from ai_service.capabilities.assistant.requirements.verifier import SatisfactionReport
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    PendingActionV1,
    TaskClarificationReplyV1,
    UnitClarificationReplyV1,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.react import (
    ReactModel,
    ReactState,
    ToolRecord,
    add_react_nodes,
)


class BuildGraphState(TypedDict, total=False):
    message: str
    message_id: str
    requirement_state: dict[str, Any]
    draft: dict[str, Any]
    resolution: dict[str, Any]
    catalog: list[dict[str, Any]]
    receipt: dict[str, Any]
    satisfaction: dict[str, Any] | None
    outcome: str
    answer: str
    consultation: dict[str, Any]
    public_build: dict[str, Any] | None
    phase: str
    pending_clarification: dict[str, Any] | None
    clarification_reply: dict[str, Any] | None
    unit_clarification_reply: dict[str, Any] | None
    public_clarification: dict[str, Any] | None
    choice_draft: dict[str, Any] | None
    unit_choice_draft: dict[str, Any] | None
    pending_amendment: dict[str, Any] | None
    ledger: list[dict[str, Any]]
    decision: dict[str, Any]
    model_calls: int
    repairs: int
    repair: bool
    react_answer: dict[str, Any]
    task_resolution: dict[str, Any]
    task_source_message: str
    task_source_message_id: str
    pending_tasks: dict[str, Any] | None
    task_clarification_reply: dict[str, Any] | None
    public_task_clarification: dict[str, Any] | None
    task_choice_draft: dict[str, Any] | None
    task_amendment: dict[str, Any] | None
    catalog_loaded: bool
    base_requirements_hash: str
    saved_build_evidence: dict[str, Any] | None
    pending_action: dict[str, Any] | None
    pending_action_receipt: dict[str, Any] | None


@dataclass(frozen=True)
class ReactComposition:
    model: ReactModel
    dispatcher: ToolDispatcher
    context: ToolContext
    catalog: TaskCatalog
    attempts: AttemptBoundary
    task_clarifier: TaskClarificationExtractor | None = None
    task_amender: TaskAmender | None = None


def build_requirement_graph(
    extractor: RequirementExtractor,
    catalog: RequirementCatalog,
    *,
    service: PCBuildApplicationService | None = None,
    explainer: BuildExplainer | None = None,
    clarifier: ClarificationExtractor | None = None,
    unit_clarifier: UnitClarificationExtractor | None = None,
    amender: PendingAmender | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    max_state_bytes: int = 512_000,
    max_catalog_rows: int = 100,
    react: ReactComposition | None = None,
) -> CompiledStateGraph:
    if max_state_bytes <= 0 or max_catalog_rows <= 0:
        raise ValueError("Graph bounds must be positive")
    optimizer = service or PCBuildApplicationService()

    def requirements(state: BuildGraphState) -> CanonicalRequirementState:
        return CanonicalRequirementState.model_validate(state["requirement_state"])

    def rows(state: BuildGraphState) -> list[CatalogRowV1]:
        return [CatalogRowV1.model_validate(row) for row in state["catalog"]]

    def bounded(state: BuildGraphState, update: dict[str, Any]) -> dict[str, Any]:
        if (
            len(json.dumps(dict(state) | update, ensure_ascii=False).encode())
            > max_state_bytes
        ):
            raise ValueError("Graph state exceeds configured bound")
        return update

    async def extract(state: BuildGraphState) -> dict[str, Any]:
        # Reset current-turn channels, never preserve a prior success answer.
        if not state["message"].strip():
            raise ValueError("Message is required")
        source_id = UUID(state["message_id"])
        source_message = state["message"]
        bounded(state, {})
        pending = (
            PendingDraft.model_validate(state["pending_clarification"])
            if state.get("pending_clarification")
            else None
        )
        pending_tasks = (
            PendingTaskDraft.model_validate(state["pending_tasks"])
            if state.get("pending_tasks")
            else None
        )
        action_receipt = None
        if state.get("pending_action"):
            if any(
                state.get(field)
                for field in (
                    "clarification_reply",
                    "task_clarification_reply",
                    "unit_clarification_reply",
                )
            ):
                raise ValueError(
                    "Cannot combine pending action with clarification reply"
                )
            action_receipt = validate_pending_action(
                PendingActionV1.model_validate(state["pending_action"]),
                requirements(state),
                pending,
                pending_tasks,
            )
            pending, pending_tasks = None, None
        if pending_tasks is not None and (react is None or pending is not None):
            raise ValueError("Incompatible pending task execution")
        if state.get("task_clarification_reply") and pending_tasks is None:
            raise ValueError("No pending task clarification")
        if state.get("clarification_reply") and pending is None:
            raise ValueError("No pending clarification")
        if state.get("unit_clarification_reply") and pending is None:
            raise ValueError("No pending unit clarification")
        discarded = (
            action_receipt is not None and action_receipt.action.action == "DISCARD"
        )
        if discarded:
            draft = Draft(items=[])
        elif pending is not None:
            if pending.constraints_revision != requirements(state).revision:
                raise ValueError("Stale clarification revision")
            draft = pending.resolution.draft
            source_id, source_message = pending.message_id, pending.message
        elif pending_tasks is not None:
            if pending_tasks.constraints_revision != requirements(state).revision:
                raise ValueError("Stale task clarification revision")
            # Requirements were committed before task resolution. Do not reapply
            # the original patch or let the choice become an unrelated ANSWER task.
            draft = Draft(items=[], tasks=pending_tasks.tasks)
            source_id, source_message = pending_tasks.message_id, pending_tasks.message
        else:
            result = await extractor.extract(state["message"], requirements(state))
            draft = Draft.model_validate(result.model_dump())
        return bounded(
            state,
            {
                "draft": draft.model_dump(mode="json"),
                "receipt": {},
                "catalog": [],
                "resolution": {},
                "satisfaction": None,
                "outcome": DISCARDED_OUTCOME if discarded else "",
                "answer": DISCARDED_ANSWER if discarded else "",
                "phase": "EXTRACTING",
                "choice_draft": None,
                "unit_choice_draft": None,
                "pending_amendment": None,
                "task_choice_draft": None,
                "task_amendment": None,
                "pending_action": None,
                "pending_action_receipt": action_receipt.model_dump(mode="json")
                if action_receipt
                else None,
                **(
                    {
                        "pending_clarification": None,
                        "pending_tasks": None,
                        "public_clarification": None,
                        "public_task_clarification": None,
                    }
                    if action_receipt
                    else {}
                ),
                "ledger": [],
                "decision": {},
                "model_calls": 0,
                "repairs": 0,
                "repair": False,
                "react_answer": {},
                "task_resolution": {},
                "catalog_loaded": False,
                "task_source_message": source_message,
                "task_source_message_id": str(source_id),
                "base_requirements_hash": canonical_hash(requirements(state)),
                "saved_build_evidence": react.context.saved_build.model_dump(
                    mode="json"
                )
                if react and react.context.saved_build
                else None,
            },
        )

    async def choose(state: BuildGraphState) -> dict[str, Any]:
        if (
            not state.get("pending_clarification")
            or state.get("clarification_reply")
            or clarifier is None
        ):
            return {}
        pending = PendingDraft.model_validate(state["pending_clarification"])
        offered = PublicClarificationV1.model_validate(state["public_clarification"])
        if not any(item.options for item in offered.items):
            return {}
        if (
            offered.pending_message_id != pending.message_id
            or offered.constraints_revision != pending.constraints_revision
        ):
            raise ValueError("Invalid clarification options/source")
        choice = await clarifier.choose(state["message"], offered)
        reply = choice_reply(choice, state["message"], pending)
        if reply is not None:
            selection_for_pending(pending, reply, requirements(state))
        return bounded(
            state,
            {
                "choice_draft": choice.model_dump(mode="json"),
                "clarification_reply": reply.model_dump(mode="json") if reply else None,
            },
        )

    async def choose_unit(state: BuildGraphState) -> dict[str, Any]:
        if (
            unit_clarifier is None
            or not state.get("pending_clarification")
            or state.get("unit_clarification_reply")
        ):
            return {}
        pending = PendingDraft.model_validate(state["pending_clarification"])
        offered = PublicClarificationV1.model_validate(state["public_clarification"])
        if (
            offered.pending_message_id != pending.message_id
            or offered.constraints_revision != pending.constraints_revision
        ):
            raise ValueError("Invalid unit clarification options/source")
        if not any(item.reason == "QUANTITY_UNIT_REQUIRED" for item in offered.items):
            return {}
        choice = await unit_clarifier.choose_unit(state["message"], offered)
        reply = unit_choice_reply(choice, pending)
        if reply is not None:
            confirmations_for_pending(
                pending,
                reply,
                requirements(state),
                message_id=UUID(state["message_id"]),
                message=state["message"],
            )
        return bounded(
            state,
            {
                "unit_choice_draft": choice.model_dump(mode="json"),
                "unit_clarification_reply": reply.model_dump(mode="json")
                if reply
                else None,
            },
        )

    async def choose_task(state: BuildGraphState) -> dict[str, Any]:
        if (
            react is None
            or react.task_clarifier is None
            or not state.get("pending_tasks")
            or state.get("task_clarification_reply")
        ):
            return {}
        pending = PendingTaskDraft.model_validate(state["pending_tasks"])
        offered = public_task_options(pending)
        if offered.model_dump(mode="json") != state.get("public_task_clarification"):
            raise ValueError("Invalid task clarification options/source")
        if not any(
            item.options and item.subject_index is not None for item in offered.items
        ):
            return {}
        choice = await react.task_clarifier.choose_task(state["message"], offered)
        reply = task_choice_reply(choice, state["message"], pending)
        if reply is not None:
            selections_for_pending_task(pending, reply, requirements(state))
        return bounded(
            state,
            {
                "task_choice_draft": choice.model_dump(mode="json"),
                "task_clarification_reply": reply.model_dump(mode="json")
                if reply
                else None,
            },
        )

    async def amend(state: BuildGraphState) -> dict[str, Any]:
        if (
            amender is None
            or not state.get("pending_clarification")
            or state.get("clarification_reply")
            or state.get("unit_clarification_reply")
        ):
            return {}
        pending = PendingDraft.model_validate(state["pending_clarification"])
        amendment = await amender.amend(state["message"], pending)
        modified = amend_pending(
            pending,
            amendment,
            requirements(state),
            message_id=UUID(state["message_id"]),
            message=state["message"],
        )
        return bounded(
            state,
            {
                "pending_amendment": amendment.model_dump(mode="json"),
                "pending_clarification": modified.model_dump(mode="json"),
                "draft": modified.resolution.draft.model_dump(mode="json"),
            },
        )

    async def amend_task(state: BuildGraphState) -> dict[str, Any]:
        if (
            react is None
            or react.task_amender is None
            or not state.get("pending_tasks")
            or state.get("task_clarification_reply")
        ):
            return {}
        pending = PendingTaskDraft.model_validate(state["pending_tasks"])
        if not any(task.kind == "COMPARE" for task in pending.tasks):
            return {}
        amendment = await react.task_amender.amend_task(state["message"], pending)
        changed = amend_pending_task(
            pending,
            amendment,
            requirements(state),
            message_id=UUID(state["message_id"]),
            message=state["message"],
        )
        return bounded(
            state,
            {
                "task_amendment": amendment.model_dump(mode="json"),
                "pending_tasks": changed.model_dump(mode="json"),
            },
        )

    async def resolve(state: BuildGraphState) -> dict[str, Any]:
        draft = Draft.model_validate(state["draft"])
        load_catalog = (
            react is None
            or bool(draft.items)
            or any(task.kind == "BUILD" for task in draft.tasks)
        )
        canonical = (
            await catalog.load(draft, requirements(state)) if load_catalog else []
        )
        if len(canonical) > max_catalog_rows:
            raise ValueError("Canonical candidate set exceeds configured bound")
        pending = (
            PendingDraft.model_validate(state["pending_clarification"])
            if state.get("pending_clarification")
            else None
        )
        selections = pending.selections if pending else {}
        confirmations = pending.quantity_confirmations if pending else {}
        if state.get("unit_clarification_reply"):
            assert pending is not None
            confirmations = confirmations_for_pending(
                pending,
                UnitClarificationReplyV1.model_validate(
                    state["unit_clarification_reply"]
                ),
                requirements(state),
                message_id=UUID(state["message_id"]),
                message=state["message"],
            )
        if state.get("clarification_reply"):
            assert pending is not None
            selections = selection_for_pending(
                pending,
                ClarificationReplyV1.model_validate(state["clarification_reply"]),
                requirements(state),
            )
        result = resolve_draft(
            draft,
            message_id=UUID(state["task_source_message_id"]),
            message=state["task_source_message"],
            state=requirements(state),
            catalog=canonical,
            selections=selections,
            quantity_confirmations=confirmations,
            item_sources=pending.item_sources if pending else {},
            discarded_items=pending.discarded_items if pending else [],
        )
        artifact = PendingDraft(
            message_id=UUID(state["task_source_message_id"]),
            message=state["task_source_message"],
            constraints_revision=requirements(state).revision,
            resolution=result,
            selections=selections,
            quantity_confirmations=confirmations,
            item_sources=pending.item_sources if pending else {},
            discarded_items=pending.discarded_items if pending else [],
        )
        return bounded(
            state,
            {
                "resolution": result.model_dump(mode="json"),
                "catalog": [row.model_dump(mode="json") for row in canonical],
                "pending_clarification": artifact.model_dump(mode="json"),
                "clarification_reply": None,
                "unit_clarification_reply": None,
                "catalog_loaded": load_catalog,
            },
        )

    async def merge(state: BuildGraphState) -> dict[str, Any]:
        resolution = Resolution.model_validate(state["resolution"])
        if resolution.patch is None:
            by_id = {row.variant_id: row for row in rows(state)}
            lines = []
            for item in resolution.pending:
                lines.append(f"Yêu cầu {item.item_index}: {item.reason}")
                for identifier in item.candidate_ids:
                    row = by_id[identifier]
                    lines.append(f"- {row.sku}: {row.component.name}")
            return bounded(
                state,
                {
                    "outcome": "CLARIFICATION",
                    "answer": "\n".join(lines),
                },
            )
        pending = PendingDraft.model_validate(state["pending_clarification"])
        merged = merge_requirements(
            requirements(state),
            resolution.patch,
            pending.source_messages(),
        )
        return bounded(
            state,
            {
                "requirement_state": merged.model_dump(mode="json"),
                "pending_clarification": None,
            },
        )

    async def optimize(state: BuildGraphState) -> dict[str, Any]:
        current, catalog_rows = requirements(state), rows(state)
        receipt = await run_cpu(
            lambda: optimize_requirements(current, catalog_rows, optimizer)
        )
        return bounded(
            state,
            {"receipt": receipt.model_dump(mode="json"), "outcome": receipt.outcome},
        )

    async def verify(state: BuildGraphState) -> dict[str, Any]:
        receipt = BuildReceipt.model_validate(state["receipt"])
        report = verify_receipt(requirements(state), receipt, rows(state))
        return bounded(
            state, {"satisfaction": report.model_dump(mode="json") if report else None}
        )

    async def explain(state: BuildGraphState) -> dict[str, Any]:
        receipt = BuildReceipt.model_validate(state["receipt"])
        report = (
            SatisfactionReport.model_validate(state["satisfaction"])
            if state["satisfaction"]
            else None
        )
        if explainer is not None and receipt.outcome == "BUILT":
            # Prose can fail/retry separately, but is not promoted to verified facts.
            await explainer.explain(receipt)
        return bounded(state, {"answer": grounded_summary(receipt, report)})

    async def resolve_completion(state: BuildGraphState) -> dict[str, Any]:
        assert react is not None
        pending = (
            PendingTaskDraft.model_validate(state["pending_tasks"])
            if state.get("pending_tasks")
            else None
        )
        selected = pending.resolution.bindings if pending else []
        if state.get("task_clarification_reply"):
            if pending is None:
                raise ValueError("No pending task clarification")
            selected = selections_for_pending_task(
                pending,
                TaskClarificationReplyV1.model_validate(
                    state["task_clarification_reply"]
                ),
                requirements(state),
            )
        result = await resolve_tasks(
            Draft.model_validate(state["draft"]).tasks,
            state["task_source_message"],
            requirements(state),
            react.catalog,
            saved_build_ref=react.context.saved_build.evidence_ref
            if react.context.saved_build
            else None,
            selections=selected,
            mention_overrides=pending.mention_overrides if pending else [],
        )
        artifact = (
            PendingTaskDraft(
                message_id=UUID(state["task_source_message_id"]),
                message=state["task_source_message"],
                constraints_revision=requirements(state).revision,
                tasks=Draft.model_validate(state["draft"]).tasks,
                resolution=result,
                mention_overrides=pending.mention_overrides if pending else [],
            )
            if result.pending
            else None
        )
        update: dict[str, Any] = {
            "task_resolution": result.model_dump(mode="json"),
            "pending_tasks": artifact.model_dump(mode="json") if artifact else None,
            "task_clarification_reply": None,
        }
        if result.pending:
            lines = []
            for item in result.pending:
                lines.append(f"Yêu cầu {item.task_index}: {item.reason}")
                lines.extend(f"- {p.name}" for p in item.candidates)
            update.update(outcome="CLARIFICATION", answer="\n".join(lines))
        elif (
            any(task.kind == "BUILD" for task in result.requirements)
            and not compile_requirements(requirements(state), rows(state)).ready
        ):
            receipt = await run_cpu(
                lambda: optimize_requirements(
                    requirements(state), rows(state), optimizer
                )
            )
            update.update(
                receipt=receipt.model_dump(mode="json"),
                outcome=receipt.outcome,
                answer=grounded_summary(receipt, None),
            )
        return bounded(state, update)

    def context_for_react(state: ReactState) -> ToolContext:
        assert react is not None
        current = cast(BuildGraphState, state)
        return ToolContext.model_validate(
            react.context.model_dump()
            | {
                "requirements": requirements(current),
                "catalog": rows(current),
                "catalog_ready": current["catalog_loaded"],
                "task_message": current["task_source_message"],
                "completion_requirements": TaskResolution.model_validate(
                    current["task_resolution"]
                ).requirements,
            }
        )

    def completion_for_react(state: ReactState) -> list[CompletionRequirement]:
        return TaskResolution.model_validate(
            cast(BuildGraphState, state)["task_resolution"]
        ).requirements

    async def accept_react(state: BuildGraphState) -> dict[str, Any]:
        answer = ValidatedAnswer.model_validate(state["react_answer"])
        update: dict[str, Any] = {"answer": answer.text, "outcome": "ANSWERED"}
        for raw in state["ledger"]:
            record = ToolRecord.model_validate(raw)
            payload = record.result.result if record.result else None
            if (
                isinstance(payload, BuildResult)
                and payload.build_ref == answer.build_ref
            ):
                update.update(
                    receipt=payload.receipt.model_dump(mode="json"),
                    outcome=payload.receipt.outcome,
                    satisfaction=payload.satisfaction.model_dump(mode="json")
                    if payload.satisfaction
                    else None,
                )
        return bounded(state, update)

    async def prepare_publish(state: BuildGraphState) -> dict[str, Any]:
        receipt = (
            BuildReceipt.model_validate(state["receipt"])
            if state.get("receipt")
            else None
        )
        previous = (
            PublicBuildV1.model_validate(state["public_build"])
            if state.get("public_build")
            else None
        )
        consultation, build = public_projection(requirements(state), receipt, previous)
        if receipt is None and state["base_requirements_hash"] == canonical_hash(
            requirements(state)
        ):
            if "consultation" in state:
                consultation = ConsultationStateV1.model_validate(
                    state.get("consultation", {})
                )
            build = previous
        clarification = None
        task_clarification = (
            public_task_options(PendingTaskDraft.model_validate(state["pending_tasks"]))
            if state.get("pending_tasks")
            else None
        )
        if state.get("pending_clarification"):
            pending = PendingDraft.model_validate(state["pending_clarification"])
            clarification = public_resolution_options(pending, rows(state))
        return bounded(
            state,
            {
                "consultation": consultation.model_dump(mode="json"),
                "public_build": build.model_dump(mode="json") if build else None,
                "phase": "READY_TO_PUBLISH",
                "public_clarification": clarification.model_dump(mode="json")
                if clarification
                else None,
                "public_task_clarification": task_clarification.model_dump(mode="json")
                if task_clarification
                else None,
            },
        )

    builder = StateGraph(BuildGraphState)
    for name, node in (
        ("extract", extract),
        ("choose", choose),
        ("choose_unit", choose_unit),
        ("choose_task", choose_task),
        ("amend", amend),
        ("amend_task", amend_task),
        ("resolve", resolve),
        ("merge", merge),
        ("optimize", optimize),
        ("verify", verify),
        ("explain", explain),
        ("prepare_publish", prepare_publish),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "extract")
    builder.add_conditional_edges(
        "extract",
        lambda state: (
            "discard" if state.get("outcome") == DISCARDED_OUTCOME else "continue"
        ),
        {"discard": "prepare_publish", "continue": "choose"},
    )
    builder.add_edge("choose", "choose_unit")
    builder.add_edge("choose_unit", "choose_task")
    builder.add_edge("choose_task", "amend")
    builder.add_edge("amend", "amend_task")
    builder.add_edge("amend_task", "resolve")
    builder.add_edge("resolve", "merge")

    async def route_after_merge(state: BuildGraphState) -> str:
        return "clarify" if state.get("outcome") == "CLARIFICATION" else "build"

    builder.add_conditional_edges(
        "merge",
        route_after_merge,
        {
            "clarify": "prepare_publish",
            "build": "resolve_completion" if react else "optimize",
        },
    )
    if react is not None:
        builder.add_node("resolve_completion", resolve_completion)
        builder.add_node("accept_react", accept_react)
        add_react_nodes(
            builder,
            react.model,
            react.dispatcher,
            context_for_react,
            completion_for_react,
            react.attempts,
            done_node="accept_react",
            answer_channel="react_answer",
        )
        builder.add_conditional_edges(
            "resolve_completion",
            lambda state: "stop" if state.get("outcome") else "agent",
            {"stop": "prepare_publish", "agent": "agent"},
        )
        builder.add_edge("accept_react", "prepare_publish")
    builder.add_edge("optimize", "verify")
    builder.add_edge("verify", "explain")
    builder.add_edge("explain", "prepare_publish")
    builder.add_edge("prepare_publish", END)
    return builder.compile(checkpointer=checkpointer)
