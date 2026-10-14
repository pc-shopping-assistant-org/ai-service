"""Internal V2 loop; publication and execution ownership remain application-owned.

Selected by the opt-in V2 runtime. Inputs are trusted accepted/checkpoint state,
never raw request dictionaries.
"""

import json
from typing import Any, Protocol, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import JsonValue, ValidationError

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    ActionKind,
    AskClarificationAction,
    CallToolAction,
    FinalAnswerAction,
    FlexibleDecision,
    ProposeRequirementsAction,
    RepairFeedback,
)
from ai_service.capabilities.assistant.react.contracts import (
    TOOL_SCHEMAS,
    CompletionRequirement,
    ToolContext,
    ToolName,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.outcomes import (
    InvalidTaskAnswer,
    validate_final_answer,
)
from ai_service.capabilities.assistant.react.proposal_resolution import (
    resolve_requirement_proposal,
)
from ai_service.capabilities.assistant.react.public_views import (
    public_resolution_options,
)
from ai_service.capabilities.assistant.react.resolution_reply import (
    FreeAmendmentV2,
    FreeResolutionReplyV2,
    FreeTaskAmendmentV2,
    FreeTaskReplyV2,
    apply_amendment_receipt,
    apply_resolution_command,
    apply_task_amendment_receipt,
    derive_resolution_reply,
    derive_task_reply,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexibleExecutionContext,
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.task_amendment import TaskAmender
from ai_service.capabilities.assistant.react.task_resolution import (
    TaskCatalog,
    TaskClarificationExtractor,
    TaskResolution,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    call_hash,
    project_current_tools,
)
from ai_service.capabilities.assistant.requirements.admission import (
    confirm_agent_patch,
)
from ai_service.capabilities.assistant.requirements.amendment import PendingAmender
from ai_service.capabilities.assistant.requirements.build import (
    ClarificationExtractor,
    RequirementCatalog,
    RequirementExtractor,
)
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
    ResolutionContinuation,
)
from ai_service.capabilities.assistant.requirements.proposals import (
    PendingRequirementProposal,
    ProposalConfirmation,
)
from ai_service.capabilities.assistant.requirements.quantity_clarification import (
    UnitClarificationExtractor,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.requirements.tasks import (
    TaskDraftBatch,
    TaskExtractor,
)
from ai_service.capabilities.assistant.requirements.user_admission import (
    UserRequirementReceipt,
    apply_user_receipt,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    UnitClarificationReplyV1,
)

from .react import ReactLimits

FLEXIBLE_GRAPH_VERSION = "native-flexible-react-v2-managed"


class FlexibleModel(Protocol):
    async def decide(
        self,
        message: str,
        transcript: list[ToolRecord],
        allowed: tuple[ToolName, ...],
        repair: RepairFeedback | None,
        context: ToolContext | None = None,
    ) -> FlexibleDecision | None: ...


class FlexibleState(TypedDict, total=False):
    catalog_context: dict[str, Any] | None
    message: str
    working_requirements: dict[str, Any]
    pending_proposal: dict[str, Any] | None
    pending_draft: dict[str, Any] | None
    confirmation: dict[str, Any] | None
    reference_reply: dict[str, Any] | None
    unit_reply: dict[str, Any] | None
    ledger: list[dict[str, Any]]
    decision: dict[str, Any] | None
    model_calls: int
    tool_calls: int
    repairs: int
    repair_feedback: dict[str, Any] | None
    phase: str
    answer: dict[str, Any]
    clarification: dict[str, Any] | None
    accepted_snapshot: dict[str, Any] | None
    task_draft: dict[str, Any] | None
    task_resolution: dict[str, Any] | None
    publication_context: dict[str, Any]
    free_resolution_reply: dict[str, Any] | None
    free_task_reply: dict[str, Any] | None
    free_amendment: dict[str, Any] | None
    free_task_amendment: dict[str, Any] | None
    user_receipts: list[dict[str, Any]]


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
    task_extractor: TaskExtractor | None = None,
    task_catalog: TaskCatalog | None = None,
    clarifier: ClarificationExtractor | None = None,
    unit_clarifier: UnitClarificationExtractor | None = None,
    task_clarifier: TaskClarificationExtractor | None = None,
    amender: PendingAmender | None = None,
    task_amender: TaskAmender | None = None,
    user_extractor: RequirementExtractor | None = None,
    catalog_loader: RequirementCatalog | None = None,
) -> CompiledStateGraph:
    trusted = ToolContext.model_validate(context.model_dump())
    tasks = [CompletionRequirement.model_validate(t.model_dump()) for t in required]
    registering = task_extractor is not None
    if user_extractor is not None and publication is None:
        raise ValueError("Explicit-user admission requires a stored request binding")
    if publication and publication.task_reply and not registering:
        raise ValueError("Task selection requires canonical task re-resolution")
    if (task_extractor is None) != (task_catalog is None):
        raise ValueError("Task registration requires extractor and canonical catalog")
    if registering and (tasks or publication is None):
        raise ValueError("Dynamic registration requires stored source, not fixed tasks")
    if (
        not tasks
        and not registering
        and not (publication and publication.discard_pending)
    ) or conversation_revision < 0:
        raise ValueError("Trusted tasks and publication revision are required")
    limits = limits or ReactLimits()
    publication = (
        FlexiblePublicationInput.model_validate(publication.model_dump())
        if publication
        else None
    )
    bootstrap_tasks = tasks
    if registering and publication and publication.accepted.flexible:
        bootstrap_tasks = [
            row.task for row in publication.accepted.flexible.task_outcomes
        ]
    turn_input = publication.turn_input(trusted, bootstrap_tasks) if publication else {}

    def registered_tasks(state: FlexibleState) -> list[CompletionRequirement]:
        if registering:
            return (
                TaskResolution.model_validate(state["task_resolution"]).requirements
                if state.get("task_resolution")
                else []
            )
        return tasks

    def unresolved_tasks(state: FlexibleState) -> list[str]:
        if registering and state.get("task_resolution"):
            return sorted(
                {
                    item.reason
                    for item in TaskResolution.model_validate(
                        state["task_resolution"]
                    ).pending
                }
            )
        return []

    def bounded(state: FlexibleState, update: dict[str, Any]) -> dict[str, Any]:
        if (
            len(
                json.dumps(
                    dict(state) | update, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            > limits.max_state_bytes
        ):
            raise ValueError("REACT_STATE_TOO_LARGE")
        return update

    def base_context(state: FlexibleState) -> ToolContext:
        value = trusted.model_copy(deep=True)
        catalog_context = state.get("catalog_context")
        if catalog_context is not None:
            value.catalog = [
                CatalogRowV1.model_validate(row) for row in catalog_context["rows"]
            ]
            value.catalog_ready = True
        return value

    def current(state: FlexibleState) -> ToolContext:
        value = base_context(state)
        value.requirements = CanonicalRequirementState.model_validate(
            state.get("working_requirements", trusted.requirements.model_dump())
        )
        # Trusted BUILD task remains BUILD; only its legacy adapter revision follows
        # application-approved working content. No model action can replace tasks.
        value.completion_requirements = [
            task.model_copy(
                update={"constraints_revision": value.requirements.revision}
            )
            if task.kind == "BUILD"
            else task.model_copy(deep=True)
            for task in registered_tasks(state)
        ]
        if publication:
            value.task_message = publication.task_message
        if state.get("task_draft"):
            value.requested_tasks = TaskDraftBatch.model_validate(state["task_draft"])
        return ToolContext.model_validate(value.model_dump())

    def records(state: FlexibleState) -> list[ToolRecord]:
        entries = [ToolRecord.model_validate(row) for row in state.get("ledger", [])]
        if len({(entry.origin_run_id, entry.call.call_id) for entry in entries}) != len(
            entries
        ):
            raise ValueError("REACT_DUPLICATE_CALL_ID")
        return entries

    def projection(state: FlexibleState) -> list[ToolRecord]:
        return project_current_tools(records(state), current(state))

    def repair(
        state: FlexibleState,
        code: Any,
        missing: list[str] | None = None,
        *,
        task_indices: list[int] | None = None,
    ) -> dict[str, Any]:
        count = state.get("repairs", 0) + 1
        if count > limits.max_repairs:
            raise ValueError("REACT_VALIDATION_EXHAUSTED")
        feedback = RepairFeedback(
            code=code,
            missing_fields=missing or [],
            task_indices=task_indices or [],
            allowed_next_actions=list(ActionKind),
        )
        return bounded(
            state,
            {
                "repairs": count,
                "repair_feedback": feedback.model_dump(mode="json"),
                "phase": "REPAIR",
                "answer": {},
            },
        )

    def resolution_update(continued: ResolutionContinuation) -> dict[str, Any]:
        update: dict[str, Any] = {
            "reference_reply": None,
            "unit_reply": None,
            "pending_draft": continued.pending.model_dump(mode="json")
            if continued.pending
            else None,
            "phase": "WAITING_INPUT"
            if continued.pending
            or (continued.admission and continued.admission.pending)
            else "STARTED",
        }
        if continued.admission:
            update.update(
                working_requirements=continued.admission.requirements.model_dump(
                    mode="json"
                ),
                pending_proposal=continued.admission.pending.model_dump(mode="json")
                if continued.admission.pending
                else None,
            )
        return update

    def initialize(state: FlexibleState, bound_input: dict[str, Any]) -> dict[str, Any]:
        if publication:
            if state.get("message") != publication.request_message:
                raise ValueError("Graph request differs from the stored request")
            state = cast(FlexibleState, dict(state) | bound_input)
        # New turns do not inherit the prior run's optimizer snapshot. Native
        # resume skips initialization and keeps its own committed snapshot.
        state = cast(
            FlexibleState,
            dict(state)
            | {
                "catalog_context": {
                    "rows": [row.model_dump(mode="json") for row in trusted.catalog]
                }
                if trusted.catalog_ready
                else None
            },
        )
        ctx = current(state)
        if not ctx.execution_allowed or not ctx.read_catalog_allowed:
            raise ValueError("REACT_EXECUTION_NOT_AUTHORIZED")
        if not state.get("message", "").strip() or len(state["message"]) > 4000:
            raise ValueError("REACT_MESSAGE_INVALID")
        update: dict[str, Any] = {
            **bound_input,
            "task_draft": None,
            "task_resolution": None,
            "working_requirements": ctx.requirements.model_dump(mode="json"),
            "model_calls": 0,
            "tool_calls": 0,
            "repairs": 0,
            "repair_feedback": None,
            "decision": None,
            "answer": {},
            "clarification": None,
            "phase": "STARTED",
            "catalog_context": {
                "rows": [row.model_dump(mode="json") for row in trusted.catalog]
            }
            if trusted.catalog_ready
            else None,
            "free_resolution_reply": None,
            "user_receipts": [],
            "free_task_reply": None,
            "free_amendment": None,
            "free_task_amendment": None,
        }
        if publication:
            # Previous-run evidence remains in immutable native ancestors. Window
            # only this new checkpoint; resume(None) never runs begin_turn again.
            previous = records(state)
            if any(entry.result is None for entry in previous):
                raise ValueError("REACT_UNRESOLVED_TOOL_PROTOCOL")
            update["ledger"] = [
                entry.model_dump(mode="json")
                for entry in previous
                if entry.origin_run_id == ctx.run_id
            ]
            update["publication_context"] = FlexibleExecutionContext(
                context=trusted,
                required=tasks,
                registered=registering,
                discard_only=publication.discard_pending,
            ).model_dump(mode="json")
        if publication and publication.discard_pending:
            update.update(
                pending_draft=None, pending_proposal=None, phase="READY_TO_PUBLISH"
            )
            return bounded(state, update)
        if (
            state.get("pending_draft")
            and not any(
                state.get(key)
                for key in ("reference_reply", "unit_reply", "confirmation")
            )
            and publication
            and (clarifier or unit_clarifier)
        ):
            update["phase"] = "INTERPRET_REPLY"
            return bounded(state, update)
        if state.get("reference_reply") or state.get("unit_reply"):
            if (
                state.get("confirmation")
                or state.get("pending_proposal")
                or not state.get("pending_draft")
                or bool(state.get("reference_reply")) == bool(state.get("unit_reply"))
            ):
                raise ValueError(
                    "Continuation requires one reply kind and one pending resolution"
                )
            pending_resolution = PendingResolution.model_validate(
                state["pending_draft"]
            )
            continued = apply_resolution_command(
                ctx,
                pending_resolution,
                state["message"],
                conversation_revision,
                ClarificationReplyV1.model_validate(state["reference_reply"])
                if state.get("reference_reply")
                else None,
                UnitClarificationReplyV1.model_validate(state["unit_reply"])
                if state.get("unit_reply")
                else None,
            )
            update.update(resolution_update(continued))
            return bounded(state, update)
        if state.get("confirmation"):
            pending = (
                PendingRequirementProposal.model_validate(state["pending_proposal"])
                if state.get("pending_proposal")
                else None
            )
            admission = confirm_agent_patch(
                ctx.requirements,
                pending,
                ProposalConfirmation.model_validate(state["confirmation"]),
                owner=ctx.account_id,
                conversation_id=ctx.conversation_id,
                conversation_revision=conversation_revision,
            )
            update.update(
                working_requirements=admission.requirements.model_dump(mode="json"),
                pending_proposal=None,
                confirmation=None,
            )
        return bounded(state, update)

    async def load_catalog(state: FlexibleState) -> dict[str, Any]:
        ctx = base_context(state)
        if ctx.catalog_ready or catalog_loader is None:
            return {}
        rows = await catalog_loader.load(Draft(items=[]), current(state).requirements)
        ctx.catalog = [CatalogRowV1.model_validate(row.model_dump()) for row in rows]
        ctx.catalog_ready = True
        update: dict[str, Any] = {
            "catalog_context": {
                "rows": [row.model_dump(mode="json") for row in ctx.catalog]
            }
        }
        if publication:
            update["publication_context"] = FlexibleExecutionContext(
                context=ctx,
                required=tasks,
                registered=registering,
                discard_only=publication.discard_pending,
            ).model_dump(mode="json")
        return bounded(state, update)

    async def prepare_agent(state: FlexibleState) -> dict[str, Any]:
        if any(task.kind == "BUILD" for task in registered_tasks(state)):
            return await load_catalog(state)
        return {}

    def begin(state: FlexibleState) -> dict[str, Any]:
        return initialize(state, turn_input)

    async def interpret_reply(state: FlexibleState) -> dict[str, Any]:
        assert publication is not None
        ctx = current(state)
        pending = PendingResolution.model_validate(state["pending_draft"])
        offered = public_resolution_options(pending.draft, ctx.catalog)
        if (
            any(item.reason == "QUANTITY_UNIT_REQUIRED" for item in offered.items)
            and unit_clarifier
        ):
            receipt = FreeResolutionReplyV2(
                run_id=ctx.run_id,
                pending_message_id=pending.draft.message_id,
                request_message_hash=canonical_hash(state["message"]),
                unit_choice=await unit_clarifier.choose_unit(state["message"], offered),
            )
        elif clarifier:
            receipt = FreeResolutionReplyV2(
                run_id=ctx.run_id,
                pending_message_id=pending.draft.message_id,
                request_message_hash=canonical_hash(state["message"]),
                reference_choice=await clarifier.choose(state["message"], offered),
            )
        else:
            raise ValueError("Pending resolution has no configured interpreter")
        reference, unit = derive_resolution_reply(
            receipt, state["message"], pending, ctx.run_id
        )
        if reference is None and unit is None:
            return bounded(
                state,
                {
                    "free_resolution_reply": receipt.model_dump(mode="json"),
                    "phase": "AMEND_PENDING" if amender else "WAITING_INPUT",
                    "model_calls": state.get("model_calls", 0) + 1,
                },
            )
        # Reuse the typed continuation lane. Its deterministic validators and
        # publication preparation remain the only transition authorities.
        interpreted = cast(
            FlexibleState,
            dict(state)
            | {
                "reference_reply": reference.model_dump(mode="json")
                if reference
                else None,
                "unit_reply": unit.model_dump(mode="json") if unit else None,
            },
        )
        effective = publication.model_copy(
            update={"reference_reply": reference, "unit_reply": unit}
        )
        update = initialize(interpreted, effective.turn_input(trusted, tasks))
        return update | {
            "free_resolution_reply": receipt.model_dump(mode="json"),
            "model_calls": state.get("model_calls", 0) + 1,
        }

    async def amend_reply(state: FlexibleState) -> dict[str, Any]:
        assert publication is not None and amender is not None
        pending = (
            publication.accepted.flexible.pending_resolution
            if publication.accepted.flexible
            else None
        )
        if pending is None:
            raise ValueError("Amendment requires an accepted pending resolution")
        ctx = current(state)
        receipt = FreeAmendmentV2(
            run_id=ctx.run_id,
            pending_message_id=pending.draft.message_id,
            request_message_hash=canonical_hash(state["message"]),
            amendment=await amender.amend(state["message"], pending.draft),
        )
        continued = apply_amendment_receipt(
            receipt, ctx, pending, state["message"], conversation_revision
        )
        return bounded(
            state,
            resolution_update(continued)
            | {
                "free_amendment": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
            },
        )

    async def register(state: FlexibleState) -> dict[str, Any]:
        assert task_extractor is not None and publication is not None
        pending = publication.accepted.flexible
        if pending and pending.pending_task_resolution:
            draft = TaskDraftBatch(tasks=pending.pending_task_resolution.tasks)
        else:
            draft = TaskDraftBatch.model_validate(
                (
                    await task_extractor.extract_tasks(publication.task_message)
                ).model_dump()
            )
        draft.validate_source(publication.task_message)
        return bounded(state, {"task_draft": draft.model_dump(mode="json")})

    async def resolve_registered(state: FlexibleState) -> dict[str, Any]:
        assert task_catalog is not None and publication is not None
        ctx = current(state)
        pending = publication.accepted.flexible
        previous = pending.pending_task_resolution if pending else None
        if state.get("free_task_amendment") is not None:
            if previous is None:
                raise ValueError("Task amendment requires exact accepted source")
            previous = apply_task_amendment_receipt(
                FreeTaskAmendmentV2.model_validate(state["free_task_amendment"]),
                ctx,
                previous,
                state["message"],
            )
        selections = previous.resolution.bindings if previous else None
        task_reply = publication.task_reply
        if state.get("free_task_reply") is not None:
            if previous is None or task_reply is not None:
                raise ValueError("Free task choice requires exact pending source")
            task_reply = derive_task_reply(
                FreeTaskReplyV2.model_validate(state["free_task_reply"]),
                state["message"],
                previous,
                ctx.run_id,
            )
        if task_reply:
            if previous is None:
                raise ValueError("Task selection requires accepted pending task")
            selections = selections_for_pending_task(
                previous, task_reply, ctx.requirements
            )
        result = await resolve_tasks(
            TaskDraftBatch.model_validate(state["task_draft"]).tasks,
            publication.task_message,
            ctx.requirements,
            task_catalog,
            saved_build_ref=ctx.saved_build.evidence_ref if ctx.saved_build else None,
            selections=selections,
            mention_overrides=previous.mention_overrides if previous else None,
        )
        return bounded(state, {"task_resolution": result.model_dump(mode="json")})

    async def interpret_task_reply(state: FlexibleState) -> dict[str, Any]:
        assert publication is not None and task_clarifier is not None
        pending = publication.accepted.flexible
        previous = pending.pending_task_resolution if pending else None
        if previous is None or publication.task_reply is not None:
            return {}
        choice = await task_clarifier.choose_task(
            state["message"], public_task_options(previous)
        )
        receipt = FreeTaskReplyV2(
            run_id=trusted.run_id,
            pending_message_id=previous.message_id,
            request_message_hash=canonical_hash(state["message"]),
            choice=choice,
        )
        reply = derive_task_reply(receipt, state["message"], previous, trusted.run_id)
        if reply:
            selections_for_pending_task(previous, reply, trusted.requirements)
        return bounded(
            state,
            {
                "free_task_reply": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
                "phase": "AMEND_TASK" if reply is None and task_amender else "STARTED",
            },
        )

    async def amend_task_reply(state: FlexibleState) -> dict[str, Any]:
        assert publication is not None and task_amender is not None
        pending = publication.accepted.flexible
        previous = pending.pending_task_resolution if pending else None
        if previous is None:
            raise ValueError("Task amendment requires accepted pending task")
        receipt = FreeTaskAmendmentV2(
            run_id=trusted.run_id,
            pending_message_id=previous.message_id,
            request_message_hash=canonical_hash(state["message"]),
            amendment=await task_amender.amend_task(state["message"], previous),
        )
        apply_task_amendment_receipt(receipt, trusted, previous, state["message"])
        return bounded(
            state,
            {
                "free_task_amendment": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
                "phase": "STARTED",
            },
        )

    async def agent(state: FlexibleState) -> dict[str, Any]:
        ctx, transcript = current(state), projection(state)
        count = state.get("model_calls", 0)
        if count >= limits.max_model_calls:
            raise ValueError("REACT_MODEL_LIMIT")
        feedback = (
            RepairFeedback.model_validate(state["repair_feedback"])
            if state.get("repair_feedback")
            else None
        )
        exposed = allowed_tools(ctx)
        payload: dict[str, JsonValue] = {
            "message": state["message"],
            "transcript": [r.model_dump(mode="json") for r in transcript],
            "allowed": list(exposed),
            "feedback": feedback.model_dump(mode="json") if feedback else None,
            "context": ctx.model_dump(mode="json"),
            "graph_version": FLEXIBLE_GRAPH_VERSION,
        }
        decision = await attempts.invoke(
            AttemptKind.MODEL,
            f"flexible:agent:{count}",
            canonical_hash(payload),
            lambda: model.decide(state["message"], transcript, exposed, feedback, ctx),
        )
        return bounded(
            state,
            {
                "model_calls": count + 1,
                "decision": FlexibleDecision.model_validate(
                    decision.model_dump()
                ).model_dump(mode="json")
                if decision
                else None,
                "phase": "DECIDED",
            },
        )

    def validate(state: FlexibleState) -> dict[str, Any]:
        if state.get("decision") is None:
            return repair(state, "INVALID_ACTION")
        action = FlexibleDecision.model_validate(state["decision"]).action
        ctx = current(state)
        if isinstance(action, CallToolAction):
            entries = records(state)
            if state.get("tool_calls", 0) >= limits.max_tool_calls:
                raise ValueError("REACT_TOOL_LIMIT")
            if action.call.call_id in {
                entry.call.call_id
                for entry in entries
                if entry.origin_run_id == ctx.run_id
            }:
                return repair(state, "INVALID_ACTION")
            try:
                name = ToolName(action.call.name)
                if name not in allowed_tools(ctx):
                    return repair(state, "TOOL_NOT_ALLOWED")
                TOOL_SCHEMAS[name].model_validate(action.call.arguments)
            except (ValueError, ValidationError):
                return repair(state, "BAD_TOOL_ARGUMENTS")
            entries.append(ToolRecord(call=action.call, origin_run_id=ctx.run_id))
            return bounded(
                state,
                {
                    "ledger": [entry.model_dump(mode="json") for entry in entries],
                    "tool_calls": state.get("tool_calls", 0) + 1,
                    "repair_feedback": None,
                    "phase": "TOOL_PENDING",
                },
            )
        if isinstance(action, ProposeRequirementsAction):
            return {"phase": "RESOLVE_PROPOSAL"}
        if isinstance(action, AskClarificationAction):
            pending_fields = unresolved_tasks(state)
            if pending_fields:
                if not set(action.missing_fields) <= set(pending_fields):
                    return repair(state, "INVALID_ACTION", pending_fields)
                return bounded(
                    state,
                    {
                        "clarification": action.model_dump(mode="json"),
                        "phase": "WAITING_INPUT",
                    },
                )
            mapping = compile_requirements(ctx.requirements, ctx.catalog).mapping
            missing = {
                item.reason for item in mapping if item.status == "NEEDS_CLARIFICATION"
            }
            if (
                not any(task.kind == "BUILD" for task in ctx.completion_requirements)
                or not set(action.missing_fields) <= missing
            ):
                return repair(state, "INVALID_ACTION", sorted(missing))
            return bounded(
                state,
                {
                    "clarification": action.model_dump(mode="json"),
                    "phase": "WAITING_INPUT",
                },
            )
        assert isinstance(action, FinalAnswerAction)
        if unresolved_tasks(state):
            return repair(state, "TASK_INCOMPLETE", unresolved_tasks(state))
        transcript = projection(state)
        try:
            answer = validate_final_answer(
                action,
                [entry.result for entry in transcript if entry.result],
                ctx,
                ctx.completion_requirements,
                dispatcher.service,
            )
        except InvalidTaskAnswer as exc:
            return repair(
                state,
                "TASK_INCOMPLETE",
                ["task_answers"],
                task_indices=exc.task_indices,
            )
        except ValueError:
            if action.outcome in {"INFEASIBLE", "SEARCH_INCOMPLETE"}:
                return repair(
                    state,
                    "SEARCH_PROOF_REQUIRED",
                    ["supporting_tool_call_ids", "search_evidence"],
                )
            if action.outcome in {"INSUFFICIENT_EVIDENCE", "DEPENDENCY_UNAVAILABLE"}:
                return repair(state, "MISSING_EVIDENCE", ["supporting_tool_call_ids"])
            return repair(state, "TASK_INCOMPLETE")
        return bounded(
            state,
            {
                "answer": answer.model_dump(mode="json"),
                "repair_feedback": None,
                "phase": "READY_TO_PUBLISH",
            },
        )

    async def execute(state: FlexibleState) -> dict[str, Any]:
        ctx, entries = current(state), records(state)
        entry = entries[-1]
        if entry.result is None:
            # Budget every tool invocation; individual adapter retries must remain
            # independently reserved. Native recovery reuses committed outcomes.
            entry.result = await attempts.invoke(
                AttemptKind.TOOL_IO,
                f"flexible:tool:{entry.call.call_id}",
                call_hash(entry.call, ctx),
                lambda: dispatcher.execute(entry.call, ctx),
            )
        if len(entry.result.model_dump_json().encode()) > limits.max_tool_bytes:
            raise ValueError("REACT_TOOL_RESULT_TOO_LARGE")
        return bounded(
            state,
            {
                "ledger": [item.model_dump(mode="json") for item in entries],
                "phase": "TOOL_COMMITTED",
            },
        )

    async def resolve(state: FlexibleState) -> dict[str, Any]:
        action = FlexibleDecision.model_validate(state["decision"]).action
        assert isinstance(action, ProposeRequirementsAction)
        ctx = current(state)
        if state.get("pending_proposal"):
            return repair(state, "PROPOSAL_CONFIRMATION_REQUIRED")
        if state.get("pending_draft"):
            return repair(state, "INVALID_ACTION", ["pending_draft"])
        extra_calls = int(user_extractor is not None and action.source_text is None)
        if (
            user_extractor is not None
            and action.source_text is None
            and state.get("model_calls", 0) >= limits.max_model_calls
        ):
            raise ValueError("REACT_MODEL_LIMIT")
        try:
            result = await resolve_requirement_proposal(
                action,
                ctx,
                state["message"],
                conversation_revision,
                user_extractor=user_extractor,
            )
            update: dict[str, Any] = {
                "model_calls": state.get("model_calls", 0) + extra_calls,
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
            return bounded(
                state,
                repair(state, "INVALID_ACTION")
                | {"model_calls": state.get("model_calls", 0) + extra_calls},
            )
        return bounded(state, update)

    builder = StateGraph(FlexibleState)
    builder.add_node("begin_turn", begin)
    builder.add_node("agent", agent)
    builder.add_node("validate_action", validate)
    builder.add_node("execute_tool", execute)
    builder.add_node("resolve_proposal", resolve)
    builder.add_node("load_proposal_catalog", load_catalog)
    builder.add_node("prepare_agent", prepare_agent)
    builder.add_edge("load_proposal_catalog", "resolve_proposal")
    builder.add_edge("prepare_agent", "agent")
    builder.add_node("interpret_reply", interpret_reply)
    builder.add_node("amend_pending", amend_reply)
    terminal = END
    if publication:

        def prepare(state: FlexibleState) -> dict[str, Any]:
            assert publication is not None
            snapshot = publication.prepare(
                state,
                base_context(state),
                registered_tasks(state),
                dispatcher.service,
                conversation_revision,
            )
            return bounded(
                state, {"accepted_snapshot": snapshot.model_dump(mode="json")}
            )

        terminal = "prepare_publish"
        builder.add_node(terminal, prepare)
        builder.add_edge(terminal, END)
    builder.add_edge(START, "begin_turn")
    start_agent = "prepare_agent"
    if registering:
        start_agent = "register_request"
        builder.add_node(start_agent, register)
        builder.add_node("resolve_registered_tasks", resolve_registered)
        if task_clarifier:
            builder.add_node("interpret_task_reply", interpret_task_reply)
            builder.add_edge(start_agent, "interpret_task_reply")
            if task_amender:
                builder.add_node("amend_task", amend_task_reply)
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
            builder.add_edge(start_agent, "resolve_registered_tasks")
        builder.add_edge("resolve_registered_tasks", "prepare_agent")
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
    builder.add_edge("agent", "validate_action")
    builder.add_conditional_edges(
        "validate_action",
        lambda s: s["phase"],
        {
            "REPAIR": "agent",
            "TOOL_PENDING": "execute_tool",
            "RESOLVE_PROPOSAL": "load_proposal_catalog",
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
            "REQUIREMENTS_UNCHANGED": "agent",
            "REQUIREMENTS_ACCEPTED": "agent",
        },
    )
    return builder.compile(checkpointer=checkpointer)
