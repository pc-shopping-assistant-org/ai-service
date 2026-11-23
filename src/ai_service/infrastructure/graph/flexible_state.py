"""State types and runtime helpers for the flexible ReAct graph."""

from __future__ import annotations

import json
from collections.abc import Coroutine
from typing import Any, Protocol, TypedDict, cast

from ai_service.capabilities.assistant.limits import MAX_HARDWARE_CATALOG_ROWS
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    ActionKind,
    FlexibleDecision,
    RepairFeedback,
    decode_flexible_decision,
)
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ControlRecord,
    ReactLimits,
    ToolContext,
    ToolName,
    ToolRecord,
    require_tool_results,
)
from ai_service.capabilities.assistant.react.dialogue import pending_context
from ai_service.capabilities.assistant.react.phases import ReActPhase, ReActPhaseCode
from ai_service.capabilities.assistant.react.resolution_reply import (
    apply_resolution_command,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexibleExecutionContext,
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.submissions import evidence_requirements
from ai_service.capabilities.assistant.react.task_resolution import TaskResolution
from ai_service.capabilities.assistant.react.tools import project_current_tools
from ai_service.capabilities.assistant.requirements.admission import (
    confirm_agent_patch,
)
from ai_service.capabilities.assistant.requirements.build import (
    ClarificationExtractor,
    RequirementCatalog,
)
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
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.stateful_contracts import (
    ClarificationReplyV1,
    UnitClarificationReplyV1,
)

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


class AsyncNode(Protocol):
    def __call__(self, state: FlexibleState) -> Coroutine[Any, Any, FlexibleState]: ...


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
    phase: ReActPhaseCode
    answer: dict[str, Any]
    response: dict[str, Any] | None
    control_history: list[dict[str, Any]]
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


class FlexibleRuntime:
    """Shared state access, bounded persistence, and cached catalog access for one graph run."""

    def __init__(
        self,
        *,
        trusted: ToolContext,
        tasks: list[CompletionRequirement],
        registering: bool,
        limits: ReactLimits,
        publication: FlexiblePublicationInput | None,
        conversation_revision: int,
        turn_input: dict[str, Any],
        catalog_loader: RequirementCatalog | None,
        clarifier: ClarificationExtractor | None,
        unit_clarifier: UnitClarificationExtractor | None,
    ) -> None:
        self.trusted = trusted
        self.tasks = tasks
        self.registering = registering
        self.limits = limits
        self.publication = publication
        self.conversation_revision = conversation_revision
        self.turn_input = turn_input
        self.catalog_loader = catalog_loader
        self.clarifier = clarifier
        self.unit_clarifier = unit_clarifier

        self._cached_catalog: list[CatalogRowV1] | None = (
            list(trusted.catalog) if trusted.catalog_ready else None
        )
        self._cached_catalog_hash: str | None = None

    def catalog_hash(self, ctx: ToolContext) -> str | None:
        if not ctx.catalog_ready:
            return None
        if self._cached_catalog_hash is None:
            rows = (
                self._cached_catalog
                if self._cached_catalog is not None
                else ctx.catalog
            )
            self._cached_catalog_hash = canonical_hash(
                [row.model_dump(mode="json") for row in rows]
            )
        return self._cached_catalog_hash

    def _catalog_ref(self, rows: list[CatalogRowV1]) -> dict[str, Any]:
        self._cached_catalog = rows
        self._cached_catalog_hash = canonical_hash(
            [row.model_dump(mode="json") for row in rows]
        )
        return {
            "ready": True,
            "hash": self._cached_catalog_hash,
            "count": len(rows),
        }

    def context_digest_payload(self, ctx: ToolContext) -> dict[str, Any]:
        """Compact context representation for model-call idempotency hashing without dumping raw catalog rows."""
        payload = ctx.model_dump(mode="json", exclude={"catalog"})
        payload["catalog_hash"] = self.catalog_hash(ctx)
        return payload

    def registered_tasks(self, state: FlexibleState) -> list[CompletionRequirement]:
        if self.registering:
            if (state.get("task_draft") or {}).get("tasks"):
                return (
                    TaskResolution.model_validate(state["task_resolution"]).requirements
                    if state.get("task_resolution")
                    else []
                )
            working = self.base_context(state)
            working.requirements = CanonicalRequirementState.model_validate(
                state.get(
                    "working_requirements", self.trusted.requirements.model_dump()
                )
            )
            return evidence_requirements(
                state.get("decision"),
                [ToolRecord.model_validate(row) for row in state.get("ledger", [])],
                working,
            )
        return self.tasks

    def unresolved_tasks(self, state: FlexibleState) -> list[str]:
        if self.registering and state.get("task_resolution"):
            return sorted(
                {
                    item.reason
                    for item in TaskResolution.model_validate(
                        state["task_resolution"]
                    ).pending
                }
            )
        return []

    def bounded(self, state: FlexibleState, update: FlexibleState) -> FlexibleState:
        if (
            len(
                json.dumps(state | update, ensure_ascii=False, allow_nan=False).encode()
            )
            > self.limits.max_state_bytes
        ):
            raise ValueError("REACT_STATE_TOO_LARGE")
        return update

    def base_context(self, state: FlexibleState) -> ToolContext:
        value = self.trusted.model_copy(deep=True)
        catalog_context = state.get("catalog_context")
        if catalog_context is not None:
            if "rows" in catalog_context:
                rows = [
                    CatalogRowV1.model_validate(row) for row in catalog_context["rows"]
                ]
                self._cached_catalog = rows
                self._cached_catalog_hash = None
                value.catalog = rows
            elif self._cached_catalog is not None:
                value.catalog = list(self._cached_catalog)
            else:
                # Cold recovery fallback: hydrate catalog from durable publication_context if present
                pub_ctx = state.get("publication_context")
                raw_rows = (
                    (pub_ctx.get("context") or {}).get("catalog")
                    if isinstance(pub_ctx, dict)
                    else None
                )
                if raw_rows is not None:
                    rows = [CatalogRowV1.model_validate(row) for row in raw_rows]
                    self._cached_catalog = rows
                    self._cached_catalog_hash = None
                    value.catalog = rows
            value.catalog_ready = True
        return value

    def current(self, state: FlexibleState) -> ToolContext:
        value = self.base_context(state)
        value.requirements = CanonicalRequirementState.model_validate(
            state.get("working_requirements", self.trusted.requirements.model_dump())
        )
        value.control_history = [
            ControlRecord.model_validate(record)
            for record in state.get("control_history", [])
        ]
        # Trusted BUILD task remains BUILD; only its legacy adapter revision follows
        # application-approved working content. No model action can replace tasks.
        value.completion_requirements = [
            task.model_copy(
                update={"constraints_revision": value.requirements.revision}
            )
            if task.kind == "BUILD"
            else task.model_copy(deep=True)
            for task in self.registered_tasks(state)
        ]
        if self.publication:
            value.task_message = self.publication.task_message
            value.pending = (
                pending_context(self.publication.accepted.flexible)
                if not self.publication.confirmation
                and value.requirements == self.trusted.requirements
                else None
            )
        if state.get("task_draft"):
            value.requested_tasks = TaskDraftBatch.model_validate(state["task_draft"])
        value.task_registration_open = (
            self.registering
            and not (state.get("task_draft") or {}).get("tasks")
            and not (
                self.publication
                and self.publication.accepted.flexible
                and self.publication.accepted.flexible.pending_task_resolution
            )
        )
        if (
            len(value.control_history) > 100
            or len(value.completion_requirements) > 8
            or len(value.catalog) > MAX_HARDWARE_CATALOG_ROWS
        ):
            raise ValueError("ToolContext field bounds exceeded")
        return value

    def records(self, state: FlexibleState) -> list[ToolRecord]:
        entries = [ToolRecord.model_validate(row) for row in state.get("ledger", [])]
        if len({(entry.origin_run_id, entry.call.call_id) for entry in entries}) != len(
            entries
        ):
            raise ValueError("REACT_DUPLICATE_CALL_ID")
        return entries

    def projection(self, state: FlexibleState) -> list[ToolRecord]:
        return project_current_tools(self.records(state), self.current(state))

    def acknowledge_control(
        self, state: FlexibleState, update: FlexibleState
    ) -> FlexibleState:
        decision = (
            decode_flexible_decision(state["decision"])
            if state.get("decision")
            else None
        )
        call = decision.control_call if decision else None
        if call is not None:
            history = [
                ControlRecord.model_validate(row)
                for row in state.get("control_history", [])
            ]
            if call.call_id not in {entry.call.call_id for entry in history}:
                history.append(
                    ControlRecord(
                        call=call,
                        code=update.get("phase", ReActPhase.REPAIR.code),
                        model_step=state.get("model_calls", 0),
                    )
                )
            update["control_history"] = [
                entry.model_dump(mode="json") for entry in history
            ]
        return update

    def repair(
        self,
        state: FlexibleState,
        code: Any,
        missing: list[str] | None = None,
        *,
        task_indices: list[int] | None = None,
        next_actions: list[ActionKind] | None = None,
    ) -> FlexibleState:
        count = state.get("repairs", 0) + 1
        if count > self.limits.max_repairs:
            raise ValueError("REACT_VALIDATION_EXHAUSTED")
        feedback = RepairFeedback(
            code=code,
            missing_fields=missing or [],
            task_indices=task_indices or [],
            allowed_next_actions=next_actions
            if next_actions is not None
            else list(ActionKind),
        )
        return self.bounded(
            state,
            self.acknowledge_control(
                state,
                {
                    "repairs": count,
                    "repair_feedback": feedback.model_dump(mode="json"),
                    "phase": ReActPhase.REPAIR.code,
                    "answer": {},
                },
            ),
        )

    def resolution_update(self, continued: ResolutionContinuation) -> FlexibleState:
        update: FlexibleState = {
            "reference_reply": None,
            "unit_reply": None,
            "pending_draft": continued.pending.model_dump(mode="json")
            if continued.pending
            else None,
            "phase": ReActPhase.WAITING_INPUT.code
            if continued.pending
            or (continued.admission and continued.admission.pending)
            else ReActPhase.STARTED.code,
        }
        if continued.admission:
            update.update(
                {
                    "working_requirements": continued.admission.requirements.model_dump(
                        mode="json"
                    ),
                    "pending_proposal": continued.admission.pending.model_dump(
                        mode="json"
                    )
                    if continued.admission.pending
                    else None,
                }
            )
        return update

    async def load_catalog(self, state: FlexibleState) -> FlexibleState:
        ctx = self.base_context(state)
        if ctx.catalog_ready and (ctx.catalog or self._cached_catalog is not None):
            return {}
        if self.catalog_loader is None:
            return {}
        rows = await self.catalog_loader.load(
            RequirementDraft(items=[]), self.current(state).requirements
        )
        validated = [CatalogRowV1.model_validate(row.model_dump()) for row in rows]
        ctx.catalog = validated
        ctx.catalog_ready = True
        update: FlexibleState = {
            "catalog_context": self._catalog_ref(validated),
        }
        if self.publication:
            update["publication_context"] = FlexibleExecutionContext(
                context=ctx,
                required=self.tasks,
                registered=bool((state.get("task_draft") or {}).get("tasks")),
                tool_driven=self.registering
                and not bool((state.get("task_draft") or {}).get("tasks")),
                discard_only=self.publication.discard_pending,
            ).model_dump(mode="json")
        return self.bounded(state, update)

    def initialize(
        self, state: FlexibleState, bound_input: dict[str, Any]
    ) -> FlexibleState:
        if self.publication:
            if state.get("message") != self.publication.request_message:
                raise ValueError("Graph request differs from the stored request")
            state = state | cast(FlexibleState, bound_input)
        # New turns do not inherit the prior run's optimizer snapshot. Native
        # resume skips initialization and keeps its own committed snapshot.
        initial_catalog_ref = (
            self._catalog_ref(list(self.trusted.catalog))
            if self.trusted.catalog_ready
            else None
        )
        state = state | {"catalog_context": initial_catalog_ref}
        ctx = self.current(state)
        if not ctx.execution_allowed or not ctx.read_catalog_allowed:
            raise ValueError("REACT_EXECUTION_NOT_AUTHORIZED")
        if not state.get("message", "").strip() or len(state["message"]) > 4000:
            raise ValueError("REACT_MESSAGE_INVALID")
        update: FlexibleState = {
            **cast(FlexibleState, bound_input),
            "task_draft": {"tasks": []} if self.registering else None,
            "task_resolution": TaskResolution().model_dump(mode="json")
            if self.registering
            else None,
            "working_requirements": ctx.requirements.model_dump(mode="json"),
            "model_calls": 0,
            "tool_calls": 0,
            "repairs": 0,
            "repair_feedback": None,
            "decision": None,
            "answer": {},
            "response": None,
            "control_history": [],
            "clarification": None,
            "phase": ReActPhase.STARTED.code,
            "catalog_context": initial_catalog_ref,
            "free_resolution_reply": None,
            "user_receipts": [],
            "free_task_reply": None,
            "free_amendment": None,
            "free_task_amendment": None,
        }
        if self.publication:
            # Previous-run evidence remains in immutable native ancestors. Window
            # only this new checkpoint; resume(None) never runs begin_turn again.
            previous = self.records(state)
            require_tool_results(previous)
            update["ledger"] = [
                entry.model_dump(mode="json")
                for entry in previous
                if entry.origin_run_id == ctx.run_id
            ]
            update["publication_context"] = FlexibleExecutionContext(
                context=self.trusted,
                required=self.tasks,
                registered=False,
                tool_driven=self.registering,
                discard_only=self.publication.discard_pending,
            ).model_dump(mode="json")
        if self.publication and self.publication.discard_pending:
            update.update(
                {
                    "pending_draft": None,
                    "pending_proposal": None,
                    "phase": ReActPhase.READY_TO_PUBLISH.code,
                }
            )
            return self.bounded(state, update)
        if (
            state.get("pending_draft")
            and not any(
                state.get(key)
                for key in ("reference_reply", "unit_reply", "confirmation")
            )
            and self.publication
            and (self.clarifier or self.unit_clarifier)
        ):
            update["phase"] = ReActPhase.INTERPRET_REPLY.code
            return self.bounded(state, update)
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
                self.conversation_revision,
                ClarificationReplyV1.model_validate(state["reference_reply"])
                if state.get("reference_reply")
                else None,
                UnitClarificationReplyV1.model_validate(state["unit_reply"])
                if state.get("unit_reply")
                else None,
            )
            update.update(self.resolution_update(continued))
            return self.bounded(state, update)
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
                conversation_revision=self.conversation_revision,
            )
            update.update(
                {
                    "working_requirements": admission.requirements.model_dump(
                        mode="json"
                    ),
                    "pending_proposal": None,
                    "confirmation": None,
                }
            )
        return self.bounded(state, update)
