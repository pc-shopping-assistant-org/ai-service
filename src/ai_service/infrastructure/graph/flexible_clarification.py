"""Clarification, amendment, and task-registration nodes for the flexible ReAct graph."""

from __future__ import annotations

from dataclasses import dataclass

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import decode_flexible_decision
from ai_service.capabilities.assistant.react.contracts import ControlRecord
from ai_service.capabilities.assistant.react.phases import ReActPhase
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
)
from ai_service.capabilities.assistant.react.task_amendment import TaskAmender
from ai_service.capabilities.assistant.react.task_resolution import (
    TaskCatalog,
    TaskClarificationExtractor,
    public_task_options,
    resolve_tasks,
    selections_for_pending_task,
)
from ai_service.capabilities.assistant.requirements.amendment import PendingAmender
from ai_service.capabilities.assistant.requirements.catalog_policy import (
    draft_needs_catalog,
)
from ai_service.capabilities.assistant.requirements.pending_resolution import (
    PendingResolution,
)
from ai_service.capabilities.assistant.requirements.resolver import RequirementDraft
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.infrastructure.graph.flexible_state import (
    AsyncNode,
    FlexibleRuntime,
    FlexibleState,
)


@dataclass(frozen=True, slots=True)
class ClarificationNodes:
    interpret_reply: AsyncNode
    amend_reply: AsyncNode
    register: AsyncNode
    resolve_registered: AsyncNode
    interpret_task_reply: AsyncNode
    amend_task_reply: AsyncNode


def build_clarification_nodes(
    runtime: FlexibleRuntime,
    *,
    task_catalog: TaskCatalog | None,
    task_clarifier: TaskClarificationExtractor | None,
    amender: PendingAmender | None,
    task_amender: TaskAmender | None,
) -> ClarificationNodes:
    async def interpret_reply(state: FlexibleState) -> FlexibleState:
        assert runtime.publication is not None
        ctx = runtime.current(state)
        pending = PendingResolution.model_validate(state["pending_draft"])
        offered = (
            runtime.publication.accepted.flexible.resolution_options
            if runtime.publication.accepted.flexible
            else None
        ) or public_resolution_options(pending.draft, ctx.catalog)
        if (
            any(item.reason == "QUANTITY_UNIT_REQUIRED" for item in offered.items)
            and runtime.unit_clarifier
        ):
            receipt = FreeResolutionReplyV2(
                run_id=ctx.run_id,
                pending_message_id=pending.draft.message_id,
                request_message_hash=canonical_hash(state["message"]),
                unit_choice=await runtime.unit_clarifier.choose_unit(
                    state["message"], offered
                ),
            )
        elif runtime.clarifier:
            receipt = FreeResolutionReplyV2(
                run_id=ctx.run_id,
                pending_message_id=pending.draft.message_id,
                request_message_hash=canonical_hash(state["message"]),
                reference_choice=await runtime.clarifier.choose(
                    state["message"], offered
                ),
            )
        else:
            raise ValueError("Pending resolution has no configured interpreter")
        reference, unit = derive_resolution_reply(
            receipt, state["message"], pending, ctx.run_id
        )
        if reference is None and unit is None:
            return runtime.bounded(
                state,
                {
                    "free_resolution_reply": receipt.model_dump(mode="json"),
                    "phase": ReActPhase.AMEND_PENDING.code
                    if amender
                    else ReActPhase.WAITING_INPUT.code,
                    "model_calls": state.get("model_calls", 0) + 1,
                },
            )
        # A side question needs only the saved offered choices. Fetch current
        # hardware only after a real selection, before deterministic admission.
        loaded: FlexibleState = (
            await runtime.load_catalog(state)
            if draft_needs_catalog(
                RequirementDraft(items=pending.draft.resolution.draft.items)
            )
            else {}
        )
        continued = apply_resolution_command(
            runtime.current(state | loaded),
            pending,
            state["message"],
            runtime.conversation_revision,
            reference,
            unit,
        )
        return runtime.bounded(
            state,
            loaded
            | runtime.resolution_update(continued)
            | {
                "free_resolution_reply": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
            },
        )

    async def amend_reply(state: FlexibleState) -> FlexibleState:
        assert runtime.publication is not None and amender is not None
        pending = (
            runtime.publication.accepted.flexible.pending_resolution
            if runtime.publication.accepted.flexible
            else None
        )
        if pending is None:
            raise ValueError("Amendment requires an accepted pending resolution")
        ctx = runtime.current(state)
        receipt = FreeAmendmentV2(
            run_id=ctx.run_id,
            pending_message_id=pending.draft.message_id,
            request_message_hash=canonical_hash(state["message"]),
            amendment=await amender.amend(state["message"], pending.draft),
        )
        if not receipt.amendment.replacements and not receipt.amendment.discards:
            return runtime.bounded(
                state,
                {
                    "free_amendment": receipt.model_dump(mode="json"),
                    "phase": ReActPhase.STARTED.code,
                    "model_calls": state.get("model_calls", 0) + 1,
                },
            )
        discarded = set(pending.draft.discarded_items) | {
            item.item_index for item in receipt.amendment.discards
        }
        needs_catalog = draft_needs_catalog(
            RequirementDraft(
                items=[
                    item
                    for index, item in enumerate(pending.draft.resolution.draft.items)
                    if index not in discarded
                ]
            )
        )
        loaded: FlexibleState = (
            await runtime.load_catalog(state) if needs_catalog else {}
        )
        continued = apply_amendment_receipt(
            receipt,
            runtime.current(state | loaded),
            pending,
            state["message"],
            runtime.conversation_revision,
        )
        return runtime.bounded(
            state,
            loaded
            | runtime.resolution_update(continued)
            | {
                "free_amendment": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
            },
        )

    async def register(state: FlexibleState) -> FlexibleState:
        assert runtime.publication is not None
        pending = runtime.publication.accepted.flexible
        if pending and pending.pending_task_resolution:
            draft = TaskDraftBatch(tasks=pending.pending_task_resolution.tasks)
        else:
            decision = decode_flexible_decision(state.get("decision"))
            if (
                not runtime.current(state).task_registration_open
                or decision.registration is None
            ):
                raise ValueError("Task registration is not available")
            draft = decision.registration
        draft.validate_source(runtime.publication.task_message)
        return runtime.bounded(state, {"task_draft": draft.model_dump(mode="json")})

    async def resolve_registered(state: FlexibleState) -> FlexibleState:
        assert task_catalog is not None and runtime.publication is not None
        ctx = runtime.current(state)
        pending = runtime.publication.accepted.flexible
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
        task_reply = runtime.publication.task_reply
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
            runtime.publication.task_message,
            ctx.requirements,
            task_catalog,
            saved_build_ref=ctx.saved_build.evidence_ref if ctx.saved_build else None,
            selections=selections,
            mention_overrides=previous.mention_overrides if previous else None,
        )
        update: FlexibleState = {"task_resolution": result.model_dump(mode="json")}
        update["publication_context"] = FlexibleExecutionContext(
            context=runtime.base_context(state), required=[], registered=True
        ).model_dump(mode="json")
        decision = (
            decode_flexible_decision(state["decision"])
            if state.get("decision")
            else None
        )
        if decision and decision.registration is not None:
            history = list(ctx.control_history)
            if decision.registration_call:
                history.append(
                    ControlRecord(
                        call=decision.registration_call,
                        code="TASKS_REGISTERED",
                        model_step=state.get("model_calls", 0),
                    )
                )
            update.update(
                {
                    "control_history": [
                        entry.model_dump(mode="json") for entry in history
                    ],
                    "phase": ReActPhase.DECIDED.code,
                }
            )
        return runtime.bounded(state, update)

    async def interpret_task_reply(state: FlexibleState) -> FlexibleState:
        assert runtime.publication is not None and task_clarifier is not None
        pending = runtime.publication.accepted.flexible
        previous = pending.pending_task_resolution if pending else None
        if previous is None or runtime.publication.task_reply is not None:
            return {}
        choice = await task_clarifier.choose_task(
            state["message"], public_task_options(previous)
        )
        receipt = FreeTaskReplyV2(
            run_id=runtime.trusted.run_id,
            pending_message_id=previous.message_id,
            request_message_hash=canonical_hash(state["message"]),
            choice=choice,
        )
        reply = derive_task_reply(
            receipt, state["message"], previous, runtime.trusted.run_id
        )
        if reply:
            selections_for_pending_task(previous, reply, runtime.trusted.requirements)
        return runtime.bounded(
            state,
            {
                "free_task_reply": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
                "phase": ReActPhase.AMEND_TASK.code
                if reply is None and task_amender
                else ReActPhase.STARTED.code,
            },
        )

    async def amend_task_reply(state: FlexibleState) -> FlexibleState:
        assert runtime.publication is not None and task_amender is not None
        pending = runtime.publication.accepted.flexible
        previous = pending.pending_task_resolution if pending else None
        if previous is None:
            raise ValueError("Task amendment requires accepted pending task")
        receipt = FreeTaskAmendmentV2(
            run_id=runtime.trusted.run_id,
            pending_message_id=previous.message_id,
            request_message_hash=canonical_hash(state["message"]),
            amendment=await task_amender.amend_task(state["message"], previous),
        )
        apply_task_amendment_receipt(
            receipt, runtime.trusted, previous, state["message"]
        )
        return runtime.bounded(
            state,
            {
                "free_task_amendment": receipt.model_dump(mode="json"),
                "model_calls": state.get("model_calls", 0) + 1,
                "phase": ReActPhase.STARTED.code,
            },
        )

    return ClarificationNodes(
        interpret_reply=interpret_reply,
        amend_reply=amend_reply,
        register=register,
        resolve_registered=resolve_registered,
        interpret_task_reply=interpret_task_reply,
        amend_task_reply=amend_task_reply,
    )
