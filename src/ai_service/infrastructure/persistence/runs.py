"""Main-path run admission, exclusive execution and atomic publication.

No lease-only takeover: a crashed worker keeps the thread busy until an explicit
recovery path verifies termination. Opt-in transient retries require stored owning
executor stop ACK. HTTP/SSE are never execution owners.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
    RunAdmissionError,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    RunViewV1,
)
from ai_service.capabilities.assistant.react.contracts import (
    SavedBuildEvidence,
    ValidatedAnswer,
)
from ai_service.capabilities.assistant.react.snapshot_publication import (
    revalidate_flexible_publication,
)
from ai_service.capabilities.assistant.react.task_resolution import (
    PendingTaskDraft,
    TaskResolution,
)
from ai_service.capabilities.assistant.react.turn_contracts import (
    TurnRequest,
    TurnRequestV2,
    parse_turn_request,
    turn_request_payload,
)
from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    classify_failure,
    require_recovery_eligible,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.pending_action import (
    DISCARDED_ANSWER,
    DISCARDED_OUTCOME,
    PendingActionReceipt,
    validate_pending_action,
)
from ai_service.capabilities.assistant.requirements.resolver import PendingDraft
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
    RunStatus,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.execution.supervisor import ExecutionOutcome
from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION

from .accepted_state import PostgresAcceptedCheckpointReader
from .call_attempts import PostgresAttemptLedger
from .checkpoint_lineage import require_checkpoint_lineage
from .checkpoint_reconciliation import reconcile_stopped_checkpoint
from .lifecycle_events import emit_start, emit_terminal
from .retry_cleanup import cleanup_stopped_retries
from .schema import agent_call_budgets, agent_runs, conversation_messages, conversations

logger = logging.getLogger(__name__)


class PostgresRunCoordinator:
    def __init__(
        self,
        engine: AsyncEngine,
        saver: BaseCheckpointSaver,
        *,
        request_versions: tuple[int, ...] = (1,),
        flexible_service: PCBuildApplicationService | None = None,
        max_attempts: int = 1,
        auto_recovery: bool = False,
    ) -> None:
        if not request_versions or any(
            type(version) is not int or version not in (1, 2)
            for version in request_versions
        ):
            raise ValueError("Request versions must explicitly select V1/V2")
        self.request_versions = frozenset(request_versions)
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Recovery attempts must be between one and three")
        self.max_attempts = max_attempts
        if auto_recovery and max_attempts < 2:
            raise ValueError("Automatic recovery requires a retry budget")
        self.auto_recovery = auto_recovery
        self.flexible_service = flexible_service
        self.engine, self.saver = engine, saver
        self.ledger = PostgresAttemptLedger(engine)

    @staticmethod
    def _view(row) -> RunViewV1:
        return RunViewV1(
            run_id=row["id"],
            conversation_id=row["conversation_id"],
            request_id=row["request_id"],
            status=row["status"],
            base_revision=row["base_revision"],
        )

    async def submit(self, owner: UUID, request: TurnRequest) -> RunViewV1:
        request = parse_turn_request(request.model_dump())
        version = 2 if isinstance(request, TurnRequestV2) else 1
        if version not in self.request_versions:
            raise RunAdmissionError("AI_REQUEST_VERSION_UNSUPPORTED")
        payload = turn_request_payload(request)
        digest = canonical_hash(payload)
        async with self.engine.begin() as conn:
            conversation = (
                (
                    await conn.execute(
                        select(conversations)
                        .where(
                            conversations.c.id == request.conversation_id,
                            conversations.c.account_id == owner,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if conversation is None:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            existing = (
                (
                    await conn.execute(
                        select(agent_runs).where(
                            agent_runs.c.conversation_id == request.conversation_id,
                            agent_runs.c.request_id == request.request_id,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if existing["request_hash"] != digest:
                    raise RunAdmissionError("AI_REQUEST_CONFLICT")
                return self._view(existing)
            if conversation["accepted_checkpoint_id"] is None:
                raise RunAdmissionError("AI_CONVERSATION_NOT_READY")
            if conversation["revision"] != request.expected_revision:
                raise RunAdmissionError("AI_REVISION_CONFLICT")
            if conversation["active_run_id"] is not None:
                raise RunAdmissionError("AI_CONVERSATION_BUSY")
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            run_id = uuid4()
            base = CheckpointRefV1(
                thread_id=request.conversation_id,
                checkpoint_id=conversation["accepted_checkpoint_id"],
            )
            row = (
                (
                    await conn.execute(
                        insert(agent_runs)
                        .values(
                            id=run_id,
                            conversation_id=request.conversation_id,
                            request_id=request.request_id,
                            request_hash=digest,
                            request_payload=payload,
                            kind="CHAT",
                            status="PENDING",
                            base_revision=request.expected_revision,
                            base_checkpoint_ref=base.model_dump(mode="json"),
                            attempt_count=0,
                            max_attempts=self.max_attempts,
                            retry_policy_version="stopped-source-v1"
                            if self.max_attempts > 1
                            else "main-path-v1",
                            retry_deadline=now + timedelta(minutes=5),
                        )
                        .returning(*agent_runs.c)
                    )
                )
                .mappings()
                .one()
            )
            await conn.execute(
                insert(agent_call_budgets).values(
                    run_id=run_id,
                    model_limit=12,
                    tool_io_limit=16,
                    deadline=now + timedelta(minutes=5),
                )
            )
            await conn.execute(
                update(conversations)
                .where(conversations.c.id == request.conversation_id)
                .values(active_run_id=run_id)
            )
            sequence = (
                await conn.execute(
                    select(
                        func.coalesce(func.max(conversation_messages.c.sequence), 0)
                    ).where(
                        conversation_messages.c.conversation_id
                        == request.conversation_id,
                    )
                )
            ).scalar_one() + 1
            await conn.execute(
                insert(conversation_messages).values(
                    id=uuid4(),
                    conversation_id=request.conversation_id,
                    run_id=run_id,
                    sequence=sequence,
                    role="user",
                    content=request.message,
                )
            )
            await emit_start(conn, row)
            return self._view(row)

    async def get(self, owner: UUID, conversation_id: UUID, run_id: UUID) -> RunViewV1:
        async with self.engine.connect() as conn:
            row = (
                (
                    await conn.execute(
                        select(agent_runs)
                        .join(
                            conversations,
                            agent_runs.c.conversation_id == conversations.c.id,
                        )
                        .where(
                            conversations.c.account_id == owner,
                            agent_runs.c.conversation_id == conversation_id,
                            agent_runs.c.id == run_id,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
        return self._view(row)

    async def cancel(
        self, owner: UUID, conversation_id: UUID, run_id: UUID
    ) -> RunViewV1:
        async with self.engine.begin() as conn:
            conversation = (
                (
                    await conn.execute(
                        select(conversations)
                        .where(
                            conversations.c.id == conversation_id,
                            conversations.c.account_id == owner,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if conversation is None:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            run = (
                (
                    await conn.execute(
                        select(agent_runs)
                        .where(
                            agent_runs.c.id == run_id,
                            agent_runs.c.conversation_id == conversation_id,
                        )
                        .with_for_update()
                    )
                )
                .mappings()
                .one_or_none()
            )
            if run is None:
                raise RunAdmissionError("AI_CONVERSATION_NOT_FOUND")
            if run["status"] in {
                "COMPLETED",
                "WAITING_INPUT",
                "FAILED_TERMINAL",
                "CANCELLED",
            }:
                return self._view(run)
            values: dict[str, Any] = {"cancel_requested": True}
            if run["status"] == "PENDING" and run["execution_id"] is None:
                values.update(status="CANCELLED", finished_at=func.clock_timestamp())
                if conversation["active_run_id"] == run_id:
                    await conn.execute(
                        update(conversations)
                        .where(conversations.c.id == conversation_id)
                        .values(active_run_id=None)
                    )
            row = (
                (
                    await conn.execute(
                        update(agent_runs)
                        .where(agent_runs.c.id == run_id)
                        .values(**values)
                        .returning(*agent_runs.c)
                    )
                )
                .mappings()
                .one()
            )
            if row["status"] == "CANCELLED":
                await emit_terminal(conn, row, "CANCELLED", conversation["revision"])
            return self._view(row)

    async def claim_pending(self, worker_id: UUID, limit: int) -> list[ExecutionClaim]:
        if not 1 <= limit <= 32:
            raise ValueError("Invalid executor claim limit")
        await cleanup_stopped_retries(self.engine, limit)
        claims = []
        async with self.engine.begin() as conn:
            candidates = (
                (
                    await conn.execute(
                        select(conversations)
                        .join(
                            agent_runs,
                            conversations.c.active_run_id == agent_runs.c.id,
                        )
                        .where(agent_runs.c.status == "PENDING")
                        .order_by(agent_runs.c.created_at)
                        .limit(limit)
                        .with_for_update(of=conversations, skip_locked=True)
                    )
                )
                .mappings()
                .all()
            )
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            for conversation in candidates:
                run = (
                    (
                        await conn.execute(
                            select(agent_runs)
                            .where(
                                agent_runs.c.id == conversation["active_run_id"],
                                agent_runs.c.status == "PENDING",
                            )
                            .with_for_update()
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if run is None:
                    continue
                claim = ExecutionClaim(
                    run_id=run["id"],
                    conversation_id=conversation["id"],
                    execution_id=uuid4(),
                    worker_instance_id=worker_id,
                    generation=run["lease_generation"] + 1,
                )
                await conn.execute(
                    update(agent_runs)
                    .where(agent_runs.c.id == claim.run_id)
                    .values(
                        status="RUNNING",
                        execution_id=claim.execution_id,
                        worker_instance_id=worker_id,
                        lease_generation=claim.generation,
                        lease_expires_at=now + timedelta(seconds=30),
                        heartbeat_at=now,
                        started_at=now,
                        attempt_count=1,
                    )
                )
                claims.append(claim)
        if self.auto_recovery and len(claims) < limit:
            async with self.engine.connect() as conn:
                candidates = (
                    (
                        await conn.execute(
                            select(agent_runs)
                            .join(
                                conversations,
                                conversations.c.active_run_id == agent_runs.c.id,
                            )
                            .where(
                                agent_runs.c.status == "FAILED_RETRYABLE",
                                agent_runs.c.executor_stop_evidence.is_not(None),
                                agent_runs.c.cancel_requested.is_(False),
                                agent_runs.c.retry_deadline > func.clock_timestamp(),
                                agent_runs.c.attempt_count < agent_runs.c.max_attempts,
                            )
                            .order_by(agent_runs.c.created_at)
                            .limit(limit - len(claims))
                        )
                    )
                    .mappings()
                    .all()
                )
            for run in candidates:
                previous = self.ledger._identity(run)
                if previous is None:
                    logger.error("Recovery row has no execution identity")
                    continue
                try:
                    claims.append(await self.claim_recovery(previous, worker_id))
                except Exception as error:  # noqa: BLE001 -- never discard other already-committed claims
                    if (
                        not isinstance(error, RunAdmissionError)
                        or str(error) != "STALE_EXECUTION"
                    ):
                        logger.error(
                            "Recovery claim rejected (%s)", type(error).__name__
                        )
        return claims

    async def recovery_source(
        self, claim: ExecutionClaim
    ) -> tuple[CheckpointRefV1, ExecutionClaim, ExecutionStopEvidenceV1] | None:
        async with self.engine.begin() as conn:
            _, run = await self._current(conn, claim)
            if run["status"] != "RECOVERING":
                return None
            entries = run["authorized_recovery_source_refs"]
            if not entries:
                raise ValueError("Recovery claim has no authorized source")
            item = entries[-1]
            source = CheckpointRefV1.model_validate(item["source_ref"])
            previous = ExecutionClaim.model_validate(item["execution"])
            proof = ExecutionStopEvidenceV1.model_validate(item["stop_evidence"])
            if (
                previous.run_id != claim.run_id
                or previous.conversation_id != claim.conversation_id
                or previous.generation != claim.generation - 1
            ):
                raise ValueError("Recovery source claim mismatch")
            return source, previous, proof

    async def _current(self, conn, claim: ExecutionClaim):
        conversation, run = await self.ledger._lock_run(conn, claim)
        if (
            self.ledger._identity(run) != claim
            or conversation["active_run_id"] != claim.run_id
        ):
            raise RunAdmissionError("STALE_EXECUTION")
        return conversation, run

    async def heartbeat(self, claim: ExecutionClaim) -> bool:
        async with self.engine.begin() as conn:
            _, run = await self._current(conn, claim)
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            if (
                run["cancel_requested"]
                or now >= run["retry_deadline"]
                or now >= run["lease_expires_at"]
            ):
                return False
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == claim.run_id)
                .values(
                    heartbeat_at=now,
                    lease_expires_at=now + timedelta(seconds=30),
                )
            )
            return True

    async def load(
        self, claim: ExecutionClaim
    ) -> tuple[TurnRequest, ConversationHeadV1]:
        async with self.engine.begin() as conn:
            _, run = await self._current(conn, claim)
            return parse_turn_request(run["request_payload"]), ConversationHeadV1(
                conversation_id=claim.conversation_id,
                revision=run["base_revision"],
                accepted_checkpoint_ref=CheckpointRefV1.model_validate(
                    run["base_checkpoint_ref"]
                ),
            )

    async def owner(self, claim: ExecutionClaim) -> UUID:
        async with self.engine.begin() as conn:
            conversation, _ = await self._current(conn, claim)
            return conversation["account_id"]

    async def stage(self, claim: ExecutionClaim, ref: CheckpointRefV1) -> None:
        if ref.thread_id != claim.conversation_id or ref.checkpoint_ns != "":
            raise ValueError("Publication checkpoint belongs to another conversation")
        async with self.engine.begin() as conn:
            _, run = await self._current(conn, claim)
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            if (
                run["cancel_requested"]
                or now >= run["lease_expires_at"]
                or now >= run["retry_deadline"]
            ):
                raise RunAdmissionError("EXECUTION_NOT_PUBLISHABLE")
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == claim.run_id)
                .values(
                    final_checkpoint_ref=ref.model_dump(mode="json"),
                )
            )

    async def observe_checkpoint(
        self, claim: ExecutionClaim, ref: CheckpointRefV1, graph_version: str
    ) -> None:
        """Persist only an emitted, committed checkpoint of the current execution.

        This is a recovery cursor, not an accepted/public head or a stop proof.
        An unrecorded checkpoint still requires explicit lineage reconciliation.
        """
        if ref.thread_id != claim.conversation_id or ref.checkpoint_ns != "":
            raise ValueError("Working checkpoint belongs to another conversation")
        record = await self.saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": str(ref.thread_id),
                    "checkpoint_ns": "",
                    "checkpoint_id": ref.checkpoint_id,
                }
            }
        )
        if record is None:
            raise ValueError("Working checkpoint is missing")
        async with self.engine.begin() as conn:
            _, run = await self._current(conn, claim)
            request = parse_turn_request(run["request_payload"])
            base = CheckpointRefV1.model_validate(run["base_checkpoint_ref"])
            expected = {
                "app_run_id": str(claim.run_id),
                "app_execution_id": str(claim.execution_id),
                "app_generation": claim.generation,
                "app_base_checkpoint_id": base.checkpoint_id,
                "app_input_hash": canonical_hash(request.execution_input()),
                "app_graph_version": graph_version,
            }
            native = record.config.get("configurable", {})
            if (
                native.get("thread_id") != str(ref.thread_id)
                or native.get("checkpoint_ns", "") != ""
                or native.get("checkpoint_id") != ref.checkpoint_id
                or any(
                    record.metadata.get(key) != value for key, value in expected.items()
                )
            ):
                raise ValueError("Working checkpoint identity mismatch")
            if run["status"] not in {"RUNNING", "RECOVERING"}:
                raise RunAdmissionError("EXECUTION_NOT_RUNNING")
            versions = {"graph_version": graph_version}
            if run["versions"] is not None and run["versions"] != versions:
                raise ValueError("Execution graph version changed")
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == claim.run_id)
                .values(
                    last_checkpoint_ref=ref.model_dump(mode="json"),
                    versions=versions,
                )
            )

    async def _publication(self, claim: ExecutionClaim, run) -> dict[str, Any]:
        request = parse_turn_request(run["request_payload"])
        ref = CheckpointRefV1.model_validate(run["final_checkpoint_ref"])
        base = CheckpointRefV1.model_validate(run["base_checkpoint_ref"])
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(ref.thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": ref.checkpoint_id,
            }
        }
        record = await self.saver.aget_tuple(config)
        if record is None:
            raise ValueError("Publication checkpoint is missing")
        values = record.checkpoint["channel_values"]
        if isinstance(request, TurnRequestV2):
            if self.flexible_service is None:
                raise TypeError("V2 managed publication adapter is not configured")
            if ref == base:
                raise ValueError("A run must produce a new terminal checkpoint")
            await require_checkpoint_lineage(
                self.saver,
                record,
                claim,
                base,
                request,
                FLEXIBLE_GRAPH_VERSION,
                recovery_sources=run["authorized_recovery_source_refs"],
            )
            accepted_snapshot = await PostgresAcceptedCheckpointReader(self.saver).read(
                base
            )
            snapshot = revalidate_flexible_publication(
                values,
                accepted_snapshot,
                request,
                owner=await self.owner(claim),
                run_id=claim.run_id,
                base_ref=base,
                service=self.flexible_service,
            )
            assert snapshot.flexible is not None
            return dict(values) | {
                "answer": snapshot.flexible.assistant_message,
                "publication_status": "WAITING_INPUT"
                if snapshot.flexible.status == "WAITING_INPUT"
                else "COMPLETED",
            }
        if values.get("phase") != "READY_TO_PUBLISH":
            raise ValueError("Checkpoint is not ready to publish")
        # Application payload is allowlisted, not model prose promoted to facts.
        publication = AcceptedConversationSnapshot.model_validate(
            {
                "consultation": values["consultation"],
                "public_build": values["public_build"],
                "public_clarification": values.get("public_clarification"),
                "public_task_clarification": values.get("public_task_clarification"),
                "pending_tasks": values.get("pending_tasks"),
                "requirement_state": values.get("requirement_state"),
            }
        )
        if request.pending_action is not None:
            accepted = await self.saver.aget_tuple(
                {
                    "configurable": {
                        "thread_id": str(base.thread_id),
                        "checkpoint_ns": base.checkpoint_ns,
                        "checkpoint_id": base.checkpoint_id,
                    }
                }
            )
            if accepted is None:
                raise ValueError("Pending action accepted base is missing")
            before = accepted.checkpoint["channel_values"]
            expected = validate_pending_action(
                request.pending_action,
                CanonicalRequirementState.model_validate(before["requirement_state"]),
                PendingDraft.model_validate(before["pending_clarification"])
                if before.get("pending_clarification")
                else None,
                PendingTaskDraft.model_validate(before["pending_tasks"])
                if before.get("pending_tasks")
                else None,
            )
            if (
                PendingActionReceipt.model_validate(
                    values.get("pending_action_receipt")
                )
                != expected
            ):
                raise ValueError(
                    "Pending action receipt does not match accepted base/request"
                )
            if request.pending_action.action == "DISCARD":
                if (
                    values.get("outcome") != DISCARDED_OUTCOME
                    or values.get("answer") != DISCARDED_ANSWER
                    or canonical_hash(publication.requirement_state)
                    != expected.requirements_hash
                    or publication.consultation.model_dump(mode="json")
                    != before["consultation"]
                    or values["public_build"] != before.get("public_build")
                    or any(
                        values.get(field)
                        for field in (
                            "pending_clarification",
                            "pending_tasks",
                            "public_clarification",
                            "public_task_clarification",
                            "receipt",
                            "ledger",
                            "task_resolution",
                        )
                    )
                ):
                    raise ValueError(
                        "Discard must preserve accepted state and clear only pending work"
                    )
            elif values.get("outcome") == DISCARDED_OUTCOME:
                raise ValueError("Replacement cannot publish a discard outcome")
        elif (
            values.get("pending_action_receipt") is not None
            or values.get("outcome") == DISCARDED_OUTCOME
        ):
            raise ValueError("Pending action publication has no matching request")
        if publication.pending_tasks is not None and (
            values.get("outcome") != "CLARIFICATION"
            or publication.pending_tasks.resolution
            != TaskResolution.model_validate(values.get("task_resolution"))
        ):
            raise ValueError("Pending task publication is inconsistent")
        if values.get("outcome") == "BUILT" and values["public_build"] is None:
            raise ValueError("Built outcome requires a validated public build")
        if not isinstance(values.get("answer"), str) or not values["answer"].strip():
            raise ValueError("Publication answer is missing")
        if values.get("outcome") not in {
            "ANSWERED",
            "BUILT",
            "INFEASIBLE",
            "CLARIFICATION",
            "UNSUPPORTED",
            "CONFLICT",
            DISCARDED_OUTCOME,
        }:
            raise ValueError("Graph did not complete a supported outcome")
        if values["outcome"] == "ANSWERED":
            answer = ValidatedAnswer.model_validate(values["react_answer"])
            tasks = TaskResolution.model_validate(values["task_resolution"])
            if (
                answer.status != "COMPLETED"
                or answer.text != values["answer"]
                or tasks.pending
                or not tasks.requirements
                or any(t.kind == "BUILD" for t in tasks.requirements)
            ):
                raise ValueError(
                    "Read answer has incomplete or inconsistent task outcome"
                )
        tasks = (
            TaskResolution.model_validate(values["task_resolution"])
            if values.get("task_resolution")
            else TaskResolution()
        )
        saved_raw = values.get("saved_build_evidence")
        saved_ref_selected = (
            saved_raw is not None
            and bool(values.get("react_answer"))
            and SavedBuildEvidence.model_validate(saved_raw).evidence_ref
            in ValidatedAnswer.model_validate(values["react_answer"]).evidence_refs
        )
        if saved_ref_selected or any(
            t.kind == "EXPLAIN_SAVED_BUILD" for t in tasks.requirements
        ):
            saved = SavedBuildEvidence.model_validate(values["saved_build_evidence"])
            accepted = await self.saver.aget_tuple(
                {
                    "configurable": {
                        "thread_id": str(base.thread_id),
                        "checkpoint_ns": base.checkpoint_ns,
                        "checkpoint_id": base.checkpoint_id,
                    }
                }
            )
            if (
                saved.source_ref != base
                or accepted is None
                or canonical_hash(saved.snapshot)
                != canonical_hash(
                    accepted.checkpoint["channel_values"].get("public_build")
                )
                or any(
                    t.saved_build_ref != saved.evidence_ref
                    for t in tasks.requirements
                    if t.kind == "EXPLAIN_SAVED_BUILD"
                )
            ):
                raise ValueError(
                    "Saved build does not match the authorized accepted base"
                )
        await require_checkpoint_lineage(self.saver, record, claim, base, request)
        return dict(values) | {
            "publication_status": "WAITING_INPUT"
            if values["outcome"] in {"CLARIFICATION", "UNSUPPORTED", "CONFLICT"}
            else "COMPLETED"
        }

    async def stopped(
        self,
        claim: ExecutionClaim,
        evidence: ExecutionStopEvidenceV1,
        outcome: ExecutionOutcome,
    ) -> None:
        require_stopped_execution(
            ExecutionIdentityV1(
                execution_id=claim.execution_id,
                worker_instance_id=claim.worker_instance_id,
                generation=claim.generation,
            ),
            evidence,
        )
        _, loaded = await self.load(claim)
        assert loaded.accepted_checkpoint_ref is not None
        async with self.engine.connect() as conn:
            run = (
                (
                    await conn.execute(
                        select(agent_runs).where(agent_runs.c.id == claim.run_id)
                    )
                )
                .mappings()
                .one()
            )
        values = (
            await self._publication(claim, run)
            if outcome.status == "COMPLETED" and not run["cancel_requested"]
            else None
        )
        async with self.engine.begin() as conn:
            conversation, run = await self._current(conn, claim)
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            status = "CANCELLED" if outcome.status == "CANCELLED" else "FAILED_TERMINAL"
            if run["cancel_requested"]:
                values = None
                status = "CANCELLED"
            if (
                outcome.status == "FAILED"
                and outcome.failure is not None
                and classify_failure(outcome.failure) == RunStatus.FAILED_RETRYABLE
                and run["attempt_count"] < run["max_attempts"]
                and now < run["retry_deadline"]
                and not run["cancel_requested"]
                and run["last_checkpoint_ref"] is not None
            ):
                status = "FAILED_RETRYABLE"
            if values is not None:
                if (
                    conversation["revision"] != loaded.revision
                    or conversation["accepted_checkpoint_id"]
                    != loaded.accepted_checkpoint_ref.checkpoint_id
                    or run["cancel_requested"]
                    or now >= run["lease_expires_at"]
                    or now >= run["retry_deadline"]
                ):
                    raise RunAdmissionError("PUBLICATION_CONFLICT")
                status = values["publication_status"]
                sequence = (
                    await conn.execute(
                        select(func.max(conversation_messages.c.sequence)).where(
                            conversation_messages.c.conversation_id
                            == claim.conversation_id,
                        )
                    )
                ).scalar_one() + 1
                await conn.execute(
                    insert(conversation_messages).values(
                        id=uuid4(),
                        conversation_id=claim.conversation_id,
                        run_id=claim.run_id,
                        sequence=sequence,
                        role="assistant",
                        content=values["answer"],
                    )
                )
                final = CheckpointRefV1.model_validate(run["final_checkpoint_ref"])
                await conn.execute(
                    update(conversations)
                    .where(conversations.c.id == claim.conversation_id)
                    .values(
                        revision=conversation["revision"] + 1,
                        accepted_checkpoint_id=final.checkpoint_id,
                        updated_at=now,
                    )
                )
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == claim.run_id)
                .values(
                    status=status,
                    executor_stop_evidence=evidence.model_dump(mode="json"),
                    finished_at=now,
                    error={"code": "AI_RUN_FAILED"}
                    if outcome.status == "FAILED" and status != "CANCELLED"
                    else None,
                )
            )
            await emit_terminal(
                conn,
                run,
                status,
                conversation["revision"] + (1 if values is not None else 0),
                values,
            )
            if status != "FAILED_RETRYABLE":
                await conn.execute(
                    update(conversations)
                    .where(conversations.c.id == claim.conversation_id)
                    .values(active_run_id=None)
                )

    async def claim_recovery(
        self, previous: ExecutionClaim, worker_id: UUID
    ) -> ExecutionClaim:
        """Internal claim from an owning executor's stored ACK, never client proof.

        Does not start graph execution. No lease-only/process-death takeover or
        source reconciliation is enabled by this method.
        """
        async with self.engine.begin() as conn:
            conversation, run = await self._current(conn, previous)
            request = parse_turn_request(run["request_payload"])
            if not isinstance(request, TurnRequestV2) or run["cancel_requested"]:
                raise RunAdmissionError("RUN_NOT_RECOVERABLE")
            proof = ExecutionStopEvidenceV1.model_validate(
                run["executor_stop_evidence"]
            )
            require_stopped_execution(
                ExecutionIdentityV1(
                    execution_id=previous.execution_id,
                    worker_instance_id=previous.worker_instance_id,
                    generation=previous.generation,
                ),
                proof,
            )
            if proof.kind != "EXECUTOR_STOP_ACK":
                raise RunAdmissionError("EXECUTOR_STOP_REQUIRED")
            base = CheckpointRefV1.model_validate(run["base_checkpoint_ref"])
            if run["versions"] != {"graph_version": FLEXIBLE_GRAPH_VERSION}:
                raise ValueError("Recovery graph version requires explicit migration")
            source = await reconcile_stopped_checkpoint(
                self.saver,
                previous,
                base,
                request,
                FLEXIBLE_GRAPH_VERSION,
                stored=CheckpointRefV1.model_validate(run["last_checkpoint_ref"])
                if run["last_checkpoint_ref"]
                else None,
                recovery_sources=run["authorized_recovery_source_refs"],
            )
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            require_recovery_eligible(
                status=RunStatus(run["status"]),
                base_revision=run["base_revision"],
                current_revision=conversation["revision"],
                base_checkpoint_id=base.checkpoint_id,
                current_checkpoint_id=conversation["accepted_checkpoint_id"],
                versions_compatible=True,
                checkpoint_valid=True,
                attempt_count=run["attempt_count"],
                max_attempts=run["max_attempts"],
                retry_deadline=run["retry_deadline"],
                now=now,
            )
            recovered = ExecutionClaim(
                run_id=previous.run_id,
                conversation_id=previous.conversation_id,
                execution_id=uuid4(),
                worker_instance_id=worker_id,
                generation=previous.generation + 1,
            )
            sources = list(run["authorized_recovery_source_refs"])
            sources.append(
                {
                    "source_ref": source.model_dump(mode="json"),
                    "execution": previous.model_dump(mode="json"),
                    "stop_evidence": proof.model_dump(mode="json"),
                    "graph_version": FLEXIBLE_GRAPH_VERSION,
                }
            )
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == recovered.run_id)
                .values(
                    status="RECOVERING",
                    execution_id=recovered.execution_id,
                    worker_instance_id=worker_id,
                    lease_generation=recovered.generation,
                    lease_expires_at=now + timedelta(seconds=30),
                    heartbeat_at=now,
                    attempt_count=run["attempt_count"] + 1,
                    executor_stop_evidence=None,
                    finished_at=None,
                    final_checkpoint_ref=None,
                    last_checkpoint_ref=source.model_dump(mode="json"),
                    authorized_recovery_source_refs=sources,
                )
            )
            return recovered
