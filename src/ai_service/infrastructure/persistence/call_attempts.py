"""PostgreSQL transaction is the authority for external-attempt caps."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ai_service.application.call_attempts import (
    AttemptKind,
    AttemptRejected,
    AttemptReservation,
    AttemptStatus,
    CallBudget,
    ExecutionClaim,
    require_attempt_admission,
)
from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.stateful_contracts import ExecutionStopEvidenceV1

from .schema import agent_call_attempts, agent_call_budgets, agent_runs, conversations


class PostgresAttemptLedger:
    def __init__(self, engine: AsyncEngine):
        self.engine = engine

    async def _lock_run(
        self, connection: AsyncConnection, claim: ExecutionClaim
    ) -> tuple[Any, Any]:
        # Conversation -> run is the coordinator lock order as well. Never use
        # status/heartbeat alone as proof of shared-thread ownership.
        conversation = (
            (
                await connection.execute(
                    select(conversations)
                    .where(conversations.c.id == claim.conversation_id)
                    .with_for_update()
                )
            )
            .mappings()
            .one_or_none()
        )
        run = (
            (
                await connection.execute(
                    select(agent_runs)
                    .where(
                        agent_runs.c.id == claim.run_id,
                        agent_runs.c.conversation_id == claim.conversation_id,
                    )
                    .with_for_update()
                )
            )
            .mappings()
            .one_or_none()
        )
        if conversation is None or run is None:
            raise AttemptRejected("STALE_EXECUTION")
        return conversation, run

    @staticmethod
    def _identity(run) -> ExecutionClaim | None:
        if (
            not run["execution_id"]
            or not run["worker_instance_id"]
            or run["lease_generation"] < 1
        ):
            return None
        return ExecutionClaim(
            run_id=run["id"],
            conversation_id=run["conversation_id"],
            execution_id=run["execution_id"],
            worker_instance_id=run["worker_instance_id"],
            generation=run["lease_generation"],
        )

    async def reserve(
        self,
        claim: ExecutionClaim,
        kind: AttemptKind,
        operation_key: str,
        request_hash: str,
    ) -> AttemptReservation:
        # Validate untrusted shapes before touching the database. The number is
        # reassigned under the run lock, never accepted from a caller/provider.
        receipt = AttemptReservation(
            id=uuid4(),
            claim=claim,
            kind=kind,
            operation_key=operation_key,
            request_hash=request_hash,
            attempt_number=1,
        )
        async with self.engine.begin() as connection:
            conversation, run = await self._lock_run(connection, claim)
            record = (
                (
                    await connection.execute(
                        select(agent_call_budgets).where(
                            agent_call_budgets.c.run_id == claim.run_id
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
            if record is None:
                raise AttemptRejected("CALL_BUDGET_NOT_CONFIGURED")
            now: datetime = (
                await connection.execute(select(func.clock_timestamp()))
            ).scalar_one()
            used = (
                await connection.execute(
                    select(func.count())
                    .select_from(agent_call_attempts)
                    .where(
                        agent_call_attempts.c.run_id == claim.run_id,
                        agent_call_attempts.c.kind == kind.value,
                    )
                )
            ).scalar_one()
            require_attempt_admission(
                claim=claim,
                current=self._identity(run),
                status=run["status"],
                active_run_id=conversation["active_run_id"],
                cancel_requested=run["cancel_requested"],
                lease_expires_at=run["lease_expires_at"],
                budget=CallBudget(
                    model_limit=record["model_limit"],
                    tool_io_limit=record["tool_io_limit"],
                    deadline=record["deadline"],
                ),
                kind=kind,
                used=used,
                now=now,
            )
            prior = (
                await connection.execute(
                    select(func.max(agent_call_attempts.c.attempt_number)).where(
                        agent_call_attempts.c.run_id == claim.run_id,
                        agent_call_attempts.c.operation_key == receipt.operation_key,
                    )
                )
            ).scalar_one()
            receipt = receipt.model_copy(update={"attempt_number": (prior or 0) + 1})
            await connection.execute(
                insert(agent_call_attempts).values(
                    id=receipt.id,
                    run_id=claim.run_id,
                    execution_id=claim.execution_id,
                    worker_instance_id=claim.worker_instance_id,
                    generation=claim.generation,
                    kind=kind.value,
                    operation_key=receipt.operation_key,
                    request_hash=receipt.request_hash,
                    attempt_number=receipt.attempt_number,
                    status="RESERVED",
                )
            )
        # Transaction must commit successfully BEFORE any remote dispatch.
        return receipt

    async def finish(
        self, reservation: AttemptReservation, status: AttemptStatus
    ) -> None:
        if status not in {AttemptStatus.SUCCEEDED, AttemptStatus.FAILED}:
            raise AttemptRejected("INVALID_ATTEMPT_FINISH")
        claim = reservation.claim
        async with self.engine.begin() as connection:
            _, run = await self._lock_run(connection, claim)
            if self._identity(run) != claim:
                raise AttemptRejected("STALE_EXECUTION")
            changed = (
                await connection.execute(
                    update(agent_call_attempts)
                    .where(
                        agent_call_attempts.c.id == reservation.id,
                        agent_call_attempts.c.run_id == claim.run_id,
                        agent_call_attempts.c.execution_id == claim.execution_id,
                        agent_call_attempts.c.worker_instance_id
                        == claim.worker_instance_id,
                        agent_call_attempts.c.generation == claim.generation,
                        agent_call_attempts.c.status == "RESERVED",
                    )
                    .values(status=status.value, finished_at=func.clock_timestamp())
                    .returning(agent_call_attempts.c.id)
                )
            ).scalar_one_or_none()
            if changed is None:
                raise AttemptRejected("ATTEMPT_ALREADY_SETTLED")

    async def settle_stopped(self, claim: ExecutionClaim) -> None:
        """Only database-stored coordinator stop evidence permits UNKNOWN cleanup.

        This does not grant takeover or free the conversation slot. Caller must
        verify/persist trusted executor evidence through the coordinator first.
        """
        async with self.engine.begin() as connection:
            _, run = await self._lock_run(connection, claim)
            if self._identity(run) != claim or run["executor_stop_evidence"] is None:
                raise AttemptRejected("EXECUTION_STOP_REQUIRED")
            evidence = ExecutionStopEvidenceV1.model_validate(
                run["executor_stop_evidence"]
            )
            require_stopped_execution(
                ExecutionIdentityV1(
                    execution_id=claim.execution_id,
                    worker_instance_id=claim.worker_instance_id,
                    generation=claim.generation,
                ),
                evidence,
            )
            await connection.execute(
                update(agent_call_attempts)
                .where(
                    agent_call_attempts.c.run_id == claim.run_id,
                    agent_call_attempts.c.execution_id == claim.execution_id,
                    agent_call_attempts.c.worker_instance_id
                    == claim.worker_instance_id,
                    agent_call_attempts.c.generation == claim.generation,
                    agent_call_attempts.c.status == "RESERVED",
                )
                .values(status="UNKNOWN", finished_at=func.clock_timestamp())
            )
