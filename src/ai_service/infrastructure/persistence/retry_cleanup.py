"""Bounded cleanup of stopped retries; deadlines never prove execution stopped."""

import logging
from datetime import datetime

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.stateful_contracts import ExecutionStopEvidenceV1

from .schema import agent_runs, conversations

logger = logging.getLogger(__name__)


async def cleanup_stopped_retries(engine: AsyncEngine, limit: int) -> None:
    if not 1 <= limit <= 32:
        raise ValueError("Invalid cleanup limit")
    async with engine.begin() as conn:
        owners = (
            (
                await conn.execute(
                    select(conversations)
                    .join(
                        agent_runs,
                        conversations.c.active_run_id == agent_runs.c.id,
                    )
                    .where(
                        agent_runs.c.kind == "CHAT",
                        agent_runs.c.status == "FAILED_RETRYABLE",
                        or_(
                            agent_runs.c.retry_deadline <= func.clock_timestamp(),
                            agent_runs.c.attempt_count >= agent_runs.c.max_attempts,
                            agent_runs.c.cancel_requested.is_(True),
                        ),
                    )
                    .order_by(agent_runs.c.created_at)
                    .limit(limit)
                    .with_for_update(of=conversations, skip_locked=True)
                )
            )
            .mappings()
            .all()
        )
        for owner in owners:
            run = (
                (
                    await conn.execute(
                        select(agent_runs)
                        .where(agent_runs.c.id == owner["active_run_id"])
                        .with_for_update()
                    )
                )
                .mappings()
                .one()
            )
            now: datetime = (
                await conn.execute(select(func.clock_timestamp()))
            ).scalar_one()
            if run["status"] != "FAILED_RETRYABLE" or (
                not run["cancel_requested"]
                and now < run["retry_deadline"]
                and run["attempt_count"] < run["max_attempts"]
            ):
                continue
            try:
                proof = ExecutionStopEvidenceV1.model_validate(
                    run["executor_stop_evidence"]
                )
                require_stopped_execution(
                    ExecutionIdentityV1(
                        execution_id=run["execution_id"],
                        worker_instance_id=run["worker_instance_id"],
                        generation=run["lease_generation"],
                    ),
                    proof,
                )
                if proof.kind != "EXECUTOR_STOP_ACK":
                    raise ValueError("Unsupported stop authority")
            except ValueError:
                logger.error("Stopped retry cleanup rejected invalid stop evidence")
                continue
            await conn.execute(
                update(agent_runs)
                .where(agent_runs.c.id == run["id"])
                .values(
                    status="CANCELLED"
                    if run["cancel_requested"]
                    else "FAILED_TERMINAL",
                    finished_at=now,
                    error=None
                    if run["cancel_requested"]
                    else {"code": "AI_RETRY_BUDGET_EXHAUSTED"},
                )
            )
            await conn.execute(
                update(conversations)
                .where(conversations.c.id == owner["id"])
                .values(active_run_id=None, updated_at=now)
            )
