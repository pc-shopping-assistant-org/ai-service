"""Native reservation serialization/restart gate; requires Alembic head."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import create_async_engine

from ai_service.application.call_attempts import (
    AttemptKind,
    AttemptRejected,
    ExecutionClaim,
)
from ai_service.infrastructure.persistence.call_attempts import PostgresAttemptLedger
from ai_service.infrastructure.persistence.schema import (
    agent_call_attempts,
    agent_call_budgets,
    agent_runs,
    conversations,
)


@pytest.mark.asyncio
async def test_reservations_serialize_and_remain_counted_after_reopen(postgres_dsn):
    dsn = postgres_dsn
    engine = create_async_engine(
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
        hide_parameters=True,
        connect_args={"connect_timeout": 3},
    )
    owner = ExecutionClaim(
        run_id=uuid4(),
        conversation_id=uuid4(),
        execution_id=uuid4(),
        worker_instance_id=uuid4(),
        generation=1,
    )
    deadline = datetime.now(UTC) + timedelta(minutes=5)
    inserted = False
    try:
        async with engine.begin() as connection:
            await connection.execute(
                insert(conversations).values(
                    id=owner.conversation_id, account_id=uuid4()
                )
            )
            await connection.execute(
                insert(agent_runs).values(
                    id=owner.run_id,
                    conversation_id=owner.conversation_id,
                    request_id=uuid4(),
                    request_hash="0" * 64,
                    request_payload={},
                    kind="CHAT",
                    status="RUNNING",
                    base_revision=0,
                    execution_id=owner.execution_id,
                    worker_instance_id=owner.worker_instance_id,
                    lease_generation=1,
                    lease_expires_at=deadline,
                    attempt_count=1,
                    max_attempts=2,
                    retry_policy_version="gate-v1",
                    retry_deadline=deadline,
                )
            )
            await connection.execute(
                conversations.update()
                .where(conversations.c.id == owner.conversation_id)
                .values(active_run_id=owner.run_id)
            )
            await connection.execute(
                insert(agent_call_budgets).values(
                    run_id=owner.run_id,
                    model_limit=1,
                    tool_io_limit=1,
                    deadline=deadline,
                )
            )
        inserted = True
        ledger = PostgresAttemptLedger(engine)
        results = await asyncio.gather(
            *[
                ledger.reserve(owner, AttemptKind.MODEL, "extract", "0" * 64)
                for _ in range(2)
            ],
            return_exceptions=True,
        )
        assert sum(isinstance(result, AttemptRejected) for result in results) == 1
        await engine.dispose()
        # Prior call might have reached a provider. RESERVED still consumes cap.
        with pytest.raises(AttemptRejected, match="CALL_BUDGET_EXHAUSTED"):
            await PostgresAttemptLedger(engine).reserve(
                owner, AttemptKind.MODEL, "extract", "0" * 64
            )
        with pytest.raises(AttemptRejected, match="STALE_EXECUTION"):
            await ledger.reserve(
                owner.model_copy(update={"generation": 2}),
                AttemptKind.TOOL_IO,
                "catalog",
                "0" * 64,
            )
        async with engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        select(agent_call_attempts).where(
                            agent_call_attempts.c.run_id == owner.run_id
                        )
                    )
                )
                .mappings()
                .all()
            )
            assert len(rows) == 1 and rows[0]["status"] == "RESERVED"
    finally:
        try:
            if inserted:
                async with engine.begin() as connection:
                    await connection.execute(
                        delete(conversations).where(
                            conversations.c.id == owner.conversation_id
                        )
                    )
        finally:
            await engine.dispose()
