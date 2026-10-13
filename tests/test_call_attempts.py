import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ai_service.application.call_attempts import (
    AttemptBoundary,
    AttemptKind,
    AttemptRejected,
    AttemptReservation,
    AttemptStatus,
    CallBudget,
    ExecutionClaim,
    require_attempt_admission,
)


def claim():
    return ExecutionClaim(
        run_id=uuid4(),
        conversation_id=uuid4(),
        execution_id=uuid4(),
        worker_instance_id=uuid4(),
        generation=1,
    )


def test_attempt_admission_counts_unknown_and_rejects_stale_cancelled_expired():
    owner = claim()
    now = datetime.now(UTC)
    budget = CallBudget(
        model_limit=2, tool_io_limit=3, deadline=now + timedelta(minutes=1)
    )
    arguments = {
        "claim": owner,
        "current": owner,
        "status": "RUNNING",
        "active_run_id": owner.run_id,
        "cancel_requested": False,
        "lease_expires_at": budget.deadline,
        "budget": budget,
        "kind": AttemptKind.MODEL,
        "used": 1,
        "now": now,
    }
    require_attempt_admission(**arguments)
    for update, reason in [
        ({"used": 2}, "CALL_BUDGET_EXHAUSTED"),
        ({"current": owner.model_copy(update={"generation": 2})}, "STALE_EXECUTION"),
        ({"active_run_id": None}, "STALE_EXECUTION"),
        ({"cancel_requested": True}, "CANCEL_REQUESTED"),
        ({"now": budget.deadline}, "EXECUTION_DEADLINE"),
        ({"status": "COMPLETED"}, "STALE_EXECUTION"),
    ]:
        with pytest.raises(AttemptRejected, match=reason):
            require_attempt_admission(**(arguments | update))


class Ledger:
    def __init__(self, reject=False):
        self.reject = reject
        self.records = []
        self.finished = []

    async def reserve(self, execution, kind, operation_key, request_hash):
        if self.reject:
            raise AttemptRejected("CALL_BUDGET_EXHAUSTED")
        receipt = AttemptReservation(
            id=uuid4(),
            claim=execution,
            kind=kind,
            operation_key=operation_key,
            request_hash=request_hash,
            attempt_number=len(self.records) + 1,
        )
        self.records.append(receipt)
        return receipt

    async def finish(self, reservation, status):
        self.finished.append((reservation.id, status))


@pytest.mark.asyncio
async def test_no_dispatch_without_durable_reservation_and_each_retry_uses_new_slot():
    calls = 0

    async def remote():
        nonlocal calls
        calls += 1
        raise RuntimeError("provider timeout")

    blocked = AttemptBoundary(Ledger(reject=True), claim())
    with pytest.raises(AttemptRejected):
        await blocked.invoke(AttemptKind.MODEL, "extract", "0" * 64, remote)
    assert calls == 0
    ledger = Ledger()
    boundary = AttemptBoundary(ledger, claim())
    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider timeout"):
            await boundary.invoke(AttemptKind.MODEL, "extract", "0" * 64, remote)
    assert calls == 2 and len(ledger.records) == 2
    assert ledger.records[0].id != ledger.records[1].id
    assert [status for _, status in ledger.finished] == [AttemptStatus.FAILED] * 2


@pytest.mark.asyncio
async def test_cancel_does_not_refund_or_mark_unknown_remote_cost_as_success():
    ledger = Ledger()
    boundary = AttemptBoundary(ledger, claim())

    async def cancelled():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await boundary.invoke(AttemptKind.TOOL_IO, "catalog", "0" * 64, cancelled)
    assert len(ledger.records) == 1
    assert ledger.finished == []  # RESERVED remains counted; recovery settles UNKNOWN.
