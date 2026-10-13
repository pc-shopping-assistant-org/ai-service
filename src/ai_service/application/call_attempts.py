"""Conservative attempt accounting, separate from checkpoint tool outcomes.

All provider/backend retries must use invoke independently, with SDK automatic
retries disabled. This bounds dispatched attempts, never provider billing.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field

from ai_service.capabilities.assistant.recovery_policy import ExecutionIdentityV1
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    NonNegativeInt,
    PositiveInt,
    Text,
)


class AttemptKind(StrEnum):
    MODEL = "MODEL"
    TOOL_IO = "TOOL_IO"


class AttemptStatus(StrEnum):
    RESERVED = "RESERVED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class AttemptRejected(ValueError):
    """No permission to dispatch an external attempt."""


class ExecutionClaim(ExecutionIdentityV1):
    run_id: UUID
    conversation_id: UUID


class CallBudget(Contract):
    model_limit: NonNegativeInt
    tool_io_limit: NonNegativeInt
    deadline: AwareDatetime


class AttemptReservation(Contract):
    id: UUID
    claim: ExecutionClaim
    kind: AttemptKind
    operation_key: Text = Field(max_length=200)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_number: PositiveInt


def require_attempt_admission(
    *,
    claim: ExecutionClaim,
    current: ExecutionClaim | None,
    status: str,
    active_run_id: UUID | None,
    cancel_requested: bool,
    lease_expires_at: datetime | None,
    budget: CallBudget,
    kind: AttemptKind,
    used: int,
    now: datetime,
) -> None:
    """Evaluate persisted data while its conversation/run locks are held."""
    if now.utcoffset() is None:
        raise AttemptRejected("UTC_TIME_REQUIRED")
    if (
        current != claim
        or active_run_id != claim.run_id
        or status not in {"RUNNING", "RECOVERING"}
    ):
        raise AttemptRejected("STALE_EXECUTION")
    if cancel_requested:
        raise AttemptRejected("CANCEL_REQUESTED")
    if now >= budget.deadline:
        raise AttemptRejected("EXECUTION_DEADLINE")
    if lease_expires_at is None or now >= lease_expires_at:
        raise AttemptRejected("EXECUTION_LEASE_EXPIRED")
    limit = budget.model_limit if kind == AttemptKind.MODEL else budget.tool_io_limit
    if used < 0 or used >= limit:
        raise AttemptRejected("CALL_BUDGET_EXHAUSTED")


class AttemptLedger(Protocol):
    async def reserve(
        self,
        claim: ExecutionClaim,
        kind: AttemptKind,
        operation_key: str,
        request_hash: str,
    ) -> AttemptReservation: ...

    async def finish(
        self,
        reservation: AttemptReservation,
        status: AttemptStatus,
    ) -> None: ...


class AttemptBoundary:
    def __init__(self, ledger: AttemptLedger, claim: ExecutionClaim):
        self.ledger = ledger
        self.claim = ExecutionClaim.model_validate(claim.model_dump())

    async def invoke[T](
        self,
        kind: AttemptKind,
        operation_key: str,
        request_hash: str,
        dispatch: Callable[[], Awaitable[T]],
    ) -> T:
        # Only a newly committed reservation can reach dispatch. Stored receipts
        # are never accepted as retry input, including after a process restart.
        receipt = await self.ledger.reserve(
            self.claim, kind, operation_key, request_hash
        )
        try:
            result = await dispatch()
        except Exception:
            await self.ledger.finish(receipt, AttemptStatus.FAILED)
            raise
        # Cancel/process death leaves RESERVED counted and eventually UNKNOWN.
        await self.ledger.finish(receipt, AttemptStatus.SUCCEEDED)
        return result
