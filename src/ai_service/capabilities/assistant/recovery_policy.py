"""Pure P4 eligibility rules, NOT thread admission or DB finalization.

The coordinator must authenticate/verify stop proofs, read native checkpoints,
and recheck these conditions atomically with its claim/finalize transaction.
Neither an untrusted proof DTO nor a passing function grants a thread lease.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    ExecutionStopEvidenceV1,
    PositiveInt,
    RunStatus,
)


class RecoveryRejected(ValueError):
    """A deterministic recovery prerequisite has not been satisfied."""


class ExecutionIdentityV1(Contract):
    execution_id: UUID
    worker_instance_id: UUID
    generation: PositiveInt


class RetryFailure(StrEnum):
    UPSTREAM_TIMEOUT = "UPSTREAM_TIMEOUT"
    UPSTREAM_RATE_LIMIT = "UPSTREAM_RATE_LIMIT"
    UPSTREAM_SERVER_ERROR = "UPSTREAM_SERVER_ERROR"
    EXECUTION_INTERRUPTED = "EXECUTION_INTERRUPTED"
    AUTHORIZATION = "AUTHORIZATION"
    INVALID_INPUT = "INVALID_INPUT"
    CORRUPTED_CHECKPOINT = "CORRUPTED_CHECKPOINT"
    LINEAGE_MISMATCH = "LINEAGE_MISMATCH"
    UNEXPECTED = "UNEXPECTED"


def classify_failure(failure: RetryFailure) -> RunStatus:
    retryable = {
        RetryFailure.UPSTREAM_TIMEOUT,
        RetryFailure.UPSTREAM_RATE_LIMIT,
        RetryFailure.UPSTREAM_SERVER_ERROR,
        RetryFailure.EXECUTION_INTERRUPTED,
    }
    return (
        RunStatus.FAILED_RETRYABLE
        if failure in retryable
        else RunStatus.FAILED_TERMINAL
    )


def require_stopped_execution(
    previous: ExecutionIdentityV1,
    evidence: ExecutionStopEvidenceV1 | None,
) -> None:
    """Match already verified coordinator evidence; run status is irrelevant."""
    previous = ExecutionIdentityV1.model_validate(previous.model_dump())
    if evidence is None:
        raise RecoveryRejected("EXECUTION_STOP_REQUIRED")
    checked = ExecutionStopEvidenceV1.model_validate(evidence.model_dump())
    if any(
        getattr(checked, field) != getattr(previous, field)
        for field in ExecutionIdentityV1.model_fields
    ):
        raise RecoveryRejected("EXECUTION_STOP_IDENTITY_MISMATCH")


def require_recovery_eligible(
    *,
    status: RunStatus,
    base_revision: int,
    current_revision: int,
    base_checkpoint_id: str | None,
    current_checkpoint_id: str | None,
    versions_compatible: bool,
    checkpoint_valid: bool,
    attempt_count: int,
    max_attempts: int,
    retry_deadline: datetime,
    now: datetime,
) -> None:
    """Evaluate stored retry budget/head, separately from execution-stop proof.

    RUNNING is eligible only as a process-death recovery candidate; the caller
    still must satisfy require_stopped_execution and atomic coordinator gates.
    """
    if status not in {RunStatus.FAILED_RETRYABLE, RunStatus.RUNNING}:
        raise RecoveryRejected("RUN_NOT_RECOVERABLE")
    if base_revision != current_revision or base_checkpoint_id != current_checkpoint_id:
        raise RecoveryRejected("ACCEPTED_HEAD_CHANGED")
    if not versions_compatible or not checkpoint_valid:
        raise RecoveryRejected("RECOVERY_CONTEXT_INVALID")
    if retry_deadline.utcoffset() is None or now.utcoffset() is None:
        raise RecoveryRejected("RETRY_TIMEZONE_REQUIRED")
    if (
        attempt_count < 1
        or max_attempts < 1
        or attempt_count >= max_attempts
        or now >= retry_deadline
    ):
        raise RecoveryRejected("RETRY_BUDGET_EXHAUSTED")
