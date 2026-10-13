from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    RecoveryRejected,
    RetryFailure,
    classify_failure,
    require_recovery_eligible,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    ExecutionStopEvidenceV1,
    RunStatus,
)


def evidence(identity):
    return ExecutionStopEvidenceV1(
        kind="EXECUTOR_STOP_ACK",
        **identity.model_dump(),
        stopped_at=datetime.now(UTC),
        checkpoint_writes_drained=True,
        proof_ref="executor/joined-and-drained",
    )


def test_terminal_status_or_lease_expiry_is_not_stop_evidence():
    identity = ExecutionIdentityV1(
        execution_id=uuid4(), worker_instance_id=uuid4(), generation=1
    )
    with pytest.raises(RecoveryRejected):
        require_stopped_execution(identity, None)
    stopped = evidence(identity)
    require_stopped_execution(identity, stopped)
    wrong = stopped.model_copy(update={"worker_instance_id": uuid4()})
    with pytest.raises(RecoveryRejected):
        require_stopped_execution(identity, wrong)


def test_only_retryable_run_with_unchanged_head_versions_and_budget_can_resume():
    now = datetime.now(UTC)
    args = {
        "status": RunStatus.FAILED_RETRYABLE,
        "base_revision": 2,
        "current_revision": 2,
        "base_checkpoint_id": "accepted",
        "current_checkpoint_id": "accepted",
        "versions_compatible": True,
        "checkpoint_valid": True,
        "attempt_count": 1,
        "max_attempts": 3,
        "retry_deadline": now + timedelta(minutes=5),
        "now": now,
    }
    require_recovery_eligible(**args)
    for override in [
        {"status": RunStatus.FAILED_TERMINAL},
        {"status": RunStatus.CANCELLED},
        {"status": RunStatus.COMPLETED},
        {"current_checkpoint_id": "other-branch"},
        {"current_revision": 3},
        {"versions_compatible": False},
        {"checkpoint_valid": False},
        {"attempt_count": 3},
        {"retry_deadline": now},
    ]:
        with pytest.raises(RecoveryRejected):
            require_recovery_eligible(**(args | override))


def test_failure_allowlist_does_not_retry_auth_corruption_or_unexpected_bugs():
    assert classify_failure(RetryFailure.UPSTREAM_TIMEOUT) == RunStatus.FAILED_RETRYABLE
    assert (
        classify_failure(RetryFailure.UPSTREAM_RATE_LIMIT) == RunStatus.FAILED_RETRYABLE
    )
    assert (
        classify_failure(RetryFailure.UPSTREAM_SERVER_ERROR)
        == RunStatus.FAILED_RETRYABLE
    )
    for failure in [
        RetryFailure.AUTHORIZATION,
        RetryFailure.INVALID_INPUT,
        RetryFailure.CORRUPTED_CHECKPOINT,
        RetryFailure.LINEAGE_MISMATCH,
        RetryFailure.UNEXPECTED,
    ]:
        assert classify_failure(failure) == RunStatus.FAILED_TERMINAL
