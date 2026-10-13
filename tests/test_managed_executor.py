import asyncio

import pytest
from test_call_attempts import claim

from ai_service.infrastructure.execution.supervisor import ManagedExecutor


class Coordinator:
    def __init__(self, execution):
        self.pending = [execution]
        self.acks = []

    async def claim_pending(self, worker_id, limit):
        rows, self.pending = self.pending[:limit], self.pending[limit:]
        return rows

    async def heartbeat(self, execution):
        return True

    async def stopped(self, execution, evidence, outcome):
        self.acks.append((execution, evidence, outcome))


@pytest.mark.asyncio
async def test_subscriber_disconnect_does_not_cancel_execution_and_children_drain_before_ack():
    execution = claim()
    started, release, child_drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
    coordinator = Coordinator(execution)

    async def run(current, tasks):
        assert current == execution

        async def child():
            try:
                await release.wait()
            finally:
                child_drained.set()

        tasks.create_task(child())
        started.set()
        await release.wait()

    executor = ManagedExecutor(
        coordinator,
        run,
        worker_id=execution.worker_instance_id,
        poll_seconds=0.01,
        heartbeat_seconds=1,
    )
    await executor.start()
    try:
        await asyncio.wait_for(started.wait(), 1)

        async def subscriber():
            await asyncio.Event().wait()

        subscription = asyncio.create_task(subscriber())
        subscription.cancel()
        with pytest.raises(asyncio.CancelledError):
            await subscription
        assert not child_drained.is_set() and not coordinator.acks
        assert await executor.stop(grace_seconds=1)
        assert child_drained.is_set() and len(coordinator.acks) == 1
        proof = coordinator.acks[0][1]
        assert proof.execution_id == execution.execution_id
        assert (
            proof.generation == execution.generation
            and proof.checkpoint_writes_drained is True
        )
    finally:
        release.set()
        await executor.stop(grace_seconds=1)


@pytest.mark.asyncio
async def test_timeout_is_not_stop_evidence_for_uncooperative_child():
    started, release = asyncio.Event(), asyncio.Event()
    execution = claim()
    coordinator = Coordinator(execution)

    async def run(current, tasks):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()  # provider cleanup has not stopped yet

    executor = ManagedExecutor(
        coordinator,
        run,
        worker_id=execution.worker_instance_id,
        poll_seconds=0.01,
        heartbeat_seconds=1,
    )
    await executor.start()
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not await executor.stop(grace_seconds=0.01)
        assert coordinator.acks == []
        release.set()
        assert await executor.stop(grace_seconds=1)
        assert len(coordinator.acks) == 1
    finally:
        release.set()
        await executor.stop(grace_seconds=1)


@pytest.mark.asyncio
async def test_execution_failure_does_not_kill_supervisor_or_publish_success():
    first = claim()
    coordinator = Coordinator(first)
    completed = asyncio.Event()

    async def fail(current, tasks):
        completed.set()
        raise ValueError("invalid graph output")

    executor = ManagedExecutor(
        coordinator,
        fail,
        worker_id=first.worker_instance_id,
        poll_seconds=0.01,
        heartbeat_seconds=1,
    )
    await executor.start()
    await asyncio.wait_for(completed.wait(), 1)
    assert await executor.stop(grace_seconds=1)
    assert len(coordinator.acks) == 1
    assert coordinator.acks[0][2].status == "FAILED"
    assert coordinator.acks[0][2].failure == "UNEXPECTED"


@pytest.mark.asyncio
async def test_shutdown_during_claim_commit_drains_claim_without_starting_graph():
    execution = claim()
    committed, release = asyncio.Event(), asyncio.Event()

    class ClaimInFlight(Coordinator):
        async def claim_pending(self, worker_id, limit):
            committed.set()
            await release.wait()
            return await super().claim_pending(worker_id, limit)

    coordinator = ClaimInFlight(execution)

    async def forbidden(current, tasks):
        raise AssertionError("closed executor must not start graph")

    executor = ManagedExecutor(
        coordinator, forbidden, worker_id=execution.worker_instance_id
    )
    await executor.start()
    await asyncio.wait_for(committed.wait(), 1)
    stopping = asyncio.create_task(executor.stop(grace_seconds=1))
    await asyncio.sleep(0)
    release.set()
    assert await stopping
    assert len(coordinator.acks) == 1 and coordinator.acks[0][2].status == "CANCELLED"


@pytest.mark.asyncio
async def test_retryable_failure_keeps_classification_separate_from_stop_evidence():
    from ai_service.capabilities.assistant.recovery_policy import (
        RetryFailure,
        classify_failure,
    )
    from ai_service.infrastructure.execution.supervisor import ExecutionFailed

    execution = claim()
    coordinator = Coordinator(execution)
    failed = asyncio.Event()

    async def timeout(current, tasks):
        failed.set()
        raise ExecutionFailed(RetryFailure.UPSTREAM_TIMEOUT)

    executor = ManagedExecutor(
        coordinator, timeout, worker_id=execution.worker_instance_id
    )
    await executor.start()
    await asyncio.wait_for(failed.wait(), 1)
    assert await executor.stop(grace_seconds=1)
    outcome = coordinator.acks[0][2]
    assert outcome.status == "FAILED"
    assert classify_failure(outcome.failure) == "FAILED_RETRYABLE"
    assert coordinator.acks[0][1].checkpoint_writes_drained is True
