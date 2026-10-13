"""Bounded managed tasks. DB coordinator remains the admission authority.

No HTTP handler owns graph tasks. Every child/stream/checkpointer operation must
be awaited or registered in the supplied TaskGroup; untracked children violate
this contract. A failed drain must keep resources alive until trusted process
termination, never be converted into stop evidence.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID

from pydantic import model_validator

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.recovery_policy import RetryFailure
from ai_service.capabilities.assistant.stateful_contracts import (
    Contract,
    ExecutionStopEvidenceV1,
)

Execution = Callable[[ExecutionClaim, asyncio.TaskGroup], Awaitable[None]]
logger = logging.getLogger(__name__)


class ExecutionFailed(Exception):
    """Adapters classify known failures; unknown errors stay terminal by default."""

    def __init__(self, failure: RetryFailure):
        self.failure = failure
        super().__init__(failure.value)


class ExecutionOutcome(Contract):
    status: Literal["COMPLETED", "FAILED", "CANCELLED"]
    failure: RetryFailure | None = None

    @model_validator(mode="after")
    def failure_matches_status(self) -> "ExecutionOutcome":
        if (self.status == "FAILED") != (self.failure is not None):
            raise ValueError("Only failed executions have a failure classification")
        return self


def _failure(error: BaseException) -> RetryFailure:
    # TaskGroup wraps even one failure. Mixed/unrecognized failures conservatively
    # remain terminal; no timeout heuristic accidentally retries corrupt state.
    if isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        return _failure(error.exceptions[0])
    return (
        error.failure if isinstance(error, ExecutionFailed) else RetryFailure.UNEXPECTED
    )


class ExecutionCoordinator(Protocol):
    async def claim_pending(
        self, worker_id: UUID, limit: int
    ) -> list[ExecutionClaim]: ...

    async def heartbeat(self, claim: ExecutionClaim) -> bool: ...

    async def stopped(
        self,
        claim: ExecutionClaim,
        evidence: ExecutionStopEvidenceV1,
        outcome: ExecutionOutcome,
    ) -> None: ...


class ManagedExecutor:
    def __init__(
        self,
        coordinator: ExecutionCoordinator,
        execute: Execution,
        *,
        worker_id: UUID,
        concurrency: int = 2,
        poll_seconds: float = 1,
        heartbeat_seconds: float = 5,
    ):
        if concurrency < 1 or poll_seconds <= 0 or heartbeat_seconds <= 0:
            raise ValueError("Executor limits must be positive")
        self.coordinator, self.execute = coordinator, execute
        self.worker_id = worker_id
        self.concurrency, self.poll_seconds = concurrency, poll_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self._group: asyncio.TaskGroup | None = None
        self._scanner: asyncio.Task[None] | None = None
        self._active: dict[UUID, tuple[ExecutionClaim, asyncio.Task[None]]] = {}
        self._started: set[UUID] = set()
        self._cancel_pending: set[UUID] = set()
        self._closing = False
        self._wake = asyncio.Event()
        self.last_error: str | None = None

    async def start(self) -> None:
        if self._group is not None or self._closing:
            raise RuntimeError("Executor cannot be started twice")
        self._group = asyncio.TaskGroup()
        await self._group.__aenter__()
        self._scanner = self._group.create_task(self._scan(), name="ai-run-scan")

    def wake(self) -> None:
        """Latency hint only: durable rows, not this event, are delivery truth."""
        self._wake.set()

    def cancel(self, execution_id: UUID) -> bool:
        """Called after durable cancel_requested; not by SSE disconnect."""
        entry = self._active.get(execution_id)
        if entry is None:
            return False
        if execution_id not in self._started:
            # Cancelling a task before its first instruction skips its finally
            # block. Let it start only the drain/ACK path, never the graph.
            self._cancel_pending.add(execution_id)
        elif not entry[1].cancelling():
            entry[1].cancel()
        return True

    def cancel_run(self, run_id: UUID) -> bool:
        for execution_id, (claim, _) in list(self._active.items()):
            if claim.run_id == run_id:
                return self.cancel(execution_id)
        return False  # Other worker / claim-registration race: durable flag wins.

    async def _heartbeat(self, claim: ExecutionClaim) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            if not await self.coordinator.heartbeat(claim):
                raise RuntimeError("EXECUTION_OWNERSHIP_LOST")

    async def _run(self, claim: ExecutionClaim) -> None:
        self._started.add(claim.execution_id)
        outcome = ExecutionOutcome(status="COMPLETED")
        try:
            try:
                if self._closing or claim.execution_id in self._cancel_pending:
                    raise asyncio.CancelledError
                async with asyncio.TaskGroup() as children:
                    heartbeat = children.create_task(self._heartbeat(claim))
                    try:
                        await self.execute(claim, children)
                    finally:
                        heartbeat.cancel()
            except asyncio.CancelledError:
                outcome = ExecutionOutcome(status="CANCELLED")
            except Exception as error:  # noqa: BLE001 -- isolate run failure, redact exception content
                logger.error("Managed execution failed (%s)", type(error).__name__)
                outcome = ExecutionOutcome(status="FAILED", failure=_failure(error))
            current = asyncio.current_task()
            if (
                current is not None
                and current.cancelling()
                and outcome.status == "COMPLETED"
            ):
                outcome = ExecutionOutcome(status="CANCELLED")
            # Reaching here proves our execution and registered children have
            # finished. Not a heartbeat timeout, run status or HTTP cancellation.
            proof = ExecutionStopEvidenceV1(
                kind="EXECUTOR_STOP_ACK",
                execution_id=claim.execution_id,
                worker_instance_id=claim.worker_instance_id,
                generation=claim.generation,
                stopped_at=datetime.now(UTC),
                checkpoint_writes_drained=True,
                proof_ref=f"executor:{claim.worker_instance_id}:{claim.execution_id}:{claim.generation}",
            )
            try:
                await self.coordinator.stopped(claim, proof, outcome)
            except Exception as error:  # noqa: BLE001 -- retain DB busy on any ACK failure
                logger.error("Stop ACK persistence failed (%s)", type(error).__name__)
                # Keep DB busy when ACK persistence fails. No fake local release.
                self.last_error = "STOP_ACK_NOT_PERSISTED"
        finally:
            self._active.pop(claim.execution_id, None)
            self._started.discard(claim.execution_id)
            self._cancel_pending.discard(claim.execution_id)
            self.wake()

    async def _scan(self) -> None:
        assert self._group is not None
        while not self._closing:
            self._wake.clear()
            available = self.concurrency - len(self._active)
            if available:
                try:
                    claims = await self.coordinator.claim_pending(
                        self.worker_id, available
                    )
                    if len(claims) > available:
                        raise RuntimeError("CLAIM_BATCH_EXCEEDS_LIMIT")
                    for claim in claims:
                        if claim.worker_instance_id != self.worker_id or any(
                            prior.conversation_id == claim.conversation_id
                            for prior, _ in self._active.values()
                        ):
                            raise RuntimeError("COORDINATOR_OWNERSHIP_INVARIANT")
                        task = self._group.create_task(
                            self._run(claim), name=f"ai-execution:{claim.execution_id}"
                        )
                        self._active[claim.execution_id] = (claim, task)
                        if self._closing:
                            self.cancel(claim.execution_id)
                except Exception as error:  # noqa: BLE001 -- supervisor survives store/claim failures
                    logger.error("Execution claim failed (%s)", type(error).__name__)
                    self.last_error = "CLAIM_FAILED"
            if not self._closing:
                try:
                    await asyncio.wait_for(self._wake.wait(), self.poll_seconds)
                except TimeoutError:
                    pass

    async def stop(self, *, grace_seconds: float) -> bool:
        if grace_seconds < 0:
            raise ValueError("Grace must be non-negative")
        self._closing = True
        self.wake()
        if self._group is None:
            return True
        for execution_id in list(self._active):
            self.cancel(execution_id)
        deadline = asyncio.get_running_loop().time() + grace_seconds
        while True:
            tasks = [task for _, task in self._active.values()]
            if self._scanner is not None and not self._scanner.done():
                tasks.append(self._scanner)
            if not tasks:
                break
            _, pending = await asyncio.wait(
                tasks, timeout=max(0, deadline - asyncio.get_running_loop().time())
            )
            if pending:
                return False  # No ACK, no resource close; shutdown has not drained.
            # A claim in flight may have committed during shutdown. Re-read the
            # registry and drain those tasks too before closing shared resources.
        await self._group.__aexit__(None, None, None)
        self._group = None
        return True
