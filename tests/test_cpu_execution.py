import asyncio
import threading

import pytest
from test_call_attempts import claim
from test_managed_executor import Coordinator

from ai_service.application.cpu_work import run_cpu
from ai_service.infrastructure.execution.supervisor import ManagedExecutor


@pytest.mark.asyncio
async def test_cpu_execution_keeps_event_loop_responsive():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    def work():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(2)
        return 42

    task = asyncio.create_task(run_cpu(work))
    try:
        await asyncio.wait_for(started.wait(), 1)
        # If the operation occupies the event loop this instruction cannot run
        # before the worker's timeout. No timing/performance threshold assertion.
        assert not task.done()
        release.set()
        assert await task == 42
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_cancelled_cpu_work_must_drain_before_executor_stop_ack():
    execution = claim()
    store = Coordinator(execution)
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    def work():
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(2)
        finally:
            finished.set()

    async def execute(current, children):
        await run_cpu(work)

    executor = ManagedExecutor(
        store, execute, worker_id=execution.worker_instance_id, poll_seconds=0.01
    )
    await executor.start()
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert not await executor.stop(grace_seconds=0.01)
        assert not store.acks and not finished.is_set()
        release.set()
        assert await executor.stop(grace_seconds=1)
        assert finished.is_set()
        assert len(store.acks) == 1 and store.acks[0][2].status == "CANCELLED"
    finally:
        release.set()
        await executor.stop(grace_seconds=1)
