"""Awaited CPU work with cancellation drain, never a detached thread.

The managed executor bounds concurrent runs; each serial graph dispatches one
operation at a time. asyncio owns the standard bounded thread pool. This is not
process isolation or immediate termination of Python computation.
"""

import asyncio
from collections.abc import Callable


async def run_cpu[T](operation: Callable[[], T]) -> T:
    future = asyncio.get_running_loop().run_in_executor(None, operation)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # Cancelling the await cannot stop a running thread. Keep the execution
        # alive until the operation actually returns; only then may its parent
        # provide a stop ACK or release the shared conversation thread.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 -- cancelled work failure is consumed, not published
                break
        if not future.cancelled():
            future.exception()
        raise
