"""Managed main-path execution; request/subscriber lifetimes do not own this work."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.persistence.runs import PostgresRunCoordinator


class ConversationExecution:
    def __init__(
        self,
        coordinator: PostgresRunCoordinator,
        graph_factory: Callable[
            [ExecutionClaim],
            DurableConversationGraph | Awaitable[DurableConversationGraph],
        ],
        graph_version: str,
    ) -> None:
        self.coordinator, self.graph_factory, self.graph_version = (
            coordinator,
            graph_factory,
            graph_version,
        )

    async def __call__(
        self, claim: ExecutionClaim, children: asyncio.TaskGroup
    ) -> None:
        request, head = await self.coordinator.load(claim)
        graph = self.graph_factory(claim)
        if inspect.isawaitable(graph):
            graph = await graph

        async def observed(ref):
            await self.coordinator.observe_checkpoint(claim, ref, self.graph_version)

        recovery = await self.coordinator.recovery_source(claim)
        if recovery:
            source, previous, evidence = recovery
            ref = await graph.resume(
                head,
                source,
                previous,
                claim,
                request.execution_input(),
                self.graph_version,
                evidence=evidence,
                checkpoint_sink=observed,
            )
        else:
            ref = await graph.new_turn(
                head,
                claim.run_id,
                request.execution_input(),
                self.graph_version,
                execution=claim,
                checkpoint_sink=observed,
            )
        await self.coordinator.stage(claim, ref)
        # Only ManagedExecutor's drained stop ACK can finalize and publish.
