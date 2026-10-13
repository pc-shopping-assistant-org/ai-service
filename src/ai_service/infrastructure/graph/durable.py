"""Native accepted-head invocation. Admission/fencing/publication belong to P4.

Internal adapter only: callers MUST hold the exclusive conversation execution
slot, including stop/drain evidence before takeover. Returning a ref is NOT
permission to publish it. Never reconstruct execution by copying snapshot values.
"""

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot

from ai_service.application.call_attempts import ExecutionClaim
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.recovery_policy import (
    ExecutionIdentityV1,
    require_stopped_execution,
)
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    ConversationHeadV1,
    ExecutionStopEvidenceV1,
)


class DurableConversationGraph:
    def __init__(
        self,
        graph: CompiledStateGraph,
        *,
        accepted_versions: set[str] | None = None,
    ) -> None:
        self.graph = graph
        if accepted_versions is not None and not accepted_versions:
            raise ValueError("Supported checkpoint versions cannot be empty")
        # None preserves existing V1 policy. New runtimes must declare compatible
        # bases explicitly; declaring compatibility is not an implicit migration.
        self.accepted_versions = (
            frozenset(accepted_versions) if accepted_versions is not None else None
        )

    @staticmethod
    def config(ref: CheckpointRefV1) -> RunnableConfig:
        if ref.checkpoint_ns != "":
            raise ValueError("Only root conversation checkpoints are supported")
        return {
            "configurable": {
                "thread_id": str(ref.thread_id),
                "checkpoint_ns": "",
                "checkpoint_id": ref.checkpoint_id,
            }
        }

    async def read(self, ref: CheckpointRefV1) -> StateSnapshot:
        snapshot = await self.graph.aget_state(self.config(ref))
        if (
            snapshot.config.get("configurable", {}).get("checkpoint_id")
            != ref.checkpoint_id
        ):
            raise ValueError("Checkpoint not found")
        return snapshot

    @staticmethod
    def require_supported_base(
        metadata: dict, head: ConversationHeadV1, versions: set[str] | frozenset[str]
    ) -> None:
        initial_root = (
            metadata.get("app_graph_version") == "conversation-root-v1"
            and metadata.get("app_checkpoint_kind") == "initial_root"
            and metadata.get("app_initialization_id") == str(head.conversation_id)
            and head.revision == 0
        )
        if not initial_root and metadata.get("app_graph_version") not in versions:
            raise ValueError("Accepted checkpoint version requires explicit migration")

    async def new_turn(
        self,
        head: ConversationHeadV1,
        run_id: UUID,
        inputs: dict[str, Any],
        graph_version: str,
        *,
        execution: ExecutionClaim | None = None,
        checkpoint_sink: Callable[[CheckpointRefV1], Awaitable[None]] | None = None,
    ) -> CheckpointRefV1:
        if (
            self.accepted_versions is not None
            and graph_version not in self.accepted_versions
        ):
            raise ValueError("Execution graph version is not supported")
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(head.conversation_id),
                "checkpoint_ns": "",
            }
        }
        base = head.accepted_checkpoint_ref
        if base is not None:
            snapshot = await self.read(base)
            if snapshot.next or snapshot.tasks:
                raise ValueError("New turn requires a terminal accepted head")
            if self.accepted_versions is not None:
                self.require_supported_base(
                    dict(snapshot.metadata or {}), head, self.accepted_versions
                )
            config = self.config(base)
        else:
            if self.accepted_versions is not None:
                raise ValueError(
                    "Versioned execution requires an initialized accepted root"
                )
            # Diagnostic existence check only, never a new-turn base. A failed
            # first run needs P4 recovery/explicit cleanup, not implicit latest.
            existing = await self.graph.aget_state(config)
            if existing.config.get("configurable", {}).get("checkpoint_id"):
                raise ValueError(
                    "Unaccepted checkpoint exists; recover the original run first"
                )
        context: dict[str, Any] = {
            "app_run_id": str(run_id),
            "app_base_checkpoint_id": base.checkpoint_id if base else None,
            "app_input_hash": canonical_hash(inputs),
            "app_graph_version": graph_version,
        }
        if execution is not None:
            if (
                execution.run_id != run_id
                or execution.conversation_id != head.conversation_id
            ):
                raise ValueError("Execution belongs to another run")
            context.update(
                {
                    "app_execution_id": str(execution.execution_id),
                    "app_generation": execution.generation,
                }
            )
        config["metadata"] = context
        return await self._stream(
            head.conversation_id, inputs, config, context, checkpoint_sink
        )

    async def resume(
        self,
        head: ConversationHeadV1,
        source: CheckpointRefV1,
        previous: ExecutionClaim,
        execution: ExecutionClaim,
        inputs: dict[str, Any],
        graph_version: str,
        *,
        evidence: ExecutionStopEvidenceV1,
        checkpoint_sink: Callable[[CheckpointRefV1], Awaitable[None]] | None = None,
    ) -> CheckpointRefV1:
        """Internal native resume after coordinator-authorized stopped-source claim.

        Matching a proof DTO is not proof authenticity or SQL claim authority.
        The coordinator must verify both before calling; this method never grants
        an execution slot, authorizes old lineage or publishes a result.
        """
        require_stopped_execution(
            ExecutionIdentityV1(
                execution_id=previous.execution_id,
                worker_instance_id=previous.worker_instance_id,
                generation=previous.generation,
            ),
            evidence,
        )
        if (
            head.accepted_checkpoint_ref is None
            or source.thread_id != head.conversation_id
            or previous.conversation_id != head.conversation_id
            or execution.conversation_id != head.conversation_id
            or previous.run_id != execution.run_id
            or execution.generation != previous.generation + 1
            or execution.execution_id == previous.execution_id
        ):
            raise ValueError("Resume execution context mismatch")
        if (
            self.accepted_versions is not None
            and graph_version not in self.accepted_versions
        ):
            raise ValueError("Resume graph version is not supported")
        snapshot = await self.read(source)
        context = {
            "app_run_id": str(previous.run_id),
            "app_base_checkpoint_id": head.accepted_checkpoint_ref.checkpoint_id,
            "app_input_hash": canonical_hash(inputs),
            "app_graph_version": graph_version,
            "app_execution_id": str(previous.execution_id),
            "app_generation": previous.generation,
        }
        if snapshot.metadata is None or any(
            snapshot.metadata.get(key) != value for key, value in context.items()
        ):
            raise ValueError("Resume source context mismatch")
        if not snapshot.next or any(task.interrupts for task in snapshot.tasks):
            raise ValueError(
                "Resume requires failed pending work, not terminal/interrupt state"
            )
        context.update(
            app_execution_id=str(execution.execution_id),
            app_generation=execution.generation,
        )
        config = self.config(source)
        config["metadata"] = context
        return await self._stream(
            head.conversation_id, None, config, context, checkpoint_sink, source=source
        )

    async def _stream(
        self,
        conversation_id: UUID,
        inputs: dict[str, Any] | None,
        config: RunnableConfig,
        context: dict[str, Any],
        checkpoint_sink: Callable[[CheckpointRefV1], Awaitable[None]] | None,
        *,
        source: CheckpointRefV1 | None = None,
    ) -> CheckpointRefV1:
        own_ref: CheckpointRefV1 | None = None
        async for event in self.graph.astream(
            inputs, config, stream_mode="checkpoints", durability="sync"
        ):
            native = event["config"]["configurable"]
            if (
                native["thread_id"] != str(conversation_id)
                or native.get("checkpoint_ns", "") != ""
            ):
                raise ValueError("Checkpoint thread mismatch")
            own_ref = CheckpointRefV1(
                thread_id=conversation_id, checkpoint_id=native["checkpoint_id"]
            )
            if own_ref == source:
                own_ref = None  # Existing resume source is not a newly committed write.
                continue
            if checkpoint_sink is not None:
                await checkpoint_sink(own_ref)
        if own_ref is None:
            raise ValueError("Execution produced no checkpoint")
        # Read our emitted ID, not latest thread checkpoint. P4 checks readiness,
        # lineage/versions and fenced CAS before any customer publication.
        final = await self.read(own_ref)
        if final.next or final.tasks:
            raise ValueError("Execution did not reach a terminal checkpoint")
        if final.metadata is None or any(
            final.metadata.get(key) != value for key, value in context.items()
        ):
            raise ValueError("Checkpoint run context mismatch")
        return own_ref
