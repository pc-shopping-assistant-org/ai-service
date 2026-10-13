"""Read-only public projection from an exact accepted native checkpoint.

Reading domain values is permitted here; this never seeds/reconstructs execution.
Native recovery remains the graph's responsibility. Corruption fails closed.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver

from ai_service.application.ports.stateful_conversation import (
    AcceptedConversationSnapshot,
)
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1


class PostgresAcceptedCheckpointReader:
    def __init__(self, saver: BaseCheckpointSaver) -> None:
        self.saver = saver

    async def read(self, ref: CheckpointRefV1) -> AcceptedConversationSnapshot:
        if ref.checkpoint_ns != "":
            raise ValueError("Only root conversation checkpoints are accepted")
        record = await self.saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": str(ref.thread_id),
                    "checkpoint_ns": "",
                    "checkpoint_id": ref.checkpoint_id,
                }
            }
        )
        if record is None:
            raise ValueError("Accepted checkpoint is missing; never substitute latest")
        actual = record.config.get("configurable", {})
        if (
            actual.get("thread_id"),
            actual.get("checkpoint_ns", ""),
            actual.get("checkpoint_id"),
        ) != (
            str(ref.thread_id),
            "",
            ref.checkpoint_id,
        ):
            raise ValueError("Accepted checkpoint identity mismatch")
        channels = record.checkpoint["channel_values"]
        if "accepted_snapshot" in channels:
            snapshot = AcceptedConversationSnapshot.model_validate(
                channels["accepted_snapshot"]
            )
            if (
                snapshot.flexible is None
                or snapshot.flexible.conversation_id != ref.thread_id
            ):
                raise ValueError("V2 accepted snapshot schema/conversation mismatch")
            return snapshot
        if "consultation" not in channels or "public_build" not in channels:
            raise ValueError(
                "Accepted checkpoint has an incompatible application schema"
            )
        return AcceptedConversationSnapshot.model_validate(
            {
                "consultation": channels["consultation"],
                "public_build": channels["public_build"],
                "public_clarification": channels.get("public_clarification"),
                "public_task_clarification": channels.get("public_task_clarification"),
                "pending_tasks": channels.get("pending_tasks"),
                "requirement_state": channels.get("requirement_state"),
            }
        )
