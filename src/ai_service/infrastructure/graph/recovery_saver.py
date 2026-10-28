"""Bind native recovery to an authorized checkpoint, including pending writes.

Passing checkpoint_id to the pinned LangGraph runtime requests replay. Recovery
instead uses its ordinary resume path, while this saver resolves the root lookup
to the coordinator-authorized ref rather than the thread's latest checkpoint.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1


class RecoverySaver(BaseCheckpointSaver):
    def __init__(self, saver: BaseCheckpointSaver, source: CheckpointRefV1):
        super().__init__(serde=saver.serde)
        self.saver, self.source = saver, source

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        native = config.get("configurable", {})
        if native.get("thread_id") != str(self.source.thread_id):
            raise ValueError("Recovery thread mismatch")
        if (
            not native.get("checkpoint_id")
            and native.get("checkpoint_ns", "") == self.source.checkpoint_ns
        ):
            config = {
                **config,
                "configurable": {**native, "checkpoint_id": self.source.checkpoint_id},
            }
        return await self.saver.aget_tuple(config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await self.saver.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await self.saver.aput_writes(config, writes, task_id, task_path)

    def get_next_version(self, current: Any, channel: None) -> Any:
        return self.saver.get_next_version(current, channel)
