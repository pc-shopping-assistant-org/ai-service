"""Native empty checkpoint; no model, catalog or graph-turn execution."""

from typing import Any, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1


class RootState(TypedDict):
    consultation: dict[str, Any]
    requirement_state: dict[str, Any]
    public_build: dict[str, Any] | None


class NativeConversationInitializer:
    def __init__(self, saver: BaseCheckpointSaver) -> None:
        self.saver = saver
        builder = StateGraph(RootState)

        def initialize_root(state: RootState) -> dict[str, Any]:
            raise RuntimeError(
                "Root must be created by native state update, not invocation"
            )

        builder.add_node("initialize_root", initialize_root)
        builder.add_edge(START, "initialize_root")
        builder.add_edge("initialize_root", END)
        self.graph = builder.compile(checkpointer=saver)

    async def initialize(self, conversation_id: UUID) -> CheckpointRefV1:
        thread: RunnableConfig = {
            "configurable": {"thread_id": str(conversation_id), "checkpoint_ns": ""}
        }
        marker = {
            "app_checkpoint_kind": "initial_root",
            "app_graph_version": "conversation-root-v1",
            "app_initialization_id": str(conversation_id),
        }
        candidates = [
            entry async for entry in self.saver.alist(thread, filter=marker, limit=2)
        ]
        if len(candidates) > 1:
            raise ValueError("Ambiguous initial root requires explicit reconciliation")
        if candidates:
            candidate = candidates[0]
            parent = (
                (candidate.parent_config or {})
                .get("configurable", {})
                .get("checkpoint_id")
            )
            snapshot = await self.graph.aget_state(candidate.config)
            expected = {
                "consultation": ConsultationStateV1().model_dump(mode="json"),
                "requirement_state": CanonicalRequirementState().model_dump(
                    mode="json"
                ),
                "public_build": None,
            }
            if parent or snapshot.next or snapshot.tasks or snapshot.values != expected:
                raise ValueError(
                    "Invalid initial root requires explicit reconciliation"
                )
            # A later native update may attach completed writes to this parent.
            # Exact snapshot next/tasks, not presence of write rows, define terminality.
            return CheckpointRefV1(
                thread_id=conversation_id,
                checkpoint_id=candidate.config["configurable"]["checkpoint_id"],
            )
        # Existence check only, never adopt latest or reset a populated legacy thread.
        if await self.saver.aget_tuple(thread) is not None:
            raise ValueError(
                "Populated native conversation requires explicit migration"
            )
        config = await self.graph.aupdate_state(
            {
                "configurable": {
                    "thread_id": str(conversation_id),
                    "checkpoint_ns": "",
                    # Explicit empty base: never seed from an unaccepted latest.
                    "checkpoint_id": "",
                },
                "metadata": {
                    "app_checkpoint_kind": "initial_root",
                    "app_graph_version": "conversation-root-v1",
                    "app_initialization_id": str(conversation_id),
                },
            },
            {
                "consultation": ConsultationStateV1().model_dump(mode="json"),
                "requirement_state": CanonicalRequirementState().model_dump(
                    mode="json"
                ),
                "public_build": None,
            },
            as_node="initialize_root",
        )
        snapshot = await self.graph.aget_state(config)
        if (
            snapshot.next
            or snapshot.tasks
            or snapshot.values["public_build"] is not None
        ):
            raise ValueError("Initial checkpoint is not an empty terminal root")
        return CheckpointRefV1(
            thread_id=conversation_id,
            checkpoint_id=config["configurable"]["checkpoint_id"],
        )
