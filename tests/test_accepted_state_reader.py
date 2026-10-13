from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.stateful_contracts import CheckpointRefV1
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)


class State(TypedDict):
    consultation: dict
    public_build: dict | None


@pytest.mark.asyncio
async def test_reader_uses_exact_ref_not_latest_and_corruption_never_becomes_default():
    async def finish(state):
        return {}

    saver = InMemorySaver()
    builder = StateGraph(State)
    builder.add_node("finish", finish)
    builder.add_edge(START, "finish")
    builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=saver)
    conversation_id = uuid4()
    config = {"configurable": {"thread_id": str(conversation_id)}}
    original = ConsultationStateV1()
    await graph.ainvoke(
        {"consultation": original.model_dump(mode="json"), "public_build": None}, config
    )
    accepted = await graph.aget_state(config)
    ref = CheckpointRefV1(
        thread_id=conversation_id,
        checkpoint_id=accepted.config["configurable"]["checkpoint_id"],
    )
    reader = PostgresAcceptedCheckpointReader(saver)
    await graph.ainvoke({"consultation": {"schema_version": 999}}, config)
    assert (await reader.read(ref)).consultation == original
    orphan = await graph.aget_state(config)
    bad_ref = ref.model_copy(
        update={"checkpoint_id": orphan.config["configurable"]["checkpoint_id"]}
    )
    with pytest.raises(ValueError):
        await reader.read(bad_ref)
    with pytest.raises(ValueError, match="missing"):
        await reader.read(ref.model_copy(update={"checkpoint_id": "does-not-exist"}))
