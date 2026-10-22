"""The initializer only accepts a unique, empty, native initial root."""

from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["AMBIGUOUS", "NONEMPTY", "PARENT"])
async def test_root_reconciliation_rejects_ambiguous_nonempty_or_nonroot_lineage(
    problem,
):
    saver, conversation = InMemorySaver(), uuid4()
    initializer = NativeConversationInitializer(saver)
    marker = {
        "app_checkpoint_kind": "initial_root",
        "app_graph_version": "conversation-root-v1",
        "app_initialization_id": str(conversation),
    }
    empty = {
        "consultation": ConsultationStateV1().model_dump(mode="json"),
        "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
        "public_build": None,
    }
    config = {
        "configurable": {"thread_id": str(conversation), "checkpoint_id": ""},
        "metadata": marker,
    }
    if problem == "AMBIGUOUS":
        await initializer.initialize(conversation)
    if problem == "NONEMPTY":
        empty["public_build"] = {"unaccepted": True}
    if problem == "PARENT":
        parent = await initializer.graph.aupdate_state(
            config | {"metadata": marker | {"app_checkpoint_kind": "debug"}},
            empty,
            as_node="initialize_root",
        )
        config = parent | {"metadata": marker}
    await initializer.graph.aupdate_state(config, empty, as_node="initialize_root")
    before = len(
        [
            row
            async for row in saver.alist(
                {"configurable": {"thread_id": str(conversation)}}
            )
        ]
    )
    with pytest.raises(ValueError, match="reconciliation"):
        await initializer.initialize(conversation)
    after = len(
        [
            row
            async for row in saver.alist(
                {"configurable": {"thread_id": str(conversation)}}
            )
        ]
    )
    assert after == before
