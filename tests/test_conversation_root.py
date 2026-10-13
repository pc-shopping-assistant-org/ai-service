"""Core path: a newly created conversation starts from an accepted empty root."""

from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_service.capabilities.assistant.conversation_core import ConsultationStateV1
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer
from ai_service.infrastructure.persistence.accepted_state import (
    PostgresAcceptedCheckpointReader,
)


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


@pytest.mark.asyncio
async def test_build_then_next_turn_keeps_requirements_from_accepted_root(
    realistic_catalog,
):
    from test_requirement_graph import Catalog, Extractor
    from test_requirement_pipeline import rows

    from ai_service.capabilities.assistant.requirements.resolver import Draft
    from ai_service.infrastructure.graph.durable import DurableConversationGraph
    from ai_service.infrastructure.graph.requirement_build import (
        build_requirement_graph,
    )

    class MultiTurnExtractor(Extractor):
        async def extract(self, message, state):
            return (
                await super().extract(message, state)
                if not state.requirements
                else Draft(items=[])
            )

    saver = InMemorySaver()
    conversation_id = uuid4()
    root = await NativeConversationInitializer(saver).initialize(conversation_id)
    graph = DurableConversationGraph(
        build_requirement_graph(
            MultiTurnExtractor(),
            Catalog(rows(realistic_catalog)),
            checkpointer=saver,
        )
    )
    head = ConversationHeadV1(
        conversation_id=conversation_id, revision=0, accepted_checkpoint_ref=root
    )
    first = await graph.new_turn(
        head,
        uuid4(),
        {
            "message": "Build 30 triệu gaming, RAM ít nhất 32GB",
            "message_id": str(uuid4()),
        },
        "requirement-build-v1",
    )
    first_state = await graph.read(first)
    assert first_state.values["outcome"] == "BUILT"
    head = head.model_copy(update={"revision": 1, "accepted_checkpoint_ref": first})
    second = await graph.new_turn(
        head,
        uuid4(),
        {
            "message": "Giữ nguyên cấu hình",
            "message_id": str(uuid4()),
        },
        "requirement-build-v1",
    )
    second_state = await graph.read(second)
    assert second_state.values["outcome"] == "BUILT"
    assert (
        second_state.values["requirement_state"]["requirements"]
        == first_state.values["requirement_state"]["requirements"]
    )
    assert (
        second_state.values["receipt"]["build"]["total_price"]
        == first_state.values["receipt"]["build"]["total_price"]
    )
    assert "32" in second_state.values["answer"]


@pytest.mark.asyncio
async def test_initial_root_is_terminal_empty_and_readable_without_running_a_turn():
    saver = InMemorySaver()
    initializer = NativeConversationInitializer(saver)
    conversation_id = uuid4()
    ref = await initializer.initialize(conversation_id)
    head = ConversationHeadV1(
        conversation_id=conversation_id, revision=0, accepted_checkpoint_ref=ref
    )
    snapshot = await initializer.graph.aget_state(
        {
            "configurable": {
                "thread_id": str(conversation_id),
                "checkpoint_id": ref.checkpoint_id,
            }
        }
    )
    assert head.accepted_checkpoint_ref == ref
    assert snapshot.next == snapshot.tasks == ()
    assert snapshot.values["requirement_state"]["revision"] == 0
    assert snapshot.values["requirement_state"]["requirements"] == {}
    assert snapshot.metadata["app_checkpoint_kind"] == "initial_root"
    accepted = await PostgresAcceptedCheckpointReader(saver).read(ref)
    assert accepted.consultation.constraints.target_budget_vnd.value is None
    assert accepted.public_build is None
