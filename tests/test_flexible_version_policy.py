from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_flexible_graph import graph
from test_react_tools import Catalog, context

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.stateful_contracts import ConversationHeadV1
from ai_service.infrastructure.graph.durable import DurableConversationGraph
from ai_service.infrastructure.graph.flexible import FLEXIBLE_GRAPH_VERSION
from ai_service.infrastructure.graph.initial_root import NativeConversationInitializer


class Model:
    def __init__(self):
        self.calls = 0

    async def decide(self, message, transcript, allowed, repair, context=None):
        self.calls += 1
        if not transcript or (message == "orphan" and len(transcript) == 1):
            return FlexibleDecision(
                action={
                    "kind": "CALL_TOOL",
                    "call": {
                        "call_id": "orphan" if transcript else "accepted",
                        "name": "search_catalog",
                        "arguments": {"query": "GPU"},
                    },
                }
            )
        return FlexibleDecision(
            action={
                "kind": "FINAL_ANSWER",
                "answer": {
                    "text": "GPU",
                    "evidence_refs": transcript[-1].result.evidence_refs,
                },
            }
        )


@pytest.mark.asyncio
async def test_v2_uses_exact_accepted_head_despite_newer_orphan():
    saver, trusted, model = InMemorySaver(), context(), Model()
    root = await NativeConversationInitializer(saver).initialize(
        trusted.conversation_id
    )
    flow = DurableConversationGraph(
        graph(model, Catalog(), trusted, saver=saver),
        accepted_versions={FLEXIBLE_GRAPH_VERSION},
    )
    head = ConversationHeadV1(
        conversation_id=trusted.conversation_id,
        revision=0,
        accepted_checkpoint_ref=root,
    )
    first = await flow.new_turn(
        head, uuid4(), {"message": "first"}, FLEXIBLE_GRAPH_VERSION
    )
    head = head.model_copy(update={"revision": 1, "accepted_checkpoint_ref": first})
    orphan = await flow.new_turn(
        head, uuid4(), {"message": "orphan"}, FLEXIBLE_GRAPH_VERSION
    )
    assert len((await flow.read(orphan)).values["ledger"]) == 2
    third = await flow.new_turn(
        head, uuid4(), {"message": "third"}, FLEXIBLE_GRAPH_VERSION
    )
    result = await flow.read(third)
    assert result.values["message"] == "third"
    assert [entry["call"]["call_id"] for entry in result.values["ledger"]] == [
        "accepted"
    ]
    assert result.metadata["app_base_checkpoint_id"] == first.checkpoint_id


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_version", ["legacy-guided-v1", ""])
async def test_legacy_base_is_rejected_before_model_or_checkpoint_write(legacy_version):
    saver, trusted, model = InMemorySaver(), context(), Model()
    root = await NativeConversationInitializer(saver).initialize(
        trusted.conversation_id
    )
    compiled = graph(model, Catalog(), trusted, saver=saver)
    legacy = DurableConversationGraph(compiled)
    head = ConversationHeadV1(
        conversation_id=trusted.conversation_id,
        revision=0,
        accepted_checkpoint_ref=root,
    )
    old = await legacy.new_turn(head, uuid4(), {"message": "legacy"}, legacy_version)
    head = head.model_copy(update={"revision": 1, "accepted_checkpoint_ref": old})
    calls = model.calls
    guarded = DurableConversationGraph(
        compiled, accepted_versions={FLEXIBLE_GRAPH_VERSION}
    )
    with pytest.raises(ValueError, match="version"):
        await guarded.new_turn(
            head, uuid4(), {"message": "new"}, FLEXIBLE_GRAPH_VERSION
        )
    assert model.calls == calls


@pytest.mark.asyncio
async def test_versioned_execution_requires_initialized_root_and_declared_execution_version():
    saver, trusted, model = InMemorySaver(), context(), Model()
    flow = DurableConversationGraph(
        graph(model, Catalog(), trusted, saver=saver),
        accepted_versions={FLEXIBLE_GRAPH_VERSION},
    )
    head = ConversationHeadV1(conversation_id=trusted.conversation_id, revision=0)
    with pytest.raises(ValueError, match="initialized"):
        await flow.new_turn(head, uuid4(), {"message": "new"}, FLEXIBLE_GRAPH_VERSION)
    with pytest.raises(ValueError, match="version"):
        await flow.new_turn(head, uuid4(), {"message": "new"}, "not-supported")
    assert model.calls == 0
