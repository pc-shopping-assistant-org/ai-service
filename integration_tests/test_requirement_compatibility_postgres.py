"""Reopen exact legacy requirement artifacts through the native PostgreSQL saver."""

from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import decode_flexible_decision
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    requirements_hash,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.requirements.user_admission import (
    apply_user_receipt,
    decode_user_requirement_receipt,
)


class LegacyCheckpoint(TypedDict, total=False):
    decision: dict
    user_receipts: list[dict]


def compile_checkpoint_graph(saver):
    async def persist(state):
        return {}

    builder = StateGraph(LegacyCheckpoint)
    builder.add_node("persist", persist)
    builder.add_edge(START, "persist")
    builder.add_edge("persist", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.asyncio
async def test_legacy_decision_and_receipt_survive_postgres_saver_reopen(postgres_dsn):
    message = "Build PC RAM 32GB"
    start = message.index("RAM 32GB")
    draft = Draft(
        items=[
            {
                "subject": "RAM",
                "predicate": "capacity",
                "start": start,
                "end": start + len("RAM 32GB"),
                "value": {"kind": "QUANTITY", "amount": 32, "unit": "GB"},
            }
        ],
        tasks=[
            {
                "kind": "BUILD",
                "start": 0,
                "end": len(message),
                "quote": message,
            }
        ],
    )
    proposed = Draft(items=draft.items)
    state, run_id, thread_id = CanonicalRequirementState(), uuid4(), uuid4()
    old_action = {
        "kind": "PROPOSE_REQUIREMENTS",
        "source_text": None,
        "draft": proposed.model_dump(mode="json"),
    }
    old_decision = {"schema_version": 2, "action": old_action}
    old_receipt = {
        "policy_version": "explicit-draft-match-v1",
        "run_id": str(run_id),
        "message_hash": canonical_hash(message),
        "base_requirements_hash": requirements_hash(state),
        "proposed": old_action,
        "extracted": draft.model_dump(mode="json"),
    }
    config = {"configurable": {"thread_id": str(thread_id)}}

    try:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = compile_checkpoint_graph(saver)
            await graph.ainvoke(
                {"decision": old_decision, "user_receipts": [old_receipt]}, config
            )
            before_close = await graph.aget_state(config)
            assert before_close.values["decision"] == old_decision
            assert before_close.values["user_receipts"] == [old_receipt]

        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.setup()
            graph = compile_checkpoint_graph(saver)
            reopened = await graph.aget_state(config)
            assert reopened.values["decision"] == old_decision
            assert reopened.values["user_receipts"] == [old_receipt]

            decision = decode_flexible_decision(reopened.values["decision"])
            receipt = decode_user_requirement_receipt(
                reopened.values["user_receipts"][0], message=message
            )
            merged = apply_user_receipt(
                state, receipt, run_id=run_id, message=message, catalog=[]
            )
            assert decision.action.draft.schema_version == 2
            assert receipt.extracted.schema_version == 2
            assert next(iter(merged.requirements.values())).value.amount == 32
            await saver.adelete_thread(str(thread_id))
    finally:
        async with AsyncPostgresSaver.from_conn_string(postgres_dsn) as saver:
            await saver.adelete_thread(str(thread_id))
