"""Catalog-scale checkpoints must fit even though model observations stay small."""

import json
from uuid import uuid4

import pytest
from test_flexible_publication import base_snapshot
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.react.actions import FlexibleDecision
from ai_service.capabilities.assistant.react.contracts import CompletionRequirement
from ai_service.capabilities.assistant.react.snapshot_publication import (
    FlexiblePublicationInput,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.providers.tool_projection import observation_json


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_complete", [True, False])
async def test_catalog_sized_build_preserves_provenance_and_only_projects_chosen_parts(
    realistic_catalog,
    metadata_complete,
):
    parts = list(realistic_catalog)
    expensive_gpu = next(p for p in parts if p.category == "GPU")
    parts += [
        expensive_gpu.model_copy(update={"price": 100_000_000, "name": f"Premium {i}"})
        for i in range(2391 - len(parts))
    ]
    if not metadata_complete:
        parts = [
            part.model_copy(update={"tdp_watts": None})
            if part.category == "GPU"
            else part.model_copy(update={"has_integrated_graphics": False})
            if part.category == "CPU"
            else part
            for part in parts
        ]
    trusted = context(state(), rows(parts))

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "build",
                            "name": "build_pc",
                            "arguments": {},
                        },
                    }
                )
            result = transcript[0].result
            projected = json.loads(observation_json(result, 12000))
            assert "Premium 2000" not in json.dumps(projected)
            assert projected["result"]["build_ref"] == str(result.result.build_ref)
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "BUILT" if metadata_complete else "SEARCH_INCOMPLETE",
                    "supporting_tool_call_ids": [] if metadata_complete else ["build"],
                    "answer": {
                        "text": "Untrusted estimate",
                        "build_ref": result.result.build_ref
                        if metadata_complete
                        else None,
                        "evidence_refs": result.evidence_refs,
                    },
                }
            )

    graph = build_flexible_graph(
        Model(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        Budget(),
        conversation_revision=0,
        publication=FlexiblePublicationInput(
            accepted=base_snapshot(trusted),
            task_message_id=uuid4(),
            task_message="Build PC",
            request_message="Build PC",
        ),
    )
    terminal = await graph.ainvoke({"message": "Build PC"})
    assert terminal["answer"]["outcome"] == (
        "BUILT" if metadata_complete else "SEARCH_INCOMPLETE"
    )
    assert bool(terminal["accepted_snapshot"]["public_build"]) == metadata_complete
    assert (
        len(
            terminal["ledger"][0]["result"]["result"]["receipt"]["compilation"][
                "optimizer_input"
            ]["candidates"]
        )
        == 2391
    )
