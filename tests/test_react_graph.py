from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_react_tools import Catalog, context
from test_requirement_pipeline import rows, state

from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
)
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.react import (
    ReactDecision,
    ReactLimits,
    build_react_graph,
)


class Budget:
    """Test-only attempt boundary; runtime must inject the SQL-backed boundary."""

    def __init__(self):
        self.calls = 0

    async def invoke(self, kind, operation_key, request_hash, dispatch):
        self.calls += 1
        return await dispatch()


class BuildModel:
    def __init__(self, *, invalid_first=False):
        self.transcripts = []
        self.invalid_first = invalid_first

    async def decide(self, message, transcript, allowed, repair, context=None):
        self.transcripts.append(transcript)
        if len(self.transcripts) == 1 and self.invalid_first:
            return ReactDecision(
                calls=[
                    ToolCall(call_id="x", name="build_pc", arguments={}),
                    ToolCall(call_id="y", name="build_pc", arguments={}),
                ]
            )
        if not transcript:
            return ReactDecision(
                calls=[ToolCall(call_id="build", name="build_pc", arguments={})]
            )
        result = transcript[-1].result
        return ReactDecision(
            answer=AnswerCandidate(
                text="LLM thinks every game is smooth",
                evidence_refs=result.evidence_refs,
                build_ref=result.result.build_ref,
            )
        )


@pytest.mark.asyncio
async def test_react_rejects_whole_multi_call_and_projects_only_valid_ledger(
    realistic_catalog,
):
    trusted = context(state(), rows(realistic_catalog))
    model, budget = BuildModel(invalid_first=True), Budget()
    graph = build_react_graph(
        model,
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        budget,
    )
    result = await graph.ainvoke({"message": "Build PC", "ledger": []})
    assert result["phase"] == "READY_TO_PUBLISH"
    assert len(result["ledger"]) == 1
    assert result["ledger"][0]["call"]["call_id"] == "build"
    assert model.transcripts[0] == model.transcripts[1] == []
    assert budget.calls == 3
    assert "every game" not in result["answer"]["text"]


@pytest.mark.asyncio
async def test_react_completion_repairs_cannot_silently_downgrade_build_task(
    realistic_catalog,
):
    class Advice:
        async def decide(self, *args):
            return ReactDecision(answer=AnswerCandidate(text="Buy a faster GPU"))

    trusted = context(state(), rows(realistic_catalog))
    budget = Budget()
    graph = build_react_graph(
        Advice(),
        ToolDispatcher(Catalog(), PCBuildApplicationService()),
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        budget,
        limits=ReactLimits(max_repairs=1),
    )
    with pytest.raises(ValueError, match="REACT_VALIDATION_EXHAUSTED"):
        await graph.ainvoke({"message": "Build PC", "ledger": []})
    assert budget.calls == 2


@pytest.mark.asyncio
async def test_react_resume_after_model_failure_reuses_committed_tool(
    realistic_catalog,
):
    class OnceFailure(BuildModel):
        failed = False

        async def decide(self, message, transcript, allowed, repair, context=None):
            if transcript and not self.failed:
                self.failed = True
                raise TimeoutError("provider failed after tool commit")
            return await super().decide(message, transcript, allowed, repair, context)

    class CountingBuild(PCBuildApplicationService):
        count = 0

        def build_pc(self, *args):
            self.count += 1
            return super().build_pc(*args)

    trusted = context(state(), rows(realistic_catalog))
    model, service, budget = OnceFailure(), CountingBuild(), Budget()
    graph = build_react_graph(
        model,
        ToolDispatcher(Catalog(), service),
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            )
        ],
        budget,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(TimeoutError):
        await graph.ainvoke(
            {"message": "Build PC", "ledger": []}, config, durability="sync"
        )
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("agent",)
    assert service.count == 1
    result = await graph.ainvoke(None, snapshot.config, durability="sync")
    assert result["phase"] == "READY_TO_PUBLISH" and service.count == 1
    assert budget.calls == 3
