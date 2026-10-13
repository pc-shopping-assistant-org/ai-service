import pytest
from langgraph.graph import END, START, StateGraph
from test_react_tools import Catalog, context
from test_react_web_flow import Budget

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
    ReactState,
    add_react_nodes,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("has_build", [False, True])
async def test_requirement_flow_runs_react_for_read_task_without_optimizer(has_build):
    from uuid import uuid4

    from ai_service.capabilities.assistant.requirements.contracts import (
        CanonicalRequirementState,
    )
    from ai_service.capabilities.assistant.requirements.resolver import Draft
    from ai_service.infrastructure.graph.requirement_build import (
        ReactComposition,
        build_requirement_graph,
    )

    catalog = Catalog()

    class Extractor:
        async def extract(self, message, state):
            return Draft(
                items=[],
                tasks=[
                    {
                        "kind": "ANSWER",
                        "start": 0,
                        "end": len(message),
                        "quote": message,
                    }
                ],
            )

    class HardwareCatalog:
        async def load(self, draft, state):
            pytest.fail("Read-only turn must not scan hardware/optimize")

    class TaskCatalog:
        async def resolve(self, mention):
            pytest.fail("ANSWER has no product-reference resolution")

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return ReactDecision(
                    calls=[
                        ToolCall(
                            call_id="search",
                            name="search_catalog",
                            arguments={"query": "GPU"},
                        )
                    ]
                )
            return ReactDecision(
                answer=AnswerCandidate(
                    text="GPU costs 1 VND",
                    evidence_refs=transcript[0].result.evidence_refs,
                )
            )

    graph = build_requirement_graph(
        Extractor(),
        HardwareCatalog(),
        react=ReactComposition(
            Model(),
            ToolDispatcher(catalog, PCBuildApplicationService()),
            context(),
            TaskCatalog(),
            Budget(),
        ),
    )
    previous = (
        {
            "objective": "BALANCED",
            "parts": [
                {
                    "slot": "GPU",
                    "variant_id": str(uuid4()),
                    "name": "Saved GPU",
                    "spending_vnd": 8_000_000,
                }
            ],
            "total_spending_vnd": 8_000_000,
            "stale": False,
        }
        if has_build
        else None
    )
    result = await graph.ainvoke(
        {
            "message": "Find GPU",
            "message_id": str(uuid4()),
            "requirement_state": CanonicalRequirementState().model_dump(mode="json"),
            "public_build": previous,
        }
    )
    assert result["outcome"] == "ANSWERED"
    assert "8,000,000 VND" in result["answer"]
    assert result["receipt"] == {}
    assert result["phase"] == "READY_TO_PUBLISH"
    if previous is not None:
        from ai_service.capabilities.assistant.public_contracts import PublicBuildV1

        assert result["public_build"] == PublicBuildV1.model_validate(
            previous
        ).model_dump(mode="json")
    else:
        assert result["public_build"] is None


class OuterState(ReactState, total=False):
    published_answer: str


@pytest.mark.asyncio
async def test_build_request_cannot_finish_as_generic_advice(realistic_catalog):
    from uuid import uuid4

    from test_requirement_pipeline import rows, state

    from ai_service.capabilities.assistant.requirements.resolver import Draft
    from ai_service.infrastructure.graph.requirement_build import (
        ReactComposition,
        build_requirement_graph,
    )

    class Extractor:
        async def extract(self, message, requirements):
            return Draft(
                items=[],
                tasks=[{"kind": "BUILD", "start": 0, "end": 8, "quote": "Build PC"}],
            )

    class Catalog:
        async def load(self, draft, requirements):
            return rows(realistic_catalog)

        async def resolve(self, mention):
            pytest.fail("BUILD does not resolve comparison mentions")

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript and not repair:
                return ReactDecision(answer=AnswerCandidate(text="Ưu tiên GPU mạnh"))
            if not transcript:
                return ReactDecision(
                    calls=[ToolCall(call_id="build", name="build_pc", arguments={})]
                )
            result = transcript[0].result
            return ReactDecision(
                answer=AnswerCandidate(
                    text="mọi game đều mượt",
                    evidence_refs=result.evidence_refs,
                    build_ref=result.result.build_ref,
                )
            )

    catalog = Catalog()
    graph = build_requirement_graph(
        Extractor(),
        catalog,
        react=ReactComposition(
            Model(),
            ToolDispatcher(catalog, PCBuildApplicationService()),
            context(),
            catalog,
            Budget(),
        ),
    )
    result = await graph.ainvoke(
        {
            "message": "Build PC",
            "message_id": str(uuid4()),
            "requirement_state": state().model_dump(mode="json"),
            "public_build": None,
        }
    )
    assert result["outcome"] == "BUILT"
    assert result["public_build"] is not None
    assert result["repairs"] == 1
    assert "Tổng chi phí" in result["answer"]
    assert "mọi game" not in result["answer"]


@pytest.mark.asyncio
async def test_react_nodes_share_outer_graph_and_keep_public_answer_separate():
    catalog = Catalog()
    trusted = context()

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                return ReactDecision(
                    calls=[
                        ToolCall(
                            call_id="search",
                            name="search_catalog",
                            arguments={"query": "GPU"},
                        )
                    ]
                )
            return ReactDecision(
                answer=AnswerCandidate(
                    text="GPU costs 1 VND",
                    evidence_refs=transcript[0].result.evidence_refs,
                )
            )

    builder = StateGraph(OuterState)
    add_react_nodes(
        builder,
        Model(),
        ToolDispatcher(catalog, PCBuildApplicationService()),
        lambda state: trusted,
        lambda state: [CompletionRequirement(kind="ANSWER")],
        Budget(),
        done_node="publish",
        answer_channel="react_answer",
    )

    def publish(state):
        return {"published_answer": state["react_answer"]["text"]}

    builder.add_node("publish", publish)
    builder.add_edge(START, "agent")
    builder.add_edge("publish", END)
    result = await builder.compile().ainvoke({"message": "Find GPU"})
    assert "8,000,000 VND" in result["published_answer"]
    assert "costs 1 VND" not in result["published_answer"]
    assert len(result["ledger"]) == 1
