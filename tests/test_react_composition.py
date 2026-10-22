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


class OuterState(ReactState, total=False):
    published_answer: str


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
