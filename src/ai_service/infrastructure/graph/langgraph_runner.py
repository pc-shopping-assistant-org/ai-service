"""LangGraph adapter for current request-scoped planning flows.

No checkpointer is configured here. Durable conversation threads, accepted heads
and run ownership belong to the separately tracked stateful implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ai_service.application.ports.graph_runner import GraphRunner


class PlanningState(TypedDict):
    inputs: BaseModel
    output: NotRequired[BaseModel]


class LangGraphRunner[InputT: BaseModel, OutputT: BaseModel](
    GraphRunner[InputT, OutputT]
):
    """Compile once; each invocation owns its inputs and has no shared state."""

    def __init__(self, normalize: Callable[[InputT], OutputT]) -> None:
        async def capture(state: PlanningState) -> dict[str, BaseModel]:
            return {"output": normalize(cast(InputT, state["inputs"]))}

        builder = StateGraph(PlanningState)
        builder.add_node("normalize", capture)
        builder.add_edge(START, "normalize")
        builder.add_edge("normalize", END)
        self._graph = builder.compile()

    async def run(self, inputs: InputT) -> OutputT:
        result = await self._graph.ainvoke({"inputs": inputs})
        return cast(OutputT, result["output"])


__all__ = ["LangGraphRunner"]
