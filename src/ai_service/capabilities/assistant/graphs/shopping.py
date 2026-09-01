"""Deterministic graph that normalizes a shopping request."""

from typing import Any, cast

from pydantic import BaseModel
from pydantic_graph import GraphBuilder

from ai_service.capabilities.assistant.schemas import AssistantIntent

ShoppingIntent = AssistantIntent


class ShoppingState(BaseModel):
    query: str = ""
    intent: ShoppingIntent = AssistantIntent.SEARCH


class ShoppingInput(BaseModel):
    query: str
    intent: ShoppingIntent = AssistantIntent.SEARCH


class ShoppingOutput(BaseModel):
    query: str
    intent: ShoppingIntent


builder = GraphBuilder(
    name="shopping_graph",
    state_type=ShoppingState,
    input_type=ShoppingInput,
    output_type=ShoppingOutput,
)


@builder.step
async def capture_query(ctx: Any) -> ShoppingOutput:
    state = cast(ShoppingState, ctx.state)
    inputs = cast(ShoppingInput, ctx.inputs)
    state.query = " ".join(inputs.query.split())
    state.intent = inputs.intent
    return ShoppingOutput(query=state.query, intent=state.intent)


builder.add(
    builder.edge_from(builder.start_node).to(capture_query),
    builder.edge_from(capture_query).to(builder.end_node),
)
shopping_graph = builder.build()


__all__ = [
    "ShoppingInput",
    "ShoppingIntent",
    "ShoppingOutput",
    "ShoppingState",
    "shopping_graph",
]
