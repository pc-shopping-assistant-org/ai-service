from typing import Any, Literal, cast

from pydantic import BaseModel
from pydantic_graph import GraphBuilder

ShoppingIntent = Literal["SEARCH", "CONSULT", "COMPARE", "EVALUATE"]


class ShoppingState(BaseModel):
    query: str = ""
    intent: ShoppingIntent = "SEARCH"


class ShoppingInput(BaseModel):
    query: str
    intent: ShoppingIntent = "SEARCH"


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
