from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext, StartNode


class ShoppingState(BaseModel):
    query: str = ""


class ShoppingInput(BaseModel):
    query: str


class ShoppingOutput(BaseModel):
    query: str


@dataclass
class CaptureQuery(BaseNode[ShoppingState, None, ShoppingOutput]):
    query: str

    async def run(self, ctx: GraphRunContext[ShoppingState, None]) -> End[ShoppingOutput]:
        ctx.state.query = self.query
        return End(ShoppingOutput(query=self.query))


builder = GraphBuilder(
    name="shopping_graph",
    state_type=ShoppingState,
    input_type=ShoppingInput,
    output_type=ShoppingOutput,
)
builder.add_edge(StartNode, CaptureQuery)  # type: ignore[arg-type]
builder.add(builder.node(CaptureQuery))
shopping_graph = builder.build()
