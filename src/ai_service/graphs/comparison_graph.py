from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, Field
from pydantic_graph import GraphBuilder


class ComparisonState(BaseModel):
    """State carried by the deterministic comparison graph."""

    product_ids: list[UUID] = Field(default_factory=list)
    question: str | None = None


class ComparisonInput(BaseModel):
    product_ids: list[UUID] = Field(min_length=2, max_length=5)
    question: str | None = None


class ComparisonOutput(BaseModel):
    product_ids: list[UUID]
    question: str | None = None


comparison_builder = GraphBuilder(
    name="comparison_graph",
    state_type=ComparisonState,
    input_type=ComparisonInput,
    output_type=ComparisonOutput,
)


@comparison_builder.step
async def capture_comparison(ctx: Any) -> ComparisonOutput:
    state = cast(ComparisonState, ctx.state)
    inputs = cast(ComparisonInput, ctx.inputs)
    state.product_ids = list(dict.fromkeys(inputs.product_ids))
    state.question = inputs.question
    return ComparisonOutput(
        product_ids=state.product_ids,
        question=state.question,
    )


comparison_builder.add(
    comparison_builder.edge_from(comparison_builder.start_node).to(capture_comparison),
    comparison_builder.edge_from(capture_comparison).to(comparison_builder.end_node),
)
comparison_graph = comparison_builder.build()
