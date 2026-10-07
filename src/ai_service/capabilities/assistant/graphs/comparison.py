"""Vendor-neutral comparison planning contracts and deterministic normalization."""

from uuid import UUID

from pydantic import BaseModel, Field


class ComparisonInput(BaseModel):
    product_ids: list[UUID] = Field(min_length=2, max_length=5)
    question: str | None = None


class ComparisonOutput(BaseModel):
    product_ids: list[UUID]
    question: str | None = None


def normalize_comparison(inputs: ComparisonInput) -> ComparisonOutput:
    return ComparisonOutput(
        product_ids=list(dict.fromkeys(inputs.product_ids)), question=inputs.question,
    )


__all__ = ["ComparisonInput", "ComparisonOutput", "normalize_comparison"]
