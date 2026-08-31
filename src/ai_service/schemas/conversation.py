from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: UUID | None = None


class ChatData(BaseModel):
    conversation_id: UUID
    intent: str
    answer: str
    products: list[ProductCard] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)


class ProductCard(BaseModel):
    id: UUID | None = None
    name: str
    seo_name: str | None = None
    list_price: int | None = None
    image_url: str | None = None
    status: str | None = None
    specifications: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None


class SearchData(BaseModel):
    query: str
    products: list[ProductCard] = Field(default_factory=list)


class ConsultRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class ConsultData(BaseModel):
    query: str
    answer: str
    products: list[ProductCard] = Field(default_factory=list)


class CompareRequest(BaseModel):
    product_ids: list[UUID] = Field(min_length=2, max_length=5)
    question: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def require_distinct_products(self) -> CompareRequest:
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("product_ids must contain distinct products")
        return self


class ProductComparison(BaseModel):
    product_id: UUID | None = None
    name: str
    list_price: int | None = None
    image_url: str | None = None
    specifications: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    status: str | None = None


class CompareData(BaseModel):
    products: list[ProductComparison] = Field(default_factory=list)
    answer: str


class EvaluateRequest(BaseModel):
    product_id: UUID
    question: str | None = Field(default=None, max_length=1000)


class EvaluateData(BaseModel):
    product: ProductCard
    answer: str


# Forward reference used by ChatData so ProductCard can remain close to the
# other product response contracts without introducing a second DTO shape.
ChatData.model_rebuild()
