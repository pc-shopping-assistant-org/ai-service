from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_service.capabilities.assistant.schemas import ProductCard


class BackendProduct(BaseModel):
    id: UUID | None = None
    name: str
    seoName: str | None = None
    listPrice: int | None = None
    imageUrl: str | None = None
    status: str | None = None
    specifications: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None

    def to_card(self) -> ProductCard:
        from ai_service.capabilities.assistant.schemas import ProductCard

        return ProductCard(
            id=self.id,
            name=self.name,
            seo_name=self.seoName,
            list_price=self.listPrice,
            image_url=self.imageUrl,
            status=self.status,
        )
