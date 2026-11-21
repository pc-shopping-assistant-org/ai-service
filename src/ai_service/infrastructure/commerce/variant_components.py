"""Reads the spec of one chosen variant from the catalog, for the compatibility check."""

from typing import Any
from uuid import UUID

from pydantic import ValidationError

from ai_service.application.ports.hardware import (
    ComponentCategory,
    ComponentSource,
    ComponentSpec,
)
from ai_service.infrastructure.commerce.hardware_catalog import SPEC_FIELDS
from ai_service.services.backend_client import BackendClient


class BackendComponentSource(ComponentSource):
    def __init__(self, backend_client: BackendClient) -> None:
        self.backend_client = backend_client

    async def component(self, variant_id: UUID, category: ComponentCategory) -> ComponentSpec | None:
        variant = await self.backend_client.get_variant(variant_id)
        if variant is None or variant.get("productId") is None:
            return None
        product = await self.backend_client.get_product(UUID(str(variant["productId"])))
        if product is None:
            return None
        specifications = product.get("specifications") or {}
        fields = {name: specifications[name] for name in SPEC_FIELDS if name in specifications}
        return _spec(
            {
                **fields,
                "id": variant_id,
                "name": product.get("name") or str(variant_id),
                "category": category,
                "price": variant.get("price") or 0,
                "brand": product.get("brandName"),
            }
        )


def _spec(values: dict[str, Any]) -> ComponentSpec:
    """A catalog value of the wrong shape (a text where a number belongs) leaves that one attribute unknown, not the part."""
    try:
        return ComponentSpec.model_validate(values)
    except ValidationError as error:
        unreadable = {str(problem["loc"][0]) for problem in error.errors() if problem["loc"]}
        return ComponentSpec.model_validate({k: v for k, v in values.items() if k not in unreadable - {"name", "category"}})
