"""Catalog-owned hardware taxonomy and optional custom attribute mapping."""

import json
from datetime import UTC, datetime
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, JsonValue, StrictBool, model_validator

from ai_service.application.call_attempts import AttemptBoundary
from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.accessories import AccessorySpec, AccessoryType
from ai_service.application.ports.hardware import ComponentCategory, ComponentSpec
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
)
from ai_service.capabilities.assistant.requirements.resolver import Draft
from ai_service.capabilities.assistant.stateful_contracts import Contract

from .product_catalog import BackendProductCatalog, Summary, Variant

SPEC_FIELDS = (
    set(ComponentSpec.model_fields) - {"id", "name", "category", "price", "extra_specs"}
) | {"storage_media"}


class HardwareCatalogMapping(Contract):
    version: str = Field(default="catalog-hardware-v1", min_length=1)
    # Destination canonical field -> source product attribute key.
    spec_fields: dict[str, str] = Field(default_factory=dict)
    # Option IDs are stable even though the current backend DTO omits option type.
    # Explicit null means a cosmetic option; missing mapping is an error.
    option_fields: dict[UUID, str | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def supported_fields(self) -> "HardwareCatalogMapping":
        fields = set(self.spec_fields) | {
            v for v in self.option_fields.values() if v is not None
        }
        if not fields <= SPEC_FIELDS or any(
            not key.strip() for key in self.spec_fields.values()
        ):
            raise ValueError("Unsupported hardware mapping field")
        return self


class HardwareSummary(Summary):
    category_id: UUID = Field(alias="categoryId")
    hardware_type: ComponentCategory | AccessoryType | None = Field(alias="hardwareType")


class Option(BaseModel):
    id: UUID
    name: str
    value: str


class HardwareVariant(Variant):
    sku: str = Field(min_length=1)
    model: str | None = None
    options: list[Option] = Field(default_factory=list)


class HardwareDetail(HardwareSummary):
    brand_name: str | None = Field(default=None, alias="brandName")
    specifications: dict[str, JsonValue]
    variants: list[HardwareVariant]


class HardwarePage(BaseModel):
    items: list[HardwareDetail]
    has_next: StrictBool = Field(alias="hasNext")
    next_cursor: UUID | None = Field(default=None, alias="nextCursor")


class BackendRequirementCatalog(BackendProductCatalog):
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        attempts: AttemptBoundary,
        mapping: HardwareCatalogMapping,
        *,
        max_rows: int = 100,
        max_pages: int = 20,
    ):
        super().__init__(client, base_url, attempts)
        if max_rows < 1 or max_pages < 1:
            raise ValueError("Catalog bounds must be positive")
        self.mapping = HardwareCatalogMapping.model_validate(mapping.model_dump())
        self.max_rows, self.max_pages = max_rows, max_pages

    async def load(
        self, draft: Draft, state: CanonicalRequirementState
    ) -> list[CatalogRowV1]:
        # Scan the configured taxonomy completely, not a keyword-ranked shortlist:
        # ambiguity/pins must not become unique merely because another SKU was cut.
        Draft.model_validate(draft.model_dump())
        CanonicalRequirementState.model_validate(state.model_dump())
        cursor = None
        seen: set[UUID] = set()
        rows: list[CatalogRowV1] = []
        try:
            for _ in range(self.max_pages):
                params: dict[str, JsonValue] = {"limit": 100}
                if cursor is not None:
                    params["cursor"] = str(cursor)
                page = HardwarePage.model_validate(
                    await self._get("/products/snapshot", params)
                )
                if len(page.items) > 100:
                    raise ValueError("Catalog page bound exceeded")
                for detail in page.items:
                    if detail.id in seen:
                        raise ValueError("Duplicate catalog page identity")
                    seen.add(detail.id)
                    if (
                        detail.status != "ACTIVE"
                        or detail.hardware_type is None
                    ):
                        continue
                    rows.extend(self._map(detail))
                    if len(rows) > self.max_rows:
                        raise ValueError("Canonical candidate bound exceeded")
                if not page.has_next:
                    return rows
                if (
                    not page.items
                    or page.next_cursor is None
                    or page.next_cursor == cursor
                ):
                    raise ValueError("Catalog pagination cannot advance")
                cursor = page.next_cursor
            raise ValueError("Catalog page bound exceeded")
        except (ValueError, TypeError) as exc:
            from ai_service.application.call_attempts import AttemptRejected

            if isinstance(exc, AttemptRejected):
                raise
            raise BackendUnavailableError(
                f"Hardware catalog mapping/bound invalid: {type(exc).__name__}"
            ) from exc

    def _map(self, detail: HardwareDetail) -> list[CatalogRowV1]:
        fields: dict[str, JsonValue] = {
            field: detail.specifications[key]
            for field in SPEC_FIELDS
            if (key := self.mapping.spec_fields.get(field, field))
            in detail.specifications
        }
        if detail.brand_name is not None:
            fields["brand"] = detail.brand_name
        identities = [variant.id for variant in detail.variants]
        if len(identities) != len(set(identities)) or any(
            v.product_id != detail.id for v in detail.variants
        ):
            raise ValueError("Variant identity mismatch")
        mapped = []
        for variant in detail.variants:
            if variant.status != "ACTIVE":
                continue
            specs = dict(fields)
            used: set[str] = set()
            for option in variant.options:
                if option.id not in self.mapping.option_fields:
                    raise ValueError("Unknown variant option mapping")
                field = self.mapping.option_fields[option.id]
                if field is None:
                    continue
                if field in used:
                    raise ValueError("Duplicate variant option field")
                used.add(field)
                try:
                    specs[field] = json.loads(option.value)
                except json.JSONDecodeError:
                    specs[field] = option.value
            extra = (
                {"storage_media": specs.pop("storage_media")}
                if "storage_media" in specs
                else {}
            )
            category = detail.hardware_type
            component = (
                ComponentSpec
                if isinstance(category, ComponentCategory)
                else AccessorySpec
            ).model_validate(
                {
                    **(
                        specs
                        if isinstance(category, ComponentCategory)
                        else {"brand": detail.brand_name}
                    ),
                    "id": variant.id,
                    "name": variant.model or detail.name,
                    "category": category,
                    "price": variant.price,
                    "extra_specs": extra
                    if isinstance(category, ComponentCategory)
                    else detail.specifications,
                },
                strict=True,
            )
            if any(
                isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0
                for v in specs.values()
            ):
                raise ValueError("Negative hardware measurement")
            ref = (
                "catalog:variant:"
                + str(variant.id)
                + ":"
                + canonical_hash(
                    {
                        "product": detail.model_dump(mode="json"),
                        "variant": variant.model_dump(mode="json"),
                        "mapping": self.mapping.model_dump(mode="json"),
                    }
                )
            )
            mapped.append(
                CatalogRowV1(
                    variant_id=variant.id,
                    sku=variant.sku,
                    product_status="ACTIVE",
                    variant_status="ACTIVE",
                    quantity=variant.quantity,
                    component=component,
                    retrieved_at=datetime.now(UTC),
                    evidence_refs=[ref],
                )
            )
        return mapped
