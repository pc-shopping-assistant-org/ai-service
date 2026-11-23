"""Budgeted reads of the actual catalog-service product DTOs, not variant IDs."""

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, JsonValue, StrictBool, StrictInt, ValidationError

from ai_service.application.call_attempts import (
    AttemptBoundary,
    AttemptKind,
    AttemptRejected,
)
from ai_service.application.errors import (
    BackendUnavailableError,
    CatalogCategoryUnresolvedError,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.contracts import ProductEvidence

Status = Literal["ACTIVE", "INACTIVE", "DELETED"]
Money = StrictInt
QueryParamScalar = str | int | float | bool | None
RESOLUTION_PAGE_SIZE = 100
MAX_RESOLUTION_PRODUCTS = 100
MAX_RESOLUTION_PAGES = 20
RRF_K = 60
GENERAL_LISTING_QUERIES = frozenset(
    {
        "*",
        "all",
        "tất cả",
        "tat ca",
        "danh sách sản phẩm",
        "danh sach san pham",
        "sản phẩm hiện có",
        "san pham hien co",
    }
)


def rrf_fuse_by_product_id(
    *ranked_lists: list[UUID],
    k: int = RRF_K,
) -> list[UUID]:
    """Reciprocal Rank Fusion (RRF, k=60) merging ranked product_id lists in Python."""
    scores: dict[UUID, float] = {}
    first_seen: dict[UUID, int] = {}
    order = 0
    for ranked in ranked_lists:
        seen_in_list: set[UUID] = set()
        for rank, product_id in enumerate(ranked, start=1):
            if product_id in seen_in_list:
                continue
            seen_in_list.add(product_id)
            if product_id not in first_seen:
                first_seen[product_id] = order
                order += 1
            scores[product_id] = scores.get(product_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda pid: (-scores[pid], first_seen[pid]))


class Summary(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    status: Status
    seo_name: str | None = Field(default=None, alias="seoName")
    main_image_url: str | None = Field(default=None, alias="mainImageUrl")
    brand_name: str | None = Field(default=None, alias="brandName")
    category_id: UUID | None = Field(default=None, alias="categoryId")
    category_name: str | None = Field(default=None, alias="categoryName")
    min_price: Money | None = Field(default=None, alias="minPrice", ge=0)
    max_price: Money | None = Field(default=None, alias="maxPrice", ge=0)
    total_quantity: int | None = Field(default=None, alias="totalQuantity")


class SearchHitSummary(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    brand_name: str | None = Field(default=None, alias="brandName")
    category_name: str | None = Field(default=None, alias="categoryName")
    min_price: Money | None = Field(default=None, alias="minPrice", ge=0)
    max_price: Money | None = Field(default=None, alias="maxPrice", ge=0)
    in_stock: bool = Field(default=False, alias="inStock")


class Variant(BaseModel):
    id: UUID
    product_id: UUID = Field(alias="productId")
    price: Money = Field(ge=0)
    quantity: StrictInt = Field(ge=0)
    status: Status


class CatalogCategory(BaseModel):
    id: UUID
    name: str
    hardware_type: str | None = Field(alias="hardwareType")
    status: Status


class ResolutionPage(BaseModel):
    items: list[Summary] = Field(max_length=RESOLUTION_PAGE_SIZE)
    has_next: StrictBool = Field(alias="hasNext")
    next_cursor: str | None = Field(default=None, alias="nextCursor", max_length=2000)


class ProductImage(BaseModel):
    url: str | None = None
    image_url: str | None = Field(default=None, alias="imageUrl")
    main: bool = False

    def resolved_url(self) -> str | None:
        raw = self.url or self.image_url
        return raw.strip() if isinstance(raw, str) and raw.strip() else None


class Detail(BaseModel):
    id: UUID
    name: str = Field(min_length=1)
    status: Status
    seo_name: str | None = Field(default=None, alias="seoName")
    main_image_url: str | None = Field(default=None, alias="mainImageUrl")
    image_url: str | None = Field(default=None, alias="imageUrl")
    images: list[ProductImage] = Field(default_factory=list)
    brand_name: str | None = Field(default=None, alias="brandName")
    category_id: UUID | None = Field(default=None, alias="categoryId")
    category_name: str | None = Field(default=None, alias="categoryName")
    specifications: dict[str, JsonValue] = Field(default_factory=dict)
    variants: list[Variant] = Field(default_factory=list)

    def resolved_image_url(self) -> str | None:
        for candidate in (self.main_image_url, self.image_url):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        for image in self.images:
            if image.main and (url := image.resolved_url()):
                return url
        for image in self.images:
            if url := image.resolved_url():
                return url
        return None


class BackendProductCatalog:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        attempts: AttemptBoundary,
        search_base_url: str | None = None,
    ):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.attempts = attempts
        self.search_base_url = search_base_url.rstrip("/") if search_base_url else None

    @staticmethod
    def _unwrap_envelope(response: httpx.Response) -> Any:
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        if (
            not isinstance(body, dict)
            or set(body) != {"data", "message", "errors"}
            or body["message"] != "SUCCESS"
            or body["errors"] != []
        ):
            raise ValueError("Invalid catalog response envelope")
        return body["data"]

    async def _get(
        self, path: str, params: dict[str, QueryParamScalar] | None = None
    ) -> Any:
        return await self._get_from(self.base_url, path, params)

    async def _get_from(
        self,
        base_url: str,
        path: str,
        params: dict[str, QueryParamScalar] | None = None,
    ) -> Any:
        async def dispatch() -> Any:
            from ai_service.infrastructure.observability import trace_headers

            headers = trace_headers()
            response = (
                await self.client.get(base_url + path, params=params, headers=headers)
                if headers
                else await self.client.get(base_url + path, params=params)
            )
            return self._unwrap_envelope(response)

        try:
            return await self.attempts.invoke(
                AttemptKind.TOOL_IO,
                ("catalog:" if base_url == self.base_url else "search:") + path,
                canonical_hash(
                    {
                        "base_url": base_url,
                        "path": path,
                        "params": {**params} if params is not None else None,
                    }
                ),
                dispatch,
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Admission failures must propagate, not masquerade as catalog failures.
            if isinstance(exc, AttemptRejected):
                raise
            raise BackendUnavailableError(
                "Catalog returned unavailable or invalid data"
            ) from exc

    async def search(
        self,
        query: str,
        limit: int,
        *,
        semantic_query: str | None = None,
        brand: str | None = None,
        category: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        in_stock_only: bool = False,
    ) -> list[ProductEvidence]:
        raw_query = query.strip()
        semantic_q = (semantic_query or "").strip()
        category_term = (category or "").strip()
        has_filters = any(
            (
                brand is not None and bool(brand.strip()),
                bool(category_term),
                min_price is not None,
                max_price is not None,
                in_stock_only,
            )
        )
        if (
            not raw_query and not semantic_q and not has_filters
        ) or not 1 <= limit <= 20:
            raise ValueError("Invalid catalog search")

        category_ids = (
            await self._resolve_category_ids(category_term) if category_term else ()
        )
        is_general_listing = raw_query.casefold() in GENERAL_LISTING_QUERIES
        lexical_q = "" if is_general_listing else raw_query

        if self.search_base_url and (lexical_q or semantic_q or has_filters):
            return await self._hybrid_search_and_ground(
                lexical_q=lexical_q,
                semantic_q=semantic_q,
                brand=brand,
                category_ids=category_ids,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
                limit=limit,
            )

        return await self._search_catalog_listing(
            keyword=lexical_q or semantic_q,
            limit=limit,
            brand=brand,
            category_ids=category_ids,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
        )

    async def _resolve_category_ids(self, category: str) -> tuple[UUID, ...]:
        body = await self._get("/categories")
        try:
            if not isinstance(body, list):
                raise TypeError("Invalid category list")
            rows = [CatalogCategory.model_validate(row) for row in body]
        except (ValidationError, TypeError) as exc:
            raise BackendUnavailableError("Catalog category list is invalid") from exc
        active = [row for row in rows if row.status == "ACTIVE"]
        term = category.casefold()
        matches = {
            row.id for row in active if (row.hardware_type or "").casefold() == term
        }
        if not matches:
            matches = {row.id for row in active if term in row.name.casefold()}
        if not matches:
            raise CatalogCategoryUnresolvedError("Unknown active catalog category")
        return tuple(sorted(matches))

    async def _search_catalog_listing(
        self,
        *,
        keyword: str,
        limit: int,
        brand: str | None,
        category_ids: tuple[UUID, ...],
        min_price: int | None,
        max_price: int | None,
        in_stock_only: bool,
    ) -> list[ProductEvidence]:
        params: dict[str, QueryParamScalar] = {"limit": limit}
        if keyword:
            params["keyword"] = keyword
        if min_price is not None:
            params["minPrice"] = min_price
        if max_price is not None:
            params["maxPrice"] = max_price

        candidates: dict[UUID, ProductEvidence] = {}
        for category_id in category_ids or (None,):
            scoped = (
                {**params, "categoryId": str(category_id)} if category_id else params
            )
            body = await self._get("/products", scoped)
            page = self._parse_catalog_summary_page(
                body,
                limit=limit,
                brand=brand,
                category_ids=category_ids,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
            )
            for item in page:
                candidates.setdefault(item.product_id, item)
        return await self._ground_filtered_candidates(
            list(candidates.values()),
            limit=limit,
            brand=brand,
            category_ids=category_ids,
            min_price=min_price,
            max_price=max_price,
            in_stock_only=in_stock_only,
        )

    async def _ground_filtered_candidates(
        self,
        candidates: list[ProductEvidence],
        *,
        limit: int,
        brand: str | None,
        category_ids: tuple[UUID, ...],
        min_price: int | None,
        max_price: int | None,
        in_stock_only: bool,
    ) -> list[ProductEvidence]:
        if min_price is None and max_price is None and not in_stock_only:
            return candidates[:limit]
        results = []
        for candidate in candidates:
            body = await self._get(f"/products/{candidate.product_id}")
            if body is None:
                continue
            item = self._ground_detail(
                body,
                expected_id=candidate.product_id,
                brand=brand,
                category_ids=category_ids,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
            )
            if item is not None:
                results.append(item)
                if len(results) == limit:
                    break
        return results

    async def _hybrid_search_and_ground(
        self,
        *,
        lexical_q: str,
        semantic_q: str,
        brand: str | None,
        category_ids: tuple[UUID, ...],
        min_price: int | None,
        max_price: int | None,
        in_stock_only: bool,
        limit: int,
    ) -> list[ProductEvidence]:
        assert self.search_base_url is not None
        search_base_url = self.search_base_url
        window_size = min(50, max(limit * 2, 20))
        base_filter_params: dict[str, QueryParamScalar] = {"size": window_size}
        if brand and brand.strip():
            base_filter_params["brand"] = brand.strip()
        if min_price is not None:
            base_filter_params["minPrice"] = min_price
        if max_price is not None:
            base_filter_params["maxPrice"] = max_price
        if in_stock_only:
            base_filter_params["inStockOnly"] = True

        async def dispatch() -> list[ProductEvidence]:
            branches = []
            if lexical_q:
                branches.append({"q": lexical_q})
            if semantic_q:
                branches.append({"semanticQuery": semantic_q})
            ranked_lists = []
            for category_id in category_ids or (None,):
                scoped = dict(base_filter_params)
                if category_id is not None:
                    scoped["categoryId"] = str(category_id)
                for branch in branches or [{}]:
                    data = await self._get_from(
                        search_base_url, "/products/search", {**scoped, **branch}
                    )
                    ranked_lists.append(self._parse_search_hit_ids(data))

            fused_ids = rrf_fuse_by_product_id(*ranked_lists)

            grounded: list[ProductEvidence] = []
            for pid in fused_ids:
                if len(grounded) >= limit:
                    break
                detail_body = await self._get(f"/products/{pid}")
                if detail_body is None:
                    continue
                evidence = self._ground_detail(
                    detail_body,
                    expected_id=pid,
                    min_price=min_price,
                    max_price=max_price,
                    in_stock_only=in_stock_only,
                    brand=brand,
                    category_ids=category_ids,
                )
                if evidence is not None:
                    grounded.append(evidence)

            if grounded:
                return grounded

            return await self._search_catalog_listing(
                keyword=lexical_q or semantic_q,
                limit=limit,
                brand=brand,
                category_ids=category_ids,
                min_price=min_price,
                max_price=max_price,
                in_stock_only=in_stock_only,
            )

        try:
            return await dispatch()
        except (httpx.HTTPError, ValidationError, ValueError, TypeError) as exc:
            if isinstance(exc, AttemptRejected):
                raise
            raise BackendUnavailableError(
                "Catalog search returned unavailable or invalid data"
            ) from exc

    @staticmethod
    def _parse_search_hit_ids(data: Any) -> list[UUID]:
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise TypeError("Invalid search-service page")
        hits = [SearchHitSummary.model_validate(item) for item in data["content"]]
        return [hit.id for hit in hits]

    def _ground_detail(
        self,
        body: Any,
        *,
        expected_id: UUID,
        min_price: int | None = None,
        max_price: int | None = None,
        in_stock_only: bool = False,
        brand: str | None = None,
        category_ids: tuple[UUID, ...] = (),
    ) -> ProductEvidence | None:
        row = Detail.model_validate(body)
        if row.id != expected_id or any(v.product_id != row.id for v in row.variants):
            raise ValueError("Product/variant identity mismatch")
        if row.status != "ACTIVE":
            return None
        if brand and (
            not row.brand_name
            or brand.strip().casefold() not in row.brand_name.casefold()
        ):
            return None
        if category_ids and row.category_id not in category_ids:
            return None
        active_variants = [v for v in row.variants if v.status == "ACTIVE"]
        matching_variants = [
            v
            for v in active_variants
            if (min_price is None or v.price >= min_price)
            and (max_price is None or v.price <= max_price)
            and (not in_stock_only or v.quantity > 0)
        ]
        if (
            min_price is not None or max_price is not None or in_stock_only
        ) and not matching_variants:
            return None
        candidate_variants = matching_variants or active_variants
        prices = {v.price for v in candidate_variants}
        price = next(iter(prices)) if len(prices) == 1 else None
        return self._evidence(
            row.id,
            row.name,
            price,
            row.specifications,
            row.model_dump(mode="json"),
            seo_name=row.seo_name,
            image_url=row.resolved_image_url(),
        )

    def _parse_catalog_summary_page(
        self,
        body: Any,
        *,
        limit: int,
        brand: str | None = None,
        category_ids: tuple[UUID, ...] = (),
        min_price: int | None = None,
        max_price: int | None = None,
        in_stock_only: bool = False,
    ) -> list[ProductEvidence]:
        try:
            if not isinstance(body, dict) or not isinstance(body.get("items"), list):
                raise TypeError("Invalid product page")
            rows = [Summary.model_validate(row) for row in body["items"]]
            if len(rows) > limit or len({row.id for row in rows}) != len(rows):
                raise ValueError("Invalid product page identities/limit")
            results: list[ProductEvidence] = []
            for row in rows:
                if row.status != "ACTIVE":
                    continue
                if brand and (
                    not row.brand_name
                    or brand.strip().casefold() not in row.brand_name.casefold()
                ):
                    continue
                if category_ids and row.category_id not in category_ids:
                    continue
                if (
                    in_stock_only
                    and row.total_quantity is not None
                    and row.total_quantity <= 0
                ):
                    continue
                if (
                    min_price is not None
                    and row.max_price is not None
                    and row.max_price < min_price
                ):
                    continue
                if (
                    max_price is not None
                    and row.min_price is not None
                    and row.min_price > max_price
                ):
                    continue
                results.append(
                    self._evidence(
                        row.id,
                        row.name,
                        row.min_price if row.min_price == row.max_price else None,
                        {},
                        row.model_dump(mode="json"),
                        seo_name=row.seo_name,
                        image_url=row.main_image_url,
                    )
                )
            return results
        except (ValidationError, ValueError, TypeError) as exc:
            raise BackendUnavailableError("Catalog product page is invalid") from exc

    async def resolve(self, mention: str) -> list[ProductEvidence]:
        """Exhaust bounded keyword pages; never treat top-k as uniqueness proof."""
        normalized = " ".join(mention.casefold().split())
        if not normalized or len(mention) > 1000:
            raise ValueError("Invalid product mention")
        matches: list[ProductEvidence] = []
        seen_ids: set[UUID] = set()
        seen_cursors: set[str] = set()
        cursor: str | None = None
        for _ in range(MAX_RESOLUTION_PAGES):
            params: dict[str, QueryParamScalar] = {
                "keyword": mention,
                "limit": RESOLUTION_PAGE_SIZE,
            }
            if cursor is not None:
                params["cursor"] = cursor
            body = await self._get("/products", params)
            try:
                page = ResolutionPage.model_validate(body)
                for row in page.items:
                    if row.id in seen_ids:
                        raise ValueError("Duplicate product across resolution pages")
                    seen_ids.add(row.id)
                    if len(seen_ids) > MAX_RESOLUTION_PRODUCTS:
                        raise ValueError("Product resolution exceeds candidate bound")
                    if row.status == "ACTIVE" and normalized in " ".join(
                        row.name.casefold().split()
                    ):
                        matches.append(
                            self._evidence(
                                row.id,
                                row.name,
                                row.min_price
                                if row.min_price == row.max_price
                                else None,
                                {},
                                row.model_dump(mode="json"),
                                seo_name=row.seo_name,
                                image_url=row.main_image_url,
                            )
                        )
                if not page.has_next:
                    return matches
                cursor = page.next_cursor
                if not cursor or cursor in seen_cursors:
                    raise ValueError("Product resolution cursor does not advance")
                seen_cursors.add(cursor)
            except (ValidationError, ValueError, TypeError) as exc:
                raise BackendUnavailableError(
                    "Canonical product resolution is incomplete or invalid"
                ) from exc
        raise BackendUnavailableError("Product resolution exceeds page bound")

    async def get(self, product_id: UUID) -> ProductEvidence | None:
        body = await self._get(f"/products/{product_id}")
        if body is None:
            return None
        try:
            return self._ground_detail(body, expected_id=product_id)
        except (ValidationError, ValueError, TypeError) as exc:
            raise BackendUnavailableError("Catalog product detail is invalid") from exc

    @staticmethod
    def _evidence(
        identifier: UUID,
        name: str,
        price: int | None,
        specifications: dict[str, JsonValue],
        snapshot: dict[str, Any],
        *,
        seo_name: str | None = None,
        image_url: str | None = None,
    ) -> ProductEvidence:
        return ProductEvidence(
            product_id=identifier,
            name=name,
            price_vnd=price,
            seo_name=seo_name.strip()
            if isinstance(seo_name, str) and seo_name.strip()
            else None,
            image_url=image_url.strip()
            if isinstance(image_url, str) and image_url.strip()
            else None,
            specifications=specifications,
            evidence_ref=f"catalog:product:{identifier}:{canonical_hash(snapshot)}",
            retrieved_at=datetime.now(UTC),
        )
