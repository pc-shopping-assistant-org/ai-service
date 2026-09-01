from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.catalog import BackendCatalogClient, CatalogRetriever
from ai_service.config.enums import EmbeddingProviderKind, RetrievalBackend
from ai_service.config.settings import Settings
from ai_service.services.retriever import BackendCatalogRetriever

LOGGER = logging.getLogger(__name__)


def _is_active_catalog_row(row: dict[str, Any]) -> bool:
    """Keep stale vector payloads from leaking hidden catalog records.

    The backend catalog is the source of truth. Index records created before
    a product was hidden/deleted may still exist in Qdrant until the next
    synchronization, so a status that is present in a payload must be
    ``ACTIVE`` before it is exposed to the assistant. Fixtures and older
    index rows without a status remain compatible and can be validated by
    the backend path when necessary.
    """
    status = row.get("status")
    return status is None or str(status).strip().upper() == "ACTIVE"


class EmbeddingProvider(Protocol):
    """Create vectors for catalog documents and natural-language queries."""

    async def embed(self, text: str) -> list[float]:
        """Return one embedding vector for ``text``."""


class VectorStore(Protocol):
    """Minimal vector-store port used by semantic retrieval and indexing."""

    async def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        """Return Qdrant-like hits with a score and payload."""

    async def ensure_collection(self, dimension: int) -> None:
        """Create the collection when it does not exist."""

    async def upsert(self, points: list[dict[str, Any]]) -> None:
        """Persist embedded catalog points."""


class HashEmbeddingProvider:
    """Small deterministic local vectorizer for development and tests.

    This is deliberately not presented as a production language model. It
    makes the Qdrant contract executable without credentials; production should
    use :class:`HttpEmbeddingProvider` with an approved embedding provider.
    """

    _TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 2:
            raise ValueError("Embedding dimension must be at least 2")
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._TOKEN_PATTERN.findall(text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class HttpEmbeddingProvider:
    """OpenAI-compatible-ish HTTP embedding adapter.

    The endpoint may return either ``{"embedding": [...]}`` or the common
    ``{"data": [{"embedding": [...]}]}`` shape. The provider is kept behind a
    protocol so a vendor-specific adapter can be substituted without touching
    the assistant or HTTP schemas.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    async def embed(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    self.endpoint,
                    json={"input": text},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Embedding service is unavailable") from exc

        embedding: Any = payload.get("embedding") if isinstance(payload, dict) else None
        if embedding is None and isinstance(payload, dict):
            rows = payload.get("data")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                embedding = rows[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise BackendUnavailableError("Embedding service returned an invalid vector")
        try:
            vector = [float(value) for value in embedding]
        except (TypeError, ValueError) as exc:
            raise BackendUnavailableError("Embedding service returned an invalid vector") from exc
        if not all(math.isfinite(value) for value in vector):
            raise BackendUnavailableError("Embedding service returned a non-finite vector")
        return vector


class HttpQdrantVectorStore:
    """HTTP client for the Qdrant collection used by semantic retrieval."""

    def __init__(
        self,
        base_url: str,
        collection: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_path = quote(collection, safe="")
        self.timeout_seconds = timeout_seconds

    @property
    def _collection_url(self) -> str:
        return f"{self.base_url}/collections/{self.collection_path}"

    async def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self._collection_url}/points/query",
                    json={
                        "query": vector,
                        "limit": limit,
                        "with_payload": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Vector store is unavailable") from exc

        results = payload.get("result", []) if isinstance(payload, dict) else []
        if isinstance(results, dict):
            results = results.get("points", [])
        return [hit for hit in results if isinstance(hit, dict)] if isinstance(results, list) else []

    async def ensure_collection(self, dimension: int) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(self._collection_url)
                if response.status_code == httpx.codes.NOT_FOUND:
                    response = await client.put(
                        self._collection_url,
                        json={"vectors": {"size": dimension, "distance": "Cosine"}},
                    )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Vector store is unavailable") from exc

    async def upsert(self, points: list[dict[str, Any]]) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.put(
                    f"{self._collection_url}/points?wait=true",
                    json={"points": points},
                )
                response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise BackendUnavailableError("Vector store is unavailable") from exc


class QdrantCatalogRetriever:
    """Retrieve backend-shaped catalog rows from a vector store."""

    def __init__(
        self,
        backend_client: BackendCatalogClient,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.backend_client = backend_client
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        vector = await self.embedding_provider.embed(" ".join(query.split()))
        hits = await self.vector_store.search(vector, limit)
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits:
            payload = hit.get("payload")
            if not isinstance(payload, dict):
                continue
            row = self._payload_row(payload)
            if row is None:
                row = await self._load_product(payload)
            if row is None or not await self._is_visible_row(row):
                continue
            key = self._record_key(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows

    async def _is_visible_row(self, row: dict[str, Any]) -> bool:
        """Validate status-less legacy payloads against the backend source.

        New index points carry ``status`` and can be filtered locally.  Older
        points may not, so an ID-bearing row is rechecked against the backend
        before it enters model context.  This keeps compatibility with old
        indexes while ensuring a hidden/deleted catalog row cannot bypass the
        visibility boundary merely by omitting its status field.
        """
        if not _is_active_catalog_row(row):
            return False
        if row.get("status") is not None:
            return True

        raw_id = row.get("id", row.get("product_id"))
        get_product = getattr(self.backend_client, "get_product", None)
        if raw_id is None or not callable(get_product):
            return False
        try:
            canonical = await get_product(raw_id)
        except BackendUnavailableError:
            raise
        except (ValueError, TypeError):
            return False
        if not isinstance(canonical, dict):
            return False
        return _is_active_catalog_row(canonical)

    @staticmethod
    def _payload_row(payload: dict[str, Any]) -> dict[str, Any] | None:
        nested = payload.get("product")
        if isinstance(nested, dict) and nested.get("name"):
            # Some indexers wrap the canonical product under ``product`` and
            # keep freshness/status metadata on the hit payload. Carry that
            # status into the row so the visibility gate cannot be bypassed by
            # a stale wrapper.
            if "status" not in nested and payload.get("status") is not None:
                return {**nested, "status": payload["status"]}
            return nested
        return payload if payload.get("name") else None

    async def _load_product(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        raw_id = payload.get("product_id", payload.get("id"))
        if raw_id is None:
            return None
        get_product = getattr(self.backend_client, "get_product", None)
        if not callable(get_product):
            return None
        try:
            result = await get_product(raw_id)
        except BackendUnavailableError:
            # Do not silently turn a source-of-truth outage into an empty
            # result when the vector payload needs backend hydration.
            raise
        except (ValueError, TypeError):
            return None
        return result if isinstance(result, dict) and result.get("name") else None

    @staticmethod
    def _record_key(row: dict[str, Any]) -> str:
        for field in ("id", "sku", "seoName", "seo_name", "name"):
            value = row.get(field)
            if value is not None:
                return f"{field}:{value}"
        return repr(sorted(row.items()))


class HybridCatalogRetriever:
    """Prefer vector retrieval and keep the backend retriever as a safe fallback."""

    def __init__(
        self,
        semantic_retriever: CatalogRetriever,
        fallback_retriever: CatalogRetriever,
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.fallback_retriever = fallback_retriever

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            semantic_rows = await self.semantic_retriever.search(query, limit)
            if semantic_rows:
                return semantic_rows[:limit]
        except BackendUnavailableError as exc:
            LOGGER.warning("Semantic retrieval unavailable; using backend fallback: %s", exc)
        return await self.fallback_retriever.search(query, limit)


class QdrantCatalogIndexer:
    """Create/update the Qdrant product collection from backend catalog rows."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def index(self, products: list[dict[str, Any]]) -> int:
        active_products = [
            product
            for product in products
            if product.get("id") is not None and _is_active_catalog_row(product)
        ]
        if not active_products:
            return 0
        vectors = await self._embed_products(active_products)
        dimension = len(vectors[0])
        await self.vector_store.ensure_collection(dimension)
        points = [
            {
                "id": str(product.get("id")),
                "vector": vector,
                "payload": product,
            }
            for product, vector in zip(active_products, vectors, strict=True)
        ]
        if not points:
            return 0
        await self.vector_store.upsert(points)
        return len(points)

    async def _embed_products(self, products: list[dict[str, Any]]) -> list[list[float]]:
        return await _gather_embeddings(self.embedding_provider, products)


async def _gather_embeddings(
    provider: EmbeddingProvider,
    products: list[dict[str, Any]],
) -> list[list[float]]:
        return await asyncio.gather(
            *(provider.embed(_catalog_text(product)) for product in products)
        )


def build_catalog_retriever(
    backend_client: BackendCatalogClient,
    settings: Settings,
) -> CatalogRetriever:
    """Build the configured retrieval port without changing route contracts.

    ``backend`` remains the safe default. ``qdrant`` requires both a vector
    store URL and an embedding endpoint (or the explicit local ``hash``
    provider); ``hybrid`` falls back to the backend keyword retriever whenever
    vector infrastructure is unavailable.
    """
    fallback = BackendCatalogRetriever(backend_client)
    if settings.retrieval_backend == RetrievalBackend.BACKEND or not settings.qdrant_url:
        return fallback

    if settings.embedding_provider == EmbeddingProviderKind.HASH:
        embedding_provider: EmbeddingProvider = HashEmbeddingProvider(settings.embedding_dimension)
    elif settings.embedding_api_url:
        embedding_provider = HttpEmbeddingProvider(
            settings.embedding_api_url,
            api_key=settings.embedding_api_key,
            timeout_seconds=settings.request_timeout_seconds,
        )
    else:
        LOGGER.warning(
            "Vector retrieval requested without an embedding endpoint; using backend fallback"
        )
        return fallback

    semantic = QdrantCatalogRetriever(
        backend_client,
        embedding_provider,
        HttpQdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_collection,
            timeout_seconds=settings.request_timeout_seconds,
        ),
    )
    return (
        semantic
        if settings.retrieval_backend == RetrievalBackend.QDRANT
        else HybridCatalogRetriever(semantic, fallback)
    )


def _catalog_text(product: dict[str, Any]) -> str:
    fields = (
        product.get("name"),
        product.get("seoName", product.get("seo_name")),
        product.get("description"),
        product.get("specifications"),
    )
    return " ".join(str(field) for field in fields if field is not None)
