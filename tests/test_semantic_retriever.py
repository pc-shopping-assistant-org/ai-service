from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from ai_service.config.settings import Settings
from ai_service.index_catalog import _fetch_catalog
from ai_service.services.backend_client import BackendUnavailableError, CatalogPage
from ai_service.services.semantic_retriever import (
    HashEmbeddingProvider,
    HybridCatalogRetriever,
    QdrantCatalogIndexer,
    QdrantCatalogRetriever,
    build_catalog_retriever,
)


class FakeBackend:
    async def search_products(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        return [{"id": "fallback", "name": f"Fallback {query}"}][:limit]

    async def get_product(self, product_id: UUID | str) -> dict[str, Any] | None:
        return {"id": str(product_id), "name": "Hydrated product"}


class FakeEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class FakeVectorStore:
    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        self.hits = hits or []
        self.search_vectors: list[list[float]] = []
        self.collections: list[int] = []
        self.points: list[dict[str, Any]] = []

    async def search(self, vector: list[float], limit: int) -> list[dict[str, Any]]:
        self.search_vectors.append(vector)
        return self.hits[:limit]

    async def ensure_collection(self, dimension: int) -> None:
        self.collections.append(dimension)

    async def upsert(self, points: list[dict[str, Any]]) -> None:
        self.points.extend(points)


@pytest.mark.asyncio
async def test_qdrant_retriever_returns_payload_rows_and_hydrates_id_only_hits() -> None:
    store = FakeVectorStore(
        [
            {"score": 0.9, "payload": {"id": "payload-1", "name": "Payload PC"}},
            {"score": 0.8, "payload": {"product_id": "hydrated-2"}},
        ]
    )

    rows = await QdrantCatalogRetriever(FakeBackend(), FakeEmbedding(), store).search(
        "laptop gaming", 5
    )

    assert [row["name"] for row in rows] == ["Payload PC", "Hydrated product"]
    assert store.search_vectors == [[13.0, 1.0]]


@pytest.mark.asyncio
async def test_qdrant_retriever_drops_hidden_catalog_payloads() -> None:
    store = FakeVectorStore(
        [
            {"score": 0.95, "payload": {"id": "hidden", "name": "Hidden PC", "status": "INACTIVE"}},
            {"score": 0.90, "payload": {"id": "deleted", "name": "Deleted PC", "status": "DELETED"}},
            {"score": 0.85, "payload": {"id": "active", "name": "Active PC", "status": " active "}},
        ]
    )

    rows = await QdrantCatalogRetriever(FakeBackend(), FakeEmbedding(), store).search(
        "laptop gaming", 5
    )

    assert [row["name"] for row in rows] == ["Active PC"]


@pytest.mark.asyncio
async def test_qdrant_retriever_rechecks_statusless_legacy_payloads() -> None:
    class CanonicalBackend(FakeBackend):
        async def get_product(self, product_id: UUID | str) -> dict[str, Any] | None:
            if str(product_id) == "hidden":
                return {"id": "hidden", "name": "Hidden PC", "status": "INACTIVE"}
            return {"id": str(product_id), "name": "Active PC", "status": "ACTIVE"}

    store = FakeVectorStore(
        [
            {"score": 0.95, "payload": {"id": "hidden", "name": "Hidden PC"}},
            {"score": 0.90, "payload": {"id": "active", "name": "Active PC"}},
        ]
    )

    rows = await QdrantCatalogRetriever(CanonicalBackend(), FakeEmbedding(), store).search(
        "laptop gaming", 5
    )

    assert [row["name"] for row in rows] == ["Active PC"]


@pytest.mark.asyncio
async def test_hybrid_retriever_falls_back_when_vector_store_is_unavailable() -> None:
    class UnavailableRetriever:
        async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
            raise BackendUnavailableError("qdrant down")

    fallback = build_catalog_retriever(FakeBackend(), Settings())
    result = await HybridCatalogRetriever(UnavailableRetriever(), fallback).search("gaming", 2)

    assert result == [{"id": "fallback", "name": "Fallback gaming"}]


@pytest.mark.asyncio
async def test_indexer_creates_collection_and_upserts_embedded_products() -> None:
    store = FakeVectorStore()
    count = await QdrantCatalogIndexer(FakeEmbedding(), store).index(
        [
            {"id": "product-1", "name": "Gaming laptop", "description": "fast"},
            {"id": "product-2", "name": "Hidden laptop", "status": "DELETED"},
        ]
    )

    assert count == 1
    assert store.collections == [2]
    assert store.points[0]["id"] == "product-1"
    assert store.points[0]["payload"]["name"] == "Gaming laptop"


@pytest.mark.asyncio
async def test_hash_embedding_is_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimension=8)
    first = await provider.embed("laptop gaming")
    second = await provider.embed("laptop gaming")

    assert first == second
    assert len(first) == 8
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_factory_keeps_backend_default_and_builds_configured_hybrid() -> None:
    backend = FakeBackend()
    assert type(build_catalog_retriever(backend, Settings())).__name__ == "BackendCatalogRetriever"
    configured = build_catalog_retriever(
        backend,
        Settings(
            retrieval_backend="hybrid",
            qdrant_url="http://qdrant:6333",
            embedding_provider="hash",
        ),
    )
    assert isinstance(configured, HybridCatalogRetriever)


@pytest.mark.asyncio
async def test_catalog_indexer_fetches_all_backend_cursor_pages() -> None:
    class PagedBackend:
        def __init__(self) -> None:
            self.cursors: list[str | None] = []

        async def list_products(self, *, cursor: str | None, limit: int) -> CatalogPage:
            self.cursors.append(cursor)
            if cursor is None:
                return CatalogPage([{"id": "one"}], next_cursor="cursor-1", has_next=True)
            return CatalogPage([{"id": "two"}], has_next=False)

    backend = PagedBackend()
    products = await _fetch_catalog(backend)  # type: ignore[arg-type]

    assert products == [{"id": "one"}, {"id": "two"}]
    assert backend.cursors == [None, "cursor-1"]
