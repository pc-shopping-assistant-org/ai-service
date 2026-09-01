from __future__ import annotations

import asyncio

from ai_service.application.ports.catalog import CatalogPage
from ai_service.config.enums import EmbeddingProviderKind
from ai_service.config.settings import Settings, get_settings
from ai_service.services.backend_client import BackendClient
from ai_service.services.semantic_retriever import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    HttpEmbeddingProvider,
    HttpQdrantVectorStore,
    QdrantCatalogIndexer,
)


def _embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == EmbeddingProviderKind.HASH:
        return HashEmbeddingProvider(settings.embedding_dimension)
    if not settings.embedding_api_url:
        raise RuntimeError(
            "AI_EMBEDDING_API_URL is required when AI_EMBEDDING_PROVIDER=http"
        )
    return HttpEmbeddingProvider(
        settings.embedding_api_url,
        api_key=settings.embedding_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )


async def index_catalog(settings: Settings | None = None) -> int:
    """Fetch the complete active catalog through cursor pages and index it."""
    runtime_settings = settings or get_settings()
    if not runtime_settings.qdrant_url:
        raise RuntimeError("AI_QDRANT_URL is required to index the catalog")

    backend = BackendClient(runtime_settings)
    products = await _fetch_catalog(backend)
    indexer = QdrantCatalogIndexer(
        _embedding_provider(runtime_settings),
        HttpQdrantVectorStore(
            runtime_settings.qdrant_url,
            runtime_settings.qdrant_collection,
            timeout_seconds=runtime_settings.request_timeout_seconds,
        ),
    )
    return await indexer.index(products)


async def _fetch_catalog(backend: BackendClient, page_size: int = 100) -> list[dict]:
    """Read every catalog page while guarding against a broken cursor."""
    products: list[dict] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        page: CatalogPage = await backend.list_products(cursor=cursor, limit=page_size)
        products.extend(page.items)
        if not page.has_next:
            return products
        if not page.next_cursor or page.next_cursor in seen_cursors:
            raise RuntimeError("Catalog endpoint returned an invalid pagination cursor")
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor


def run() -> None:
    count = asyncio.run(index_catalog())
    print(f"Indexed {count} catalog products")


if __name__ == "__main__":
    run()
