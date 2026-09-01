from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ai_service.application.ports.catalog import (
    BackendCatalogClient,
    CatalogRetriever,
)

__all__ = ["BackendCatalogClient", "BackendCatalogRetriever", "CatalogRetriever"]


class BackendCatalogRetriever:
    """Grounded fallback retriever backed by the canonical catalog API.

    The backend currently exposes keyword search rather than embeddings.  A
    full natural-language query is attempted first; when it has no hits, a
    small set of meaningful terms is queried and de-duplicated.  This makes
    local Vietnamese requests useful without misrepresenting the fallback as
    semantic/vector retrieval.
    """

    _STOP_WORDS = frozenset(
        {
            "cho",
            "cần",
            "có",
            "giúp",
            "mình",
            "một",
            "muốn",
            "nào",
            "này",
            "phù",
            "tôi",
            "tìm",
            "và",
            "với",
        }
    )
    _MAX_FALLBACK_TERMS = 6

    def __init__(self, backend_client: BackendCatalogClient) -> None:
        self.backend_client = backend_client

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        normalized_query = " ".join(query.split())
        primary = await self.backend_client.search_products(normalized_query, limit)
        if primary:
            return primary[:limit]

        terms = self._meaningful_terms(normalized_query)
        if len(terms) <= 1:
            return primary

        term_results = await asyncio.gather(
            *(self.backend_client.search_products(term, limit) for term in terms)
        )
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rows in term_results:
            for row in rows:
                key = self._record_key(row)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(row)
                if len(merged) >= limit:
                    return merged
        return merged

    @classmethod
    def _meaningful_terms(cls, query: str) -> list[str]:
        terms = [
            term.casefold()
            for term in re.findall(r"[^\W_]+", query, flags=re.UNICODE)
            if len(term) >= 2 and term.casefold() not in cls._STOP_WORDS
        ]
        return list(dict.fromkeys(terms))[: cls._MAX_FALLBACK_TERMS]

    @staticmethod
    def _record_key(row: dict[str, Any]) -> str:
        for field in ("id", "sku", "seoName", "seo_name", "name"):
            value = row.get(field)
            if value is not None:
                return f"{field}:{value}"
        return "row:" + json.dumps(row, sort_keys=True, default=str)
