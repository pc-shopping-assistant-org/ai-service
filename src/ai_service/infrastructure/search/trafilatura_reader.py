"""HTTPX + Trafilatura web page reader for extracting clean content and tables."""

from __future__ import annotations

import hashlib
import logging

import httpx
import trafilatura

from ai_service.application.cpu_work import run_cpu
from ai_service.application.ports.web_search import (
    ExtractedWebFact,
    WebExtractionStatus,
    WebPageReader,
    WebPageReadResult,
    WebPageReadStatus,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.config.settings import Settings, get_settings
from ai_service.infrastructure.search.asus_techspec import extract_asus_dimensions
from ai_service.infrastructure.search.techpowerup_gpu_specs import (
    extract_techpowerup_gpu_dimensions,
)
from ai_service.infrastructure.search.tomshardware_gpu_specs import (
    extract_tomshardware_gpu_dimensions,
)

LOGGER = logging.getLogger(__name__)

CURATED_PARSERS = [
    extract_asus_dimensions,
    extract_techpowerup_gpu_dimensions,
    extract_tomshardware_gpu_dimensions,
]


def _extract_html(
    html: str, final_url: str
) -> tuple[str | None, str | None, list[ExtractedWebFact] | None]:
    content = trafilatura.extract(html, include_tables=True, output_format="markdown")
    metadata = trafilatura.extract_metadata(html)
    facts: list[ExtractedWebFact] | None = None
    for parser in CURATED_PARSERS:
        facts = parser(html, final_url)
        if facts:
            break
    if not content and facts:
        content = "\n".join(f.model + "\n" + f.quote for f in facts)
    return content, metadata.title if metadata else None, facts


class TrafilaturaReader(WebPageReader):
    """Fetches URL via httpx and extracts readable content/tables using Trafilatura."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def read(self, url: str) -> WebPageReadResult:
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
                },
            ) as client:
                res = await client.get(url)
                final_url = str(res.url)

                if not res.is_success or not res.text.strip():
                    return WebPageReadResult(
                        url=url,
                        final_url=final_url,
                        status=WebPageReadStatus.EMPTY
                        if res.is_success
                        else WebPageReadStatus.UNREADABLE,
                    )

                extracted, title, facts = await run_cpu(
                    lambda: _extract_html(res.text, final_url)
                )

                if not extracted or not extracted.strip():
                    return WebPageReadResult(
                        url=url,
                        final_url=final_url,
                        title=title,
                        status=WebPageReadStatus.EMPTY
                        if not res.text.strip()
                        else WebPageReadStatus.UNREADABLE,
                    )

                cleaned_content = extracted.strip()
                content_hash = hashlib.sha256(
                    cleaned_content.encode("utf-8")
                ).hexdigest()
                evidence_ref = "web:" + canonical_hash(
                    {"final_url": final_url, "content_hash": content_hash}
                )

                return WebPageReadResult(
                    url=url,
                    final_url=final_url,
                    title=title,
                    status=WebPageReadStatus.READABLE,
                    content_hash=content_hash,
                    content=cleaned_content,
                    evidence_ref=evidence_ref,
                    extraction_status=WebExtractionStatus.STRUCTURED
                    if facts
                    else WebExtractionStatus.CONTENT_ONLY,
                    document_hash=hashlib.sha256(res.text.encode()).hexdigest(),
                    facts=facts or [],
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch or extract web page %s: %s", url, exc)
            return WebPageReadResult(
                url=url,
                final_url=url,
                status=WebPageReadStatus.UNREADABLE,
            )
