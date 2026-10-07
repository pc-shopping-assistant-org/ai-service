"""DuckDuckGo HTML web search adapter for live hardware intelligence."""

from __future__ import annotations

import html
import logging
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchClient,
    WebSearchItem,
    WebSearchQuery,
    WebSearchResult,
)
from ai_service.config.settings import Settings, get_settings

LOGGER = logging.getLogger(__name__)

DOMAIN_SUGGESTIONS: dict[WebSearchCategory, list[str]] = {
    WebSearchCategory.COMPATIBILITY: ["reddit.com", "tomshardware.com", "techpowerup.com"],
    WebSearchCategory.REVIEWS: ["techpowerup.com", "tomshardware.com", "gamersnexus.net"],
    WebSearchCategory.BUILD_GUIDES: ["reddit.com/r/buildapc", "pcpartpicker.com"],
    WebSearchCategory.GAME_REQUIREMENTS: ["steampowered.com", "systemreqs.com"],
    WebSearchCategory.HARDWARE_ISSUES: ["tomshardware.com", "techpowerup.com", "reddit.com"],
    WebSearchCategory.PSU_TIER: ["cultists.network", "linustechtips.com"],
}


class DuckDuckGoSearchAdapter(WebSearchClient):
    """Zero-dependency, keyless search adapter querying DuckDuckGo HTML endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def search(self, query: WebSearchQuery) -> WebSearchResult:
        enriched_query = self._enrich_query(query)
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.request_timeout_seconds,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
            ) as client:
                res = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": enriched_query},
                )
                if res.is_success:
                    items = self._parse_html(res.text, limit=query.limit)
                    return WebSearchResult(
                        query=query.query, category=query.category,
                        items=items, total_results=len(items),
                        note=None if items else "WEB_SEARCH_NO_RESULTS",
                    )
        except Exception as exc:  # noqa: BLE001 - network or parsing error falls back gracefully
            LOGGER.warning("DuckDuckGo search unavailable: %s", exc)

        return self._fallback_result(query)

    def _enrich_query(self, query: WebSearchQuery) -> str:
        base = query.query.strip()
        if query.target_domains:
            domains_filter = " OR ".join(f"site:{d}" for d in query.target_domains)
            return f"{base} ({domains_filter})"

        if query.category == WebSearchCategory.PSU_TIER:
            return f"{base} PSU Tier List Cultists Network"
        if query.category == WebSearchCategory.GAME_REQUIREMENTS:
            return f"{base} system requirements PC minimum recommended"
        if query.category == WebSearchCategory.HARDWARE_ISSUES:
            return f"{base} known issues defect stability temperature"

        return base

    def _clean_text(self, raw_html: str) -> str:
        clean = re.sub(r"<[^>]+>", "", raw_html)
        return html.unescape(clean).strip()

    def _clean_url(self, raw_url: str) -> str:
        parsed = urlparse(raw_url)
        if "/l/" in parsed.path:
            qs = parse_qs(parsed.query)
            if qs.get("uddg"):
                return unquote(qs["uddg"][0])
        return raw_url

    def _parse_html(self, html_text: str, limit: int = 5) -> list[WebSearchItem]:
        items: list[WebSearchItem] = []
        # Pattern to capture result titles and snippets
        pattern = re.compile(
            r'<h2 class="result__title">\s*<a class="result__url"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>\s*</h2>.*?'
            r'<a class="result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )

        for match in pattern.finditer(html_text):
            raw_url = match.group("url")
            raw_title = match.group("title")
            raw_snippet = match.group("snippet")

            url = self._clean_url(raw_url)
            title = self._clean_text(raw_title)
            snippet = self._clean_text(raw_snippet)

            if title and url:
                items.append(WebSearchItem(title=title, url=url, snippet=snippet))
            if len(items) >= limit:
                break

        # Fallback simpler regex if DDG markup variant differs
        if not items:
            simple_pattern = re.compile(
                r'<a class="result__snippet[^"]*" href="(?P<url>[^"]+)"[^>]*>(?P<snippet>.*?)</a>',
                re.DOTALL,
            )
            for m in simple_pattern.finditer(html_text):
                url = self._clean_url(m.group("url"))
                snippet = self._clean_text(m.group("snippet"))
                items.append(
                    WebSearchItem(
                        title=f"Kết quả tra cứu: {snippet[:50]}...",
                        url=url,
                        snippet=snippet,
                    )
                )
                if len(items) >= limit:
                    break

        return items

    def _fallback_result(self, query: WebSearchQuery) -> WebSearchResult:
        return WebSearchResult(
            query=query.query,
            category=query.category,
            note="WEB_SEARCH_UNAVAILABLE",
        )


__all__ = ["DuckDuckGoSearchAdapter"]
