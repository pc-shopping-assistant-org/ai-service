"""DuckDuckGo HTML web search adapter for live hardware intelligence."""

from __future__ import annotations

import html
import logging
from collections.abc import Callable
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

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


class _ResultParser(HTMLParser):
    """Pair each result heading with its snippet; never invent source titles."""

    def __init__(self, clean_url: Callable[[str], str], limit: int):
        super().__init__(convert_charrefs=True)
        self.clean_url, self.limit = clean_url, limit
        self.items: list[WebSearchItem] = []
        self.no_results = False
        self.in_heading = False
        self.capture: str | None = None
        self.url = ""
        self.title: list[str] = []
        self.snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if "no-results" in classes:
            self.no_results = True
        if tag == "h2" and "result__title" in classes:
            self.in_heading = True
            self.capture = None
            self.url, self.title, self.snippet = "", [], []
        elif tag == "a":
            if self.in_heading and classes.intersection({"result__a", "result__url"}):
                self.url = self.clean_url(values.get("href") or "")
                self.capture = "title"
            elif "result__snippet" in classes and self.url:
                self.capture = "snippet"

    def handle_data(self, data: str) -> None:
        if self.capture == "title":
            self.title.append(data)
        elif self.capture == "snippet":
            self.snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self.in_heading = False
        if tag != "a":
            return
        if self.capture == "snippet":
            title = " ".join("".join(self.title).split())
            snippet = " ".join("".join(self.snippet).split())
            if title and len(self.items) < self.limit:
                self.items.append(WebSearchItem(title=title, url=self.url, snippet=snippet))
            self.url = ""
        self.capture = None

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
                if res.status_code == 200:
                    parsed = _ResultParser(self._clean_url, query.limit)
                    parsed.feed(res.text)
                    items = parsed.items
                    if not items and not parsed.no_results:
                        return self._fallback_result(query)
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

    def _clean_url(self, raw_url: str) -> str:
        if not raw_url:
            return ""
        url = urljoin("https://duckduckgo.com/", html.unescape(raw_url))
        parsed = urlparse(url)
        if parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"} and parsed.path == "/y.js":
            return ""  # Advertising click-through, not a research source.
        if parsed.path == "/l/":
            qs = parse_qs(parsed.query)
            if qs.get("uddg"):
                url = qs["uddg"][0]  # parse_qs already decodes once.
        parsed = urlparse(url)
        return url if parsed.scheme in {"http", "https"} and parsed.hostname else ""

    def _parse_html(self, html_text: str, limit: int = 5) -> list[WebSearchItem]:
        parsed = _ResultParser(self._clean_url, limit)
        parsed.feed(html_text)
        return parsed.items

    def _fallback_result(self, query: WebSearchQuery) -> WebSearchResult:
        return WebSearchResult(
            query=query.query,
            category=query.category,
            note="WEB_SEARCH_UNAVAILABLE",
        )


__all__ = ["DuckDuckGoSearchAdapter"]
