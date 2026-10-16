"""Reserve each search/read before multi-source research dispatches external I/O."""

from ai_service.application.call_attempts import AttemptBoundary, AttemptKind
from ai_service.application.ports.web_search import (
    WebPageReader,
    WebPageReadResult,
    WebSearchClient,
    WebSearchQuery,
    WebSearchResult,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.infrastructure.observability import step


class BudgetedWebSearch:
    def __init__(self, client: WebSearchClient, attempts: AttemptBoundary):
        self.client, self.attempts = client, attempts

    async def search(self, query: WebSearchQuery) -> WebSearchResult:
        digest = canonical_hash(query)
        with step("web.search"):
            return await self.attempts.invoke(
                AttemptKind.TOOL_IO,
                "web:search:" + digest,
                digest,
                lambda: self.client.search(query),
            )


class BudgetedWebReader:
    def __init__(self, reader: WebPageReader, attempts: AttemptBoundary):
        self.reader, self.attempts = reader, attempts

    async def read(self, url: str) -> WebPageReadResult:
        digest = canonical_hash({"url": url})
        with step("web.read") as observation:
            result = await self.attempts.invoke(
                AttemptKind.TOOL_IO,
                "web:read:" + digest,
                digest,
                lambda: self.reader.read(url),
            )
            observation.result_status = result.status
            return result
