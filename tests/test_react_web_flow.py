from pathlib import Path

import httpx
import pytest
from test_react_tools import Catalog, context

from ai_service.application.ports.web_search import (
    WebPageReader,
    WebPageReadResult,
    WebPageReadStatus,
    WebSearchClient,
    WebSearchItem,
    WebSearchQuery,
    WebSearchResult,
)
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ProductReview,
    ReadWebPageResult,
    SearchWebResult,
    ToolCall,
    ToolError,
    ToolName,
    WebExcerpt,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    validate_answer,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.web_search.presets import WebSearchPresets
from ai_service.infrastructure.graph.react import (
    ReactDecision,
    build_react_graph,
)
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader


class MockSearchClient(WebSearchClient):
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: WebSearchQuery) -> WebSearchResult:
        self.queries.append(query.query)
        return WebSearchResult(
            query=query.query,
            category=query.category,
            items=[
                WebSearchItem(
                    title="RTX 4070 Super Black Myth Wukong Benchmark",
                    url="https://example.com/reviews/rtx-4070-super-wukong",
                    snippet="Detailed FPS benchmark for RTX 4070 Super in Black Myth Wukong at 1440p.",
                )
            ],
            total_results=1,
        )


class MockWebPageReader(WebPageReader):
    def __init__(
        self,
        content: str | None = None,
        status: WebPageReadStatus = WebPageReadStatus.READABLE,
    ) -> None:
        self.urls: list[str] = []
        self.content = content or (
            "# RTX 4070 Super Benchmark Review\n\n"
            "Here are the test results for Black Myth Wukong at 1440p:\n"
            "Resolution: 1440p Very High -> Average 72 FPS.\n"
        )
        self.status = status

    async def read(self, url: str) -> WebPageReadResult:
        self.urls.append(url)
        content_hash = "a" * 64
        return WebPageReadResult(
            url=url,
            final_url=url,
            title="RTX 4070 Super Benchmark Review",
            status=self.status,
            content_hash=content_hash
            if self.status == WebPageReadStatus.READABLE
            else None,
            content=self.content if self.status == WebPageReadStatus.READABLE else None,
            evidence_ref=f"web:{content_hash[:16]}"
            if self.status == WebPageReadStatus.READABLE
            else None,
        )


class Budget:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, kind, operation_key, request_hash, dispatch):
        self.calls += 1
        return await dispatch()


def test_dispatcher_requires_injected_web_ports_not_global_adapters():
    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService())
    with pytest.raises(ValueError, match="configured"):
        _ = dispatcher.search_client
    with pytest.raises(ValueError, match="configured"):
        _ = dispatcher.reader


@pytest.mark.asyncio
async def test_search_result_without_snippet_remains_readable():
    class SearchWithoutSnippet:
        async def search(self, query):
            return WebSearchResult(
                query=query.query,
                category=query.category,
                items=[WebSearchItem(title="Official specs", url="https://example.com/specs", snippet="")],
            )

    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService(), SearchWithoutSnippet())
    result = await dispatcher.execute(
        ToolCall(call_id="search", name="search_web", arguments={"query": "GPU specs"}),
        context(),
    )
    assert result.status == "COMPLETED"
    assert isinstance(result.result, SearchWebResult)
    assert result.result.results[0].snippet == ""
    assert result.result.results[0].url == "https://example.com/specs"
    assert result.evidence_refs == []


@pytest.mark.asyncio
async def test_mixed_catalog_web_answer_preserves_quote_not_model_prose():
    catalog = Catalog()
    dispatcher = ToolDispatcher(catalog, PCBuildApplicationService(), reader=MockWebPageReader())
    ctx = context()
    details = await dispatcher.execute(
        ToolCall(call_id="details", name="get_product_details", arguments={"product_id": str(catalog.product.product_id)}), ctx
    )
    page = await dispatcher.execute(
        ToolCall(call_id="page", name="read_web_page", arguments={"url": "https://example.com/benchmark"}), ctx
    )
    quote = "Resolution: 1440p Very High -> Average 72 FPS."
    answer = validate_answer(
        AnswerCandidate(
            text="Invented claim: 999 FPS",
            evidence_refs=details.evidence_refs + page.evidence_refs,
            web_excerpts=[WebExcerpt(evidence_ref=page.evidence_refs[0], quote=quote)],
        ),
        [details, page], ctx, [CompletionRequirement(kind="ANSWER")],
    )
    assert "8,000,000 VND" in answer.text
    assert quote in answer.text
    assert "chưa xác thực" in answer.text
    assert "999 FPS" not in answer.text
    assert "https://example.com/benchmark" in answer.text


@pytest.mark.asyncio
async def test_unavailable_search_is_a_tool_failure_not_empty_success():
    class UnavailableSearch:
        async def search(self, query):
            return WebSearchResult(query=query.query, category=query.category, note="WEB_SEARCH_UNAVAILABLE")

    dispatcher = ToolDispatcher(Catalog(), PCBuildApplicationService(), UnavailableSearch())
    result = await dispatcher.execute(
        ToolCall(call_id="search", name="search_web", arguments={"query": "GPU benchmark"}),
        context(),
    )
    assert result.status == "FAILED"
    assert result.error == ToolError.WEB_SEARCH_UNAVAILABLE
    assert result.result is None
    assert result.evidence_refs == []


@pytest.mark.asyncio
async def test_web_citation_alone_cannot_publish_invented_benchmark():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="page",
            name="read_web_page",
            arguments={"url": "https://example.com/benchmark"},
        ),
        ctx,
    )
    with pytest.raises(ValueError, match="Web answer requires source excerpts"):
        validate_answer(
            AnswerCandidate(
                text="RTX 4070 Super đạt 999 FPS ở 4K Ultra.",
                evidence_refs=page.evidence_refs,
            ),
            [page],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
        )


@pytest.mark.asyncio
async def test_web_excerpt_cannot_change_number_condition_or_source():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="page",
            name="read_web_page",
            arguments={"url": "https://example.com/benchmark"},
        ),
        ctx,
    )
    quote = "Resolution: 1440p Very High -> Average 72 FPS."
    for ref, excerpt in [
        (page.evidence_refs[0], quote.replace("72 FPS", "999 FPS")),
        (page.evidence_refs[0], quote.replace("1440p", "4K")),
        ("web:invented", quote),
    ]:
        with pytest.raises(ValueError, match="Web excerpt must match"):
            validate_answer(
                AnswerCandidate(
                    text="Invented answer",
                    evidence_refs=page.evidence_refs,
                    web_excerpts=[WebExcerpt(evidence_ref=ref, quote=excerpt)],
                ),
                [page],
                ctx,
                [CompletionRequirement(kind="ANSWER")],
            )
    answer = validate_answer(
        AnswerCandidate(
            text="999 FPS at 4K Ultra",
            evidence_refs=page.evidence_refs,
            web_excerpts=[WebExcerpt(evidence_ref=page.evidence_refs[0], quote=quote)],
        ),
        [page],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert quote in answer.text
    assert "999 FPS" not in answer.text
    assert "chưa xác thực" in answer.text


@pytest.mark.asyncio
async def test_search_and_read_tool_execution_ledger() -> None:
    search_client = MockSearchClient()
    reader = MockWebPageReader()
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), search_client, reader
    )
    ctx = context()

    assert ToolName.SEARCH_WEB in allowed_tools(ctx)
    assert ToolName.READ_WEB_PAGE in allowed_tools(ctx)

    # 1. Execute search_web
    search_call = ToolCall(
        call_id="call-search-1",
        name="search_web",
        arguments={
            "query": WebSearchPresets.product_reviews(
                "RTX 4070 Super", focus_aspect="GAMING_FPS"
            )
        },
    )
    search_res = await dispatcher.execute(search_call, ctx)
    assert search_res.status == "COMPLETED"
    assert isinstance(search_res.result, SearchWebResult)
    assert len(search_res.result.results) == 1
    # Search snippet must NOT produce citation evidence
    assert search_res.evidence_refs == []

    # 2. Execute read_web_page
    read_call = ToolCall(
        call_id="call-read-1",
        name="read_web_page",
        arguments={"url": search_res.result.results[0].url},
    )
    read_res = await dispatcher.execute(read_call, ctx)
    assert read_res.status == "COMPLETED"
    assert isinstance(read_res.result, ReadWebPageResult)
    assert read_res.result.status == WebPageReadStatus.READABLE
    assert "72 FPS" in (read_res.result.content or "")
    assert len(read_res.evidence_refs) == 1
    assert read_res.evidence_refs[0].startswith("web:")


@pytest.mark.asyncio
async def test_validate_answer_synthesizes_prose_and_server_renders_citations() -> None:
    reader = MockWebPageReader()
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), reader
    )
    ctx = context()

    read_res = await dispatcher.execute(
        ToolCall(
            call_id="read-1",
            name="read_web_page",
            arguments={"url": "https://example.com/test"},
        ),
        ctx,
    )

    candidate = AnswerCandidate(
        text="Ở độ phân giải 1440p thiết lập Very High, RTX 4070 Super đạt trung bình 72 FPS trong Black Myth Wukong.",
        evidence_refs=read_res.evidence_refs,
        web_excerpts=[
            WebExcerpt(
                evidence_ref=read_res.evidence_refs[0],
                quote="Resolution: 1440p Very High -> Average 72 FPS.",
            )
        ],
    )

    answer = validate_answer(
        candidate,
        [read_res],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )

    assert answer.status == "COMPLETED"
    assert "72 FPS" in answer.text
    # Server rendered citation link from result metadata
    assert "Nguồn tham khảo:" in answer.text
    assert "[RTX 4070 Super Benchmark Review](https://example.com/test)" in answer.text


@pytest.mark.asyncio
async def test_validate_answer_rejects_web_evidence_for_build_requirement() -> None:
    reader = MockWebPageReader()
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), reader
    )
    ctx = context()

    read_res = await dispatcher.execute(
        ToolCall(
            call_id="read-1",
            name="read_web_page",
            arguments={"url": "https://example.com/test"},
        ),
        ctx,
    )

    candidate = AnswerCandidate(
        text="Build PC hoàn tất",
        evidence_refs=read_res.evidence_refs,
    )

    with pytest.raises(
        ValueError, match="Build completion requires a verified optimizer outcome"
    ):
        validate_answer(
            candidate,
            [read_res],
            ctx,
            [
                CompletionRequirement(
                    kind="BUILD", constraints_revision=ctx.requirements.revision
                )
            ],
        )


class WebResearchModel:
    def __init__(self, fps_found: bool = True) -> None:
        self.fps_found = fps_found
        self.step = 0

    async def decide(
        self, message: str, transcript: list, allowed: tuple, repair: bool, context=None
    ) -> ReactDecision:
        self.step += 1
        if len(transcript) == 0:
            return ReactDecision(
                calls=[
                    ToolCall(
                        call_id="s1",
                        name="search_web",
                        arguments={
                            "query": "RTX 4070 Super Black Myth Wukong 1440p FPS benchmark"
                        },
                    )
                ]
            )
        if len(transcript) == 1:
            search_item = transcript[0].result.result.results[0]
            return ReactDecision(
                calls=[
                    ToolCall(
                        call_id="r1",
                        name="read_web_page",
                        arguments={"url": search_item.url},
                    )
                ]
            )
        read_res = transcript[1].result
        if self.fps_found:
            text = "Theo benchmark thực tế, RTX 4070 Super đạt trung bình 72 FPS ở 1440p thiết lập Very High."
        else:
            text = (
                "Nguồn bài viết không cung cấp số liệu FPS cụ thể cho thiết lập 1440p."
            )
        return ReactDecision(
            answer=AnswerCandidate(
                text=text,
                evidence_refs=read_res.evidence_refs,
                web_excerpts=[
                    WebExcerpt(
                        evidence_ref=read_res.evidence_refs[0],
                        quote=read_res.result.content[:2000],
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_full_react_web_research_loop() -> None:
    search_client = MockSearchClient()
    reader = MockWebPageReader(
        content="RTX 4070 Super Black Myth Wukong: 1440p Very High average 72 FPS."
    )
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), search_client, reader
    )
    ctx = context()
    budget = Budget()

    model = WebResearchModel(fps_found=True)
    graph = build_react_graph(
        model,
        dispatcher,
        ctx,
        [CompletionRequirement(kind="ANSWER")],
        budget,
    )

    result = await graph.ainvoke(
        {
            "message": "RTX 4070 Super chơi Black Myth Wukong ở 1440p được bao nhiêu FPS?",
            "ledger": [],
        }
    )

    assert result["phase"] == "READY_TO_PUBLISH"
    assert len(result["ledger"]) == 2
    assert result["ledger"][0]["call"]["name"] == "search_web"
    assert result["ledger"][1]["call"]["name"] == "read_web_page"
    assert "72 FPS" in result["answer"]["text"]
    assert "Nguồn tham khảo:" in result["answer"]["text"]
    assert (
        "https://example.com/reviews/rtx-4070-super-wukong" in result["answer"]["text"]
    )
    assert budget.calls == result["model_calls"] + 2


@pytest.mark.asyncio
async def test_long_web_article_is_bounded_and_can_continue() -> None:
    content = "Thông số phần cứng. " * 10_000 + "\n1440p Very High: 72 FPS."
    dispatcher = ToolDispatcher(
        Catalog(),
        PCBuildApplicationService(),
        MockSearchClient(),
        MockWebPageReader(content=content),
    )
    ctx = context()
    first = await dispatcher.execute(
        ToolCall(
            call_id="first",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )
    assert first.status == "COMPLETED"
    assert isinstance(first.result, ReadWebPageResult)
    assert len(first.model_dump_json().encode()) < 128_000
    assert first.result.offset == 0
    assert first.result.next_offset == len(first.result.content)
    assert first.result.total_chars == len(content)

    last = await dispatcher.execute(
        ToolCall(
            call_id="last",
            name="read_web_page",
            arguments={
                "url": "https://example.com/review",
                "offset": len(content) - 100,
            },
        ),
        ctx,
    )
    assert isinstance(last.result, ReadWebPageResult)
    assert "72 FPS" in last.result.content
    assert last.result.next_offset is None
    assert first.evidence_refs != last.evidence_refs

    graph = build_react_graph(
        WebResearchModel(),
        dispatcher,
        ctx,
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
    )
    result = await graph.ainvoke({"message": "Read this review"})
    assert result["phase"] == "READY_TO_PUBLISH"


@pytest.mark.asyncio
async def test_web_prose_cannot_override_canonical_catalog_answer():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    catalog = await dispatcher.execute(
        ToolCall(call_id="catalog", name="search_catalog", arguments={"query": "GPU"}),
        ctx,
    )
    page = await dispatcher.execute(
        ToolCall(
            call_id="page",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )
    answer = validate_answer(
        AnswerCandidate(
            text="Canonical GPU giá 1 VND",
            evidence_refs=catalog.evidence_refs + page.evidence_refs,
        ),
        [catalog, page],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "8,000,000 VND" in answer.text
    assert "giá 1 VND" not in answer.text
    assert "https://example.com/review" in answer.text


@pytest.mark.asyncio
async def test_web_answer_rejects_model_authored_links():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="page",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )
    with pytest.raises(ValueError, match="Server owns web citation links"):
        validate_answer(
            AnswerCandidate(
                text="72 FPS [nguồn](https://invented.example/fake)",
                evidence_refs=page.evidence_refs,
            ),
            [page],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
        )


@pytest.mark.asyncio
async def test_react_web_research_handles_missing_benchmark_without_hallucinating() -> (
    None
):
    search_client = MockSearchClient()
    reader = MockWebPageReader(
        content="RTX 4070 Super is a powerful card with 12GB VRAM. Game runs smoothly."
    )
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), search_client, reader
    )
    ctx = context()
    budget = Budget()

    model = WebResearchModel(fps_found=False)
    graph = build_react_graph(
        model,
        dispatcher,
        ctx,
        [CompletionRequirement(kind="ANSWER")],
        budget,
    )

    result = await graph.ainvoke(
        {
            "message": "RTX 4070 Super chơi Black Myth Wukong ở 1440p được bao nhiêu FPS?",
            "ledger": [],
        }
    )

    assert result["phase"] == "READY_TO_PUBLISH"
    # Model explicitly reports missing benchmark rather than inventing numbers
    assert "chưa xác thực thông số/điều kiện đo" in result["answer"]["text"]
    assert "12GB VRAM" in result["answer"]["text"]
    assert "72 FPS" not in result["answer"]["text"]
    assert "Nguồn tham khảo:" in result["answer"]["text"]


@pytest.mark.asyncio
async def test_product_review_renders_structured_sections_with_quotes_and_citations():
    reader = MockWebPageReader(
        content=(
            "# RTX 4070 Super Review\n\n"
            "The RTX 4070 Super offers great 1440p gaming performance with lower power draw.\n"
            "However, 12GB VRAM may be limiting for future 4K textures, and pricing remains high.\n"
        )
    )
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), reader
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="read-review",
            name="read_web_page",
            arguments={"url": "https://example.com/rtx-4070-super-review"},
        ),
        ctx,
    )

    review = ProductReview(
        product_name="Nvidia GeForce RTX 4070 Super",
        summary="Card đồ họa mạnh mẽ trong phân khúc cận cao cấp, tối ưu cho màn hình 2K/1440p.",
        pros=[
            "Hiệu năng gaming 1440p xuất sắc với DLSS 3",
            "Mức tiêu thụ điện năng ấn tượng",
        ],
        cons=[
            "Bộ nhớ VRAM 12GB có thể hạn chế ở 4K lâu dài",
            "Giá khởi điểm vẫn còn tương đối cao",
        ],
        verdict="Rất đáng mua nếu bạn đang nâng cấp từ GTX 10-series hoặc RTX 20-series để chơi game 1440p.",
    )

    candidate = AnswerCandidate(
        text="Tổng hợp đánh giá RTX 4070 Super",
        evidence_refs=page.evidence_refs,
        web_excerpts=[
            WebExcerpt(
                evidence_ref=page.evidence_refs[0],
                quote="The RTX 4070 Super offers great 1440p gaming performance with lower power draw.",
            )
        ],
        review=review,
    )

    answer = validate_answer(
        candidate,
        [page],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )

    assert answer.status == "COMPLETED"
    assert "### Đánh giá: Nvidia GeForce RTX 4070 Super" in answer.text
    assert "Card đồ họa mạnh mẽ trong phân khúc cận cao cấp" in answer.text
    assert "**Ưu điểm:**" in answer.text
    assert "- Hiệu năng gaming 1440p xuất sắc với DLSS 3" in answer.text
    assert "- Mức tiêu thụ điện năng ấn tượng" in answer.text
    assert "**Nhược điểm:**" in answer.text
    assert "- Bộ nhớ VRAM 12GB có thể hạn chế ở 4K lâu dài" in answer.text
    assert "- Giá khởi điểm vẫn còn tương đối cao" in answer.text
    assert "**Tổng kết:**" in answer.text
    assert "Rất đáng mua nếu bạn đang nâng cấp" in answer.text
    assert "Nội dung nguồn tham khảo (chưa xác thực thông số/điều kiện đo):" in answer.text
    assert "The RTX 4070 Super offers great 1440p gaming performance" in answer.text
    assert "Nguồn tham khảo:" in answer.text
    assert "[RTX 4070 Super Benchmark Review](https://example.com/rtx-4070-super-review)" in answer.text


@pytest.mark.asyncio
async def test_product_review_rejects_raw_urls_in_review_fields():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="read-page",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )

    quote = "Resolution: 1440p Very High -> Average 72 FPS."
    valid_excerpts = [WebExcerpt(evidence_ref=page.evidence_refs[0], quote=quote)]

    bad_reviews = [
        ProductReview(
            product_name="RTX 4070 Super (https://bad.com)",
            summary="Đánh giá chi tiết",
            verdict="Tốt",
        ),
        ProductReview(
            product_name="RTX 4070 Super",
            summary="Xem thêm tại https://bad.com/review",
            verdict="Tốt",
        ),
        ProductReview(
            product_name="RTX 4070 Super",
            summary="Đánh giá chi tiết",
            pros=["Ưu điểm 1", "Chi tiết xem www.bad.com/pros"],
            verdict="Tốt",
        ),
        ProductReview(
            product_name="RTX 4070 Super",
            summary="Đánh giá chi tiết",
            cons=["Nhược điểm tại http://bad.com/cons"],
            verdict="Tốt",
        ),
        ProductReview(
            product_name="RTX 4070 Super",
            summary="Đánh giá chi tiết",
            verdict="Tổng kết ở https://bad.com/verdict",
        ),
    ]

    for bad in bad_reviews:
        with pytest.raises(ValueError, match="Server owns web citation links"):
            validate_answer(
                AnswerCandidate(
                    text="Đánh giá sản phẩm",
                    evidence_refs=page.evidence_refs,
                    web_excerpts=valid_excerpts,
                    review=bad,
                ),
                [page],
                ctx,
                [CompletionRequirement(kind="ANSWER")],
            )


@pytest.mark.asyncio
async def test_product_review_requires_source_excerpts_and_web_evidence():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    page = await dispatcher.execute(
        ToolCall(
            call_id="read-page",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )
    catalog = await dispatcher.execute(
        ToolCall(call_id="cat", name="search_catalog", arguments={"query": "GPU"}),
        ctx,
    )

    review = ProductReview(
        product_name="RTX 4070 Super",
        summary="Đánh giá chi tiết",
        verdict="Tổng kết đánh giá",
    )

    # 1. Missing web_excerpts
    with pytest.raises(ValueError, match="Product review requires source excerpts"):
        validate_answer(
            AnswerCandidate(
                text="Đánh giá sản phẩm",
                evidence_refs=page.evidence_refs,
                web_excerpts=[],
                review=review,
            ),
            [page],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
        )

    # 2. Only catalog evidence, no web pages
    with pytest.raises(ValueError, match="Product review requires read web page evidence"):
        validate_answer(
            AnswerCandidate(
                text="Đánh giá sản phẩm",
                evidence_refs=catalog.evidence_refs,
                web_excerpts=[],
                review=review,
            ),
            [catalog],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
        )


@pytest.mark.asyncio
async def test_product_review_with_catalog_preserves_canonical_catalog_and_review():
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), MockWebPageReader()
    )
    ctx = context()
    catalog = await dispatcher.execute(
        ToolCall(call_id="cat", name="search_catalog", arguments={"query": "GPU"}),
        ctx,
    )
    page = await dispatcher.execute(
        ToolCall(
            call_id="page",
            name="read_web_page",
            arguments={"url": "https://example.com/review"},
        ),
        ctx,
    )

    quote = "Resolution: 1440p Very High -> Average 72 FPS."
    review = ProductReview(
        product_name="RTX 4070 Super",
        summary="Đánh giá nhanh về hiệu năng chơi game.",
        pros=["Chơi mượt 1440p"],
        cons=["Giá bán chưa giảm sâu"],
        verdict="Rất đáng mua.",
    )

    candidate = AnswerCandidate(
        text="Tổng hợp từ catalog và bài đánh giá",
        evidence_refs=catalog.evidence_refs + page.evidence_refs,
        web_excerpts=[WebExcerpt(evidence_ref=page.evidence_refs[0], quote=quote)],
        review=review,
    )

    answer = validate_answer(
        candidate,
        [catalog, page],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )

    assert "8,000,000 VND" in answer.text
    assert "### Đánh giá: RTX 4070 Super" in answer.text
    assert "Chơi mượt 1440p" in answer.text
    assert "Giá bán chưa giảm sâu" in answer.text
    assert quote in answer.text
    assert "https://example.com/review" in answer.text


@pytest.mark.asyncio
async def test_product_review_with_real_techpowerup_review_data(monkeypatch):
    fixture_path = Path(__file__).with_name("fixtures").joinpath("techpowerup_gpu_review.html")
    real_html = fixture_path.read_text(encoding="utf-8")

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=real_html)
            ),
            **kwargs,
        ),
    )

    reader = TrafilaturaReader()
    target_url = "https://www.techpowerup.com/review/nvidia-geforce-rtx-4070-super-founders-edition/44.html"

    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), MockSearchClient(), reader
    )
    ctx = context()

    # 1. Execute read_web_page with real HTML
    read_call = ToolCall(
        call_id="call-read-tpu",
        name="read_web_page",
        arguments={"url": target_url},
    )
    read_res = await dispatcher.execute(read_call, ctx)
    assert read_res.status == "COMPLETED"
    assert read_res.result.status == WebPageReadStatus.READABLE
    assert read_res.result.title == "NVIDIA GeForce RTX 4070 Super Founders Edition Review"
    assert "15% faster than the RTX 4070 non-Super" in read_res.result.content

    # 2. Extract verbatim quotes from the real TechPowerUp article content
    real_quote_1 = (
        "Averaged over the 25 games in our test suite, at 1440p, we find the RTX 4070 Super "
        "Founders Edition 15% faster than the RTX 4070 non-Super, which is a pretty substantial "
        "improvement for a refresh—unlike what Intel did with their 14th Gen Raptor Lake. "
        "This means that the card is able to match last generation's RTX 3090 flagship, "
        "and the gap to RTX 4070 Ti shrinks to just 8%."
    )
    real_quote_2 = (
        "With these performance numbers RTX 4070 Super is a perfect match for 1440p "
        "with maximum settings. You should be able to enable ray tracing in most titles, too."
    )
    assert real_quote_1 in read_res.result.content
    assert real_quote_2 in read_res.result.content

    # 3. Model synthesizes qualitative review grounded in the real article
    review = ProductReview(
        product_name="NVIDIA GeForce RTX 4070 Super Founders Edition",
        summary=(
            "Đợt làm mới Ada Lovelace 2024 mang lại hiệu năng ấn tượng, "
            "vượt 15% so với RTX 4070 non-Super và bắt kịp flagship RTX 3090 ở độ phân giải 1440p."
        ),
        pros=[
            "Nhanh hơn 15% so với bản 4070 thường qua 25 tựa game ở 1440p",
            "Hiệu năng bắt kịp flagship RTX 3090 thế hệ trước",
            "Khả năng xử lý Ray Tracing và DLSS 3 Frame Generation vượt trội hơn RX 7800 XT",
            "Thiết kế Founders Edition all-black đẹp mắt và tản nhiệt hiệu quả",
        ],
        cons=[
            "Bộ nhớ VRAM 12GB có thể thành điểm nghẽn ở độ phân giải 4K tương lai",
            "Mức giá $600 vẫn là khoản đầu tư đáng kể",
        ],
        verdict=(
            "Lựa chọn tuyệt vời cho người dùng muốn chơi game 1440p thiết lập tối đa "
            "cùng Ray Tracing mượt mà ở mức giá $600."
        ),
    )

    candidate = AnswerCandidate(
        text="Tổng hợp đánh giá chi tiết RTX 4070 Super Founders Edition từ TechPowerUp",
        evidence_refs=read_res.evidence_refs,
        web_excerpts=[
            WebExcerpt(evidence_ref=read_res.evidence_refs[0], quote=real_quote_1),
            WebExcerpt(evidence_ref=read_res.evidence_refs[0], quote=real_quote_2),
        ],
        review=review,
    )

    answer = validate_answer(
        candidate,
        [read_res],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )

    assert answer.status == "COMPLETED"
    assert "### Đánh giá: NVIDIA GeForce RTX 4070 Super Founders Edition" in answer.text
    assert "Đợt làm mới Ada Lovelace 2024" in answer.text
    assert "**Ưu điểm:**" in answer.text
    assert "- Nhanh hơn 15% so với bản 4070 thường" in answer.text
    assert "- Hiệu năng bắt kịp flagship RTX 3090" in answer.text
    assert "**Nhược điểm:**" in answer.text
    assert "- Bộ nhớ VRAM 12GB có thể thành điểm nghẽn" in answer.text
    assert "**Tổng kết:**" in answer.text
    assert "Lựa chọn tuyệt vời cho người dùng" in answer.text
    assert real_quote_1 in answer.text
    assert real_quote_2 in answer.text
    assert "Nguồn tham khảo:" in answer.text
    assert f"[NVIDIA GeForce RTX 4070 Super Founders Edition Review]({target_url})" in answer.text

    # 4. Invariant verification: Tampering with real quote is strictly rejected
    tampered_quote = real_quote_1.replace("15% faster", "50% faster")
    with pytest.raises(ValueError, match="Web excerpt must match selected source content"):
        validate_answer(
            AnswerCandidate(
                text="Tampered benchmark claim",
                evidence_refs=read_res.evidence_refs,
                web_excerpts=[WebExcerpt(evidence_ref=read_res.evidence_refs[0], quote=tampered_quote)],
                review=review,
            ),
            [read_res],
            ctx,
            [CompletionRequirement(kind="ANSWER")],
        )



