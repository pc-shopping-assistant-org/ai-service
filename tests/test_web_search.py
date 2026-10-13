import httpx
import pytest

from ai_service.application.ports.web_search import (
    WebSearchCategory,
    WebSearchQuery,
)
from ai_service.capabilities.web_search.schemas import (
    BuildGuideSearchArgs,
    GameRequirementSearchArgs,
    HardwareCompatibilitySearchArgs,
    HardwareIssueSearchArgs,
    ProductReviewSearchArgs,
    PsuTierSearchArgs,
)
from ai_service.capabilities.web_search.tools import WebSearchTools
from ai_service.infrastructure.search.duckduckgo_adapter import (
    DuckDuckGoSearchAdapter,
)


@pytest.fixture
def search_adapter() -> DuckDuckGoSearchAdapter:
    return DuckDuckGoSearchAdapter()


@pytest.fixture
def search_tools(monkeypatch: pytest.MonkeyPatch) -> WebSearchTools:
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='''
            <h2 class="result__title"><a class="result__url" href="https://example.org/review">Fixture review</a></h2>
            <a class="result__snippet" href="https://example.org/review">RAM clearance fixture evidence</a>
        ''')

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    return WebSearchTools()


def test_duckduckgo_query_enrichment(search_adapter: DuckDuckGoSearchAdapter) -> None:
    # 1. PSU Tier enrichment
    query_psu = WebSearchQuery(query="Corsair RM850e", category=WebSearchCategory.PSU_TIER)
    enriched = search_adapter._enrich_query(query_psu)
    assert "Cultists Network" in enriched

    # 2. Game requirements enrichment
    query_game = WebSearchQuery(query="Cyberpunk 2077", category=WebSearchCategory.GAME_REQUIREMENTS)
    enriched_game = search_adapter._enrich_query(query_game)
    assert "system requirements" in enriched_game

    # 3. Target domains filter injection
    query_domains = WebSearchQuery(
        query="RTX 4070",
        target_domains=["techpowerup.com", "tomshardware.com"],
    )
    enriched_domains = search_adapter._enrich_query(query_domains)
    assert "site:techpowerup.com" in enriched_domains


def test_duckduckgo_html_parsing(search_adapter: DuckDuckGoSearchAdapter) -> None:
    sample_html = """
    <h2 class="result__title">
        <a class="result__url" href="/l/?uddg=https%3A%2F%2Fwww.techpowerup.com%2Freview%2Frtx-4060">
            NVIDIA GeForce RTX 4060 8 GB Review - TechPowerUp
        </a>
    </h2>
    <a class="result__snippet" href="/l/?uddg=https%3A%2F%2Fwww.techpowerup.com%2Freview%2Frtx-4060">
        The GeForce RTX 4060 is NVIDIA&#39;s latest mainstream offering. In our testing it delivers solid 1080p performance.
    </a>
    """
    items = search_adapter._parse_html(sample_html, limit=2)
    assert len(items) == 1
    assert "RTX 4060" in items[0].title
    assert items[0].url == "https://www.techpowerup.com/review/rtx-4060"
    assert "solid 1080p performance" in items[0].snippet


def test_duckduckgo_current_markup_preserves_organic_titles_and_urls(search_adapter):
    # Current HTML endpoint: result__a, arbitrary attribute order, nested text.
    markup = '''
    <h2 class="result__title"><a rel="nofollow" class="result__a"
      href="https://duckduckgo.com/y.js?ad_domain=shop.example&amp;u3=ad">Ad</a></h2>
    <a class="result__snippet" href="https://duckduckgo.com/y.js?ad_domain=shop.example">Buy now</a>
    <h2 class="result__title"><a rel="nofollow" class="result__a"
      href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.techpowerup.com%2Freview%2Frtx-4070%3Fq%3DA%252FB&amp;rut=tracking">
      RTX <b>4070 Super</b> &amp; Review</a></h2>
    <a href="//duckduckgo.com/l/?uddg=ignored" class="result__snippet">Measured <b>1440p</b> performance.</a>
    <h2 class="result__title"><a href="https://example.org/specs" class="result__a">Official specifications</a></h2>
    <a class="result__snippet" href="https://example.org/specs">Dimensions and power.</a>
    '''
    items = search_adapter._parse_html(markup, limit=2)
    assert [(item.title, item.url, item.snippet) for item in items] == [
        (
            "RTX 4070 Super & Review",
            "https://www.techpowerup.com/review/rtx-4070?q=A%2FB",
            "Measured 1440p performance.",
        ),
        ("Official specifications", "https://example.org/specs", "Dimensions and power."),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,html,expected_note",
    [
        (202, '<form id="challenge-form">Verify you are human</form>', "WEB_SEARCH_UNAVAILABLE"),
        (200, '<form id="challenge-form">Verify you are human</form>', "WEB_SEARCH_UNAVAILABLE"),
        (200, '<div class="no-results">No results found</div>', "WEB_SEARCH_NO_RESULTS"),
    ],
)
async def test_search_distinguishes_challenge_and_empty_results(monkeypatch, status, html, expected_note):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text=html, request=request))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    result = await DuckDuckGoSearchAdapter().search(WebSearchQuery(query="GPU benchmark"))
    assert result.items == []
    assert result.note == expected_note


@pytest.mark.asyncio
async def test_search_hardware_compatibility(search_tools: WebSearchTools) -> None:
    args = HardwareCompatibilitySearchArgs(
        part_a="DeepCool AK620",
        part_b="Corsair Vengeance RGB DDR5",
        specific_concern="RAM clearance height",
    )
    res = await search_tools.search_hardware_compatibility(args)
    assert res.category == "COMPATIBILITY"
    assert len(res.results) > 0
    assert "RAM clearance" in res.analysis_prompt


@pytest.mark.asyncio
async def test_search_product_reviews(search_tools: WebSearchTools) -> None:
    args = ProductReviewSearchArgs(
        product_name="RTX 4060",
        compare_with="RX 6700 XT",
        focus_aspect="GAMING_FPS",
    )
    res = await search_tools.search_product_reviews(args)
    assert res.category == "REVIEWS"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_build_guides(search_tools: WebSearchTools) -> None:
    args = BuildGuideSearchArgs(
        budget_vnd=25_000_000,
        target_use_case="Black Myth Wukong 2K",
    )
    res = await search_tools.search_build_guides(args)
    assert res.category == "BUILD_GUIDES"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_game_requirements(search_tools: WebSearchTools) -> None:
    args = GameRequirementSearchArgs(
        game_title="Black Myth Wukong",
        target_resolution="1440P",
    )
    res = await search_tools.search_game_requirements(args)
    assert res.category == "GAME_REQUIREMENTS"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_hardware_issues(search_tools: WebSearchTools) -> None:
    args = HardwareIssueSearchArgs(
        product_name="Intel Core i9-14900K",
        concern="instability crash Vmin shift",
    )
    res = await search_tools.search_hardware_issues(args)
    assert res.category == "HARDWARE_ISSUES"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_psu_tier(search_tools: WebSearchTools) -> None:
    args = PsuTierSearchArgs(psu_model_name="MSI MAG A650BN")
    res = await search_tools.search_psu_tier(args)
    assert res.category == "PSU_TIER"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_driver_and_software(search_tools: WebSearchTools) -> None:
    from ai_service.capabilities.web_search.schemas import DriverSoftwareSearchArgs

    args = DriverSoftwareSearchArgs(
        component_name="RTX 4070 SUPER",
        software_type="VGA_DRIVER",
    )
    res = await search_tools.search_driver_and_software(args)
    assert res.category == "DRIVER_SOFTWARE"
    assert len(res.results) > 0


@pytest.mark.asyncio
async def test_search_tech_specs_from_vendor(search_tools: WebSearchTools) -> None:
    from ai_service.capabilities.web_search.schemas import VendorSpecsSearchArgs

    args = VendorSpecsSearchArgs(
        product_model="Thermalright Peerless Assassin 120 SE",
        spec_attribute="chiều cao mm",
    )
    res = await search_tools.search_tech_specs_from_vendor(args)
    assert res.category == "VENDOR_SPECS"
    assert len(res.results) > 0
