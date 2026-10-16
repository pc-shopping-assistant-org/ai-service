import pytest
from test_react_research_tools import runner
from test_react_tools import context
from test_react_web_flow import MockWebPageReader

from ai_service.application.ports.web_search import WebSearchItem, WebSearchResult
from ai_service.capabilities.assistant.react.contracts import ToolCall
from ai_service.capabilities.web_search.source_policy import ResearchSourcePolicy


class ScopedSearch:
    def __init__(self, *, official_empty=False):
        self.queries = []
        self.official_empty = official_empty

    async def search(self, query):
        self.queries.append(query)
        domains = query.target_domains
        if self.official_empty and domains == ["corsair.com"]:
            return WebSearchResult(query=query.query, category=query.category)
        host = domains[0] if domains else "unscoped.example"
        return WebSearchResult(
            query=query.query,
            category=query.category,
            items=[
                WebSearchItem(
                    title="Unrelated result",
                    url="https://untrusted.example/post",
                    snippet="Corsair RM850x",
                ),
                WebSearchItem(
                    title="Corsair RM850x product specifications",
                    url=f"https://{host}/rm850x",
                    snippet="RM850x official specifications and tests",
                ),
            ],
        )


@pytest.mark.asyncio
async def test_manufacturer_is_searched_and_read_before_review_domains():
    search, reader = ScopedSearch(), MockWebPageReader()
    result = await runner(search, reader).execute(
        ToolCall(
            call_id="psu",
            name="research_hardware",
            arguments={"model": "Corsair RM850x", "focus": "PSU_QUALITY"},
        ),
        context(),
    )
    assert search.queries[0].target_domains == ["corsair.com"]
    assert "hwbusters.com" in search.queries[1].target_domains
    assert all(query.target_domains for query in search.queries)
    assert reader.urls[0] == "https://corsair.com/rm850x"
    assert all("untrusted.example" not in url for url in reader.urls)
    section = result.result.sections[0]
    assert section.official_source_status == "RESOLVED"
    assert [step.tier for step in section.search_steps] == [
        "OFFICIAL",
        "INDEPENDENT_REVIEW",
    ]
    assert result.result.source_policy_version


@pytest.mark.asyncio
async def test_missing_manufacturer_source_keeps_failure_and_uses_review_scope():
    search = ScopedSearch(official_empty=True)
    result = await runner(search).execute(
        ToolCall(
            call_id="psu",
            name="research_hardware",
            arguments={"model": "Corsair RM850x", "focus": "PSU_QUALITY"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.search_steps[0].status == "NO_RESULTS"
    assert section.search_steps[1].status == "READABLE"
    assert section.status == "READABLE"


@pytest.mark.asyncio
async def test_unknown_manufacturer_is_explicit_and_never_triggers_implicit_all_web_search():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="unknown",
            name="research_hardware",
            arguments={"model": "Unregistered Model Q", "focus": "PSU_QUALITY"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.official_source_status == "UNKNOWN"
    assert all(query.target_domains for query in search.queries)
    assert all(step.tier != "OFFICIAL" for step in section.search_steps)


@pytest.mark.asyncio
async def test_official_specs_stops_after_a_readable_manufacturer_source():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="specs",
            name="research_hardware",
            arguments={"model": "Corsair RM850x", "focus": "OFFICIAL_SPECS"},
        ),
        context(),
    )
    assert len(search.queries) == 1 and search.queries[0].target_domains == [
        "corsair.com"
    ]
    assert result.result.sections[0].search_steps[0].tier == "OFFICIAL"


def test_registry_prefers_board_partner_over_chip_family_and_respects_canonical_brand():
    policy = ResearchSourcePolicy()
    assert policy.official_domains("ASUS TUF NVIDIA RTX 5070 Ti") == ["asus.com"]
    assert policy.official_domains("RM850x", brand="Corsair") == ["corsair.com"]
    assert policy.official_domains("RTX 5070 Ti", brand="Unregistered Brand") == []
    assert policy.official_domains("NotCorsair RM850x") == []


@pytest.mark.asyncio
async def test_source_registry_can_be_overridden_without_changing_tool_arguments():
    dispatcher = runner(ScopedSearch())
    dispatcher.source_policy = ResearchSourcePolicy(
        version="local-sources-v2",
        manufacturers={"acme": ["acme.example"]},
        reviews={"GENERAL": ["review.example"]},
        aliases={},
    )
    result = await dispatcher.execute(
        ToolCall(
            call_id="custom",
            name="research_hardware",
            arguments={"model": "ACME PSU"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert [step.target_domains for step in section.search_steps] == [
        ["acme.example"],
        ["review.example"],
    ]
    assert result.result.source_policy_version == "local-sources-v2"
    assert result.result.source_policy_hash == dispatcher.source_policy.digest


@pytest.mark.asyncio
async def test_off_scope_redirect_cannot_become_official_evidence():
    class RedirectReader(MockWebPageReader):
        async def read(self, url):
            page = await super().read(url)
            if "corsair.com" in url:
                page.final_url = "https://corsair.com.untrusted.example/marketing"
            return page

    result = await runner(ScopedSearch(), RedirectReader()).execute(
        ToolCall(
            call_id="redirect",
            name="research_hardware",
            arguments={"model": "Corsair RM850x"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.search_steps[0].status == "UNREADABLE"
    assert section.search_steps[1].status == "READABLE"
    assert all("untrusted.example" not in page.final_url for page in section.pages)


@pytest.mark.asyncio
async def test_benchmark_does_not_mistake_gpu_manufacturer_for_game_publisher():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="game",
            name="research_workload",
            arguments={
                "workload": "Unregistered Game",
                "hardware": ["NVIDIA RTX 5070 Ti"],
            },
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.official_source_status == "UNKNOWN"
    assert all("nvidia.com" not in query.target_domains for query in search.queries)


@pytest.mark.asyncio
async def test_general_web_fallback_requires_explicit_configuration():
    class EmptyScopedSearch(ScopedSearch):
        async def search(self, query):
            if not query.target_domains:
                return await super().search(query)
            self.queries.append(query)
            return WebSearchResult(query=query.query, category=query.category)

    search = EmptyScopedSearch()
    dispatcher = runner(search)
    dispatcher.source_policy = ResearchSourcePolicy(allow_general_fallback=True)
    result = await dispatcher.execute(
        ToolCall(
            call_id="fallback",
            name="research_hardware",
            arguments={"model": "Corsair RM850x"},
        ),
        context(),
    )
    assert [step.tier for step in result.result.sections[0].search_steps] == [
        "OFFICIAL",
        "INDEPENDENT_REVIEW",
        "GENERAL_WEB",
    ]
    assert search.queries[-1].target_domains == []


@pytest.mark.asyncio
async def test_full_build_uses_canonical_brand_when_model_name_omits_it(
    realistic_catalog,
):
    from test_requirement_pipeline import rows, state

    from ai_service.capabilities.assistant.react.contracts import ToolRecord

    for part in realistic_catalog:
        if part.brand == "AMD":
            part.name = part.name.removeprefix("AMD Ryzen ")
    trusted = context(state(), rows(realistic_catalog))
    dispatcher = runner(ScopedSearch())
    call = ToolCall(call_id="build", name="build_pc", arguments={})
    built = await dispatcher.execute(call, trusted)
    result = await dispatcher.execute(
        ToolCall(
            call_id="research",
            name="research_pc_build",
            arguments={"build_ref": str(built.result.build_ref)},
        ),
        trusted,
        records=[ToolRecord(call=call, result=built, origin_run_id=trusted.run_id)],
    )
    cpu = built.result.receipt.build.parts["CPU"]
    section = next(
        section for section in result.result.sections if section.subjects == [cpu.name]
    )
    expected = dispatcher.source_policy.official_domains(cpu.name, cpu.brand)
    assert section.search_steps[0].target_domains == expected
    assert section.search_steps[0].tier == "OFFICIAL"
