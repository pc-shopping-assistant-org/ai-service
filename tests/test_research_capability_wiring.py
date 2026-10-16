import pytest
from test_react_research_tools import MockSearchClient, MockWebPageReader, runner
from test_react_tools import context
from test_research_source_policy import ScopedSearch

from ai_service.application.ports.web_search import WebSearchResult
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
)
from ai_service.capabilities.assistant.react.tools import validate_answer
from ai_service.capabilities.web_search.source_policy import ResearchSourcePolicy


@pytest.mark.asyncio
async def test_requirements_use_steam_not_review_sites_for_an_unknown_publisher():
    search = ScopedSearch()
    dispatcher = runner(search)
    dispatcher.source_policy = ResearchSourcePolicy(allow_general_fallback=True)
    result = await dispatcher.execute(
        ToolCall(
            call_id="requirements",
            name="research_workload",
            arguments={"workload": "Black Myth Wukong", "mode": "REQUIREMENTS"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.official_source_status == "UNKNOWN"
    assert [step.tier for step in section.search_steps] == ["STORE_LISTING"]
    assert all(
        query.target_domains == ["store.steampowered.com"] for query in search.queries
    )
    assert result.evidence_refs and section.pages


@pytest.mark.asyncio
async def test_requirements_resolve_application_publisher_without_brand_in_the_name():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="premiere",
            name="research_workload",
            arguments={"workload": "Premiere Pro 2026", "mode": "REQUIREMENTS"},
        ),
        context(),
    )
    section = result.result.sections[0]
    assert section.official_source_status == "RESOLVED"
    assert len(search.queries) == 1
    assert "helpx.adobe.com" in search.queries[0].target_domains
    assert section.search_steps[0].tier == "OFFICIAL"


@pytest.mark.asyncio
async def test_requirements_try_store_after_empty_publisher_result():
    class Search(ScopedSearch):
        async def search(self, query):
            if "blender.org" in query.target_domains:
                self.queries.append(query)
                return WebSearchResult(query=query.query, category=query.category)
            return await super().search(query)

    result = await runner(Search()).execute(
        ToolCall(
            call_id="blender",
            name="research_workload",
            arguments={"workload": "Blender", "mode": "REQUIREMENTS"},
        ),
        context(),
    )
    assert [
        (step.tier, step.status) for step in result.result.sections[0].search_steps
    ] == [
        ("OFFICIAL", "NO_RESULTS"),
        ("STORE_LISTING", "READABLE"),
    ]


@pytest.mark.asyncio
async def test_driver_uses_only_manufacturer_even_if_general_fallback_is_enabled():
    search, reader = ScopedSearch(), MockWebPageReader()
    dispatcher = runner(search, reader)
    dispatcher.source_policy = ResearchSourcePolicy(allow_general_fallback=True)
    result = await dispatcher.execute(
        ToolCall(
            call_id="bios",
            name="research_driver",
            arguments={
                "component_name": "MSI B650M",
                "software_type": "BIOS_UPDATE",
                "operating_system": "Windows 11",
                "version": "7D77v1J",
            },
        ),
        context(),
    )
    assert result.status == "COMPLETED"
    assert search.queries and all(
        query.target_domains == ["msi.com"] for query in search.queries
    )
    assert (
        "7D77v1J" in search.queries[0].query and "Windows 11" in search.queries[0].query
    )
    assert all("untrusted.example" not in url for url in reader.urls)
    assert all(
        step.tier == "OFFICIAL" for step in result.result.sections[0].search_steps
    )
    assert result.evidence_refs


@pytest.mark.asyncio
async def test_unknown_driver_manufacturer_makes_no_unscoped_requests():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="unknown",
            name="research_driver",
            arguments={"component_name": "Unknown Model Q"},
        ),
        context(),
    )
    assert (
        result.status == "COMPLETED" and not search.queries and not result.evidence_refs
    )
    section = result.result.sections[0]
    assert (
        section.official_source_status == "UNKNOWN"
        and section.status == "SOURCE_UNRESOLVED"
    )


@pytest.mark.asyncio
async def test_driver_does_not_assume_an_operating_system():
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="gpu",
            name="research_driver",
            arguments={"component_name": "NVIDIA RTX 4070 SUPER"},
        ),
        context(),
    )
    assert result.status == "COMPLETED"
    assert (
        "Windows" not in search.queries[0].query
        and "Linux" not in search.queries[0].query
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"software_type": "INSTALL"},
        {"download": True},
        {"target_domains": ["example.com"]},
    ],
)
async def test_driver_rejects_mutation_and_model_owned_domains(extra):
    search = ScopedSearch()
    result = await runner(search).execute(
        ToolCall(
            call_id="invalid",
            name="research_driver",
            arguments={"component_name": "MSI B650M", **extra},
        ),
        context(),
    )
    assert (
        result.status == "REJECTED"
        and result.error == "BAD_ARGUMENTS"
        and not search.queries
    )


@pytest.mark.asyncio
async def test_driver_answer_uses_read_page_evidence_not_search_snippets():
    trusted = context()
    quote = "BIOS 7D77v1J includes a firmware update."
    result = await runner(MockSearchClient(), MockWebPageReader(content=quote)).execute(
        ToolCall(
            call_id="bios",
            name="research_driver",
            arguments={"component_name": "MSI B650M", "software_type": "BIOS_UPDATE"},
        ),
        trusted,
    )
    assert result.status == "COMPLETED"
    answer = validate_answer(
        AnswerCandidate(
            text="ignore",
            evidence_refs=result.evidence_refs,
            web_excerpts=[{"evidence_ref": result.evidence_refs[0], "quote": quote}],
        ),
        [result],
        trusted,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert quote in answer.text and "https://msi.com" in answer.text


@pytest.mark.asyncio
async def test_legacy_helpers_share_presets_and_official_source_rules():
    from ai_service.capabilities.web_search.presets import WebSearchPresets
    from ai_service.capabilities.web_search.schemas import DriverSoftwareSearchArgs
    from ai_service.capabilities.web_search.tools import WebSearchTools

    search = ScopedSearch()
    helper = WebSearchTools(search)
    result = await helper.search_driver_and_software(
        DriverSoftwareSearchArgs(
            component_name="MSI B650M",
            software_type="BIOS_UPDATE",
        )
    )
    assert search.queries[0].query == WebSearchPresets.driver_and_software(
        "MSI B650M", "BIOS_UPDATE"
    )
    assert search.queries[0].target_domains == ["msi.com"]
    assert all("untrusted.example" not in item.url for item in result.results)
    assert "Snippet chỉ dùng discovery" in result.analysis_prompt
    search.queries.clear()
    unresolved = await helper.search_driver_and_software(
        DriverSoftwareSearchArgs(component_name="Unknown Model Q")
    )
    assert not search.queries and unresolved.note == "SOURCE_UNRESOLVED"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
async def test_native_provider_exposes_driver_schema(provider_name):
    from langchain_core.messages import AIMessage
    from test_native_react_model import Provider

    from ai_service.capabilities.assistant.react.contracts import ToolName
    from ai_service.infrastructure.providers.react_model import NativeFlexibleModel

    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "bios",
                    "name": ToolName.RESEARCH_DRIVER.value,
                    "args": {
                        "component_name": "MSI B650M",
                        "software_type": "BIOS_UPDATE",
                    },
                }
            ],
        ),
        name=provider_name,
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Tra BIOS", [], (ToolName.RESEARCH_DRIVER,), None
    )
    assert decision.action.call.name == ToolName.RESEARCH_DRIVER
    schema = provider.model.tools[0]["parameters"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["SoftwareType"]["enum"] == [
        "VGA_DRIVER",
        "BIOS_UPDATE",
        "CHIPSET_DRIVER",
        "RGB_CONTROL",
    ]


@pytest.mark.asyncio
async def test_native_resume_keeps_committed_driver_evidence_without_refetch():
    from uuid import uuid4

    from langgraph.checkpoint.memory import InMemorySaver
    from test_react_graph import Budget

    from ai_service.capabilities.assistant.react.actions import FlexibleDecision
    from ai_service.infrastructure.graph.flexible import build_flexible_graph

    quote = "BIOS update for this exact MSI B650M model."
    search, reader = MockSearchClient(), MockWebPageReader(content=quote)

    class Model:
        fail_once = True

        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                assert "research_driver" in allowed
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "bios",
                            "name": "research_driver",
                            "arguments": {
                                "component_name": "MSI B650M",
                                "software_type": "BIOS_UPDATE",
                            },
                        },
                    }
                )
            if self.fail_once:
                self.fail_once = False
                raise TimeoutError("explanation failed")
            result = transcript[-1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "ignored",
                        "evidence_refs": result.evidence_refs,
                        "web_excerpts": [
                            {"evidence_ref": result.evidence_refs[0], "quote": quote}
                        ],
                    },
                }
            )

    graph = build_flexible_graph(
        Model(),
        runner(search, reader),
        context(),
        [CompletionRequirement(kind="ANSWER")],
        Budget(),
        conversation_revision=0,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(TimeoutError):
        await graph.ainvoke({"message": "Tra BIOS"}, config, durability="sync")
    before = list(reader.urls), list(search.queries)
    result = await graph.ainvoke(
        None, (await graph.aget_state(config)).config, durability="sync"
    )
    assert result["phase"] == "READY_TO_PUBLISH"
    assert (reader.urls, search.queries) == before
