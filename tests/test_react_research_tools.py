from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from test_react_graph import Budget
from test_react_tools import Catalog, context
from test_react_web_flow import MockSearchClient as GeneralSearch
from test_react_web_flow import MockWebPageReader as GeneralReader
from test_requirement_pipeline import rows, state

from ai_service.application.call_attempts import AttemptRejected
from ai_service.application.ports.web_search import WebPageReadStatus, WebSearchResult
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.react.actions import (
    FinalAnswerAction,
    FlexibleDecision,
)
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
    ToolRecord,
)
from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    allowed_tools,
    validate_answer,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.graph.flexible import build_flexible_graph
from ai_service.infrastructure.search.budgeted import (
    BudgetedWebReader,
    BudgetedWebSearch,
)


class MockSearchClient(GeneralSearch):
    """Search fake honors scope; production must still reject out-of-scope results."""

    async def search(self, query):
        result = await super().search(query)
        domain = query.target_domains[0] if query.target_domains else "example.com"
        for item in result.items:
            item.url = item.url.replace("example.com", domain)
        return result


class MockWebPageReader(GeneralReader):
    async def read(self, url):
        page = await super().read(url)
        if page.evidence_ref:
            page.evidence_ref = "web:" + canonical_hash(
                {"url": url, "content_hash": page.content_hash}
            )
        return page


def runner(search=None, reader=None):
    return ToolDispatcher(
        Catalog(),
        PCBuildApplicationService(),
        search or MockSearchClient(),
        reader or MockWebPageReader(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,arguments,words",
    [
        (
            "research_hardware",
            {"model": "Corsair RM850x", "focus": "PSU_QUALITY"},
            ["Corsair RM850x", "ripple", "protection"],
        ),
        (
            "research_hardware",
            {"model": "MSI B650M", "focus": "MOTHERBOARD"},
            ["MSI B650M", "VRM", "BIOS"],
        ),
        (
            "research_compatibility",
            {
                "part_a": "Ryzen 7 7800X3D",
                "part_b": "MSI B650M",
                "concern": "minimum BIOS version",
            },
            ["Ryzen 7 7800X3D", "MSI B650M", "minimum BIOS version"],
        ),
        (
            "research_workload",
            {
                "workload": "Black Myth Wukong",
                "hardware": ["RTX 4070 SUPER"],
                "resolution": "1440p",
                "preset": "Very High",
                "upscaling": "off",
            },
            ["Black Myth Wukong", "RTX 4070 SUPER", "1440p", "Very High", "off"],
        ),
    ],
)
async def test_research_searches_then_reads_actual_sources(name, arguments, words):
    search, reader = MockSearchClient(), MockWebPageReader()
    result = await runner(search, reader).execute(
        ToolCall(call_id="research", name=name, arguments=arguments), context()
    )
    assert result.status == "COMPLETED"
    section = result.result.sections[0]
    assert all(word in section.query for word in words)
    assert section.status == "READABLE" and reader.urls
    assert section.pages[0].content and result.evidence_refs
    assert result.evidence_refs == list(
        dict.fromkeys(page.evidence_ref for page in section.pages)
    )


@pytest.mark.asyncio
async def test_workload_research_never_invents_measurement_conditions():
    result = await runner().execute(
        ToolCall(
            call_id="requirements",
            name="research_workload",
            arguments={"workload": "Blender", "mode": "REQUIREMENTS"},
        ),
        context(),
    )
    query = result.result.sections[0].query
    assert "Blender" in query and "requirements" in query
    assert "1080" not in query and "1440" not in query and "Ultra" not in query


@pytest.mark.asyncio
async def test_research_web_evidence_supports_grounded_review_without_snippet_citations():
    trusted = context()
    result = await runner().execute(
        ToolCall(
            call_id="review",
            name="research_hardware",
            arguments={"model": "RTX 4070 SUPER", "focus": "PERFORMANCE"},
        ),
        trusted,
    )
    quote = "Resolution: 1440p Very High -> Average 72 FPS."
    answer = validate_answer(
        AnswerCandidate(
            text="invented 999 FPS",
            evidence_refs=result.evidence_refs,
            web_excerpts=[{"evidence_ref": result.evidence_refs[0], "quote": quote}],
            review={
                "product_name": "RTX 4070 SUPER",
                "summary": "Nhận xét theo nguồn đã đọc.",
                "pros": [],
                "cons": [],
                "verdict": "Cần đối chiếu đúng điều kiện đo.",
            },
        ),
        [result],
        trusted,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert quote in answer.text and "Nhận xét theo nguồn" in answer.text
    assert "999 FPS" not in answer.text and "https://nvidia.com" in answer.text
    with pytest.raises(ValueError, match="excerpt"):
        validate_answer(
            AnswerCandidate(
                text="false",
                evidence_refs=result.evidence_refs,
                web_excerpts=[
                    {
                        "evidence_ref": result.evidence_refs[0],
                        "quote": "Invented result",
                    }
                ],
            ),
            [result],
            trusted,
            [CompletionRequirement(kind="ANSWER")],
        )


@pytest.mark.asyncio
async def test_research_failure_is_recorded_without_discarding_readable_sections(
    realistic_catalog,
):
    trusted = context(state(), rows(realistic_catalog))
    dispatcher = runner()
    build_call = ToolCall(call_id="build", name="build_pc", arguments={})
    build = await dispatcher.execute(build_call, trusted)
    record = ToolRecord(call=build_call, result=build, origin_run_id=trusted.run_id)

    class PartialSearch:
        async def search(self, query):
            if "VRM" in query.query:
                return WebSearchResult(
                    query=query.query,
                    category=query.category,
                    note="WEB_SEARCH_UNAVAILABLE",
                )
            return await MockSearchClient().search(query)

    result = await runner(PartialSearch()).execute(
        ToolCall(
            call_id="whole-build",
            name="research_pc_build",
            arguments={"build_ref": str(build.result.build_ref)},
        ),
        trusted,
        records=[record],
    )
    assert result.status == "COMPLETED"
    sections = result.result.sections
    assert len(sections) == len(build.result.receipt.build.parts)
    names = {part.name for part in build.result.receipt.build.parts.values()}
    assert {section.subjects[0] for section in sections} == names
    assert any(
        step.status == "UNAVAILABLE"
        for section in sections
        for step in section.search_steps
    )
    assert any(section.status == "READABLE" for section in sections)
    assert result.result.build_source_ref == f"build:{build.result.build_ref}"
    assert "research_pc_build" in allowed_tools(trusted, [record])
    assert "research_pc_build" not in allowed_tools(context())


@pytest.mark.asyncio
async def test_build_research_rejects_foreign_or_stale_build_records(realistic_catalog):
    trusted = context(state(), rows(realistic_catalog))
    dispatcher = runner()
    call = ToolCall(call_id="build", name="build_pc", arguments={})
    build = await dispatcher.execute(call, trusted)
    foreign = ToolRecord(call=call, result=build, origin_run_id=uuid4())
    result = await dispatcher.execute(
        ToolCall(
            call_id="review",
            name="research_pc_build",
            arguments={"build_ref": str(build.result.build_ref)},
        ),
        trusted,
        records=[foreign],
    )
    assert result.status == "REJECTED" and result.error == "PREREQUISITE_MISSING"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"model": "GPU", "price_vnd": 100},
        {"model": "GPU", "focus": "MAKE_UP_FPS"},
        {"model": "GPU", "max_sources": True},
    ],
)
async def test_research_contracts_reject_unknown_fields_and_invalid_limits(arguments):
    result = await runner().execute(
        ToolCall(call_id="bad", name="research_hardware", arguments=arguments),
        context(),
    )
    assert result.status == "REJECTED" and result.error == "BAD_ARGUMENTS"


@pytest.mark.asyncio
async def test_native_resume_reuses_completed_research_sources():
    search, reader = MockSearchClient(), MockWebPageReader()

    class Model:
        fail_once = True

        async def decide(self, message, transcript, allowed, repair, context=None):
            if not transcript:
                assert "research_hardware" in allowed
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "review",
                            "name": "research_hardware",
                            "arguments": {"model": "RTX 4070 SUPER"},
                        },
                    }
                )
            if self.fail_once:
                self.fail_once = False
                raise TimeoutError("after research commit")
            result = transcript[-1].result
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "answer": {
                        "text": "ignored",
                        "evidence_refs": result.evidence_refs,
                        "web_excerpts": [
                            {
                                "evidence_ref": result.evidence_refs[0],
                                "quote": "Resolution: 1440p Very High -> Average 72 FPS.",
                            }
                        ],
                    },
                }
            )

    flow = build_flexible_graph(
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
        await flow.ainvoke({"message": "Đánh giá GPU"}, config, durability="sync")
    urls = list(reader.urls)
    snapshot = await flow.aget_state(config)
    result = await flow.ainvoke(None, snapshot.config, durability="sync")
    assert result["phase"] == "READY_TO_PUBLISH" and reader.urls == urls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_state,outcome,reason",
    [
        ("NO_RESULTS", "INSUFFICIENT_EVIDENCE", "NO_RESULTS"),
        ("UNREADABLE", "INSUFFICIENT_EVIDENCE", "SOURCE_UNAVAILABLE"),
        ("UNAVAILABLE", "DEPENDENCY_UNAVAILABLE", "DEPENDENCY_UNAVAILABLE"),
    ],
)
async def test_research_missing_sources_remain_explicitly_unresolved(
    source_state, outcome, reason
):
    class Search:
        async def search(self, query):
            if source_state == "UNREADABLE":
                return await MockSearchClient().search(query)
            return WebSearchResult(
                query=query.query,
                category=query.category,
                note="WEB_SEARCH_UNAVAILABLE"
                if source_state == "UNAVAILABLE"
                else None,
            )

    trusted = context()
    result = await runner(
        Search(), MockWebPageReader(status=WebPageReadStatus.UNREADABLE)
    ).execute(
        ToolCall(
            call_id="empty",
            name="research_hardware",
            arguments={"model": "Exact PSU model"},
        ),
        trusted,
    )
    assert result.result.sections[0].status == source_state and not result.evidence_refs
    answer = validate_final_answer(
        FinalAnswerAction(
            outcome=outcome,
            supporting_tool_call_ids=[result.tool_call_id],
            answer=AnswerCandidate(text="ignored"),
        ),
        [result],
        trusted,
        [CompletionRequirement(kind="ANSWER")],
        PCBuildApplicationService(),
    )
    assert answer.status == "UNRESOLVED" and answer.reason == reason
    assert "Exact PSU model" in answer.text


@pytest.mark.asyncio
async def test_saved_build_research_uses_accepted_names_and_rejects_absent_slots():
    from test_saved_build_explanation import saved_context

    trusted = saved_context(stale=True)
    result = await runner().execute(
        ToolCall(call_id="saved", name="research_pc_build", arguments={}), trusted
    )
    assert result.status == "COMPLETED"
    assert result.result.build_source_ref == trusted.saved_build.evidence_ref
    assert result.result.sections[0].subjects == ["Historical GPU"]
    invalid = await runner().execute(
        ToolCall(
            call_id="slot", name="research_pc_build", arguments={"slots": ["PSU"]}
        ),
        trusted,
    )
    assert invalid.status == "REJECTED" and invalid.error == "BAD_ARGUMENTS"


@pytest.mark.asyncio
async def test_research_reserves_search_and_read_and_propagates_budget_rejection():
    from test_react_web_flow import Budget as CountingBudget

    attempts = CountingBudget()
    call = ToolCall(
        call_id="budgeted",
        name="research_hardware",
        arguments={"model": "Corsair RM850x"},
    )
    await runner(
        BudgetedWebSearch(MockSearchClient(), attempts),
        BudgetedWebReader(MockWebPageReader(), attempts),
    ).execute(call, context())
    assert attempts.calls == 4  # Both official/review search + page reads reserve I/O.

    class Exhausted:
        async def invoke(self, *args):
            raise AttemptRejected("CALL_BUDGET_EXHAUSTED")

    with pytest.raises(AttemptRejected, match="CALL_BUDGET_EXHAUSTED"):
        await runner(BudgetedWebSearch(MockSearchClient(), Exhausted())).execute(
            call, context()
        )


@pytest.mark.asyncio
async def test_research_citations_alone_cannot_complete_a_mixed_answer():
    trusted = context()
    dispatcher = runner()
    product = await dispatcher.execute(
        ToolCall(call_id="product", name="search_catalog", arguments={"query": "GPU"}),
        trusted,
    )
    research = await dispatcher.execute(
        ToolCall(
            call_id="research", name="research_hardware", arguments={"model": "GPU"}
        ),
        trusted,
    )
    with pytest.raises(ValueError, match="Research answer requires"):
        validate_answer(
            AnswerCandidate(
                text="It is excellent",
                evidence_refs=[*product.evidence_refs, *research.evidence_refs],
            ),
            [product, research],
            trusted,
            [CompletionRequirement(kind="ANSWER")],
        )


@pytest.mark.asyncio
async def test_unknown_research_facts_can_explain_missing_data_but_not_complete_answer():
    from ai_service.application.ports.web_search import (
        ExtractedWebFact,
        WebExtractionStatus,
        WebFactStatus,
    )

    class Reader(MockWebPageReader):
        async def read(self, url):
            page = await super().read(url)
            page.document_hash = "b" * 64
            page.extraction_status = WebExtractionStatus.STRUCTURED
            page.facts = [
                ExtractedWebFact(
                    model="GPU",
                    value=None,
                    fact_status=WebFactStatus.UNKNOWN,
                    source_url=url,
                    document_hash=page.document_hash,
                    parser_version="fixture-v1",
                    locator="table",
                    quote="Dimensions not specified",
                )
            ]
            return page

    trusted = context()
    result = await runner(reader=Reader()).execute(
        ToolCall(
            call_id="unknown", name="research_hardware", arguments={"model": "GPU"}
        ),
        trusted,
    )
    fact_ref = result.result.sections[0].pages[0].facts[0].evidence_ref
    candidate = AnswerCandidate(text="ignored", evidence_refs=[fact_ref])
    required = [CompletionRequirement(kind="ANSWER")]
    with pytest.raises(ValueError, match="Unknown/conflicting"):
        validate_final_answer(
            FinalAnswerAction(answer=candidate),
            [result],
            trusted,
            required,
            PCBuildApplicationService(),
        )
    answer = validate_final_answer(
        FinalAnswerAction(
            outcome="INSUFFICIENT_EVIDENCE",
            supporting_tool_call_ids=[result.tool_call_id],
            answer=candidate,
        ),
        [result],
        trusted,
        required,
        PCBuildApplicationService(),
    )
    assert answer.status == "UNRESOLVED" and answer.reason == "FACT_UNKNOWN"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name", ["openai", "gemini"])
async def test_native_provider_binds_and_parses_new_research_schema(provider_name):
    from langchain_core.messages import AIMessage
    from test_native_react_model import Provider

    from ai_service.capabilities.assistant.react.contracts import ToolName
    from ai_service.infrastructure.providers.react_model import NativeFlexibleModel

    provider = Provider(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "psu",
                    "name": ToolName.RESEARCH_HARDWARE.value,
                    "args": {"model": "Corsair RM850x", "focus": "PSU_QUALITY"},
                }
            ],
        ),
        name=provider_name,
    )
    decision = await NativeFlexibleModel(provider).decide(
        "Review nguồn", [], (ToolName.RESEARCH_HARDWARE,), None
    )
    assert decision.action.calls[0].name == ToolName.RESEARCH_HARDWARE
    schema = provider.model.tools[0]
    assert schema["parameters"]["additionalProperties"] is False
    assert "exact model" in schema["description"]


@pytest.mark.asyncio
async def test_v2_graph_builds_then_researches_that_exact_build(realistic_catalog):
    trusted = context(state(), rows(realistic_catalog))

    class Model:
        async def decide(self, message, transcript, allowed, repair, context=None):
            assert repair is None
            if not transcript:
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "build",
                            "name": "build_pc",
                            "arguments": {},
                        },
                    }
                )
            build = transcript[0].result.result
            if len(transcript) == 1:
                assert "research_pc_build" in allowed
                return FlexibleDecision(
                    action={
                        "kind": "CALL_TOOL",
                        "call": {
                            "call_id": "research",
                            "name": "research_pc_build",
                            "arguments": {"build_ref": str(build.build_ref)},
                        },
                    }
                )
            research = transcript[-1].result
            assert research.status == "COMPLETED"
            return FlexibleDecision(
                action={
                    "kind": "FINAL_ANSWER",
                    "outcome": "BUILT",
                    "answer": {
                        "text": "ignored",
                        "build_ref": str(build.build_ref),
                        "evidence_refs": [
                            *transcript[0].result.evidence_refs,
                            *research.evidence_refs,
                        ],
                        "web_excerpts": [
                            {
                                "evidence_ref": research.evidence_refs[0],
                                "quote": "Resolution: 1440p Very High -> Average 72 FPS.",
                            }
                        ],
                    },
                }
            )

    result = await build_flexible_graph(
        Model(),
        runner(),
        trusted,
        [
            CompletionRequirement(
                kind="BUILD", constraints_revision=trusted.requirements.revision
            ),
            CompletionRequirement(kind="ANSWER"),
        ],
        Budget(),
        conversation_revision=0,
    ).ainvoke({"message": "Build rồi tìm review từng linh kiện"})
    assert result["phase"] == "READY_TO_PUBLISH"
    assert "Nguồn nghiên cứu đã tra" in result["answer"]["text"]
    assert result["answer"]["build_ref"]
