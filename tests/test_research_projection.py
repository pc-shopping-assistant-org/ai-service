"""Model-facing research observations drop replay data and never lose evidence refs."""

import json

import pytest

from ai_service.application.ports.web_search import WebPageReadStatus
from ai_service.capabilities.assistant.react.contracts import (
    ReadWebPageResult,
    ResearchResult,
    ResearchSearchStep,
    ResearchSection,
    SearchWebItem,
    ToolName,
    ToolResult,
)
from ai_service.capabilities.web_search.research_contracts import ResearchSourceStatus
from ai_service.capabilities.web_search.source_policy import SourceTier
from ai_service.infrastructure.providers.tool_projection import observation_json

HASH = "a" * 64


def research_result(contents: list[str]) -> ToolResult:
    urls = [f"https://corsair.com/p{index}" for index in range(len(contents))]
    section = ResearchSection(
        subjects=["Corsair RM850x"],
        focus="PSU_QUALITY",
        query="Corsair RM850x review",
        status=ResearchSourceStatus.READABLE,
        search_steps=[
            ResearchSearchStep(
                tier=SourceTier.OFFICIAL,
                query="secret replay query",
                target_domains=["corsair.com"],
                status=ResearchSourceStatus.READABLE,
                result_urls=urls,
            ),
            ResearchSearchStep(
                tier=SourceTier.INDEPENDENT_REVIEW,
                query="another replay query",
                target_domains=["hwbusters.com"],
                status=ResearchSourceStatus.NO_RESULTS,
            ),
        ],
        results=[
            SearchWebItem(title=f"T{i}", url=url, snippet="duplicate snippet")
            for i, url in enumerate(urls)
        ],
        pages=[
            ReadWebPageResult(
                url=url,
                final_url=url,
                title=f"T{i}",
                status=WebPageReadStatus.READABLE,
                content_hash=HASH,
                content=content,
                evidence_ref=f"web:ref{i}",
            )
            for i, (url, content) in enumerate(zip(urls, contents, strict=True))
        ],
    )
    payload = ResearchResult(
        tool=ToolName.RESEARCH_HARDWARE,
        sections=[section],
        source_policy_version="manufacturer-first-v1",
        source_policy_hash=HASH,
    )
    return ToolResult(
        tool_call_id="research",
        tool=payload.tool,
        status="COMPLETED",
        input_hash="c" * 64,
        constraints_revision=0,
        result=payload,
        evidence_refs=[f"web:ref{i}" for i in range(len(contents))],
    )


def test_research_projection_drops_replay_data_but_keeps_sources_and_tier_status():
    output = observation_json(research_result(["Nội dung nguồn hãng."]), 12_000)
    projected = json.loads(output)
    section = projected["result"]["sections"][0]
    assert section["sources"] == [
        {
            "url": "https://corsair.com/p0",
            "status": "READABLE",
            "title": "T0",
            "evidence_ref": "web:ref0",
            "content": "Nội dung nguồn hãng.",
        }
    ]
    assert section["tiers"] == {"OFFICIAL": "READABLE", "INDEPENDENT_REVIEW": "NO_RESULTS"}
    for dropped in ("replay query", "duplicate snippet", HASH, "target_domains"):
        assert dropped not in output
    assert "projection" not in projected


def test_oversized_research_shares_budget_and_keeps_every_evidence_ref():
    contents = ["x" * 9_000, "y" * 9_000, "Ghi chú tiếng Việt " * 500]
    output = observation_json(research_result(contents), 6_000)
    assert len(output.encode()) <= 6_000
    projected = json.loads(output)
    assert projected["projection"] == {"truncated": True}
    sources = projected["result"]["sections"][0]["sources"]
    assert [s["evidence_ref"] for s in sources] == ["web:ref0", "web:ref1", "web:ref2"]
    assert all(s["content"] for s in sources)


def test_research_that_cannot_fit_even_without_content_is_rejected_not_degraded():
    with pytest.raises(ValueError, match="REACT_TOOL_CONTEXT_TOO_LARGE"):
        observation_json(research_result(["x" * 9_000]), 200)
