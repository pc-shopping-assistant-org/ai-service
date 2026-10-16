import pytest
from test_react_research_tools import MockSearchClient, runner
from test_react_tools import context

from ai_service.capabilities.assistant.react.contracts import ToolCall


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "focus,expected",
    [
        ("PERFORMANCE", {"techspot.com", "computerbase.de", "techpowerup.com"}),
        ("PRODUCTIVITY", {"pugetsystems.com", "phoronix.com"}),
        ("THERMAL_NOISE", {"gamersnexus.net", "kitguru.net"}),
        ("MEMORY", {"techpowerup.com", "tomshardware.com", "techspot.com"}),
        ("PSU_QUALITY", {"hwbusters.com", "cybenetics.com", "techpowerup.com"}),
    ],
)
async def test_hardware_uses_benchmark_sources_for_its_test_type(focus, expected):
    result = await runner().execute(
        ToolCall(
            call_id="hardware",
            name="research_hardware",
            arguments={"model": "Registered Component", "focus": focus},
        ),
        context(),
    )
    section = result.result.sections[0]
    review = next(
        step for step in section.search_steps if step.tier == "INDEPENDENT_REVIEW"
    )
    assert expected <= set(review.target_domains)
    assert section.pages and result.evidence_refs
    assert section.pages[0].final_url.startswith("https://" + review.target_domains[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,workload,expected",
    [
        ("GAMING", "Black Myth Wukong", {"techspot.com", "computerbase.de"}),
        ("PRODUCTIVITY", "Blender", {"pugetsystems.com", "phoronix.com"}),
    ],
)
async def test_workload_selects_benchmark_group_without_changing_official_priority(
    kind, workload, expected
):
    result = await runner().execute(
        ToolCall(
            call_id="workload",
            name="research_workload",
            arguments={
                "workload": workload,
                "hardware": ["RTX 5070 Ti"],
                "benchmark_kind": kind,
            },
        ),
        context(),
    )
    assert result.status == "COMPLETED"
    section = result.result.sections[0]
    review = next(
        step for step in section.search_steps if step.tier == "INDEPENDENT_REVIEW"
    )
    assert expected <= set(review.target_domains)
    if kind == "PRODUCTIVITY":
        assert section.search_steps[0].target_domains == ["blender.org"]
        assert (
            result.result.missing_conditions == []
        )  # No gaming-only resolution/DLSS questions.
    else:
        assert {condition.value for condition in result.result.missing_conditions} == {
            "resolution",
            "preset",
            "upscaling",
        }


@pytest.mark.asyncio
async def test_benchmark_kind_rejects_unknown_values():
    result = await runner().execute(
        ToolCall(
            call_id="invalid",
            name="research_workload",
            arguments={
                "workload": "Blender",
                "benchmark_kind": "MAKE_UP_SCORE",
            },
        ),
        context(),
    )
    assert result.status == "REJECTED" and result.error == "BAD_ARGUMENTS"


@pytest.mark.asyncio
async def test_configured_benchmark_domains_are_used_instead_of_builtins():
    from ai_service.capabilities.web_search.source_policy import ResearchSourcePolicy

    search = MockSearchClient()
    dispatcher = runner(search)
    dispatcher.source_policy = ResearchSourcePolicy(
        reviews={"PERFORMANCE": ["benchmark.example"]}
    )
    result = await dispatcher.execute(
        ToolCall(
            call_id="config",
            name="research_workload",
            arguments={"workload": "Game"},
        ),
        context(),
    )
    assert result.result.sections[0].search_steps[0].target_domains == [
        "benchmark.example"
    ]
