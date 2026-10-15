import httpx
import pytest
from test_react_tools import Catalog, context

from ai_service.application.ports.web_search import WebExtractionStatus, WebFactStatus
from ai_service.capabilities.assistant.react.actions import FinalAnswerAction
from ai_service.capabilities.assistant.react.contracts import (
    CompletionRequirement,
    ToolCall,
)
from ai_service.capabilities.assistant.react.outcomes import validate_final_answer
from ai_service.capabilities.assistant.react.tools import ToolDispatcher
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.search.tomshardware_gpu_specs import (
    extract_tomshardware_gpu_dimensions,
)
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

URL = "https://www.tomshardware.com/pc-components/gpus/asus-rtx-5070-ti-review"
MODEL = "ASUS TUF RTX 5070 Ti OC"


def page(table: str) -> str:
    return f"""<html><head><title>GPU review | Tom's Hardware</title>
    <link rel="canonical" href="{URL}"></head><body><article>
    <h1>GPU review</h1><p>This review discusses the graphics card design,
    dimensions and installation considerations. The source tables below record
    physical measurements; they do not establish performance or compatibility.</p>
    {table}</article></body></html>"""


def table(model: str, *dimensions: str) -> str:
    rows = "".join(
        f"<tr><td>Dimensions</td><td>{value}</td></tr>" for value in dimensions
    )
    return (
        f"<table><thead><tr><th>Graphics Card</th><th>{model}</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "markup",
    [
        "<table><tr><td>Dimensions</td><td>300 x 120 x 50 mm</td></tr></table>",
        table("", "300 x 120 x 50 mm"),
        table("Header Cell - Column 1", "300 x 120 x 50 mm"),
        table("300 x 120 x 50 mm", "300 x 120 x 50 mm"),
        (
            "<table><thead><tr><th>Graphics Card</th></tr></thead>"
            "<tbody><tr><td>Dimensions</td><td>300 x 120 x 50 mm</td></tr></tbody></table>"
        ),
    ],
    ids=[
        "no-header",
        "empty-model",
        "placeholder",
        "dimension-as-model",
        "unbound-column",
    ],
)
async def test_unbound_toms_dimensions_fall_back_without_facts(monkeypatch, markup):
    html = page(markup)
    assert extract_tomshardware_gpu_dimensions(html, URL) is None
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=html)
            ),
            **kwargs,
        ),
    )

    result = await TrafilaturaReader().read(URL)

    assert result.extraction_status == WebExtractionStatus.CONTENT_ONLY
    assert result.facts == []
    assert result.content


@pytest.mark.parametrize("separate_tables", [False, True])
def test_conflicting_toms_rows_clear_only_the_same_models_values(separate_tables):
    html = page(
        table(MODEL, "300 x 120 x 50 mm")
        + table(f"  {MODEL.lower()}  ", "310 x 120 x 50 mm")
        + table("RTX 5080 Founders Edition", "305 x 122 x 40 mm")
        if separate_tables
        else table(MODEL, "300 x 120 x 50 mm", "310 x 120 x 50 mm")
        + table("RTX 5080 Founders Edition", "305 x 122 x 40 mm")
    )

    facts = extract_tomshardware_gpu_dimensions(html, URL)

    assert facts is not None
    matching = [fact for fact in facts if fact.model.casefold() == MODEL.casefold()]
    assert len(matching) == 2
    assert all(fact.fact_status == WebFactStatus.CONFLICT for fact in matching)
    assert all(fact.value is None for fact in matching)
    assert "300 x 120 x 50 mm" in matching[0].quote
    assert "310 x 120 x 50 mm" in matching[1].quote
    other = [fact for fact in facts if fact.model.casefold() != MODEL.casefold()]
    assert len(other) == 1
    assert other[0].fact_status == WebFactStatus.SUPPORTED
    assert other[0].value == (305.0, 122.0, 40.0)


def test_equivalent_toms_mm_and_cm_rows_are_not_a_conflict():
    facts = extract_tomshardware_gpu_dimensions(
        page(table(MODEL, "300 x 120 x 50 mm", "30 x 12 x 5 cm")), URL
    )

    assert facts is not None
    assert len(facts) == 2
    assert all(fact.fact_status == WebFactStatus.SUPPORTED for fact in facts)
    assert all(fact.value == (300.0, 120.0, 50.0) for fact in facts)


@pytest.mark.asyncio
async def test_toms_conflict_reaches_agent_as_unresolved_not_a_positive_fact(
    monkeypatch,
):
    html = page(table(MODEL, "300 x 120 x 50 mm", "310 x 120 x 50 mm"))
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=html)
            ),
            **kwargs,
        ),
    )
    ctx = context()
    service = PCBuildApplicationService()
    record = await ToolDispatcher(
        Catalog(), service, reader=TrafilaturaReader()
    ).execute(
        ToolCall(call_id="page", name="read_web_page", arguments={"url": URL}), ctx
    )
    refs = [fact.evidence_ref for fact in record.result.facts]
    tasks = [CompletionRequirement(kind="ANSWER")]
    with pytest.raises(ValueError, match="insufficient|unresolved"):
        validate_final_answer(
            FinalAnswerAction(answer={"text": "300 mm fits", "evidence_refs": refs}),
            [record],
            ctx,
            tasks,
            service,
        )
    answer = validate_final_answer(
        FinalAnswerAction(
            outcome="INSUFFICIENT_EVIDENCE",
            supporting_tool_call_ids=["page"],
            answer={"text": "300 mm fits", "evidence_refs": refs},
        ),
        [record],
        ctx,
        tasks,
        service,
    )
    assert answer.status == "UNRESOLVED"
    assert answer.reason == "FACT_CONFLICT"
    assert "CONFLICT" in answer.text and URL in answer.text
    assert "300 mm fits" not in answer.text and "kích thước công bố:" not in answer.text
