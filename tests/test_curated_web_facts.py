from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from test_react_tools import Catalog, context

from ai_service.application.ports.web_search import WebExtractionStatus, WebFactStatus
from ai_service.capabilities.assistant.react.contracts import (
    AnswerCandidate,
    CompletionRequirement,
    ToolCall,
)
from ai_service.capabilities.assistant.react.tools import (
    ToolDispatcher,
    validate_answer,
)
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader

URL = "https://www.asus.com/motherboards-components/graphics-cards/tuf-gaming/tuf-rtx5070ti-o16g-gaming/techspec/"
HTML = (
    Path(__file__).with_name("fixtures").joinpath("asus_gpu_techspec.html").read_text()
)
TPU_URL = (
    "https://www.techpowerup.com/gpu-specs/asus-tuf-rtx-4070-gaming-oc.b10950"
)
TPU_HTML = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("techpowerup_gpu_specs.html")
    .read_text()
)
TH_ASUS_URL = "https://www.tomshardware.com/pc-components/gpus/asus-geforce-rtx-5080-noctua-edition-review"
TH_ASUS_HTML = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("tomshardware_asus_rtx_5080.html")
    .read_text()
)
TH_MSI_URL = "https://www.tomshardware.com/pc-components/gpus/msi-geforce-rtx-5090-lightning-z-review"
TH_MSI_HTML = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("tomshardware_msi_rtx_5090.html")
    .read_text()
)
TH_RX_9070_XT_URL = "https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9070-xt-review"
TH_RX_9070_XT_HTML = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("tomshardware_rx_9070_xt.html")
    .read_text()
)
TH_RX_9070_GRE_URL = "https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9070-gre-review"
TH_RX_9070_GRE_HTML = (
    Path(__file__)
    .with_name("fixtures")
    .joinpath("tomshardware_rx_9070_gre.html")
    .read_text()
)


def th_reader(monkeypatch, html: str = TH_ASUS_HTML):
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
    return TrafilaturaReader()


def tpu_reader(monkeypatch, html=TPU_HTML):
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
    return TrafilaturaReader()


def reader(monkeypatch, html=HTML):
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
    return TrafilaturaReader()


@pytest.mark.asyncio
async def test_real_asus_template_dimensions_reach_server_rendered_answer(monkeypatch):
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), reader=reader(monkeypatch)
    )
    ctx = context()
    result = await dispatcher.execute(
        ToolCall(call_id="page", name="read_web_page", arguments={"url": URL}), ctx
    )
    page = result.result
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    fact = page.facts[0]
    assert fact.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact.fact.value == (329.0, 140.0, 62.5)
    assert fact.fact.unit == "mm"
    assert (
        fact.fact.model == "ASUS TUF Gaming GeForce RTX™ 5070 Ti 16GB GDDR7 OC Edition"
    )
    assert "329 x 140 x 62.5 mm" in fact.fact.quote
    assert fact.fact.source_url == URL
    answer = validate_answer(
        AnswerCandidate(
            text="999 mm invented length", evidence_refs=[fact.evidence_ref]
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "329 × 140 × 62.5 mm" in answer.text
    assert "999 mm" not in answer.text
    assert "GPU length" not in answer.text  # Do not invent axis order from dimensions.
    assert URL in answer.text
    forged = fact.model_dump()
    forged["fact"]["value"] = (999, 140, 62.5)
    with pytest.raises(ValidationError, match="identity"):
        type(fact).model_validate(forged)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "html,status",
    [
        (HTML.replace("329 x 140 x 62.5 mm", "Not specified"), WebFactStatus.UNKNOWN),
        (
            HTML.replace(
                "329 x 140 x 62.5 mm", "329 x 140 x 62.5 mm<br>400 x 140 x 62.5 mm"
            ),
            WebFactStatus.CONFLICT,
        ),
    ],
)
async def test_missing_or_conflicting_dimensions_never_publish_supported_numbers(
    monkeypatch, html, status
):
    page = await reader(monkeypatch, html).read(URL)
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    assert all(fact.fact_status == status for fact in page.facts)
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), reader=TrafilaturaReader()
    )
    ctx = context()
    result = await dispatcher.execute(
        ToolCall(call_id="page", name="read_web_page", arguments={"url": URL}), ctx
    )
    answer = validate_answer(
        AnswerCandidate(
            text="Invented 999 mm", evidence_refs=[result.result.facts[0].evidence_ref]
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert status.value in answer.text
    assert "999" not in answer.text
    assert "kích thước công bố:" not in answer.text


@pytest.mark.asyncio
async def test_fact_source_and_page_status_cannot_be_reassigned(monkeypatch):
    page = await reader(monkeypatch).read(URL)
    payload = page.model_dump()
    payload["final_url"] = "https://example.com/other"
    with pytest.raises(ValidationError, match="source"):
        type(page).model_validate(payload)
    payload = page.model_dump()
    payload["extraction_status"] = "CONTENT_ONLY"
    with pytest.raises(ValidationError, match="Non-structured"):
        type(page).model_validate(payload)


@pytest.mark.asyncio
async def test_http_failure_does_not_claim_curated_extraction(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(403, text="Forbidden")
            ),
            **kwargs,
        ),
    )
    page = await TrafilaturaReader().read(URL)
    assert page.extraction_status == WebExtractionStatus.FAILED
    assert page.facts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,html",
    [
        (URL.replace("www.asus.com", "example.com"), HTML),
        (URL.replace("/techspec/", "/review/"), HTML),
        (URL, HTML.replace("rowTableTitle", "unrecognizedTitle")),
        (URL, HTML.replace("RTX™ 5070 Ti", "RTX™ 5070")),
        (URL.replace("rtx5070ti", "rtx5070"), HTML.replace("rtx5070ti", "rtx5070")),
        (
            URL,
            HTML.replace(
                "TUF Gaming GeForce RTX™ 5070 Ti 16GB GDDR7 OC Edition - Tech Specs",
                "Other model - Tech Specs",
            ),
        ),
    ],
)
async def test_domain_path_layout_or_identity_mismatch_falls_back_without_facts(
    monkeypatch, url, html
):
    page = await reader(monkeypatch, html).read(url)
    assert page.extraction_status == WebExtractionStatus.CONTENT_ONLY
    assert page.facts == []
    assert page.content


@pytest.mark.asyncio
async def test_real_techpowerup_template_dimensions_reach_server_rendered_answer(
    monkeypatch,
):
    dispatcher = ToolDispatcher(
        Catalog(), PCBuildApplicationService(), reader=tpu_reader(monkeypatch)
    )
    ctx = context()
    result = await dispatcher.execute(
        ToolCall(
            call_id="page", name="read_web_page", arguments={"url": TPU_URL}
        ),
        ctx,
    )
    page = result.result
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    fact = page.facts[0]
    assert fact.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact.fact.value == (304.0, 138.0, 65.0)
    assert fact.fact.unit == "mm"
    assert fact.fact.model == "ASUS TUF RTX 4070 Gaming OC"
    assert "Length: 304 mm" in fact.fact.quote
    assert fact.fact.source_url == TPU_URL
    answer = validate_answer(
        AnswerCandidate(
            text="999 mm invented length", evidence_refs=[fact.evidence_ref]
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "304 × 138 × 65 mm" in answer.text
    assert "999 mm" not in answer.text
    assert TPU_URL in answer.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "html,status",
    [
        (
            TPU_HTML.replace("304 mm", "Not specified"),
            WebFactStatus.UNKNOWN,
        ),
        (
            TPU_HTML.replace(
                "<dt>Length</dt>\n    <dd>304 mm<br>12.0 inches</dd>",
                "<dt>Length</dt>\n    <dd>304 mm<br>12.0 inches</dd>\n    <dt>Length</dt>\n    <dd>400 mm</dd>",
            ),
            WebFactStatus.CONFLICT,
        ),
    ],
)
async def test_techpowerup_missing_or_conflicting_dimensions(
    monkeypatch, html, status
):
    page = await tpu_reader(monkeypatch, html).read(TPU_URL)
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    assert all(fact.fact_status == status for fact in page.facts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,html",
    [
        (TPU_URL.replace("www.techpowerup.com", "example.com"), TPU_HTML),
        (TPU_URL.replace("/gpu-specs/", "/review/"), TPU_HTML),
        (TPU_URL, TPU_HTML.replace("Board Design", "Other Section")),
        (TPU_URL, TPU_HTML.replace("TechPowerUp GPU Database", "Other Database")),
    ],
)
async def test_techpowerup_domain_path_layout_mismatch_falls_back(
    monkeypatch, url, html
):
    page = await tpu_reader(monkeypatch, html).read(url)
    assert page.extraction_status == WebExtractionStatus.CONTENT_ONLY
    assert page.facts == []
    assert page.content


@pytest.mark.asyncio
async def test_real_tomshardware_asus_5080_cm_dimensions_reach_server_rendered_answer(
    monkeypatch,
):
    dispatcher = ToolDispatcher(
        Catalog(),
        PCBuildApplicationService(),
        reader=th_reader(monkeypatch, TH_ASUS_HTML),
    )
    ctx = context()
    result = await dispatcher.execute(
        ToolCall(
            call_id="page", name="read_web_page", arguments={"url": TH_ASUS_URL}
        ),
        ctx,
    )
    page = result.result
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    assert len(page.facts) == 2

    # Fact 0: RTX 5080 Founders Edition (cm converted to mm)
    fact0 = page.facts[0]
    assert fact0.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact0.fact.value == (305.0, 122.0, 40.0)
    assert fact0.fact.unit == "mm"
    assert fact0.fact.model == "RTX 5080 Founders Edition"
    assert '12" x 4.8" x 1.6" (30.5 x 12.2 x 4 cm)' in fact0.fact.quote
    assert fact0.fact.source_url == TH_ASUS_URL

    # Fact 1: Asus RTX 5080 Noctua Edition (cm converted to mm)
    fact1 = page.facts[1]
    assert fact1.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact1.fact.value == (382.0, 136.0, 81.0)
    assert fact1.fact.unit == "mm"
    assert fact1.fact.model == "Asus RTX 5080 Noctua Edition"
    assert '15" x 5.3" x 3.2" (38.2 x 13.6 x 8.1 cm)' in fact1.fact.quote
    assert fact1.fact.source_url == TH_ASUS_URL

    # Validate answer formatting and anti-hallucination guard
    answer = validate_answer(
        AnswerCandidate(
            text="999 mm invented length", evidence_refs=[fact1.evidence_ref]
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "382 × 136 × 81 mm" in answer.text
    assert "999 mm" not in answer.text
    assert "Asus RTX 5080 Noctua Edition" in answer.text
    assert TH_ASUS_URL in answer.text


@pytest.mark.asyncio
async def test_real_tomshardware_msi_5090_direct_mm_and_excludes_radiator(
    monkeypatch,
):
    dispatcher = ToolDispatcher(
        Catalog(),
        PCBuildApplicationService(),
        reader=th_reader(monkeypatch, TH_MSI_HTML),
    )
    ctx = context()
    result = await dispatcher.execute(
        ToolCall(
            call_id="page", name="read_web_page", arguments={"url": TH_MSI_URL}
        ),
        ctx,
    )
    page = result.result
    assert page.extraction_status == WebExtractionStatus.STRUCTURED
    assert len(page.facts) == 2

    # Fact 0: RTX 5090 Founders Edition (direct mm in parentheses)
    fact0 = page.facts[0]
    assert fact0.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact0.fact.value == (304.0, 137.0, 60.0)
    assert fact0.fact.unit == "mm"
    assert fact0.fact.model == "RTX 5090 Founders Edition"
    assert '12" x 5.4" x 2.4" (304 x 137 x 60 mm)' in fact0.fact.quote
    assert fact0.fact.source_url == TH_MSI_URL

    # Fact 1: MSI RTX 5090 Lightning Z (direct mm in parentheses)
    fact1 = page.facts[1]
    assert fact1.fact.fact_status == WebFactStatus.SUPPORTED
    assert fact1.fact.value == (260.0, 151.0, 61.0)
    assert fact1.fact.unit == "mm"
    assert fact1.fact.model == "MSI RTX 5090 Lightning Z"
    assert '10.2" x 5.9" x 2.4" (260 x 151 x 61 mm)' in fact1.fact.quote
    assert fact1.fact.source_url == TH_MSI_URL

    # Radiator row was excluded from facts
    assert all(
        "radiator" not in f.fact.model.lower()
        and "radiator" not in f.fact.quote.lower()
        for f in page.facts
    )

    answer = validate_answer(
        AnswerCandidate(
            text="999 mm invented length", evidence_refs=[fact1.evidence_ref]
        ),
        [result],
        ctx,
        [CompletionRequirement(kind="ANSWER")],
    )
    assert "260 × 151 × 61 mm" in answer.text
    assert "999 mm" not in answer.text
    assert "MSI RTX 5090 Lightning Z" in answer.text
    assert TH_MSI_URL in answer.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,html",
    [
        (TH_RX_9070_XT_URL, TH_RX_9070_XT_HTML),
        (TH_RX_9070_GRE_URL, TH_RX_9070_GRE_HTML),
    ],
    ids=["rx_9070_xt", "rx_9070_gre"],
)
async def test_real_tomshardware_fallback_when_no_dimensions_row(
    monkeypatch, url, html
):
    page = await th_reader(monkeypatch, html).read(url)
    assert page.extraction_status == WebExtractionStatus.CONTENT_ONLY
    assert page.facts == []
    assert page.content is not None
    assert len(page.content) > 100


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,html",
    [
        (TH_ASUS_URL.replace("www.tomshardware.com", "example.com"), TH_ASUS_HTML),
        (TH_ASUS_URL.replace("/gpus/", "/news/"), TH_ASUS_HTML),
        (TH_ASUS_URL, TH_ASUS_HTML.replace("Tom's Hardware", "Other Hardware")),
        (
            TH_ASUS_URL,
            TH_ASUS_HTML.replace(
                "asus-geforce-rtx-5080-noctua-edition-review",
                "other-gpu-review",
            ),
        ),
    ],
    ids=["domain_mismatch", "path_mismatch", "title_mismatch", "canonical_mismatch"],
)
async def test_tomshardware_domain_path_or_metadata_mismatch_falls_back(
    monkeypatch, url, html
):
    page = await th_reader(monkeypatch, html).read(url)
    assert page.extraction_status == WebExtractionStatus.CONTENT_ONLY
    assert page.facts == []
    assert page.content

