import threading

import httpx
import pytest
import trafilatura

from ai_service.application.ports.web_search import WebPageReadStatus
from ai_service.infrastructure.search.trafilatura_reader import TrafilaturaReader


@pytest.mark.asyncio
async def test_reader_does_not_download_a_software_package_body(monkeypatch):
    consumed = []

    class Package(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.append(True)
            yield b"software package bytes"

    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "application/octet-stream"},
            stream=Package(),
        )
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs)
    )
    result = await TrafilaturaReader().read("https://vendor.example/driver.exe")
    assert result.status == WebPageReadStatus.UNREADABLE and not consumed
    assert result.content is None and result.evidence_ref is None


@pytest.mark.asyncio
async def test_reader_parses_outside_event_loop_and_preserves_result(monkeypatch):
    loop_thread = threading.get_ident()
    threads = []
    extract = trafilatura.extract
    metadata = trafilatura.extract_metadata

    def extracting(*args, **kwargs):
        threads.append(threading.get_ident())
        return extract(*args, **kwargs)

    def metadata_extracting(*args, **kwargs):
        threads.append(threading.get_ident())
        return metadata(*args, **kwargs)

    monkeypatch.setattr(trafilatura, "extract", extracting)
    monkeypatch.setattr(trafilatura, "extract_metadata", metadata_extracting)
    client = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            text="<article><h1>GPU specs</h1><p>Graphics card length is 300 mm. Manufacturer specifications list the dimensions of this graphics card.</p></article>",
        )
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client(transport=transport, **kwargs)
    )
    result = await TrafilaturaReader().read("https://example.com/specs")
    assert result.status == WebPageReadStatus.READABLE
    assert "300 mm" in result.content
    assert len(threads) == 2 and all(thread != loop_thread for thread in threads)


@pytest.mark.asyncio
async def test_trafilatura_reader_extracts_content_and_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RTX 4070 Super Black Myth Wukong Benchmark</title>
    </head>
    <body>
        <nav><a href="/">Home</a><a href="/reviews">Reviews</a></nav>
        <article>
            <h1>RTX 4070 Super Performance Test</h1>
            <p>Testing Black Myth Wukong on PC with RTX 4070 Super:</p>
            <table>
                <thead>
                    <tr><th>Resolution</th><th>Settings</th><th>Average FPS</th></tr>
                </thead>
                <tbody>
                    <tr><td>1440p</td><td>Very High</td><td>72 FPS</td></tr>
                    <tr><td>4K</td><td>High DLSS Quality</td><td>58 FPS</td></tr>
                </tbody>
            </table>
        </article>
        <footer>Copyright 2026 Hardware Review</footer>
    </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_content, request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    reader = TrafilaturaReader()
    result = await reader.read("https://example.com/benchmark/rtx-4070-super")

    assert result.status == WebPageReadStatus.READABLE
    assert result.title == "RTX 4070 Super Performance Test"
    assert result.content is not None
    assert "72 FPS" in result.content
    assert "1440p" in result.content
    assert result.content_hash is not None
    assert result.evidence_ref is not None
    assert result.final_url == "https://example.com/benchmark/rtx-4070-super"


@pytest.mark.asyncio
async def test_identical_content_on_distinct_sources_has_distinct_evidence(monkeypatch):
    html = "<html><body><article><h1>GPU specifications</h1><p>GPU length is 300 millimeters. This manufacturer lists the physical dimensions of the graphics card.</p></article></body></html>"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=html, request=request)
    )
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )
    reader = TrafilaturaReader()
    first = await reader.read("https://example.com/one")
    second = await reader.read("https://example.com/two")
    assert first.status == second.status == WebPageReadStatus.READABLE
    assert first.content_hash == second.content_hash
    assert first.evidence_ref != second.evidence_ref


@pytest.mark.asyncio
async def test_trafilatura_reader_handles_http_errors_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden", request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    reader = TrafilaturaReader()
    result = await reader.read("https://example.com/blocked")

    assert result.status == WebPageReadStatus.UNREADABLE
    assert result.content is None
    assert result.evidence_ref is None


@pytest.mark.asyncio
async def test_trafilatura_reader_handles_empty_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="   ", request=request)

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=transport, **kwargs),
    )

    reader = TrafilaturaReader()
    result = await reader.read("https://example.com/empty")

    assert result.status == WebPageReadStatus.EMPTY
    assert result.content is None
