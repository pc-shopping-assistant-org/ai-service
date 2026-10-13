"""Curated template for Tom's Hardware GPU reviews.

Extracts physical board dimensions (Length x Width x Height in mm) from
structured comparison and specifications tables in Tom's Hardware GPU reviews.
Converts both centimeter and millimeter notations to canonical millimeters.
Unsupported layouts return None so the reader falls back to general extraction.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from trafilatura import load_html

from ai_service.application.ports.web_search import ExtractedWebFact, WebFactStatus

PARSER_VERSION = "tomshardware-gpu-dimensions:v1"
PATH = re.compile(
    r"^/(?:pc-components/gpus/[a-z0-9-]+-review|reviews/[a-z0-9-]+-review(?:,\d+\.html)?)/?$",
    re.IGNORECASE,
)
DIM_ROW = re.compile(r"^dimensions(?:\s*\(.*\))?$", re.IGNORECASE)

MM_PAREN = re.compile(
    r"\(\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\s*\)",
    re.IGNORECASE,
)
CM_PAREN = re.compile(
    r"\(\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*cm\s*\)",
    re.IGNORECASE,
)
MM_DIRECT = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\b",
    re.IGNORECASE,
)
CM_DIRECT = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*cm\b",
    re.IGNORECASE,
)


def _parse_dimensions(text: str) -> tuple[float, float, float] | None:
    if m := MM_PAREN.search(text):
        g = m.groups()
        return (round(float(g[0]), 1), round(float(g[1]), 1), round(float(g[2]), 1))
    if m := CM_PAREN.search(text):
        g = m.groups()
        return (
            round(float(g[0]) * 10, 1),
            round(float(g[1]) * 10, 1),
            round(float(g[2]) * 10, 1),
        )
    if m := MM_DIRECT.search(text):
        g = m.groups()
        return (round(float(g[0]), 1), round(float(g[1]), 1), round(float(g[2]), 1))
    if m := CM_DIRECT.search(text):
        g = m.groups()
        return (
            round(float(g[0]) * 10, 1),
            round(float(g[1]) * 10, 1),
            round(float(g[2]) * 10, 1),
        )
    return None


def extract_tomshardware_gpu_dimensions(
    html: str, final_url: str
) -> list[ExtractedWebFact] | None:
    url = urlparse(final_url)
    if (
        url.hostname not in {"www.tomshardware.com", "tomshardware.com"}
        or url.scheme != "https"
    ):
        return None
    if not PATH.fullmatch(url.path):
        return None

    root = load_html(html)
    if root is None:
        return None

    titles = root.xpath("//title/text()")
    if not titles or "Tom's Hardware" not in titles[0]:
        return None

    canonicals = root.xpath('//link[@rel="canonical"]/@href')
    if len(canonicals) == 1:
        canonical = urlparse(canonicals[0])
        if (canonical.scheme, canonical.netloc, canonical.path.rstrip("/")) != (
            url.scheme,
            url.netloc,
            url.path.rstrip("/"),
        ):
            return None

    doc_hash = hashlib.sha256(html.encode()).hexdigest()
    tables = root.xpath("//table")
    facts: list[ExtractedWebFact] = []

    for table in tables:
        th_cells = [
            th.text_content().strip()
            for th in table.xpath(".//thead//th | .//thead//td")
        ]
        if not th_cells:
            th_cells = [
                td.text_content().strip()
                for td in table.xpath(".//tr[1]/th | .//tr[1]/td")
            ]

        for tr in table.xpath(".//tr"):
            cells = [c.text_content().strip() for c in tr.xpath("./td | ./th")]
            if not cells:
                continue
            first_label = cells[0].strip()
            if not DIM_ROW.match(first_label):
                continue

            for col_idx in range(1, len(cells)):
                model_name = (
                    th_cells[col_idx]
                    if col_idx < len(th_cells)
                    else f"Model {col_idx}"
                )
                if model_name.startswith("Header Cell"):
                    continue

                dim_str = cells[col_idx]
                dims = _parse_dimensions(dim_str)
                quote = f"{model_name} {first_label}: {dim_str}"
                if len(quote) > 2000:
                    quote = quote[:2000]

                status = (
                    WebFactStatus.SUPPORTED
                    if dims and all(d > 0 for d in dims)
                    else WebFactStatus.UNKNOWN
                )
                facts.append(
                    ExtractedWebFact(
                        model=model_name,
                        value=dims if status == WebFactStatus.SUPPORTED else None,
                        fact_status=status,
                        source_url=final_url,
                        document_hash=doc_hash,
                        parser_version=PARSER_VERSION,
                        locator='table tr td:first-child="Dimensions"',
                        quote=quote,
                        unit="mm",
                    )
                )

    return facts if facts else None
