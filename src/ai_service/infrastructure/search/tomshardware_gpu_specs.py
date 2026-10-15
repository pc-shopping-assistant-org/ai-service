"""Curated template for Tom's Hardware GPU reviews.

Extracts physical board dimensions (source order, no inferred axes) from
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

PARSER_VERSION = "tomshardware-gpu-dimensions:v2"
MAX_FACT_QUOTE_CHARS = 2_000
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

    for table_index, table in enumerate(tables, start=1):
        headers = table.xpath("./thead/tr[1]") or table.xpath(
            "./tr[1][th] | ./tbody/tr[1][th]"
        )
        if len(headers) != 1:
            continue
        th_cells = [
            cell.text_content().strip() for cell in headers[0].xpath("./th | ./td")
        ]
        if len(th_cells) < 2 or DIM_ROW.fullmatch(th_cells[0]):
            continue

        for row_index, tr in enumerate(table.xpath(".//tr"), start=1):
            cells = [c.text_content().strip() for c in tr.xpath("./td | ./th")]
            if not cells:
                continue
            first_label = cells[0].strip()
            if not DIM_ROW.match(first_label):
                continue

            for col_idx in range(1, len(cells)):
                if col_idx >= len(th_cells):
                    continue
                model_name = th_cells[col_idx]
                if (
                    not model_name
                    or not any(char.isalpha() for char in model_name)
                    or model_name.casefold().startswith("header cell")
                    or _parse_dimensions(model_name) is not None
                ):
                    continue

                dim_str = cells[col_idx]
                dims = _parse_dimensions(dim_str)
                quote = f"{model_name} {first_label}: {dim_str}"
                if len(quote) > MAX_FACT_QUOTE_CHARS:
                    quote = quote[:MAX_FACT_QUOTE_CHARS]

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
                        locator=f"((//table)[{table_index}]//tr)[{row_index}]/*[{col_idx + 1}]",
                        quote=quote,
                        unit="mm",
                    )
                )

    values_by_model: dict[str, set[tuple[float, float, float]]] = {}
    for fact in facts:
        if fact.value is not None:
            key = " ".join(fact.model.split()).casefold()
            values_by_model.setdefault(key, set()).add(fact.value)
    conflicting_models = {
        model for model, values in values_by_model.items() if len(values) > 1
    }
    # Retain each source row, but never publish a winning number from conflicting rows.
    return [
        fact.model_copy(update={"value": None, "fact_status": WebFactStatus.CONFLICT})
        if " ".join(fact.model.split()).casefold() in conflicting_models
        else fact
        for fact in facts
    ] or None
