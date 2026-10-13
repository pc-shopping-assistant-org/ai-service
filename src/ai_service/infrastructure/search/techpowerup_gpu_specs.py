"""Curated template for TechPowerUp GPU Database specifications.

Extracts physical board dimensions (Length x Width x Height in mm).
Preserves manufacturer dimension order; unsupported layouts return None
so the reader falls back to general Trafilatura extraction.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

from trafilatura import load_html

from ai_service.application.ports.web_search import ExtractedWebFact, WebFactStatus

PARSER_VERSION = "techpowerup-gpu-specs-dimensions:v1"
PATH = re.compile(r"^/gpu-specs/[a-z0-9-]+(?:\.[a-z0-9]+)?/?$")
DIMENSIONS_3D = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\b",
    re.IGNORECASE,
)
SINGLE_MM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*mm\b", re.IGNORECASE)


def extract_techpowerup_gpu_dimensions(
    html: str, final_url: str
) -> list[ExtractedWebFact] | None:
    url = urlparse(final_url)
    match = PATH.fullmatch(url.path)
    if url.hostname != "www.techpowerup.com" or not match or url.scheme != "https":
        return None

    root = load_html(html)
    if root is None:
        return None

    titles = root.xpath("//title/text()")
    canonicals = root.xpath('//link[@rel="canonical"]/@href')
    if len(titles) != 1 or len(canonicals) > 1:
        return None

    if len(canonicals) == 1:
        canonical = urlparse(canonicals[0])
        if (canonical.scheme, canonical.netloc, canonical.path.rstrip("/")) != (
            url.scheme,
            url.netloc,
            url.path.rstrip("/"),
        ):
            return None

    title = " ".join(titles[0].split())
    if "TechPowerUp GPU Database" not in title:
        return None

    h1_names = root.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " gpudb-name ")]/text()'
    )
    if h1_names and h1_names[0].strip():
        model = " ".join(h1_names[0].split())
    elif " Specs | TechPowerUp GPU Database" in title:
        model = title.removesuffix(" Specs | TechPowerUp GPU Database").strip()
    else:
        return None

    if not model or len(model) > 500:
        return None

    h2_board = root.xpath('//h2[normalize-space(.)="Board Design"]')
    if not h2_board:
        return None

    dl_list = h2_board[0].xpath("./following-sibling::dl | ../dl")
    if not dl_list:
        return None

    dts = dl_list[0].xpath("./dt")
    if not dts:
        return None

    # Collect dt -> dd text pairs
    pairs: list[tuple[str, str]] = []
    for dt in dts:
        dt_text = " ".join(" ".join(dt.itertext()).split())
        dds = dt.xpath("./following-sibling::dd[1]")
        if dds:
            dd_text = " ".join(" ".join(dds[0].itertext()).split())
            pairs.append((dt_text, dd_text))

    values: set[tuple[float, float, float]] = set()
    quotes: list[str] = []

    # Check for combined Dimensions first
    for dt_text, dd_text in pairs:
        if dt_text.lower() in ("dimensions", "card dimensions", "board dimensions"):
            quotes.append(f"{dt_text}: {dd_text}")
            for m in DIMENSIONS_3D.findall(dd_text):
                dims = (float(m[0]), float(m[1]), float(m[2]))
                if all(part > 0 for part in dims):
                    values.add(dims)

    # Check for separate Length, Width, Height
    if not values:
        lengths = [dd for dt, dd in pairs if dt.lower() == "length"]
        widths = [dd for dt, dd in pairs if dt.lower() == "width"]
        heights = [dd for dt, dd in pairs if dt.lower() == "height"]

        if lengths or widths or heights:
            for l_dd in lengths:
                for w_dd in widths:
                    for h_dd in heights:
                        quote_str = f"Length: {l_dd}, Width: {w_dd}, Height: {h_dd}"
                        quotes.append(quote_str)
                        l_m = SINGLE_MM.search(l_dd)
                        w_m = SINGLE_MM.search(w_dd)
                        h_m = SINGLE_MM.search(h_dd)
                        if l_m and w_m and h_m:
                            dims = (
                                float(l_m.group(1)),
                                float(w_m.group(1)),
                                float(h_m.group(1)),
                            )
                            if all(part > 0 for part in dims):
                                values.add(dims)

    if not quotes:
        return None

    if len(values) == 1:
        status = WebFactStatus.SUPPORTED
    elif len(values) > 1:
        status = WebFactStatus.CONFLICT
    else:
        status = WebFactStatus.UNKNOWN

    quote = "\n".join(dict.fromkeys(quotes))
    if len(quote) > 2000:
        return None

    return [
        ExtractedWebFact(
            model=model,
            value=next(iter(values)) if status == WebFactStatus.SUPPORTED else None,
            fact_status=status,
            source_url=final_url,
            document_hash=hashlib.sha256(html.encode()).hexdigest(),
            parser_version=PARSER_VERSION,
            locator='section.details h2="Board Design" dl.clearfix',
            quote=quote,
        )
    ]
