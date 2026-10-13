"""One curated template, matched by final URL, canonical identity and DOM.

Dimension tuples retain the manufacturer's order; no inferred length/width/height.
Unsupported pages/layouts return None so the general extractor remains the fallback.
"""

import hashlib
import re
from urllib.parse import urlparse

from trafilatura import load_html

from ai_service.application.ports.web_search import ExtractedWebFact, WebFactStatus

PARSER_VERSION = "asus-gpu-techspec-dimensions:v1"
PATH = re.compile(r"^/motherboards-components/graphics-cards/[^/]+/([^/]+)/techspec/$")
GPU = re.compile(r"(?:rtx|gtx|rx)\d{3,4}(?:ti|super|xtx|xt)?", re.IGNORECASE)
DIMENSIONS = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*mm\b",
    re.IGNORECASE,
)


def extract_asus_dimensions(html: str, final_url: str) -> list[ExtractedWebFact] | None:
    url = urlparse(final_url)
    match = PATH.fullmatch(url.path)
    if url.hostname != "www.asus.com" or not match or url.scheme != "https":
        return None
    root = load_html(html)
    if root is None:
        return None
    titles = root.xpath("//title/text()")
    canonicals = root.xpath('//link[@rel="canonical"]/@href')
    if len(titles) != 1 or len(canonicals) != 1:
        return None
    canonical = urlparse(canonicals[0])
    if (canonical.scheme, canonical.netloc, canonical.path) != (
        url.scheme,
        url.netloc,
        url.path,
    ):
        return None
    title = " ".join(titles[0].split())
    gpu = GPU.search(match[1])
    normalized_title = re.sub(r"[^a-z0-9]", "", title.lower())
    title_gpu = GPU.search(normalized_title)
    if (
        not title.startswith("ASUS ")
        or not title.endswith(" - Tech Specs")
        or not gpu
        or not title_gpu
        or gpu[0].lower() != title_gpu[0].lower()
    ):
        return None
    model = title.removesuffix(" - Tech Specs")
    labels = root.xpath(
        '//*[contains(concat(" ", normalize-space(@class), " "), " rowTableTitle ") and normalize-space(.)="Dimensions"]'
    )
    if not labels:
        return None
    quotes = []
    values: set[tuple[float, float, float]] = set()
    for label in labels:
        row = label.getparent().getparent()
        cells = row.xpath(
            './/*[contains(concat(" ", normalize-space(@class), " "), " rowTableItemViewBox ")]'
        )
        if len(cells) != 1:
            return None
        quote = " ".join(cells[0].itertext())
        quote = " ".join(quote.split())
        if not quote or len(quote) > 2000:
            return None
        quotes.append(quote)
        for value in DIMENSIONS.findall(quote):
            dimensions = (float(value[0]), float(value[1]), float(value[2]))
            if any(part <= 0 for part in dimensions):
                return None
            values.add(dimensions)
    # Duplicate different values share one model/property/context: explicit conflict.
    status = (
        WebFactStatus.SUPPORTED
        if len(values) == 1
        else WebFactStatus.CONFLICT
        if values
        else WebFactStatus.UNKNOWN
    )
    quote = "\n".join(dict.fromkeys(quotes))
    if len(quote) > 2000:
        return None
    return [
        ExtractedWebFact(
            model=model,
            value=next(iter(values)) if len(values) == 1 else None,
            fact_status=status,
            source_url=final_url,
            document_hash=hashlib.sha256(html.encode()).hexdigest(),
            parser_version=PARSER_VERSION,
            locator="rowTableTitle=Dimensions / rowTableItemViewBox",
            quote=quote,
        )
    ]
