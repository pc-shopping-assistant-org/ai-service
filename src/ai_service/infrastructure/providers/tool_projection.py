"""Bound model-facing observations without modifying canonical tool results."""

import json
from typing import Any

from ai_service.capabilities.assistant.react.contracts import (
    BuildResult,
    CatalogResult,
    ComparisonResult,
    DetailsResult,
    ReadWebPageResult,
    ResearchResult,
    ResearchSection,
    ToolResult,
)


def _clip(text: str, limit: int) -> str:
    return text.encode()[: max(limit, 0)].decode("utf-8", errors="ignore")


def _base_payload(
    result: ToolResult, *, include_evidence_refs: bool = True
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "tool_call_id": result.tool_call_id,
        "tool": result.tool,
        "status": result.status,
    }
    if result.error is not None:
        payload["error"] = result.error
    if include_evidence_refs and result.evidence_refs:
        payload["evidence_refs"] = result.evidence_refs
    return payload


def _source(page: ReadWebPageResult) -> dict[str, Any]:
    source: dict[str, Any] = {
        "url": page.final_url or page.url,
        "status": page.status.value,
    }
    if page.title:
        source["title"] = page.title
    if page.evidence_ref:
        source["evidence_ref"] = page.evidence_ref
    if page.content:
        source["content"] = page.content
    if page.facts:
        source["facts"] = [item.model_dump(mode="json") for item in page.facts]
    return source


def _section(section: ResearchSection) -> dict[str, Any]:
    return {
        "subjects": section.subjects,
        "focus": section.focus,
        "status": section.status.value,
        "official_source_status": section.official_source_status.value,
        # Query history/domains/hashes are replay data; the model only needs
        # to know which source tiers produced nothing to avoid overclaiming.
        "tiers": {step.tier.value: step.status.value for step in section.search_steps},
        "sources": [_source(page) for page in section.pages],
    }


def _research_payload(result: ToolResult, research: ResearchResult, limit: int) -> str:
    payload = _base_payload(result, include_evidence_refs=False)
    projected: dict[str, Any] = {
        "sections": [_section(section) for section in research.sections],
    }
    if research.missing_conditions:
        projected["missing_conditions"] = [c.value for c in research.missing_conditions]
    if research.build_source_ref is not None:
        projected["build_source_ref"] = research.build_source_ref
    payload["result"] = projected

    def encoded() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    if len(encoded().encode()) <= limit:
        return encoded()
    # Share the remaining budget across page contents. Evidence refs and facts
    # are never dropped: a model cannot cite a source it no longer sees.
    sources = [s for sec in projected["sections"] for s in sec["sources"]]
    contents = [s for s in sources if "content" in s]
    payload["projection"] = {"truncated": True}
    for source in contents:
        source["content"] = ""
    overhead = len(encoded().encode())
    share = (limit - overhead) // max(len(contents), 1)
    for page, source in zip(
        [p for sec in research.sections for p in sec.pages if p.content], contents
    ):
        source["content"] = _clip(page.content or "", share)
    output = encoded()
    if len(output.encode()) > limit:
        raise ValueError("REACT_TOOL_CONTEXT_TOO_LARGE")
    return output


_INTERNAL_PART_KEYS = {
    "category",
    "performance_score",
    "integrated_graphics_score",
    "stock_cooler_score",
}


def _clean_part(part: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in part.items()
        if key not in _INTERNAL_PART_KEYS
        and value is not None
        and value is not False
        and value != []
        and value != {}
    }


def _clean_build(raw: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {
        "total_price": raw["total_price"],
    }
    if raw.get("objective") is not None:
        cleaned["objective"] = raw["objective"]
    if raw.get("ranking_status") is not None:
        cleaned["ranking_status"] = raw["ranking_status"]
    cleaned["compatibility_status"] = raw["compatibility_status"]
    if raw.get("owned_categories"):
        cleaned["owned_categories"] = raw["owned_categories"]
    if power := raw.get("power_estimate"):
        cleaned["power_estimate"] = {
            "estimated_sustained_peak_watts": power["estimated_sustained_peak_watts"],
            "recommended_capacity_watts": power["recommended_capacity_watts"],
        }
    cleaned["parts"] = {
        slot: _clean_part(part) for slot, part in raw.get("parts", {}).items()
    }
    return cleaned


def project_build_result(build: BuildResult) -> dict[str, Any]:
    # Optimizer inputs/pool identities are replay data, not model context.
    receipt, search = build.receipt, build.search_evidence
    projected: dict[str, Any] = {
        "build_ref": str(build.build_ref),
        "evidence_ref": f"build:{build.build_ref}",
        "outcome": search.outcome
        if search and receipt.outcome != "BUILT"
        else receipt.outcome,
        "build": _clean_build(
            receipt.build.model_dump(mode="json", exclude_none=True)
        )
        if receipt.build
        else None,
    }
    if search is not None and receipt.outcome != "BUILT":
        projected["search_evidence"] = {
            "outcome": search.outcome,
            "termination": search.termination,
        }
    return projected


def observation_json(result: ToolResult, byte_limit: int) -> str:
    if isinstance(result.result, ResearchResult):
        return _research_payload(result, result.result, byte_limit)
    if isinstance(result.result, BuildResult):
        build = result.result
        payload = _base_payload(
            result, include_evidence_refs=build.receipt.outcome != "BUILT"
        )
        payload["result"] = project_build_result(build)
    elif isinstance(result.result, (CatalogResult, DetailsResult, ComparisonResult)):
        payload = _base_payload(result, include_evidence_refs=False)
        if isinstance(result.result, DetailsResult):
            payload["result"] = {
                "tool": result.result.tool,
                "product": result.result.product.model_dump(
                    mode="json", exclude={"retrieved_at"}
                ),
            }
        else:
            payload["result"] = {
                "tool": result.result.tool,
                "products": [
                    item.model_dump(mode="json", exclude={"retrieved_at"})
                    for item in result.result.products
                ],
            }
    else:
        payload = _base_payload(result)
        if result.result is not None:
            payload["result"] = result.model_dump(mode="json")["result"]

    def encoded() -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    if len(encoded().encode()) <= byte_limit:
        return encoded()
    # Keep whole structured entries/values. Never slice a JSON document or number.
    payload["projection"] = {"truncated": True}

    def candidates() -> list[tuple[int, list | dict, str | int, str]]:
        options: list[tuple[int, list | dict, str | int, str]] = []

        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if (
                        isinstance(item, str)
                        and key in {"content", "description", "snippet", "text"}
                        and item
                    ):
                        options.append(
                            (len(item.encode()), value, key, path + "." + key)
                        )
                    walk(item, path + "." + key)
            elif isinstance(value, list):
                if len(value) > 1:
                    options.append(
                        (
                            len(json.dumps(value[-1], ensure_ascii=False).encode()),
                            value,
                            len(value) - 1,
                            path,
                        )
                    )
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")

        walk(payload.get("result"), "result")
        return options

    while len(encoded().encode()) > byte_limit:
        options = candidates()
        if not options:
            # A single indivisible entry can exceed the budget (e.g. build trace).
            # Preserve outcome identity and refs; full data stays in the ledger.
            payload["result"] = None
            break
        _, container, key, _path = max(options, key=lambda row: row[0])
        if isinstance(container, list):
            container.pop(int(key))
        else:
            original = container[str(key)].encode()
            container[str(key)] = original[
                : min(len(original) // 2, byte_limit // 2)
            ].decode("utf-8", errors="ignore")
    output = encoded()
    if len(output.encode()) > byte_limit:
        raise ValueError("REACT_TOOL_CONTEXT_TOO_LARGE")
    return output
