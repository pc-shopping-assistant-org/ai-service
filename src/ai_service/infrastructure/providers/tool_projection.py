"""Bound model-facing observations without modifying canonical tool results."""

import json
from typing import Any

from ai_service.capabilities.assistant.react.contracts import (
    BuildResult,
    ReadWebPageResult,
    ResearchResult,
    ResearchSection,
    ToolResult,
)


def _clip(text: str, limit: int) -> str:
    return text.encode()[: max(limit, 0)].decode("utf-8", errors="ignore")


def _source(page: ReadWebPageResult) -> dict[str, Any]:
    source: dict[str, Any] = {"url": page.url, "status": page.status.value}
    if page.final_url != page.url:
        source["final_url"] = page.final_url
    if page.title:
        source["title"] = page.title
    if page.evidence_ref:
        source["evidence_ref"] = page.evidence_ref
    if page.content:
        source["content"] = page.content
    if page.next_offset is not None:
        source["next_offset"] = page.next_offset
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
    payload = result.model_dump(mode="json", exclude={"result"})
    projected: dict[str, Any] = {
        "tool": research.tool.value,
        "sections": [_section(section) for section in research.sections],
        "missing_conditions": [c.value for c in research.missing_conditions],
    }
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


def observation_json(result: ToolResult, byte_limit: int) -> str:
    if isinstance(result.result, ResearchResult):
        return _research_payload(result, result.result, byte_limit)
    if isinstance(result.result, BuildResult):
        # Optimizer inputs/pool identities are replay data, not model context.
        build = result.result
        receipt, search = build.receipt, build.search_evidence
        trace = receipt.search_trace
        payload = result.model_dump(mode="json", exclude={"result"})
        payload["result"] = {
            "tool": build.tool,
            "build_ref": str(build.build_ref),
            "outcome": search.outcome
            if search and receipt.outcome != "BUILT"
            else receipt.outcome,
            "build": receipt.build.model_dump(mode="json", exclude_none=True)
            if receipt.build
            else None,
            "requirements": [
                entry.model_dump(mode="json") for entry in receipt.compilation.mapping
            ],
            "search_evidence": search.model_dump(mode="json") if search else None,
            "metadata_rejections": trace.metadata_rejections if trace else 0,
            "scoring_rejections": trace.scoring_rejections if trace else 0,
        }
        payload["projection"] = {"truncated": True}
    else:
        payload = result.model_dump(mode="json")

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
