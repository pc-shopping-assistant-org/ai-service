"""Bounded search/read workflows. Readability is not benchmark verification."""

from dataclasses import dataclass

from ai_service.application.ports.hardware import ComponentCategory
from ai_service.application.ports.web_search import (
    WebPageReader,
    WebPageReadStatus,
    WebSearchCategory,
    WebSearchClient,
    WebSearchQuery,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.build import verify_receipt
from ai_service.capabilities.web_search.presets import (
    HARDWARE_SEARCH_FOCUS,
    WebSearchPresets,
)
from ai_service.capabilities.web_search.research_contracts import (
    MAX_RESEARCH_SEARCH_RESULTS,
    RESEARCH_PAGE_CHARS,
    BenchmarkCondition,
    HardwareResearchFocus,
    ResearchCompatibilityArgs,
    ResearchDriverArgs,
    ResearchHardwareArgs,
    ResearchMode,
    ResearchPCBuildArgs,
    ResearchSourceStatus,
    ResearchWorkloadArgs,
    WorkloadBenchmarkKind,
)
from ai_service.capabilities.web_search.source_policy import (
    OfficialSourceStatus,
    ResearchSourcePolicy,
    SourceTier,
    normalized,
    within_domains,
)

from .contracts import (
    BuildResult,
    ReadWebPageArgs,
    ReadWebPageResult,
    ResearchResult,
    ResearchSearchStep,
    ResearchSection,
    SearchWebItem,
    ToolContext,
    ToolError,
    ToolName,
    WebFactEvidence,
)

COMPONENT_RESEARCH_FOCUS: dict[ComponentCategory, HardwareResearchFocus] = {
    ComponentCategory.CPU: HardwareResearchFocus.PERFORMANCE,
    ComponentCategory.GPU: HardwareResearchFocus.PERFORMANCE,
    ComponentCategory.MAINBOARD: HardwareResearchFocus.MOTHERBOARD,
    ComponentCategory.RAM: HardwareResearchFocus.MEMORY,
    ComponentCategory.STORAGE: HardwareResearchFocus.STORAGE,
    ComponentCategory.PSU: HardwareResearchFocus.PSU_QUALITY,
    ComponentCategory.COOLER: HardwareResearchFocus.THERMAL_NOISE,
    ComponentCategory.CASE: HardwareResearchFocus.CASE_AIRFLOW,
}

WORKLOAD_RESEARCH_FOCUS = {
    WorkloadBenchmarkKind.GAMING: HardwareResearchFocus.PERFORMANCE,
    WorkloadBenchmarkKind.PRODUCTIVITY: HardwareResearchFocus.PRODUCTIVITY,
}

type ResearchArgs = (
    ResearchHardwareArgs
    | ResearchCompatibilityArgs
    | ResearchWorkloadArgs
    | ResearchPCBuildArgs
    | ResearchDriverArgs
)


class PageOffsetError(ValueError):
    pass


class ResearchArgumentsError(ValueError):
    pass


class ResearchPrerequisiteError(ValueError):
    pass


def research_build_subjects(
    args: ResearchPCBuildArgs, context: ToolContext, builds: list[BuildResult]
) -> tuple[dict[ComponentCategory, str], str, dict[ComponentCategory, str | None]]:
    if args.build_ref is None:
        if context.saved_build is None:
            raise ResearchPrerequisiteError("Saved build is required")
        return (
            {
                part.slot: part.name
                for part in context.saved_build.snapshot.parts
                if isinstance(part.slot, ComponentCategory)
            },
            context.saved_build.evidence_ref,
            {},  # Saved names are immutable; never infer brand from today's catalog.
        )
    selected = next(
        (build for build in builds if build.build_ref == args.build_ref), None
    )
    if selected is None or selected.receipt.build is None:
        raise ResearchPrerequisiteError("Current build is required")
    report = verify_receipt(context.requirements, selected.receipt, context.catalog)
    if report is None or report != selected.satisfaction or not report.hard_satisfied:
        raise ResearchPrerequisiteError("Verified build is required")
    return (
        {slot: part.name for slot, part in selected.receipt.build.parts.items()},
        f"build:{selected.build_ref}",
        {slot: part.brand for slot, part in selected.receipt.build.parts.items()},
    )


@dataclass(frozen=True)
class ResearchRequest:
    subjects: list[str]
    focus: str
    query: WebSearchQuery
    max_sources: int
    brand: str | None = None
    official_subjects: list[str] | None = None
    review_focus: HardwareResearchFocus | None = None


async def read_page_evidence(
    reader: WebPageReader, args: ReadWebPageArgs
) -> ReadWebPageResult:
    page = await reader.read(args.url)
    content = page.content
    next_offset = None
    total_chars = len(content) if content is not None else None
    evidence_ref = None
    if page.status == WebPageReadStatus.READABLE and content:
        if args.offset >= len(content):
            raise PageOffsetError("WEB_PAGE_OFFSET_OUT_OF_RANGE")
        end = min(args.offset + args.max_chars, len(content))
        content = content[args.offset : end]
        next_offset = end if end < len(page.content or "") else None
        if page.evidence_ref:
            evidence_ref = "web:" + canonical_hash(
                {
                    "source_ref": page.evidence_ref,
                    "offset": args.offset,
                    "content": content,
                }
            )
    return ReadWebPageResult(
        url=args.url,
        final_url=page.final_url,
        title=page.title[:500] if page.title else None,
        status=page.status,
        content_hash=page.content_hash,
        content=content,
        evidence_ref=evidence_ref,
        offset=args.offset,
        next_offset=next_offset,
        total_chars=total_chars,
        extraction_status=page.extraction_status,
        document_hash=page.document_hash,
        facts=[
            WebFactEvidence(evidence_ref="fact:" + canonical_hash(fact), fact=fact)
            for fact in page.facts
        ],
    )


def hardware_request(
    args: ResearchHardwareArgs, brand: str | None = None
) -> ResearchRequest:
    category = HARDWARE_SEARCH_FOCUS[args.focus][0]
    query = WebSearchPresets.research_hardware(args.model, args.focus, args.question)
    return ResearchRequest(
        [args.model],
        args.focus.value,
        WebSearchQuery(
            query=query, category=category, limit=MAX_RESEARCH_SEARCH_RESULTS
        ),
        args.max_sources,
        brand=brand,
    )


def research_requests(
    args: ResearchArgs,
    parts: dict[ComponentCategory, str],
    brands: dict[ComponentCategory, str | None],
) -> list[ResearchRequest]:
    if isinstance(args, ResearchHardwareArgs):
        return [hardware_request(args)]
    if isinstance(args, ResearchDriverArgs):
        query = " ".join(
            value
            for value in (
                WebSearchPresets.driver_and_software(
                    args.component_name, args.software_type, args.version
                ),
                args.operating_system,
            )
            if value
        )
        return [
            ResearchRequest(
                [args.component_name],
                WebSearchCategory.DRIVER_SOFTWARE.value,
                WebSearchQuery(
                    query=query,
                    category=WebSearchCategory.DRIVER_SOFTWARE,
                    limit=MAX_RESEARCH_SEARCH_RESULTS,
                ),
                args.max_sources,
            )
        ]
    if isinstance(args, ResearchCompatibilityArgs):
        query = WebSearchPresets.compatibility(args.part_a, args.part_b, args.concern)
        return [
            ResearchRequest(
                [args.part_a, args.part_b],
                WebSearchCategory.COMPATIBILITY.value,
                WebSearchQuery(
                    query=query,
                    category=WebSearchCategory.COMPATIBILITY,
                    limit=MAX_RESEARCH_SEARCH_RESULTS,
                ),
                args.max_sources,
            )
        ]
    if isinstance(args, ResearchWorkloadArgs):
        review_focus = WORKLOAD_RESEARCH_FOCUS[args.benchmark_kind]
        keywords = (
            WebSearchPresets.game_requirements(args.workload)
            if args.mode == ResearchMode.REQUIREMENTS
            else HARDWARE_SEARCH_FOCUS[review_focus][1]
        )
        query = " ".join(
            value
            for value in (
                None if args.mode == ResearchMode.REQUIREMENTS else args.workload,
                *args.hardware,
                keywords,
                args.resolution,
                args.preset,
                f"upscaling {args.upscaling}" if args.upscaling else None,
            )
            if value
        )
        category = (
            WebSearchCategory.GAME_REQUIREMENTS
            if args.mode == ResearchMode.REQUIREMENTS
            else WebSearchCategory.REVIEWS
        )
        return [
            ResearchRequest(
                [args.workload, *args.hardware],
                args.mode.value,
                WebSearchQuery(
                    query=query, category=category, limit=MAX_RESEARCH_SEARCH_RESULTS
                ),
                args.max_sources,
                official_subjects=[args.workload],
                review_focus=review_focus
                if args.mode == ResearchMode.BENCHMARK
                else None,
            )
        ]
    selected = args.slots or list(parts)
    if not selected or not set(selected) <= set(parts):
        raise ResearchArgumentsError("Research slots must exist in the selected build")
    return [
        hardware_request(
            ResearchHardwareArgs(
                model=parts[slot],
                focus=COMPONENT_RESEARCH_FOCUS[slot],
                question=args.question,
                max_sources=1,
            ),
            brand=brands.get(slot),
        )
        for slot in selected
    ]


def source_plan(
    request: ResearchRequest,
    policy: ResearchSourcePolicy,
) -> tuple[OfficialSourceStatus, list[tuple[SourceTier, WebSearchQuery]]]:
    subjects = request.official_subjects or request.subjects
    is_requirements = request.focus == ResearchMode.REQUIREMENTS.value
    is_driver = request.focus == WebSearchCategory.DRIVER_SOFTWARE.value
    resolved = [
        policy.publisher_domains(subject)
        if is_requirements
        else policy.official_domains(subject, request.brand)
        for subject in subjects
    ]
    official_domains = list(
        dict.fromkeys(domain for group in resolved for domain in group)
    )
    status = (
        OfficialSourceStatus.RESOLVED
        if all(resolved)
        else OfficialSourceStatus.PARTIAL
        if any(resolved)
        else OfficialSourceStatus.UNKNOWN
    )
    stages = []
    if official_domains:
        official_query = (
            " ".join(filter(None, [request.brand, *subjects]))
            + " official product specifications support"
        )
        if request.focus in {
            WebSearchCategory.COMPATIBILITY.value,
            ResearchMode.REQUIREMENTS.value,
            HardwareResearchFocus.KNOWN_ISSUES.value,
            HardwareResearchFocus.OFFICIAL_SPECS.value,
            WebSearchCategory.DRIVER_SOFTWARE.value,
        }:
            official_query = request.query.query
        stages.append(
            (
                SourceTier.OFFICIAL,
                request.query.model_copy(
                    update={
                        "query": official_query,
                        "category": request.query.category
                        if is_requirements or is_driver
                        else WebSearchCategory.VENDOR_SPECS,
                        "target_domains": official_domains,
                    }
                ),
            )
        )
    if is_driver:
        return status, stages
    if is_requirements:
        if policy.requirement_stores:
            stages.append(
                (
                    SourceTier.STORE_LISTING,
                    request.query.model_copy(
                        update={
                            "target_domains": policy.requirement_stores,
                        }
                    ),
                )
            )
        return status, stages
    review_domains = policy.review_domains(request.review_focus or request.focus)
    if review_domains:
        stages.append(
            (
                SourceTier.INDEPENDENT_REVIEW,
                request.query.model_copy(
                    update={
                        "target_domains": review_domains,
                    }
                ),
            )
        )
    if policy.allow_general_fallback:
        stages.append((SourceTier.GENERAL_WEB, request.query))
    return status, stages


async def research_sources(
    name: ToolName,
    args: ResearchArgs,
    search: WebSearchClient,
    reader: WebPageReader,
    *,
    parts: dict[ComponentCategory, str] | None = None,
    build_source_ref: str | None = None,
    brands: dict[ComponentCategory, str | None] | None = None,
    policy: ResearchSourcePolicy | None = None,
) -> ResearchResult:
    policy = policy or ResearchSourcePolicy()
    sections = []
    # Reuse one fetched page when several component queries return the same URL.
    pages: dict[str, ReadWebPageResult] = {}
    for request in research_requests(args, parts or {}, brands or {}):
        official_status, stages = source_plan(request, policy)
        results = []
        seen_urls: set[str] = set()
        selected_pages: list[ReadWebPageResult] = []
        steps = []
        for tier, query in stages:
            if tier == SourceTier.GENERAL_WEB and any(
                page.status == WebPageReadStatus.READABLE for page in selected_pages
            ):
                continue
            found = await search.search(query)
            candidates = [
                item
                for item in found.items
                if found.note != ToolError.WEB_SEARCH_UNAVAILABLE
                and (
                    tier == SourceTier.GENERAL_WEB
                    or within_domains(item.url, query.target_domains)
                )
            ]
            # Prefer model tokens in titles; relevance still needs evidence/condition checks.
            tokens = set(normalized(" ".join(request.subjects)).split())
            candidates.sort(
                key=lambda item: len(
                    tokens.intersection(normalized(item.title).split())
                ),
                reverse=True,
            )
            stage_results = []
            stage_pages = []
            for item in candidates:
                if item.url not in seen_urls:
                    stage_results.append(
                        SearchWebItem(
                            title=item.title[:500],
                            url=item.url[:2000],
                            snippet=item.snippet[:2000],
                        )
                    )
                    seen_urls.add(item.url)
                if len(stage_results) == MAX_RESEARCH_SEARCH_RESULTS:
                    break
            if found.note != ToolError.WEB_SEARCH_UNAVAILABLE:
                for source in stage_results[: request.max_sources]:
                    if source.url not in pages:
                        pages[source.url] = await read_page_evidence(
                            reader,
                            ReadWebPageArgs(
                                url=source.url, max_chars=RESEARCH_PAGE_CHARS
                            ),
                        )
                    page = pages[source.url]
                    if tier != SourceTier.GENERAL_WEB and not within_domains(
                        page.final_url, query.target_domains
                    ):
                        # A redirect outside the tier is not that tier's evidence.
                        continue
                    stage_pages.append(page)
            status = source_status(
                stage_results,
                stage_pages,
                unavailable=found.note == ToolError.WEB_SEARCH_UNAVAILABLE,
            )
            steps.append(
                ResearchSearchStep(
                    tier=tier,
                    query=query.query,
                    target_domains=query.target_domains,
                    status=status,
                    result_urls=[item.url for item in stage_results],
                )
            )
            results.extend(stage_results)
            selected_pages.extend(stage_pages)
            if (
                tier == SourceTier.OFFICIAL
                and status == ResearchSourceStatus.READABLE
                and request.focus
                in {
                    HardwareResearchFocus.OFFICIAL_SPECS.value,
                    ResearchMode.REQUIREMENTS.value,
                }
            ):
                break
        status = source_status(
            results,
            selected_pages,
            unavailable=bool(steps)
            and all(step.status == ResearchSourceStatus.UNAVAILABLE for step in steps),
        )
        if not stages:
            status = ResearchSourceStatus.SOURCE_UNRESOLVED
        sections.append(
            ResearchSection(
                subjects=request.subjects,
                focus=request.focus,
                query=request.query.query,
                status=status,
                official_source_status=official_status,
                search_steps=steps,
                results=results,
                pages=selected_pages,
            )
        )
    missing = []
    if isinstance(args, ResearchWorkloadArgs) and args.mode == ResearchMode.BENCHMARK:
        required_conditions = (
            list(BenchmarkCondition)
            if args.benchmark_kind == WorkloadBenchmarkKind.GAMING
            else [BenchmarkCondition.HARDWARE]
        )
        missing = [
            condition
            for condition in required_conditions
            if not getattr(args, condition.value)
        ]
    return ResearchResult.model_validate(
        {
            "tool": name,
            "sections": sections,
            "build_source_ref": build_source_ref,
            "source_policy_version": policy.version,
            "source_policy_hash": policy.digest,
            "missing_conditions": missing,
        }
    )


def source_status(
    results: list[SearchWebItem],
    pages: list[ReadWebPageResult],
    *,
    unavailable: bool = False,
) -> ResearchSourceStatus:
    if any(page.status == WebPageReadStatus.READABLE for page in pages):
        return ResearchSourceStatus.READABLE
    if results:
        return ResearchSourceStatus.UNREADABLE
    return (
        ResearchSourceStatus.UNAVAILABLE
        if unavailable
        else ResearchSourceStatus.NO_RESULTS
    )
