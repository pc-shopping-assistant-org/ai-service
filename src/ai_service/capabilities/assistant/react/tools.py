"""Typed read/build tools; model narration never becomes financial/catalog truth."""

import re
from typing import TypedDict
from uuid import uuid5

from pydantic import ValidationError

from ai_service.application.cpu_work import run_cpu
from ai_service.application.errors import BackendUnavailableError
from ai_service.application.ports.web_search import (
    WebFactStatus,
    WebPageReader,
    WebPageReadStatus,
    WebSearchClient,
    WebSearchQuery,
)
from ai_service.capabilities.assistant.provenance import canonical_hash
from ai_service.capabilities.assistant.requirements.build import (
    grounded_summary,
    optimize_requirements,
    verify_receipt,
)
from ai_service.capabilities.assistant.requirements.compiler import compile_requirements
from ai_service.capabilities.assistant.requirements.search_report import search_report
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)

from .contracts import (
    TOOL_SCHEMAS,
    AnswerCandidate,
    BuildPCArgs,
    BuildResult,
    CatalogResult,
    CompareProductsArgs,
    ComparisonResult,
    CompletionRequirement,
    DetailsResult,
    ProductCatalog,
    ProductDetailsArgs,
    ProductEvidence,
    ReadWebPageArgs,
    ReadWebPageResult,
    SearchCatalogArgs,
    SearchWebArgs,
    SearchWebItem,
    SearchWebResult,
    ToolCall,
    ToolContext,
    ToolError,
    ToolName,
    ToolRecord,
    ToolResult,
    ValidatedAnswer,
    WebFactEvidence,
)


class _ResultIdentity(TypedDict):
    tool_call_id: str
    tool: str
    input_hash: str
    constraints_revision: int


def allowed_tools(context: ToolContext) -> tuple[ToolName, ...]:
    if not context.execution_allowed or not context.read_catalog_allowed:
        return ()
    names: tuple[ToolName, ...] = (
        ToolName.SEARCH_CATALOG,
        ToolName.GET_PRODUCT_DETAILS,
        ToolName.COMPARE_PRODUCTS,
        ToolName.SEARCH_WEB,
        ToolName.READ_WEB_PAGE,
    )
    if (
        context.catalog_ready
        and compile_requirements(context.requirements, context.catalog).ready
    ):
        names += (ToolName.BUILD_PC,)
    return names


def call_hash(call: ToolCall, context: ToolContext) -> str:
    return canonical_hash(
        {
            "tool_version": "v1",
            "call": call.model_dump(mode="json"),
            "requirements": context.requirements.model_dump(mode="json"),
            "catalog": [row.model_dump(mode="json") for row in context.catalog],
        }
    )


def project_current_tools(
    records: list[ToolRecord], context: ToolContext
) -> list[ToolRecord]:
    """Select complete current evidence without rewriting the canonical ledger."""
    if any(record.result is None for record in records):
        raise ValueError("REACT_UNRESOLVED_TOOL_PROTOCOL")
    return [
        record
        for record in records
        if record.origin_run_id == context.run_id
        and record.result
        and record.result.input_hash == call_hash(record.call, context)
    ]


class ToolDispatcher:
    def __init__(
        self,
        catalog: ProductCatalog,
        service: PCBuildApplicationService,
        search_client: WebSearchClient | None = None,
        reader: WebPageReader | None = None,
    ):
        self.catalog = catalog
        self.service = service
        self._search_client = search_client
        self._reader = reader

    @property
    def search_client(self) -> WebSearchClient:
        if self._search_client is None:
            raise ValueError("Web search adapter is not configured")
        return self._search_client

    @property
    def reader(self) -> WebPageReader:
        if self._reader is None:
            raise ValueError("Web reader adapter is not configured")
        return self._reader

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        call = ToolCall.model_validate(call.model_dump())
        context = ToolContext.model_validate(context.model_dump())
        base: _ResultIdentity = {
            "tool_call_id": call.call_id,
            "tool": call.name,
            "input_hash": call_hash(call, context),
            "constraints_revision": context.requirements.revision,
        }

        def rejected(error: ToolError) -> ToolResult:
            return ToolResult(**base, status="REJECTED", error=error)

        try:
            name = ToolName(call.name)
        except ValueError:
            return rejected(ToolError.UNKNOWN_TOOL)
        if name not in allowed_tools(context):
            return rejected(ToolError.PREREQUISITE_MISSING)
        try:
            args = TOOL_SCHEMAS[name].model_validate(call.arguments)
        except ValidationError:
            return rejected(ToolError.BAD_ARGUMENTS)
        # V1 cannot silently change a versioned recommendation policy using a
        # model hint. Explicit unsupported hints are rejected, never ignored.
        if isinstance(args, BuildPCArgs) and args.objective is not None:
            return rejected(ToolError.BAD_ARGUMENTS)
        try:
            payload: (
                CatalogResult
                | DetailsResult
                | ComparisonResult
                | BuildResult
                | SearchWebResult
                | ReadWebPageResult
            )
            if isinstance(args, SearchCatalogArgs):
                products = await self.catalog.search(args.query, args.limit)
                products = [
                    ProductEvidence.model_validate(p.model_dump()) for p in products
                ]
                if len(products) > args.limit or len(
                    {p.product_id for p in products}
                ) != len(products):
                    raise ValueError("Invalid canonical search projection")
                payload = CatalogResult(products=products)
            elif isinstance(args, ProductDetailsArgs):
                product = await self.catalog.get(args.product_id)
                if product is None:
                    return rejected(ToolError.PRODUCT_NOT_FOUND)
                product = ProductEvidence.model_validate(product.model_dump())
                if product.product_id != args.product_id:
                    raise ValueError("Canonical product identity mismatch")
                payload = DetailsResult(product=product)
                products = [product]
            elif isinstance(args, CompareProductsArgs):
                products = []
                for identifier in args.product_ids:
                    found = await self.catalog.get(identifier)
                    if found is None:
                        return rejected(ToolError.PRODUCT_NOT_FOUND)
                    found = ProductEvidence.model_validate(found.model_dump())
                    if found.product_id != identifier:
                        raise ValueError("Canonical product identity mismatch")
                    products.append(found)
                payload = ComparisonResult(products=products)
            elif isinstance(args, SearchWebArgs):
                raw = await self.search_client.search(
                    WebSearchQuery(query=args.query, limit=args.limit)
                )
                if raw.note == "WEB_SEARCH_UNAVAILABLE":
                    return ToolResult(
                        **base, status="FAILED", error=ToolError.WEB_SEARCH_UNAVAILABLE
                    )
                items = [
                    SearchWebItem(
                        title=it.title[:500],
                        url=it.url[:2000],
                        snippet=it.snippet[:2000],
                    )
                    for it in raw.items
                ]
                payload = SearchWebResult(query=args.query, results=items)
                return ToolResult(
                    **base,
                    status="COMPLETED",
                    result=payload,
                    evidence_refs=[],
                )
            elif isinstance(args, ReadWebPageArgs):
                page = await self.reader.read(args.url)
                content = page.content
                next_offset = None
                total_chars = len(content) if content is not None else None
                evidence_ref = None
                if page.status == WebPageReadStatus.READABLE and content:
                    if args.offset >= len(content):
                        return rejected(ToolError.BAD_ARGUMENTS)
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
                payload = ReadWebPageResult(
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
                        WebFactEvidence(
                            evidence_ref="fact:" + canonical_hash(fact), fact=fact
                        )
                        for fact in page.facts
                    ],
                )
                ev_refs = [payload.evidence_ref] if payload.evidence_ref else []
                ev_refs.extend(fact.evidence_ref for fact in payload.facts)
                return ToolResult(
                    **base,
                    status="COMPLETED",
                    result=payload,
                    evidence_refs=ev_refs,
                )
            else:
                receipt = await run_cpu(
                    lambda: optimize_requirements(
                        context.requirements, context.catalog, self.service
                    )
                )
                report = verify_receipt(context.requirements, receipt, context.catalog)
                ref = uuid5(context.run_id, canonical_hash(receipt))
                payload = BuildResult(
                    build_ref=ref,
                    receipt=receipt,
                    satisfaction=report,
                    search_evidence=search_report(
                        context.requirements,
                        context.catalog,
                        self.service,
                        receipt,
                        build_ref=ref,
                    ),
                )
                return ToolResult(
                    **base,
                    status="COMPLETED",
                    result=payload,
                    evidence_refs=[f"build:{ref}"],
                )
        except BackendUnavailableError:
            return ToolResult(
                **base, status="FAILED", error=ToolError.CATALOG_UNAVAILABLE
            )
        return ToolResult(
            **base,
            status="COMPLETED",
            result=payload,
            evidence_refs=[p.evidence_ref for p in products],
        )


def validate_answer(
    candidate: AnswerCandidate,
    ledger: list[ToolResult],
    context: ToolContext,
    required: list[CompletionRequirement],
) -> ValidatedAnswer:
    """Complete every trusted task. Render facts from DTOs, omit unverified prose.

    A caller must supply application-validated completion requirements, not a
    replacement task selected by the model. Clarification does not complete BUILD.
    """
    candidate = AnswerCandidate.model_validate(candidate.model_dump())
    context = ToolContext.model_validate(context.model_dump())
    ledger = [ToolResult.model_validate(item.model_dump()) for item in ledger]
    if not required or candidate.clarification_fields:
        raise ValueError("Unvalidated completion/clarification requirements")
    if len({item.tool_call_id for item in ledger}) != len(ledger):
        raise ValueError("Duplicate tool ledger identity")
    if any(
        item.constraints_revision != context.requirements.revision for item in ledger
    ):
        raise ValueError("Tool result constraints revision mismatch")
    results = [item for item in ledger if item.status == "COMPLETED"]
    known = {ref for item in results for ref in item.evidence_refs}
    if context.saved_build:
        known.add(context.saved_build.evidence_ref)
    selected = set(candidate.evidence_refs)
    saved = (
        context.saved_build
        if context.saved_build and context.saved_build.evidence_ref in selected
        else None
    )
    if not selected <= known or len(selected) != len(candidate.evidence_refs):
        raise ValueError("Unknown or duplicate evidence reference")
    selected_results = [
        item.result for item in results if selected.intersection(item.evidence_refs)
    ]
    builds = [p for p in selected_results if isinstance(p, BuildResult)]
    build = next((p for p in builds if p.build_ref == candidate.build_ref), None)
    if candidate.build_ref is not None and build is None:
        raise ValueError("Unknown build evidence reference")
    for task in required:
        task = CompletionRequirement.model_validate(task.model_dump())
        if task.kind == "BUILD":
            if (
                build is None
                or task.constraints_revision != context.requirements.revision
                or build.receipt.outcome not in {"BUILT", "INFEASIBLE"}
            ):
                raise ValueError(
                    "Build completion requires a verified optimizer outcome"
                )
            if build.receipt.compilation.state_hash != canonical_hash(
                context.requirements
            ):
                raise ValueError("Build completion refers to different requirements")
        elif task.kind == "COMPARE":
            if not any(
                isinstance(p, ComparisonResult)
                and set(task.product_ids)
                == {product.product_id for product in p.products}
                and all(product.evidence_ref in selected for product in p.products)
                for p in selected_results
            ):
                raise ValueError(
                    "Comparison completion requires all requested canonical products"
                )
        elif task.kind == "EXPLAIN_SAVED_BUILD":
            if saved is None or task.saved_build_ref != saved.evidence_ref:
                raise ValueError(
                    "Saved build completion requires the exact authorized snapshot"
                )
        elif task.product_ids:
            products = [
                product
                for payload in selected_results
                for product in (
                    [payload.product]
                    if isinstance(payload, DetailsResult)
                    else payload.products
                    if isinstance(payload, (CatalogResult, ComparisonResult))
                    else []
                )
                if product.evidence_ref in selected
            ]
            if not set(task.product_ids) <= {
                product.product_id for product in products
            }:
                raise ValueError("Answer must cover the selected canonical products")
        elif not selected_results and saved is None:
            raise ValueError("Answer completion requires typed evidence")
    lines: list[str] = []
    if saved:
        snapshot = saved.snapshot
        lines.append(
            "Cấu hình đã lưu"
            + (" — STALE" if snapshot.stale else "")
            + ": chi phí lịch sử, không phải báo giá hiện tại."
        )
        lines.extend(
            f"{part.slot}: {part.name}; {part.spending_vnd:,} VND"
            + ("; đã có" if part.is_owned else "; pinned" if part.is_pinned else "")
            for part in snapshot.parts
        )
        lines.append(f"Tổng chi phí đã lưu: {snapshot.total_spending_vnd:,} VND.")
    rendered: set[str] = set()
    web_pages = [
        p
        for p in selected_results
        if isinstance(p, ReadWebPageResult) and p.status == WebPageReadStatus.READABLE
    ]
    pages_by_ref = {page.evidence_ref: page for page in web_pages}
    for excerpt in candidate.web_excerpts:
        page = pages_by_ref.get(excerpt.evidence_ref)
        if (
            page is None
            or not excerpt.quote.strip()
            or excerpt.quote not in (page.content or "")
        ):
            raise ValueError("Web excerpt must match selected source content")

    if candidate.review is not None:
        if not web_pages:
            raise ValueError("Product review requires read web page evidence")
        if not candidate.web_excerpts:
            raise ValueError("Product review requires source excerpts")
        review_fields = [
            candidate.review.product_name,
            candidate.review.summary,
            *candidate.review.pros,
            *candidate.review.cons,
            candidate.review.verdict,
        ]
        if any(
            re.search(r"https?://|www\.", text, re.IGNORECASE) for text in review_fields
        ):
            raise ValueError("Server owns web citation links")

    for page in web_pages:
        for evidence in page.facts:
            if (
                evidence.evidence_ref not in selected
                or evidence.evidence_ref in rendered
            ):
                continue
            fact = evidence.fact
            if fact.fact_status == WebFactStatus.SUPPORTED:
                assert fact.value is not None
                dimensions = " × ".join(f"{value:g}" for value in fact.value)
                lines.append(
                    f"{fact.model} — kích thước công bố: {dimensions} {fact.unit} (theo nguồn, chưa kết luận fit/compatibility)."
                )
            elif fact.fact_status == WebFactStatus.UNKNOWN:
                lines.append(
                    f"{fact.model} — UNKNOWN: nguồn chưa cung cấp kích thước mm theo template hỗ trợ."
                )
            else:
                lines.append(
                    f"{fact.model} — CONFLICT: các giá trị kích thước trong nguồn mâu thuẫn; chưa kết luận."
                )
            rendered.add(evidence.evidence_ref)

    for payload in selected_results:
        if isinstance(payload, BuildResult):
            report = verify_receipt(
                context.requirements, payload.receipt, context.catalog
            )
            if report != payload.satisfaction:
                raise ValueError("Saved satisfaction evidence mismatch")
            key = str(payload.build_ref)
            if key not in rendered:
                lines.append(grounded_summary(payload.receipt, report))
                rendered.add(key)
        elif isinstance(payload, (DetailsResult, CatalogResult, ComparisonResult)):
            products = (
                [payload.product]
                if isinstance(payload, DetailsResult)
                else payload.products
            )
            for product in products:
                if (
                    product.evidence_ref not in selected
                    or product.evidence_ref in rendered
                ):
                    continue
                price = (
                    f"{product.price_vnd:,} VND"
                    if product.price_vnd is not None
                    else "chưa có giá xác thực"
                )
                lines.append(
                    f"{product.name}: {price}. Thông số: {product.specifications or 'chưa có dữ liệu xác thực'}."
                )
                rendered.add(product.evidence_ref)
        elif isinstance(payload, (ReadWebPageResult, SearchWebResult)):
            continue

    if web_pages:
        citations: list[str] = []
        for page in web_pages:
            citation_key = "source:" + page.final_url
            if (
                page.evidence_ref in selected
                or any(fact.evidence_ref in selected for fact in page.facts)
            ) and citation_key not in rendered:
                title_display = page.title or page.final_url
                citations.append(f"- [{title_display}]({page.final_url})")
                rendered.add(citation_key)
        # Catalog/build output stays canonical even when web sources are selected.
        candidate_body = ""
        if re.search(r"https?://|www\.", candidate.text, re.IGNORECASE):
            raise ValueError("Server owns web citation links")
        if not lines:
            if not candidate.text.strip():
                raise ValueError("Web answer candidate must provide synthesized text")
            if not candidate.web_excerpts:
                raise ValueError("Web answer requires source excerpts")
        if candidate.review is not None:
            review_parts = [
                f"### Đánh giá: {candidate.review.product_name}",
                candidate.review.summary,
            ]
            if candidate.review.pros:
                review_parts.append(
                    "**Ưu điểm:**\n"
                    + "\n".join(f"- {pro}" for pro in candidate.review.pros)
                )
            if candidate.review.cons:
                review_parts.append(
                    "**Nhược điểm:**\n"
                    + "\n".join(f"- {con}" for con in candidate.review.cons)
                )
            review_parts.append(f"**Tổng kết:**\n{candidate.review.verdict}")
            candidate_body = "\n\n".join(review_parts)
            candidate_body += (
                "\n\nNội dung nguồn tham khảo (chưa xác thực thông số/điều kiện đo):\n"
            )
            candidate_body += "\n\n".join(
                "\n".join("> " + line for line in excerpt.quote.splitlines())
                for excerpt in candidate.web_excerpts
            )
        elif candidate.web_excerpts:
            # Render exact quotes for both web-only and mixed answers, never
            # unrestricted model narration. Attribution is not verification.
            candidate_body = (
                "Nội dung nguồn tham khảo (chưa xác thực thông số/điều kiện đo):\n"
            )
            candidate_body += "\n\n".join(
                "\n".join("> " + line for line in excerpt.quote.splitlines())
                for excerpt in candidate.web_excerpts
            )
        if citations:
            citation_block = "Nguồn tham khảo:\n" + "\n".join(citations)
            candidate_body = f"{candidate_body}\n\n{citation_block}".strip()
        lines.append(candidate_body)

    if not lines:
        raise ValueError("Answer completion has no renderable evidence")
    if candidate.explanation is not None:
        refs = candidate.explanation.evidence_refs
        if len(set(refs)) != len(refs) or not set(refs) <= selected:
            raise ValueError("Explanation must cite selected canonical evidence")
        # Facts/financial output above remain canonical. This is explicitly an
        # LLM interpretation; reference validation is not a factuality guarantee.
        lines.append(
            "Nhận định tư vấn (diễn giải, không phải số liệu benchmark):\n"
            + candidate.explanation.text
        )
    return ValidatedAnswer(
        text="\n".join(lines),
        status="COMPLETED",
        evidence_refs=candidate.evidence_refs,
        build_ref=candidate.build_ref,
    )
