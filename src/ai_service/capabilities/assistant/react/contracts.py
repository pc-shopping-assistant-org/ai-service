"""Tool schemas contain no owner, revision, candidate price or execution authority."""

from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, StrictInt, model_validator

from ai_service.application.ports.user_memory import MAX_MEMORIES, MemoryEntry
from ai_service.application.ports.web_search import (
    ExtractedWebFact,
    WebExtractionStatus,
    WebPageReadStatus,
)
from ai_service.capabilities.assistant.limits import (
    MAX_BUILD_RESULT_BYTES,
    MAX_DIALOGUE_CHARS,
    MAX_DIALOGUE_TURNS,
    MAX_HARDWARE_CATALOG_ROWS,
    MAX_REACT_STATE_BYTES,
    MAX_SELECTED_PRODUCTS,
    MAX_TURN_MESSAGE_CHARS,
    MIN_COMPARISON_PRODUCTS,
)
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.public_contracts import (
    PublicBuildV1,
    PublicClarificationV1,
    PublicTaskClarificationV1,
)
from ai_service.capabilities.assistant.requirements.build import BuildReceipt
from ai_service.capabilities.assistant.requirements.contracts import (
    CanonicalRequirementState,
    RequirementPatch,
)
from ai_service.capabilities.assistant.requirements.tasks import TaskDraftBatch
from ai_service.capabilities.assistant.requirements.verifier import SatisfactionReport
from ai_service.capabilities.assistant.stateful_contracts import (
    CheckpointRefV1,
    Contract,
    NonNegativeInt,
    Text,
)
from ai_service.capabilities.web_search.research_contracts import (
    MAX_BUILD_RESEARCH_PARTS,
    MAX_RESEARCH_SEARCH_RESULTS,
    MAX_RESEARCH_SOURCE_TIERS,
    MAX_RESEARCH_SOURCES,
    MAX_RESEARCH_SUBJECTS,
    BenchmarkCondition,
    ResearchCompatibilityArgs,
    ResearchDriverArgs,
    ResearchHardwareArgs,
    ResearchPCBuildArgs,
    ResearchSourceStatus,
    ResearchWorkloadArgs,
)
from ai_service.capabilities.web_search.source_policy import (
    OfficialSourceStatus,
    SourceTier,
)

MAX_NATIVE_TOOL_CALLS = 16


class ReactLimits(Contract):
    max_model_calls: StrictInt = Field(default=12, ge=1, le=100)
    max_tool_calls: StrictInt = Field(default=8, ge=1, le=50)
    max_tool_calls_per_batch: StrictInt = Field(
        default=4, ge=1, le=MAX_NATIVE_TOOL_CALLS
    )
    max_model_input_bytes: StrictInt = Field(default=128_000, ge=1024)
    max_tool_context_bytes: StrictInt = Field(default=12_000, ge=512)
    max_repairs: StrictInt = Field(default=2, ge=0, le=10)
    max_state_bytes: StrictInt = Field(default=MAX_REACT_STATE_BYTES, ge=1)
    max_tool_bytes: StrictInt = Field(default=128_000, ge=1)
    max_build_result_bytes: StrictInt = Field(default=MAX_BUILD_RESULT_BYTES, ge=1)


from .search_evidence import BuildSearchEvidence


class ToolName(StrEnum):
    SEARCH_CATALOG = "search_catalog"
    GET_PRODUCT_DETAILS = "get_product_details"
    COMPARE_PRODUCTS = "compare_products"
    BUILD_PC = "build_pc"
    SEARCH_WEB = "search_web"
    READ_WEB_PAGE = "read_web_page"
    RESEARCH_HARDWARE = "research_hardware"
    RESEARCH_COMPATIBILITY = "research_compatibility"
    RESEARCH_WORKLOAD = "research_workload"
    RESEARCH_PC_BUILD = "research_pc_build"
    RESEARCH_DRIVER = "research_driver"


class SearchCatalogArgs(Contract):
    query: Text = Field(max_length=1000)
    limit: StrictInt = Field(default=10, ge=1, le=20)


class ProductDetailsArgs(Contract):
    product_id: UUID


class CompareProductsArgs(Contract):
    product_ids: list[UUID] = Field(
        min_length=MIN_COMPARISON_PRODUCTS, max_length=MAX_SELECTED_PRODUCTS
    )
    question: Text | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def distinct_products(self) -> "CompareProductsArgs":
        if len(set(self.product_ids)) != len(self.product_ids):
            raise ValueError("Product IDs must be distinct")
        return self


class BuildPCArgs(Contract):
    """Build from admitted requirements using the full canonical hardware catalog.

    Loads candidates automatically; do not search individual parts first.
    Application policy selects compatible parts and the recommendation objective.
    """


class SearchWebArgs(Contract):
    query: Text = Field(max_length=1000)
    limit: StrictInt = Field(default=5, ge=1, le=10)


class ReadWebPageArgs(Contract):
    url: Text = Field(max_length=2000)
    offset: NonNegativeInt = 0
    max_chars: StrictInt = Field(default=12_000, ge=1, le=16_000)


TOOL_SCHEMAS: dict[ToolName, type[Contract]] = {
    ToolName.SEARCH_CATALOG: SearchCatalogArgs,
    ToolName.GET_PRODUCT_DETAILS: ProductDetailsArgs,
    ToolName.COMPARE_PRODUCTS: CompareProductsArgs,
    ToolName.BUILD_PC: BuildPCArgs,
    ToolName.SEARCH_WEB: SearchWebArgs,
    ToolName.READ_WEB_PAGE: ReadWebPageArgs,
    ToolName.RESEARCH_HARDWARE: ResearchHardwareArgs,
    ToolName.RESEARCH_COMPATIBILITY: ResearchCompatibilityArgs,
    ToolName.RESEARCH_WORKLOAD: ResearchWorkloadArgs,
    ToolName.RESEARCH_PC_BUILD: ResearchPCBuildArgs,
    ToolName.RESEARCH_DRIVER: ResearchDriverArgs,
}


class ProductEvidence(Contract):
    product_id: UUID
    name: Text = Field(max_length=500)
    status: Literal["ACTIVE"] = "ACTIVE"
    price_vnd: NonNegativeInt | None = None
    specifications: dict[str, JsonValue] = Field(default_factory=dict, max_length=100)
    evidence_ref: Text
    retrieved_at: AwareDatetime


class ProductCatalog(Protocol):
    async def search(self, query: str, limit: int) -> list[ProductEvidence]: ...
    async def get(self, product_id: UUID) -> ProductEvidence | None: ...


class CatalogResult(Contract):
    tool: Literal["search_catalog"] = "search_catalog"
    products: list[ProductEvidence] = Field(max_length=20)


class DetailsResult(Contract):
    tool: Literal["get_product_details"] = "get_product_details"
    product: ProductEvidence


class ComparisonResult(Contract):
    tool: Literal["compare_products"] = "compare_products"
    products: list[ProductEvidence] = Field(
        min_length=MIN_COMPARISON_PRODUCTS, max_length=MAX_SELECTED_PRODUCTS
    )

    @model_validator(mode="after")
    def distinct_products(self) -> "ComparisonResult":
        if len({p.product_id for p in self.products}) != len(self.products):
            raise ValueError("Comparison evidence must cover distinct products")
        return self


class BuildResult(Contract):
    tool: Literal["build_pc"] = "build_pc"
    build_ref: UUID
    receipt: BuildReceipt
    satisfaction: SatisfactionReport | None
    search_evidence: BuildSearchEvidence | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class SearchWebItem(Contract):
    title: Text = Field(max_length=500)
    url: Text = Field(max_length=2000)
    snippet: str = Field(strict=True, max_length=2000)


class SearchWebResult(Contract):
    tool: Literal["search_web"] = "search_web"
    query: Text = Field(max_length=1000)
    results: list[SearchWebItem] = Field(default_factory=list, max_length=10)


class WebFactEvidence(Contract):
    evidence_ref: Text
    fact: ExtractedWebFact

    @model_validator(mode="after")
    def fact_identity(self) -> "WebFactEvidence":
        if self.evidence_ref != "fact:" + canonical_hash(self.fact):
            raise ValueError("Web fact identity does not match source/value")
        return self


class ReadWebPageResult(Contract):
    tool: Literal["read_web_page"] = "read_web_page"
    url: Text = Field(max_length=2000)
    final_url: Text = Field(max_length=2000)
    title: Text | None = Field(default=None, max_length=500)
    status: WebPageReadStatus
    content_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    # Preserve exact source offsets and segment hashes, including whitespace.
    content: (
        Annotated[str, Field(strict=True, min_length=1, max_length=16_000)] | None
    ) = None
    evidence_ref: Text | None = None
    offset: NonNegativeInt = 0
    next_offset: NonNegativeInt | None = None
    total_chars: NonNegativeInt | None = None
    extraction_status: WebExtractionStatus | None = None
    document_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    facts: list[WebFactEvidence] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def fact_boundary(self) -> "ReadWebPageResult":
        if self.extraction_status is None:
            self.extraction_status = (
                WebExtractionStatus.CONTENT_ONLY
                if self.status == WebPageReadStatus.READABLE
                else WebExtractionStatus.FAILED
            )
        if self.extraction_status == WebExtractionStatus.STRUCTURED:
            if not self.facts or self.status != WebPageReadStatus.READABLE:
                raise ValueError("Structured page requires readable facts")
        elif self.facts:
            raise ValueError("Non-structured page cannot contain facts")
        if any(
            f.fact.source_url != self.final_url
            or f.fact.document_hash != self.document_hash
            for f in self.facts
        ):
            raise ValueError("Web fact belongs to a different source document")
        return self


class ResearchSearchStep(Contract):
    tier: SourceTier
    query: Text
    target_domains: list[Text]
    status: ResearchSourceStatus
    result_urls: list[Text] = Field(default_factory=list)


class ResearchSection(Contract):
    subjects: list[Text] = Field(min_length=1, max_length=MAX_RESEARCH_SUBJECTS)
    focus: Text
    query: Text = Field(max_length=2000)
    status: ResearchSourceStatus
    official_source_status: OfficialSourceStatus = OfficialSourceStatus.UNKNOWN
    search_steps: list[ResearchSearchStep] = Field(
        default_factory=list, max_length=MAX_RESEARCH_SOURCE_TIERS
    )
    results: list[SearchWebItem] = Field(
        default_factory=list,
        max_length=MAX_RESEARCH_SEARCH_RESULTS * MAX_RESEARCH_SOURCE_TIERS,
    )
    pages: list[ReadWebPageResult] = Field(
        default_factory=list,
        max_length=MAX_RESEARCH_SOURCES * MAX_RESEARCH_SOURCE_TIERS,
    )

    @model_validator(mode="after")
    def source_status(self) -> "ResearchSection":
        readable = any(page.status == WebPageReadStatus.READABLE for page in self.pages)
        if readable != (self.status == ResearchSourceStatus.READABLE):
            raise ValueError("Research source status does not match its pages")
        if self.status in {
            ResearchSourceStatus.NO_RESULTS,
            ResearchSourceStatus.UNAVAILABLE,
            ResearchSourceStatus.SOURCE_UNRESOLVED,
        } and (self.results or self.pages):
            raise ValueError("Unavailable/empty research has no source results")
        urls = {item.url for item in self.results}
        if len(urls) != len(self.results) or any(
            page.url not in urls for page in self.pages
        ):
            raise ValueError("Research pages must belong to distinct search results")
        return self


class ResearchResult(Contract):
    tool: Literal[
        ToolName.RESEARCH_HARDWARE,
        ToolName.RESEARCH_COMPATIBILITY,
        ToolName.RESEARCH_WORKLOAD,
        ToolName.RESEARCH_PC_BUILD,
        ToolName.RESEARCH_DRIVER,
    ]
    sections: list[ResearchSection] = Field(
        min_length=1, max_length=MAX_BUILD_RESEARCH_PARTS
    )
    build_source_ref: Text | None = None
    source_policy_version: Text
    source_policy_hash: Text = Field(pattern=r"^[0-9a-f]{64}$")
    missing_conditions: list[BenchmarkCondition] = Field(
        default_factory=list, max_length=len(BenchmarkCondition)
    )

    @model_validator(mode="after")
    def build_binding(self) -> "ResearchResult":
        if (self.tool == ToolName.RESEARCH_PC_BUILD) != (
            self.build_source_ref is not None
        ):
            raise ValueError("Build research requires its canonical source reference")
        return self


ToolPayload = Annotated[
    CatalogResult
    | DetailsResult
    | ComparisonResult
    | BuildResult
    | SearchWebResult
    | ReadWebPageResult
    | ResearchResult,
    Field(discriminator="tool"),
]


def result_web_pages(payload: ToolPayload | None) -> list[ReadWebPageResult]:
    if isinstance(payload, ReadWebPageResult):
        return [payload]
    if isinstance(payload, ResearchResult):
        return [page for section in payload.sections for page in section.pages]
    return []


def web_evidence_refs(pages: list[ReadWebPageResult]) -> list[str]:
    return list(
        dict.fromkeys(
            ref
            for page in pages
            for ref in ([page.evidence_ref] if page.evidence_ref else [])
            + [item.evidence_ref for item in page.facts]
        )
    )


class ToolError(StrEnum):
    BAD_ARGUMENTS = "BAD_ARGUMENTS"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    PREREQUISITE_MISSING = "PREREQUISITE_MISSING"
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    CATALOG_UNAVAILABLE = "CATALOG_UNAVAILABLE"
    WEB_SEARCH_UNAVAILABLE = "WEB_SEARCH_UNAVAILABLE"


class ToolCall(Contract):
    call_id: Text = Field(max_length=200)
    name: Text = Field(max_length=100)
    arguments: dict[str, JsonValue]
    # Opaque native Gemini protocol data, not model arguments/business authority.
    provider_signature: Text | None = Field(default=None, max_length=16384)


class ToolResult(Contract):
    schema_version: Literal[1] = 1
    tool_call_id: Text
    tool: Text
    status: Literal["COMPLETED", "REJECTED", "FAILED"]
    input_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    constraints_revision: NonNegativeInt
    result: ToolPayload | None = None
    error: ToolError | None = None
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def outcome_matches_status(self) -> "ToolResult":
        if self.status == "COMPLETED":
            if (
                self.result is None
                or self.error is not None
                or self.result.tool != self.tool
            ):
                raise ValueError("Completed tool outcome is inconsistent")
        elif self.result is not None or self.error is None or self.evidence_refs:
            raise ValueError("Failed/rejected tool has no successful evidence")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Evidence references must be unique")
        if isinstance(
            self.result, ResearchResult
        ) and self.evidence_refs != web_evidence_refs(result_web_pages(self.result)):
            raise ValueError("Research evidence must match its actual read pages")
        return self


class ToolRecord(Contract):
    call: ToolCall
    result: ToolResult | None = None
    # Application provenance, never a model-controlled argument. Missing legacy
    # origin is not silently interpreted as evidence from the current V2 run.
    origin_run_id: UUID | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    model_step: NonNegativeInt | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def matched_pair(self) -> "ToolRecord":
        if self.result and (
            self.call.call_id != self.result.tool_call_id
            or self.call.name != self.result.tool
        ):
            raise ValueError("Tool ledger pair mismatch")
        return self


class SavedBuildEvidence(Contract):
    source_ref: CheckpointRefV1
    snapshot: PublicBuildV1

    @property
    def evidence_ref(self) -> str:
        return f"saved-build:{self.source_ref.thread_id}:{self.source_ref.checkpoint_id}:{canonical_hash(self.snapshot)}"


class CompletionRequirement(Contract):
    kind: Literal["BUILD", "COMPARE", "ANSWER", "EXPLAIN_SAVED_BUILD"]
    constraints_revision: NonNegativeInt | None = None
    product_ids: list[UUID] = Field(
        default_factory=list, max_length=MAX_SELECTED_PRODUCTS
    )
    saved_build_ref: Text | None = None

    @model_validator(mode="after")
    def required_subjects(self) -> "CompletionRequirement":
        if self.kind == "BUILD" and self.constraints_revision is None:
            raise ValueError("Build completion requires trusted constraints revision")
        if self.kind == "EXPLAIN_SAVED_BUILD" and self.saved_build_ref is None:
            raise ValueError(
                "Saved build completion requires an authorized snapshot ref"
            )
        if (
            self.kind == "COMPARE"
            and not MIN_COMPARISON_PRODUCTS
            <= len(set(self.product_ids))
            == len(self.product_ids)
            <= MAX_SELECTED_PRODUCTS
        ):
            raise ValueError("Comparison completion requires 2..5 distinct products")
        return self


class DialogueTurn(Contract):
    user_text: Text
    assistant_text: Text


class ControlRecord(Contract):
    """Checkpointed graph-transition acknowledgement, separate from tool outcomes."""

    call: ToolCall
    code: Text
    model_step: NonNegativeInt


class PendingConversationContext(Contract):
    """Model-facing pending data; never a confirmation or mutation command."""

    question: Text
    proposal: RequirementPatch | None = None
    proposal_sources: list[Text] = Field(default_factory=list, max_length=8)
    resolution: PublicClarificationV1 | None = None
    task_selection: PublicTaskClarificationV1 | None = None


class ToolContext(Contract):
    account_id: UUID
    conversation_id: UUID
    run_id: UUID
    requirements: CanonicalRequirementState
    memories: list[MemoryEntry] = Field(
        default_factory=list,
        max_length=MAX_MEMORIES,
        exclude_if=lambda value: not value,
    )
    catalog: list[CatalogRowV1] = Field(
        default_factory=list, max_length=MAX_HARDWARE_CATALOG_ROWS
    )
    read_catalog_allowed: bool
    execution_allowed: bool
    catalog_ready: bool = True
    saved_build: SavedBuildEvidence | None = None
    history: list[DialogueTurn] = Field(
        default_factory=list,
        max_length=MAX_DIALOGUE_TURNS,
        exclude_if=lambda value: not value,
    )
    task_message: Text | None = Field(default=None, max_length=MAX_TURN_MESSAGE_CHARS)
    requested_tasks: TaskDraftBatch | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    task_registration_open: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    completion_requirements: list[CompletionRequirement] = Field(
        default_factory=list, max_length=8
    )
    pending: PendingConversationContext | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    control_history: list[ControlRecord] = Field(
        default_factory=list,
        max_length=100,
        exclude_if=lambda value: not value,
    )

    @model_validator(mode="after")
    def saved_snapshot_scope(self) -> "ToolContext":
        if (
            sum(len(turn.user_text) + len(turn.assistant_text) for turn in self.history)
            > MAX_DIALOGUE_CHARS
        ):
            raise ValueError("Dialogue context exceeds its bound")
        if self.saved_build and (
            self.saved_build.source_ref.thread_id != self.conversation_id
            or self.saved_build.source_ref.checkpoint_ns
        ):
            raise ValueError("Saved build belongs to another conversation or namespace")
        return self


class WebExcerpt(Contract):
    evidence_ref: Text
    quote: Annotated[str, Field(strict=True, min_length=1, max_length=2000)]


class ProductReview(Contract):
    product_name: Text = Field(max_length=200)
    summary: Text = Field(max_length=2000)
    pros: list[Text] = Field(default_factory=list, max_length=20)
    cons: list[Text] = Field(default_factory=list, max_length=20)
    verdict: Text = Field(max_length=2000)


class ConsultationExplanation(Contract):
    """Evidence-linked interpretation, not a new machine-verified fact source."""

    text: Text = Field(max_length=3000)
    evidence_refs: list[Text] = Field(min_length=1, max_length=20)


class AnswerCandidate(Contract):
    schema_version: Literal[1] = 1
    text: Text = Field(max_length=8000)
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)
    build_ref: UUID | None = None
    clarification_fields: list[Text] = Field(default_factory=list, max_length=100)
    web_excerpts: list[WebExcerpt] = Field(default_factory=list, max_length=8)
    review: ProductReview | None = None
    explanation: ConsultationExplanation | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ConversationResponse(Contract):
    """Response to this turn, independent of business-task completion."""

    text: Text = Field(max_length=8000)
    evidence_refs: list[Text] = Field(default_factory=list, max_length=100)


class ValidatedAnswer(Contract):
    text: Text
    status: Literal["COMPLETED", "WAITING_INPUT"]
    evidence_refs: list[Text]
    build_ref: UUID | None = None
