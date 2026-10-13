"""Build application stages; no graph, provider, HTTP or publication side effects."""

from typing import Literal, Protocol

from pydantic import Field

from ai_service.capabilities.assistant.conversation_core import Digest
from ai_service.capabilities.assistant.provenance import CatalogRowV1, canonical_hash
from ai_service.capabilities.assistant.public_contracts import PublicClarificationV1
from ai_service.capabilities.assistant.stateful_contracts import Contract
from ai_service.capabilities.pc_builder.application.service import (
    PCBuildApplicationService,
)
from ai_service.capabilities.pc_builder.errors import NoFeasibleBuildError
from ai_service.capabilities.pc_builder.schemas import (
    OptimizerSearchTrace,
    RankedBuild,
    UseCaseProfile,
)

from .compiler import Compilation, compile_requirements
from .contracts import CanonicalRequirementState
from .resolver import ChoiceDraft, Draft
from .verifier import SatisfactionReport, verify_satisfaction


class RequirementExtractor(Protocol):
    async def extract(
        self, message: str, state: CanonicalRequirementState
    ) -> Draft: ...


class RequirementCatalog(Protocol):
    async def load(
        self, draft: Draft, state: CanonicalRequirementState
    ) -> list[CatalogRowV1]: ...


class ClarificationExtractor(Protocol):
    async def choose(
        self, message: str, options: PublicClarificationV1
    ) -> ChoiceDraft: ...


class BuildReceipt(Contract):
    compilation: Compilation
    outcome: Literal["BUILT", "INFEASIBLE", "CLARIFICATION", "UNSUPPORTED", "CONFLICT"]
    build: RankedBuild | None = None
    observed_use_case: UseCaseProfile | None = None
    search_trace: OptimizerSearchTrace | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    search_policy_hash: Digest | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class BuildExplainer(Protocol):
    async def explain(self, receipt: BuildReceipt) -> str: ...


def optimize_requirements(
    state: CanonicalRequirementState,
    catalog: list[CatalogRowV1],
    service: PCBuildApplicationService,
) -> BuildReceipt:
    compiled = compile_requirements(state, catalog)
    normalized = compiled.optimizer_input
    if not compiled.ready or normalized is None:
        statuses = {m.status for m in compiled.mapping}
        outcome: Literal["CLARIFICATION", "UNSUPPORTED", "CONFLICT"] = (
            "CONFLICT"
            if "CONFLICT" in statuses
            else "UNSUPPORTED"
            if "UNSUPPORTED" in statuses
            else "CLARIFICATION"
        )
        return BuildReceipt(compilation=compiled, outcome=outcome)
    policy_hash = canonical_hash(service.configuration_snapshot())
    try:
        result = service.build_pc(normalized.constraints, normalized.candidates)
    except NoFeasibleBuildError as exc:
        return BuildReceipt(
            compilation=compiled,
            outcome="INFEASIBLE",
            observed_use_case=normalized.constraints.use_case.value,
            search_trace=exc.result.search_trace,
            search_policy_hash=policy_hash,
        )
    return BuildReceipt(
        compilation=compiled,
        outcome="BUILT",
        build=result.recommendation.build,
        observed_use_case=result.optimization.use_case,
        search_trace=result.optimization.search_trace,
        search_policy_hash=policy_hash,
    )


def verify_receipt(
    state: CanonicalRequirementState, receipt: BuildReceipt, catalog: list[CatalogRowV1]
) -> SatisfactionReport | None:
    if receipt.outcome != "BUILT":
        return None
    if receipt.build is None or receipt.observed_use_case is None:
        raise ValueError("Completed optimizer result is incomplete")
    report = verify_satisfaction(
        state,
        receipt.compilation,
        receipt.build,
        catalog,
        observed_use_case=receipt.observed_use_case,
    )
    if not report.hard_satisfied:
        raise ValueError("Optimizer output failed canonical requirement satisfaction")
    return report


def grounded_summary(receipt: BuildReceipt, report: SatisfactionReport | None) -> str:
    if receipt.outcome != "BUILT":
        reasons = sorted(
            {m.reason for m in receipt.compilation.mapping if m.status != "APPLIED"}
        )
        return f"{receipt.outcome}: " + (
            ", ".join(reasons) or "Không tìm được cấu hình thỏa các yêu cầu đã hỗ trợ."
        )
    if report is None or not report.hard_satisfied or receipt.build is None:
        raise ValueError("Cannot explain an unverified build")
    lines = [f"Tổng chi phí linh kiện: {receipt.build.total_price:,} VND."]
    for slot, part in sorted(receipt.build.parts.items()):
        capacity = (
            f"; dung lượng {part.capacity_gb} GB"
            if part.capacity_gb is not None
            else ""
        )
        lines.append(
            f"{slot.value}: {part.name}{capacity}; chi phí {receipt.build.spending_prices[slot]:,} VND."
        )
    return "\n".join(lines)
