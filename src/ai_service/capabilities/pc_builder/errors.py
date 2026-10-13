"""Typed optimizer domain failures, distinct from validation/programming errors."""

from ai_service.capabilities.pc_builder.schemas import OptimizationResult


class CaseSupportMetadataError(ValueError):
    """Search could not prove feasibility because case support is missing."""

    def __init__(self) -> None:
        super().__init__("CASE_SUPPORT_METADATA_REQUIRED")


class NoFeasibleBuildError(ValueError):
    """Search returned no rankable build; diagnostics are retained.

    ValueError inheritance preserves existing library callers. Graph nodes must
    catch this specific type, never classify all ValueErrors as INFEASIBLE. This
    exception alone is not proof: pruning/unknown metadata/scoring may limit search.
    """

    def __init__(self, result: OptimizationResult) -> None:
        self.result = result
        super().__init__(
            f"No compatible build configuration found within budget {result.target_budget_vnd:,} VND."
        )
