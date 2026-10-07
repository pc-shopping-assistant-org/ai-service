"""Application service coordinating the PC Builder pipeline.

Connects constraints, catalog, deterministic combinatorial optimizer, and recommendation policy.
Pure business orchestration without LLM or UI coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.application.recommendation_policy import (
    RecommendationDecision,
    choose_recommended_build,
)
from ai_service.capabilities.pc_builder.optimizer import DeterministicPCOptimizer
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    OptimizationResult,
    PCBuildConstraints,
)


@dataclass(frozen=True, slots=True)
class BuildApplicationResult:
    """The complete result returned by the PC Build Application Service."""

    optimization: OptimizationResult
    recommendation: RecommendationDecision


class PCBuildApplicationService:
    """Application-level service for executing PC build recommendations."""

    def __init__(self, optimizer: DeterministicPCOptimizer | None = None) -> None:
        self._optimizer = optimizer or DeterministicPCOptimizer()

    def build_pc(
        self,
        constraints: PCBuildConstraints,
        catalog: list[ComponentSpec],
        requested_objective: BuildObjective | None = None,
    ) -> BuildApplicationResult:
        """Execute deterministic optimization and select the recommended build by policy."""
        optimization = self._optimizer.optimize(
            constraints=constraints,
            catalog=catalog,
        )

        recommendation = choose_recommended_build(
            result=optimization,
            requested_objective=requested_objective,
        )

        return BuildApplicationResult(
            optimization=optimization,
            recommendation=recommendation,
        )
