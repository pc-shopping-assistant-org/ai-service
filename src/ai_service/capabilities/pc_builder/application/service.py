"""Application service coordinating the PC Builder pipeline.

Connects constraints, catalog, deterministic combinatorial optimizer, and recommendation policy.
Pure business orchestration without LLM or UI coupling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic import JsonValue, TypeAdapter

from ai_service.application.ports.hardware import ComponentSpec
from ai_service.capabilities.pc_builder.application.recommendation_policy import (
    DEFAULT_OBJECTIVE_BY_PROFILE,
    FALLBACK_OBJECTIVE_ORDER,
    RECOMMENDATION_POLICY_VERSION,
    RecommendationDecision,
    choose_recommended_build,
)
from ai_service.capabilities.pc_builder.optimizer import (
    FIXED_CATEGORY_ORDER,
    POWER_SIZING_POLICY_VERSION,
    SCORING_POLICIES,
    DeterministicPCOptimizer,
)
from ai_service.capabilities.pc_builder.pruning import DEFAULT_ENVELOPES
from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    OptimizationResult,
    OptimizerSearchTrace,
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

    def configuration_snapshot(self) -> JsonValue:
        """Capture live tunable policy for reuse hashes, not a full replay artifact.

        Algorithm/calibration code changes also require an implementation-version
        change. These captured knobs are the same ones used by build_pc.
        """
        return TypeAdapter(JsonValue).validate_python(
            {
                "enable_pruning": self._optimizer.enable_pruning,
                "algorithm_version": OptimizerSearchTrace.model_fields["algorithm_version"].default,
                "pruning_version": "compatibility-aware-pareto-v2",
                "case_compatibility_version": "explicit-support-v1",
                "power_sizing_version": POWER_SIZING_POLICY_VERSION,
                "platform_policy": dict(self._optimizer.platform_policy),
                "scoring": [
                    {
                        "profile": profile.value,
                        "objective": objective.value,
                        "weights": dict(weights),
                    }
                    for (profile, objective), weights in sorted(
                        SCORING_POLICIES.items()
                    )
                ],
                "envelopes": {
                    profile.value: {
                        category.value: asdict(envelope)
                        for category, envelope in categories.items()
                    }
                    for profile, categories in DEFAULT_ENVELOPES.items()
                },
                "category_order": [category.value for category in FIXED_CATEGORY_ORDER],
                "recommendation_version": RECOMMENDATION_POLICY_VERSION,
                "default_objectives": {
                    profile.value: objective.value
                    for profile, objective in DEFAULT_OBJECTIVE_BY_PROFILE.items()
                },
                "fallback_objectives": [
                    objective.value for objective in FALLBACK_OBJECTIVE_ORDER
                ],
            }
        )

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
