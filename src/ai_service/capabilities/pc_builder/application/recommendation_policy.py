"""Application-level policy for selecting a recommended PC build among Pareto optima.

Pure application logic, versioned and isolated from the core combinatorial optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ai_service.capabilities.pc_builder.schemas import (
    BuildObjective,
    OptimizationResult,
    RankedBuild,
    UseCaseProfile,
)

RECOMMENDATION_POLICY_VERSION = "v1"


class RecommendationPolicyError(RuntimeError):
    """Raised when policy evaluation fails due to missing or invalid policy configuration."""


class RecommendationReason(StrEnum):
    """Reason code explaining why a particular build was selected as the recommendation."""

    USER_SELECTED = "USER_SELECTED_OBJECTIVE"
    PROFILE_DEFAULT = "PROFILE_DEFAULT_OBJECTIVE"
    DEFAULT_UNAVAILABLE = "DEFAULT_OBJECTIVE_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class RecommendationDecision:
    """The chosen recommendation decision with provenance and policy version."""

    objective: BuildObjective
    build: RankedBuild
    reason_code: RecommendationReason
    policy_version: str


FALLBACK_OBJECTIVE_ORDER: tuple[BuildObjective, ...] = (
    BuildObjective.BALANCED,
    BuildObjective.PERFORMANCE,
    BuildObjective.UPGRADE_FRIENDLY,
)


DEFAULT_OBJECTIVE_BY_PROFILE: dict[UseCaseProfile, BuildObjective] = {
    UseCaseProfile.GAMING_1080P: BuildObjective.BALANCED,
    UseCaseProfile.GAMING_1440P: BuildObjective.BALANCED,
    UseCaseProfile.GAMING_4K: BuildObjective.PERFORMANCE,
    UseCaseProfile.CONTENT_CREATION_3D: BuildObjective.BALANCED,
    UseCaseProfile.AI_DATA_SCIENCE: BuildObjective.PERFORMANCE,
    UseCaseProfile.OFFICE_BUDGET: BuildObjective.BALANCED,
}


def choose_recommended_build(
    result: OptimizationResult,
    requested_objective: BuildObjective | None = None,
) -> RecommendationDecision:
    """Choose the recommended build from an OptimizationResult according to policy v1.

    Rules:
    1. If the user explicitly requested an objective, select it. If unavailable, raise ValueError.
    2. Otherwise, look up the profile's configured default objective.
    3. If the default objective is not feasible in the result, fall back deterministically in order:
       FALLBACK_OBJECTIVE_ORDER (BALANCED -> PERFORMANCE -> UPGRADE_FRIENDLY).
    4. If no feasible builds exist at all, raise ValueError.
    """
    if requested_objective is not None:
        build = result.builds.get(requested_objective)
        if build is None:
            raise ValueError(
                f"No feasible build for requested objective '{requested_objective.value}'"
            )
        return RecommendationDecision(
            objective=requested_objective,
            build=build,
            reason_code=RecommendationReason.USER_SELECTED,
            policy_version=RECOMMENDATION_POLICY_VERSION,
        )

    default_objective = DEFAULT_OBJECTIVE_BY_PROFILE.get(result.use_case)
    if default_objective is None:
        raise RecommendationPolicyError(
            f"No recommendation policy configured for use-case '{result.use_case.value}' "
            f"(policy={RECOMMENDATION_POLICY_VERSION})"
        )

    build = result.builds.get(default_objective)
    if build is not None:
        return RecommendationDecision(
            objective=default_objective,
            build=build,
            reason_code=RecommendationReason.PROFILE_DEFAULT,
            policy_version=RECOMMENDATION_POLICY_VERSION,
        )

    # Deterministic fallback order — never depends on dict iteration order
    for candidate_objective in FALLBACK_OBJECTIVE_ORDER:
        candidate = result.builds.get(candidate_objective)
        if candidate is not None:
            return RecommendationDecision(
                objective=candidate_objective,
                build=candidate,
                reason_code=RecommendationReason.DEFAULT_UNAVAILABLE,
                policy_version=RECOMMENDATION_POLICY_VERSION,
            )

    raise ValueError("Optimization result contains no feasible builds.")
